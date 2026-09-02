"""A deterministic, iterator-backed training engine for real experiments.

This module defines the execution path but never launches work on import.  A
caller must explicitly construct :class:`ExperimentTrainer` and call ``run``.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import torch
from torch import Tensor, nn

from spalmer.modeling import CausalLMOutput, SPALMERCausalLM
from spalmer.training.config import TrainingConfig
from spalmer.training.device import RuntimeDevice, resolve_runtime_device, seed_everything
from spalmer.training.optim import OptimizerBundle, build_optimizers, gradients_are_finite


@dataclass(slots=True)
class CausalBatch:
    """One host-backed next-token batch.

    ``input_ids`` includes both context and final target, so a configured
    sequence length of N expects shape ``[batch, N + 1]`` and contributes N
    target tokens per row.  The engine moves this object to the accelerator;
    the complete corpus remains CPU/mmap backed. ``token_utf8_bytes`` optionally
    assigns an exact original-byte contribution to every input token; an empty
    tuple means BPB is unavailable rather than estimated.
    """

    input_ids: Tensor
    labels: Tensor | None = None
    attention_mask: Tensor | None = None
    state_reset_mask: Tensor | None = None
    strata: tuple[str, ...] = ()
    token_utf8_bytes: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2 or self.input_ids.shape[1] < 2:
            raise ValueError("input_ids must have shape [batch, sequence + 1]")
        labels = self.input_ids if self.labels is None else self.labels
        if labels.shape != self.input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")
        for name, mask in (
            ("attention_mask", self.attention_mask),
            ("state_reset_mask", self.state_reset_mask),
        ):
            if mask is not None and mask.shape != self.input_ids.shape:
                raise ValueError(f"{name} must have the same shape as input_ids")
        batch_size = self.input_ids.shape[0]
        if self.strata and len(self.strata) != batch_size:
            raise ValueError("strata must contain one entry per batch row")
        if self.token_utf8_bytes:
            if len(self.token_utf8_bytes) != batch_size:
                raise ValueError("token_utf8_bytes must contain one entry per batch row")
            width = self.input_ids.shape[1]
            if any(len(row) != width for row in self.token_utf8_bytes):
                raise ValueError("token_utf8_bytes rows must match input_ids width")
            if any(value < 0 for row in self.token_utf8_bytes for value in row):
                raise ValueError("token_utf8_bytes cannot contain negative counts")

    @property
    def resolved_labels(self) -> Tensor:
        return self.input_ids if self.labels is None else self.labels

    @property
    def target_tokens(self) -> int:
        return int((self.resolved_labels[:, 1:] != -100).sum())

    def to(self, device: torch.device) -> CausalBatch:
        return CausalBatch(
            input_ids=self.input_ids.to(device=device, dtype=torch.long, non_blocking=True),
            labels=self.resolved_labels.to(device=device, dtype=torch.long, non_blocking=True),
            attention_mask=_move_optional(self.attention_mask, device),
            state_reset_mask=_move_optional(self.state_reset_mask, device),
            strata=self.strata,
            token_utf8_bytes=self.token_utf8_bytes,
        )


class StatefulBatchSource(Protocol):
    """Minimal contract implemented by deterministic mmap samplers."""

    def next_batch(self, *, batch_size: int, sequence_length: int) -> CausalBatch: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


@dataclass(frozen=True, slots=True)
class TrainStepMetrics:
    step: int
    tokens_seen: int
    objective: float
    model_loss: float
    auxiliary_loss: float | None
    surprise_calibration_loss: float | None
    predictive_entropy: float
    learning_rate: float
    gradient_norm: float | None
    elapsed_seconds: float
    promoted_experts: tuple[int, ...]
    average_surprise: float


@dataclass(slots=True)
class TrainerProgress:
    completed_steps: int = 0
    tokens_seen: int = 0
    started_at: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "completed_steps": self.completed_steps,
            "tokens_seen": self.tokens_seen,
            "started_at": self.started_at,
        }


CheckpointCallback = Callable[["ExperimentTrainer", TrainStepMetrics], None]
ValidationCallback = Callable[[SPALMERCausalLM, int, int], Mapping[str, float]]
TelemetryCallback = Callable[[TrainStepMetrics], None]
ModelFactory = TypeVar("ModelFactory", bound=Callable[[], SPALMERCausalLM])


def initialize_model(factory: ModelFactory, config: TrainingConfig) -> SPALMERCausalLM:
    """Seed before model construction, then establish FP32 persistent weights."""

    seed_everything(config.seed, deterministic_algorithms=config.deterministic_algorithms)
    model = factory()
    return model.to(dtype=torch.float32)


class ExperimentTrainer:
    """GPU-first optimizer loop over a resumable batch source."""

    def __init__(
        self,
        model: SPALMERCausalLM,
        batches: StatefulBatchSource,
        config: TrainingConfig,
        *,
        optimizer: OptimizerBundle | None = None,
        on_checkpoint: CheckpointCallback | None = None,
        on_validation: ValidationCallback | None = None,
        on_telemetry: TelemetryCallback | None = None,
    ) -> None:
        self.config = config
        self.runtime = resolve_runtime_device(config)
        seed_everything(
            config.seed,
            deterministic_algorithms=config.deterministic_algorithms,
        )
        self.model = model.to(device=self.runtime.device, dtype=torch.float32)
        self.batches = batches
        self.optimizer = optimizer or build_optimizers(self.model, config)
        self.on_checkpoint = on_checkpoint
        self.on_validation = on_validation
        self.on_telemetry = on_telemetry
        self.progress = TrainerProgress()
        if config.activation_checkpointing:
            enable = getattr(self.model, "enable_activation_checkpointing", None)
            if not callable(enable):
                raise RuntimeError(
                    "activation checkpointing was requested but the model exposes no hook"
                )
            enable()

    def run(self) -> Iterator[TrainStepMetrics]:
        """Execute configured optimization steps and yield bounded telemetry."""

        if self.progress.started_at == 0:
            self.progress.started_at = time.time()
        self.model.train()
        while self.progress.completed_steps < self.config.max_steps:
            yield self._optimizer_step()

    def _optimizer_step(self) -> TrainStepMetrics:
        step_index = self.progress.completed_steps
        rate = self.config.learning_rate * self.config.learning_rate_multiplier(step_index)
        self.optimizer.set_learning_rate(rate)
        self.optimizer.zero_grad(set_to_none=True)
        outputs: list[CausalLMOutput] = []
        objectives: list[float] = []
        targets = 0
        for _ in range(self.config.gradient_accumulation_steps):
            host_batch = self.batches.next_batch(
                batch_size=self.config.micro_batch_size,
                sequence_length=self.config.sequence_length,
            )
            batch = host_batch.to(self.runtime.device)
            targets += batch.target_tokens
            with _autocast(self.runtime):
                output = self.model(
                    batch.input_ids,
                    labels=batch.resolved_labels,
                    attention_mask=batch.attention_mask,
                    state_reset_mask=batch.state_reset_mask,
                )
                objective = _objective(output, self.config)
                scaled = objective / self.config.gradient_accumulation_steps
            scaled.backward()
            outputs.append(output)
            objectives.append(float(objective.detach()))
            if output.token_nll is not None:
                self.model.observe_surprise(
                    output.token_nll,
                    batch.resolved_labels[:, 1:] != -100,
                )

        parameters = tuple(
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        )
        if not gradients_are_finite(parameters):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"non-finite gradient at optimizer step {step_index + 1}")
        gradient_norm = _clip_gradients(parameters, self.config.gradient_clip)
        self.optimizer.step()

        last = outputs[-1]
        promoted = self.model.update_potentiation(last.layer_metrics)
        self.progress.completed_steps += 1
        self.progress.tokens_seen += targets
        metrics = TrainStepMetrics(
            step=self.progress.completed_steps,
            tokens_seen=self.progress.tokens_seen,
            objective=sum(objectives) / len(objectives),
            model_loss=_mean_optional(outputs, "loss", required=True),
            auxiliary_loss=_mean_optional(outputs, "auxiliary_loss"),
            surprise_calibration_loss=_mean_optional(outputs, "surprise_calibration_loss"),
            predictive_entropy=_mean_optional(outputs, "predictive_entropy", required=True),
            learning_rate=rate,
            gradient_norm=gradient_norm,
            elapsed_seconds=time.time() - self.progress.started_at,
            promoted_experts=promoted,
            average_surprise=self.model.average_surprise,
        )
        self._dispatch_callbacks(metrics)
        return metrics

    def _dispatch_callbacks(self, metrics: TrainStepMetrics) -> None:
        step = metrics.step
        if self.on_telemetry is not None and (
            step == 1 or step % self.config.telemetry_interval_steps == 0
        ):
            self.on_telemetry(metrics)
        if self.on_validation is not None and step % self.config.validation_interval_steps == 0:
            was_training = self.model.training
            try:
                self.model.eval()
                self.on_validation(self.model, step, self.progress.tokens_seen)
            finally:
                self.model.train(was_training)
        if self.on_checkpoint is not None and step % self.config.checkpoint_interval_steps == 0:
            self.on_checkpoint(self, metrics)

    def state_dict(self) -> dict[str, Any]:
        """Return mutable runtime state for the versioned experiment envelope."""

        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "progress": self.progress.to_dict(),
            "batch_source": dict(self.batches.state_dict()),
            "training_config": self.config.to_dict(),
        }


def _objective(output: CausalLMOutput, config: TrainingConfig) -> Tensor:
    if output.loss is None:
        raise RuntimeError("model did not return next-token loss")
    value = output.loss
    if output.auxiliary_loss is not None:
        value = value + config.auxiliary_loss_weight * output.auxiliary_loss
    if output.surprise_calibration_loss is not None:
        value = value + config.surprise_calibration_weight * output.surprise_calibration_loss
    return value


def _autocast(runtime: RuntimeDevice):
    if not runtime.autocast_enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=runtime.device.type, dtype=runtime.compute_dtype)


def _move_optional(value: Tensor | None, device: torch.device) -> Tensor | None:
    if value is None:
        return None
    return value.to(device=device, non_blocking=True)


def _mean_optional(
    outputs: list[CausalLMOutput],
    attribute: str,
    *,
    required: bool = False,
) -> float | None:
    values = [getattr(output, attribute) for output in outputs]
    present = [value for value in values if isinstance(value, Tensor)]
    if not present:
        if required:
            raise RuntimeError(f"model did not return required {attribute}")
        return None
    return sum(float(value.detach().float().mean()) for value in present) / len(present)


def _clip_gradients(parameters: tuple[nn.Parameter, ...], maximum: float | None) -> float | None:
    if maximum is None:
        return None
    dense = [
        parameter
        for parameter in parameters
        if parameter.grad is not None and not parameter.grad.is_sparse
    ]
    sparse: list[Tensor] = []
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None or not gradient.is_sparse:
            continue
        parameter.grad = gradient.coalesce()
        sparse.append(parameter.grad.values())
    squared = torch.zeros((), device=parameters[0].device, dtype=torch.float32)
    for parameter in dense:
        squared.add_(parameter.grad.detach().float().square().sum())
    for values in sparse:
        squared.add_(values.detach().float().square().sum())
    norm = squared.sqrt()
    scale = (maximum / (norm + 1e-6)).clamp(max=1.0)
    for parameter in dense:
        parameter.grad.mul_(scale.to(parameter.grad.dtype))
    for values in sparse:
        values.mul_(scale.to(values.dtype))
    return float(norm)


__all__ = [
    "CausalBatch",
    "ExperimentTrainer",
    "StatefulBatchSource",
    "TrainStepMetrics",
    "TrainerProgress",
    "initialize_model",
]
