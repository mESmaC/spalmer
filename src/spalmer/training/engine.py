"""A deterministic, iterator-backed training engine for real experiments.

This module defines the execution path but never launches work on import.  A
caller must explicitly construct :class:`ExperimentTrainer` and call ``run``.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import torch
from torch import Tensor, nn

from spalmer.modeling import CausalLMOutput, SPALMERCausalLM
from spalmer.training.config import TrainingConfig
from spalmer.training.device import RuntimeDevice, resolve_runtime_device, seed_everything
from spalmer.training.optim import (
    BF16MasterAdamW,
    OptimizerBundle,
    build_optimizers,
    gradients_are_finite,
)
from spalmer.training.recurrence import RecurrenceSampler


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
    max_expert_group_load: int | None = None
    max_expert_group_padding_amplification: float | None = None
    max_expert_group_global_max_counterfactual_padding_amplification: float | None = None
    recurrence_steps: int | None = None
    backprop_steps: int | None = None
    effective_depth: int | None = None
    latent_delta_final: float | None = None
    latent_position_correlation: float | None = None


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


@dataclass(slots=True)
class _LayerPotentiationTotals:
    """Small detached sufficient statistics for one routed-expert layer."""

    responsibility: Tensor
    weighted_nll: Tensor
    quantization_error: Tensor
    quantization_observations: Tensor


class _PotentiationAccumulator:
    """Combine microbatch telemetry into one optimizer-step observation.

    The model reports utilization normalized by the number of supervised
    targets in each microbatch and an NLL mean for each expert. Multiplying
    those values back into responsibility counts makes aggregation independent
    of microbatch size and label masking. Quantization error is deterministic
    for an expert between optimizer updates, so averaging its non-zero
    observations preserves the existing "selected this pass" semantics.
    """

    _FIELDS = (
        "potentiation_utilization",
        "expert_attributed_nll",
        "expert_quantization_error",
    )

    def __init__(self) -> None:
        self._layers: list[_LayerPotentiationTotals | None] | None = None
        self._availability: tuple[bool, ...] | None = None
        self._target_tokens = 0

    def add(self, layer_metrics: Sequence[Mapping[str, Any]], *, target_tokens: int) -> None:
        if target_tokens < 0:
            raise ValueError("target_tokens cannot be negative")
        availability: list[bool] = []
        values_by_layer: list[tuple[Tensor, Tensor, Tensor] | None] = []
        for layer_index, metrics in enumerate(layer_metrics):
            values = tuple(metrics.get(name) for name in self._FIELDS)
            present = tuple(isinstance(value, Tensor) for value in values)
            if any(present) and not all(present):
                missing = [name for name, exists in zip(self._FIELDS, present) if not exists]
                raise RuntimeError(
                    f"layer {layer_index} has incomplete potentiation telemetry: {missing}"
                )
            complete = all(present)
            availability.append(complete)
            if complete:
                utilization, attributed_nll, quantization_error = values
                assert isinstance(utilization, Tensor)
                assert isinstance(attributed_nll, Tensor)
                assert isinstance(quantization_error, Tensor)
                values_by_layer.append((utilization, attributed_nll, quantization_error))
            else:
                values_by_layer.append(None)

        observed_availability = tuple(availability)
        if self._layers is None:
            self._availability = observed_availability
            self._layers = [None] * len(layer_metrics)
        elif len(layer_metrics) != len(self._layers):
            raise RuntimeError(
                "layer count changed between gradient-accumulation microbatches: "
                f"{len(self._layers)} != {len(layer_metrics)}"
            )
        elif observed_availability != self._availability:
            raise RuntimeError(
                "potentiation telemetry availability changed between "
                "gradient-accumulation microbatches"
            )

        self._target_tokens += target_tokens
        assert self._layers is not None
        for layer_index, values in enumerate(values_by_layer):
            if values is None:
                continue
            utilization, attributed_nll, quantization_error = (
                value.detach().float() for value in values
            )
            if not (
                utilization.shape == attributed_nll.shape == quantization_error.shape
            ):
                raise RuntimeError(
                    f"layer {layer_index} potentiation vectors must have one common shape"
                )
            responsibility = utilization * float(target_tokens)
            weighted_nll = attributed_nll * responsibility
            quantization_observations = (quantization_error > 0).to(torch.long)
            totals = self._layers[layer_index]
            if totals is None:
                self._layers[layer_index] = _LayerPotentiationTotals(
                    responsibility=responsibility.clone(),
                    weighted_nll=weighted_nll.clone(),
                    quantization_error=quantization_error.clone(),
                    quantization_observations=quantization_observations,
                )
                continue
            for name, value in (
                ("responsibility", responsibility),
                ("weighted_nll", weighted_nll),
                ("quantization_error", quantization_error),
                ("quantization_observations", quantization_observations),
            ):
                current = getattr(totals, name)
                if current.shape != value.shape or current.device != value.device:
                    raise RuntimeError(
                        f"layer {layer_index} {name} changed shape or device between microbatches"
                    )
                current.add_(value)

    def finish(self) -> tuple[Mapping[str, Any], ...]:
        if self._layers is None:
            return ()
        aggregated: list[Mapping[str, Any]] = []
        target_denominator = float(max(self._target_tokens, 1))
        for totals in self._layers:
            if totals is None:
                aggregated.append({})
                continue
            responsibility = totals.responsibility
            utilization = responsibility / target_denominator
            attributed_nll = torch.where(
                responsibility > 0,
                totals.weighted_nll / responsibility.clamp_min(1e-8),
                torch.zeros_like(totals.weighted_nll),
            )
            quantization_error = torch.where(
                totals.quantization_observations > 0,
                totals.quantization_error
                / totals.quantization_observations.clamp_min(1).to(
                    totals.quantization_error.dtype
                ),
                torch.zeros_like(totals.quantization_error),
            )
            aggregated.append(
                {
                    "potentiation_utilization": utilization,
                    "expert_attributed_nll": attributed_nll,
                    "expert_quantization_error": quantization_error,
                }
            )
        return tuple(aggregated)


class _ExecutionTelemetryAccumulator:
    """Keep only optimizer-step maxima from per-layer execution telemetry."""

    def __init__(self) -> None:
        self._max_group_load: Tensor | None = None
        self._max_padding_amplification: Tensor | None = None
        self._max_global_max_counterfactual_amplification: Tensor | None = None

    def add(self, layer_metrics: Sequence[Mapping[str, Any]]) -> None:
        for layer_index, metrics in enumerate(layer_metrics):
            group_load = metrics.get("expert_group_max_load")
            amplification = metrics.get("expert_group_padding_amplification")
            counterfactual_amplification = metrics.get(
                "expert_group_global_max_counterfactual_padding_amplification"
            )
            present = (
                isinstance(group_load, Tensor),
                isinstance(amplification, Tensor),
                isinstance(counterfactual_amplification, Tensor),
            )
            if any(present) and not all(present):
                raise RuntimeError(
                    f"layer {layer_index} has incomplete expert execution telemetry"
                )
            if not all(present):
                continue
            assert isinstance(group_load, Tensor)
            assert isinstance(amplification, Tensor)
            assert isinstance(counterfactual_amplification, Tensor)
            if (
                group_load.numel() != 1
                or amplification.numel() != 1
                or counterfactual_amplification.numel() != 1
            ):
                raise RuntimeError("expert execution telemetry values must be scalars")
            detached_load = group_load.detach().to(dtype=torch.long)
            detached_amplification = amplification.detach().float()
            detached_counterfactual = counterfactual_amplification.detach().float()
            if self._max_group_load is None:
                self._max_group_load = detached_load.clone()
                self._max_padding_amplification = detached_amplification.clone()
                self._max_global_max_counterfactual_amplification = (
                    detached_counterfactual.clone()
                )
                continue
            assert self._max_padding_amplification is not None
            assert self._max_global_max_counterfactual_amplification is not None
            if (
                detached_load.device != self._max_group_load.device
                or detached_amplification.device != self._max_padding_amplification.device
                or detached_counterfactual.device
                != self._max_global_max_counterfactual_amplification.device
            ):
                raise RuntimeError(
                    "expert execution telemetry changed device between microbatches"
                )
            self._max_group_load = torch.maximum(self._max_group_load, detached_load)
            self._max_padding_amplification = torch.maximum(
                self._max_padding_amplification,
                detached_amplification,
            )
            self._max_global_max_counterfactual_amplification = torch.maximum(
                self._max_global_max_counterfactual_amplification,
                detached_counterfactual,
            )

    def finish(self) -> tuple[int | None, float | None, float | None]:
        if self._max_group_load is None:
            return None, None, None
        assert self._max_padding_amplification is not None
        assert self._max_global_max_counterfactual_amplification is not None
        return (
            int(self._max_group_load),
            float(self._max_padding_amplification),
            float(self._max_global_max_counterfactual_amplification),
        )


CheckpointCallback = Callable[["ExperimentTrainer", TrainStepMetrics], None]
ValidationCallback = Callable[[SPALMERCausalLM, int, int], Mapping[str, float]]
TelemetryCallback = Callable[[TrainStepMetrics], None]
ModelFactory = TypeVar("ModelFactory", bound=Callable[[], SPALMERCausalLM])


def initialize_model(factory: ModelFactory, config: TrainingConfig) -> SPALMERCausalLM:
    """Seed construction, then establish the configured persistent master dtype."""

    seed_everything(config.seed, deterministic_algorithms=config.deterministic_algorithms)
    model = factory()
    model = model.to(dtype=_parameter_dtype(config))
    _validate_expert_master_dtype(model, config)
    return model


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
        self.model = model.to(
            device=self.runtime.device,
            dtype=_parameter_dtype(config),
        )
        _validate_expert_master_dtype(self.model, config)
        self.recurrence = RecurrenceSampler.from_training_config(config)
        _validate_recurrence_contract(self.model, self.recurrence)
        self.batches = batches
        self.optimizer = optimizer or build_optimizers(self.model, config)
        _validate_optimizer_precision_contract(self.optimizer, config)
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
        # One draw per optimizer step: every microbatch runs the same depth, so
        # `_PotentiationAccumulator` still sees a constant `layer_metrics` length.
        recurrence = None if self.recurrence is None else self.recurrence.sample(step_index)
        recurrence_kwargs: dict[str, int] = (
            {}
            if recurrence is None
            else {"recurrence_steps": recurrence[0], "backprop_steps": recurrence[1]}
        )
        rate = self.config.learning_rate * self.config.learning_rate_multiplier(step_index)
        self.optimizer.set_learning_rate(rate)
        self.optimizer.zero_grad(set_to_none=True)
        objectives: list[float] = []
        model_losses: list[float] = []
        auxiliary_losses: list[float] = []
        calibration_losses: list[float] = []
        predictive_entropies: list[float] = []
        latent_deltas: list[float] = []
        latent_correlations: list[float] = []
        potentiation = _PotentiationAccumulator()
        execution_telemetry = _ExecutionTelemetryAccumulator()
        targets = 0
        for _ in range(self.config.gradient_accumulation_steps):
            host_batch = self.batches.next_batch(
                batch_size=self.config.micro_batch_size,
                sequence_length=self.config.sequence_length,
            )
            batch = host_batch.to(self.runtime.device)
            batch_targets = batch.target_tokens
            targets += batch_targets
            with _autocast(self.runtime):
                output = self.model(
                    batch.input_ids,
                    labels=batch.resolved_labels,
                    attention_mask=batch.attention_mask,
                    state_reset_mask=batch.state_reset_mask,
                    **recurrence_kwargs,
                )
                objective = _objective(output, self.config)
                scaled = objective / self.config.gradient_accumulation_steps
            scaled.backward()
            objectives.append(float(objective.detach()))
            _append_optional_scalar(model_losses, output.loss)
            _append_optional_scalar(auxiliary_losses, output.auxiliary_loss)
            _append_optional_scalar(calibration_losses, output.surprise_calibration_loss)
            _append_optional_scalar(predictive_entropies, output.predictive_entropy)
            if output.token_nll is not None:
                self.model.observe_surprise(
                    output.token_nll,
                    batch.resolved_labels[:, 1:] != -100,
                )
            potentiation.add(output.layer_metrics, target_tokens=batch_targets)
            execution_telemetry.add(output.layer_metrics)
            _append_latent_telemetry(latent_deltas, latent_correlations, output)
            # Do not keep logits, router scores, recurrent states, or their
            # already-backwarded graphs alive across accumulation microbatches.
            del output, objective, scaled, batch, host_batch

        parameters = tuple(
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        )
        if not gradients_are_finite(parameters):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"non-finite gradient at optimizer step {step_index + 1}")
        gradient_norm = _clip_gradients(parameters, self.config.gradient_clip)
        self.optimizer.step()

        promoted = self.model.update_potentiation(potentiation.finish())
        (
            max_group_load,
            max_padding_amplification,
            max_global_max_counterfactual_amplification,
        ) = execution_telemetry.finish()
        self.progress.completed_steps += 1
        self.progress.tokens_seen += targets
        metrics = TrainStepMetrics(
            step=self.progress.completed_steps,
            tokens_seen=self.progress.tokens_seen,
            objective=sum(objectives) / len(objectives),
            model_loss=_mean_scalars(model_losses, "loss", required=True),
            auxiliary_loss=_mean_scalars(auxiliary_losses, "auxiliary_loss"),
            surprise_calibration_loss=_mean_scalars(
                calibration_losses,
                "surprise_calibration_loss",
            ),
            predictive_entropy=_mean_scalars(
                predictive_entropies,
                "predictive_entropy",
                required=True,
            ),
            learning_rate=rate,
            gradient_norm=gradient_norm,
            elapsed_seconds=time.time() - self.progress.started_at,
            promoted_experts=promoted,
            average_surprise=self.model.average_surprise,
            max_expert_group_load=max_group_load,
            max_expert_group_padding_amplification=max_padding_amplification,
            max_expert_group_global_max_counterfactual_padding_amplification=(
                max_global_max_counterfactual_amplification
            ),
            recurrence_steps=None if recurrence is None else recurrence[0],
            backprop_steps=None if recurrence is None else recurrence[1],
            effective_depth=(
                None if recurrence is None else _effective_depth(self.model, recurrence[0])
            ),
            latent_delta_final=_mean_scalars(latent_deltas, "latent_delta_final"),
            latent_position_correlation=_mean_scalars(
                latent_correlations,
                "latent_position_correlation",
            ),
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


def _parameter_dtype(config: TrainingConfig) -> torch.dtype:
    return torch.bfloat16 if config.parameter_dtype == "bfloat16" else torch.float32


def _validate_expert_master_dtype(
    model: SPALMERCausalLM,
    config: TrainingConfig,
) -> None:
    """Keep the checkpointed expert recipe and optimizer-visible weights aligned."""

    for layer_index, block in enumerate(model.backbone.blocks):
        bank = getattr(block.channel_mixer, "experts", None)
        expert_config = getattr(bank, "config", None)
        expected = getattr(expert_config, "expert_master_dtype", config.parameter_dtype)
        if expected != config.parameter_dtype:
            raise ValueError(
                f"expert master dtype at layer {layer_index} is {expected!r}, but "
                f"training parameter_dtype is {config.parameter_dtype!r}"
            )


def _validate_optimizer_precision_contract(
    optimizer: OptimizerBundle,
    config: TrainingConfig,
) -> None:
    """Prevent injected optimizers from weakening the selected BF16 lane."""

    if config.parameter_dtype != "bfloat16":
        return
    if not isinstance(optimizer.dense, BF16MasterAdamW) or optimizer.sparse is not None:
        raise ValueError(
            "BF16 base pretraining requires BF16MasterAdamW with FP32 moments "
            "and no sparse optimizer lane"
        )
    if optimizer.dense.optimizer_state_offload != config.optimizer_state_offload:
        raise ValueError(
            "BF16 base pretraining optimizer-state offload policy does not match "
            "TrainingConfig"
        )


def _move_optional(value: Tensor | None, device: torch.device) -> Tensor | None:
    if value is None:
        return None
    return value.to(device=device, non_blocking=True)


def _append_optional_scalar(destination: list[float], value: Any) -> None:
    if isinstance(value, Tensor):
        destination.append(float(value.detach().float().mean()))


def _model_recurrence(model: Any) -> Any:
    """The model's recurrence config, tolerating test doubles without one."""

    return getattr(getattr(model, "config", None), "recurrence", None)


def _validate_recurrence_contract(model: Any, sampler: RecurrenceSampler | None) -> None:
    """Reject a depth policy that does not match the model's physical core."""

    is_recurrent = _model_recurrence(model) is not None
    if is_recurrent and sampler is None:
        raise ValueError("recurrent model requires TrainingConfig.mean_recurrence")
    if not is_recurrent and sampler is not None:
        raise ValueError("mean_recurrence was set but the model has no recurrent core")


def _effective_depth(model: Any, steps: int) -> int | None:
    config = getattr(model, "config", None)
    resolver = getattr(config, "effective_depth", None)
    if not callable(resolver):
        return None
    return int(resolver(steps))


def _append_latent_telemetry(
    deltas: list[float],
    correlations: list[float],
    output: Any,
) -> None:
    latent_deltas = getattr(output, "latent_deltas", None)
    if isinstance(latent_deltas, Tensor) and latent_deltas.numel():
        deltas.append(float(latent_deltas.detach().float().reshape(-1)[-1]))
    _append_optional_scalar(
        correlations,
        getattr(output, "latent_position_correlation", None),
    )


def _mean_scalars(values: list[float], name: str, *, required: bool = False) -> float | None:
    if not values:
        if required:
            raise RuntimeError(f"model did not return required {name}")
        return None
    return sum(values) / len(values)


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
