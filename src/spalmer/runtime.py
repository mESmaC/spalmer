"""Minimal training and cached-generation loops for executable prototypes."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from spalmer.modeling import SPALMERCausalLM
from spalmer.training.optim import BF16MasterAdamW
from spalmer.training.recurrence import RecurrenceSampler

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spalmer.modeling import AdaptiveExit


@dataclass(slots=True)
class TrainResult:
    losses: list[float]
    tokens_seen: int
    elapsed_seconds: float
    final_model_loss: float
    final_auxiliary_loss: float | None
    final_surprise_calibration_loss: float | None
    final_predictive_entropy: float
    promoted_experts: tuple[int, ...]
    average_surprise: float = 0.0
    mean_recurrence_steps: float | None = None


@dataclass(slots=True)
class RecurrenceTrace:
    """A caller-owned receipt of the depth actually run by ``generate_tokens``."""

    max_steps: int = 0
    prefill_steps: int = 0
    decode_steps: list[int] = field(default_factory=list)
    early_exits: int = 0

    @property
    def mean_decode_steps(self) -> float:
        if not self.decode_steps:
            return 0.0
        return sum(self.decode_steps) / len(self.decode_steps)

    @property
    def min_decode_steps(self) -> int:
        return min(self.decode_steps, default=0)

    @property
    def max_decode_steps(self) -> int:
        return max(self.decode_steps, default=0)


def _recurrence_config(model: Any) -> Any:
    """The model's recurrence config, tolerating doubles that have none."""

    return getattr(getattr(model, "config", None), "recurrence", None)


def train_token_stream(
    model: SPALMERCausalLM,
    token_ids: Tensor,
    *,
    steps: int,
    batch_size: int,
    sequence_length: int,
    learning_rate: float = 3e-4,
    auxiliary_loss_weight: float = 0.01,
    surprise_calibration_weight: float = 0.05,
    gradient_clip: float | None = 1.0,
    seed: int = 0,
    log: Callable[[int, float], None] | None = None,
    recurrence: RecurrenceSampler | None = None,
) -> TrainResult:
    """Train directly from one one-dimensional token stream."""

    is_recurrent = _recurrence_config(model) is not None
    if is_recurrent and recurrence is None:
        raise ValueError("recurrent model requires a RecurrenceSampler")
    if not is_recurrent and recurrence is not None:
        raise ValueError("a RecurrenceSampler was given but the model has no recurrent core")
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be one-dimensional")
    if len(token_ids) <= sequence_length:
        raise ValueError("token stream must be longer than sequence_length")
    if min(steps, batch_size, sequence_length) <= 0:
        raise ValueError("steps, batch_size, and sequence_length must be positive")
    if auxiliary_loss_weight < 0 or surprise_calibration_weight < 0:
        raise ValueError("auxiliary loss weights must be non-negative")

    device = next(model.parameters()).device
    stream = token_ids.to(device=device, dtype=torch.long)
    generator = torch.Generator(device=device).manual_seed(seed)
    parameters, parameter_dtype = _validated_trainable_parameters(model)
    if parameter_dtype == torch.bfloat16:
        optimizer = BF16MasterAdamW(
            [{"params": parameters, "weight_decay": 0.01}],
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            stochastic_rounding=True,
            update_chunk_size=1_048_576,
        )
    else:  # _validated_trainable_parameters only permits FP32 or BF16.
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    losses: list[float] = []
    final_model_loss = float("nan")
    final_auxiliary_loss: float | None = None
    final_surprise_calibration_loss: float | None = None
    final_predictive_entropy = float("nan")
    sampled_steps: list[int] = []
    started = time.perf_counter()
    model.train()

    max_start = len(stream) - sequence_length
    for step in range(1, steps + 1):
        starts = torch.randint(
            0,
            max_start,
            (batch_size,),
            generator=generator,
            device=device,
        )
        offsets = torch.arange(sequence_length + 1, device=device)
        batch = stream[starts[:, None] + offsets[None, :]]

        optimizer.zero_grad(set_to_none=True)
        recurrence_kwargs: dict[str, int] = {}
        if recurrence is not None:
            drawn_steps, drawn_backprop = recurrence.sample(step - 1)
            sampled_steps.append(drawn_steps)
            recurrence_kwargs = {
                "recurrence_steps": drawn_steps,
                "backprop_steps": drawn_backprop,
            }
        output = model(batch, labels=batch, **recurrence_kwargs)
        if output.loss is None:
            raise RuntimeError("model did not return its next-token loss")
        objective = output.loss
        if output.auxiliary_loss is not None:
            objective = objective + auxiliary_loss_weight * output.auxiliary_loss
        if output.surprise_calibration_loss is not None:
            objective = objective + surprise_calibration_weight * output.surprise_calibration_loss
        objective.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        promoted_experts = model.update_potentiation(output.layer_metrics)
        if output.token_nll is not None:
            model.observe_surprise(output.token_nll, batch[:, 1:] != -100)

        value = float(objective.detach())
        final_model_loss = float(output.loss.detach())
        final_auxiliary_loss = (
            None if output.auxiliary_loss is None else float(output.auxiliary_loss.detach())
        )
        final_surprise_calibration_loss = (
            None
            if output.surprise_calibration_loss is None
            else float(output.surprise_calibration_loss.detach())
        )
        if output.predictive_entropy is None:
            raise RuntimeError("model did not return predictive entropy telemetry")
        final_predictive_entropy = float(output.predictive_entropy.mean())
        losses.append(value)
        if log is not None:
            log(step, value)

    return TrainResult(
        losses=losses,
        tokens_seen=steps * batch_size * sequence_length,
        elapsed_seconds=time.perf_counter() - started,
        final_model_loss=final_model_loss,
        final_auxiliary_loss=final_auxiliary_loss,
        final_surprise_calibration_loss=final_surprise_calibration_loss,
        final_predictive_entropy=final_predictive_entropy,
        promoted_experts=promoted_experts,
        average_surprise=model.average_surprise,
        mean_recurrence_steps=(
            None if not sampled_steps else sum(sampled_steps) / len(sampled_steps)
        ),
    )


def _validated_trainable_parameters(
    model: SPALMERCausalLM,
) -> tuple[tuple[torch.nn.Parameter, ...], torch.dtype]:
    """Resolve the legacy optimizer only after validating its entire parameter set."""

    named = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if not named:
        raise ValueError("model has no trainable parameters")
    non_floating = sorted(name for name, parameter in named if not parameter.is_floating_point())
    if non_floating:
        raise ValueError(
            "all trainable parameters must be floating point before optimizer construction; "
            f"non-floating={non_floating}"
        )
    by_dtype: dict[torch.dtype, list[str]] = {}
    for name, parameter in named:
        by_dtype.setdefault(parameter.dtype, []).append(name)
    if len(by_dtype) != 1:
        detail = ", ".join(
            f"{dtype}: {sorted(names)}" for dtype, names in sorted(by_dtype.items(), key=str)
        )
        raise ValueError(
            "all trainable parameters must share one dtype before optimizer construction; "
            f"observed {detail}"
        )
    dtype = next(iter(by_dtype))
    if dtype not in {torch.bfloat16, torch.float32}:
        raise ValueError(
            f"unsupported trainable parameter dtype {dtype}; expected bfloat16 or float32"
        )
    return tuple(parameter for _, parameter in named), dtype


@torch.inference_mode()
def generate_tokens(
    model: SPALMERCausalLM,
    prompt_ids: Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    seed: int = 0,
    dynamic_residency: bool = False,
    recurrence_steps: int | None = None,
    adaptive_exit: AdaptiveExit | None = None,
    warm_start: bool = False,
    latent_seed: int | None = None,
    trace: RecurrenceTrace | None = None,
) -> Tensor:
    """Generate with one prefill followed by explicit recurrent decode steps.

    With ``dynamic_residency`` the prefill runs the C13 residency controller:
    the resident candidate set expands while the prompt's effective surprise
    stays above the model's average surprise. The configured per-token top-k
    remains fixed, and decoding continues with the resident ids that were
    retained.

    For a model with a recurrent latent core, ``recurrence_steps`` selects the
    fixed depth (``None`` uses the checkpoint's ``default_steps``),
    ``adaptive_exit`` optionally stops decode iterations early, and
    ``warm_start`` seeds each decode step from the previous token's final
    latent. ``trace`` is filled in place with the depth actually run.
    """

    recurrence_config = _recurrence_config(model)
    if recurrence_config is None and (
        recurrence_steps is not None or adaptive_exit is not None or warm_start
    ):
        raise ValueError("model has no recurrent core")
    if recurrence_steps is not None and recurrence_steps < 1:
        raise ValueError("recurrence_steps must be positive or None")
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] == 0:
        raise ValueError("prompt_ids must describe one non-empty sequence")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if temperature < 0:
        raise ValueError("temperature cannot be negative")

    # Expert-offloaded models intentionally have parameters on both CPU and
    # the accelerator; never infer execution placement from parameter order.
    device = model.execution_device
    generated = prompt_ids.to(device=device, dtype=torch.long)
    if max_new_tokens == 0:
        return generated
    steps: int | None = None
    prefill_kwargs: dict[str, Any] = {}
    decode_kwargs: dict[str, Any] = {}
    if recurrence_config is not None:
        steps = int(recurrence_steps or recurrence_config.default_steps)
        latent_generator = torch.Generator(device=device).manual_seed(
            seed if latent_seed is None else latent_seed
        )
        prefill_kwargs = {"recurrence_steps": steps, "latent_generator": latent_generator}
        decode_kwargs = dict(prefill_kwargs)
        if adaptive_exit is not None:
            decode_kwargs["adaptive_exit"] = adaptive_exit
        if trace is not None:
            trace.max_steps = steps
            trace.prefill_steps = steps
    was_training = model.training
    model.eval()
    try:
        if dynamic_residency:
            from spalmer.experts.residency import choose_inference_residency

            residency_kwargs = {} if steps is None else {"recurrence_steps": steps}
            output = choose_inference_residency(model, generated, **residency_kwargs).output
        else:
            output = model(generated, **prefill_kwargs)
        token_states = output.token_mixer_states
        channel_states = output.channel_mixer_states
        generator = torch.Generator(device=device).manual_seed(seed)

        for token_index in range(max_new_tokens):
            next_token = _select_token(
                output.logits[:, -1],
                temperature=temperature,
                top_k=top_k,
                generator=generator,
            )
            generated = torch.cat((generated, next_token), dim=1)
            if token_index + 1 == max_new_tokens:
                break
            step_kwargs = dict(decode_kwargs)
            if warm_start:
                latent_states = getattr(output, "latent_states", None)
                if latent_states is not None:
                    step_kwargs["latent_init"] = latent_states[:, -1:, :]
            output = model(
                next_token,
                token_mixer_states=token_states,
                channel_mixer_states=channel_states,
                execution_mode="decode",
                **step_kwargs,
            )
            token_states = output.token_mixer_states
            channel_states = output.channel_mixer_states
            if trace is not None and steps is not None:
                ran = getattr(output, "recurrence_steps", None)
                ran = steps if ran is None else int(ran)
                trace.decode_steps.append(ran)
                trace.early_exits += int(ran < steps)
    finally:
        if dynamic_residency:
            model.end_residency_request()
        model.train(was_training)
    return generated


def _select_token(
    logits: Tensor,
    *,
    temperature: float,
    top_k: int | None,
    generator: torch.Generator,
) -> Tensor:
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        count = min(top_k, logits.shape[-1])
        cutoff = logits.topk(count, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
    probabilities = logits.float().softmax(dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


__all__ = ["RecurrenceTrace", "TrainResult", "generate_tokens", "train_token_stream"]
