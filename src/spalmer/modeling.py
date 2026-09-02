"""Composable decoder shell for SPALMER research components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from spalmer.config import SPALMERConfig
from spalmer.embeddings import AlternatingPLE
from spalmer.nn import RMSNorm


@dataclass(slots=True)
class ChannelMixerOutput:
    update: Tensor
    state: Any = None
    auxiliary_loss: Tensor | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BlockOutput:
    hidden_states: Tensor
    token_mixer_state: Any = None
    channel_mixer_state: Any = None
    auxiliary_loss: Tensor | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BackboneOutput:
    hidden_states: Tensor
    token_mixer_states: tuple[Any, ...]
    channel_mixer_states: tuple[Any, ...]
    auxiliary_loss: Tensor | None
    layer_metrics: tuple[Mapping[str, Any], ...]


@dataclass(slots=True)
class CausalLMOutput:
    logits: Tensor
    token_mixer_states: tuple[Any, ...]
    channel_mixer_states: tuple[Any, ...]
    loss: Tensor | None = None
    auxiliary_loss: Tensor | None = None
    token_nll: Tensor | None = None
    surprise_calibration_loss: Tensor | None = None
    predictive_entropy: Tensor | None = None
    layer_metrics: tuple[Mapping[str, Any], ...] = ()


class SPALMERBlock(nn.Module):
    """Conventional pre-norm residual shell around replaceable research modules.

    Token mixers must accept ``state=...`` and return either an update tensor or
    ``(update, new_state)``. Stateful decode uses an explicit ``step`` method.
    Channel mixers may return a same-shaped tensor or :class:`ChannelMixerOutput`.
    """

    def __init__(
        self,
        d_model: int,
        token_mixer: nn.Module,
        channel_mixer: nn.Module,
        *,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.token_norm = RMSNorm(d_model, eps=norm_eps)
        self.token_mixer = token_mixer
        self.channel_norm = RMSNorm(d_model, eps=norm_eps)
        self.channel_mixer = channel_mixer

    def forward(
        self,
        hidden_states: Tensor,
        *,
        token_mixer_state: Any = None,
        channel_mixer_state: Any = None,
        execution_mode: Literal["prefill", "decode"] = "prefill",
        attention_mask: Tensor | None = None,
        state_reset_mask: Tensor | None = None,
        mixer_kwargs: Mapping[str, Any] | None = None,
    ) -> BlockOutput:
        kwargs = {} if mixer_kwargs is None else dict(mixer_kwargs)
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if state_reset_mask is not None:
            kwargs["state_reset_mask"] = state_reset_mask

        normalized = self.token_norm(hidden_states)
        if execution_mode == "prefill":
            mixed = self.token_mixer(normalized, state=token_mixer_state, **kwargs)
        elif execution_mode == "decode":
            step = getattr(self.token_mixer, "step", None)
            if not callable(step):
                raise TypeError("decode mode requires the token mixer to implement step(...)")
            mixed = step(normalized, state=token_mixer_state, **kwargs)
        else:
            raise ValueError(f"unsupported execution mode: {execution_mode}")
        if isinstance(mixed, tuple):
            if len(mixed) != 2:
                raise TypeError("token mixer tuples must contain exactly (update, new_state)")
            token_update, new_state = mixed
        else:
            token_update, new_state = mixed, token_mixer_state

        _require_same_shape("token mixer", token_update, hidden_states)
        hidden_states = hidden_states + token_update

        normalized = self.channel_norm(hidden_states)
        if channel_mixer_state is None:
            channel_result = self.channel_mixer(normalized)
        else:
            channel_result = self.channel_mixer(normalized, state=channel_mixer_state)

        if isinstance(channel_result, ChannelMixerOutput):
            channel_update = channel_result.update
            new_channel_state = channel_result.state
            auxiliary_loss = channel_result.auxiliary_loss
            metrics = channel_result.metrics
        else:
            channel_update = channel_result
            new_channel_state = channel_mixer_state
            auxiliary_loss = None
            metrics = {}
        _require_same_shape("channel mixer", channel_update, hidden_states)
        if auxiliary_loss is not None and not isinstance(auxiliary_loss, Tensor):
            raise TypeError("channel mixer auxiliary_loss must be a tensor or None")
        hidden_states = hidden_states + channel_update
        return BlockOutput(
            hidden_states=hidden_states,
            token_mixer_state=new_state,
            channel_mixer_state=new_channel_state,
            auxiliary_loss=auxiliary_loss,
            metrics=metrics,
        )


class SPALMERBackbone(nn.Module):
    """Inject per-layer lexical identity and execute composable decoder blocks."""

    def __init__(
        self,
        config: SPALMERConfig,
        blocks: Sequence[SPALMERBlock],
        *,
        embeddings: AlternatingPLE | None = None,
    ) -> None:
        super().__init__()
        if len(blocks) != config.n_layers:
            raise ValueError(f"expected {config.n_layers} blocks, got {len(blocks)}")
        self.config = config
        self.embeddings = embeddings or AlternatingPLE(config.ple_config())
        if self.embeddings.config != config.ple_config():
            raise ValueError("embedding configuration does not match the model configuration")
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)

    def forward(
        self,
        input_ids: Tensor,
        *,
        token_mixer_states: Sequence[Any] | None = None,
        channel_mixer_states: Sequence[Any] | None = None,
        execution_mode: Literal["prefill", "decode"] = "prefill",
        attention_mask: Tensor | None = None,
        state_reset_mask: Tensor | None = None,
        layer_mixer_kwargs: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> BackboneOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, sequence], got {input_ids.shape}")
        if execution_mode == "decode" and input_ids.shape[1] != 1:
            raise ValueError("decode mode requires exactly one token per sequence")
        _validate_mask("attention_mask", attention_mask, input_ids)
        _validate_mask("state_reset_mask", state_reset_mask, input_ids)

        token_states = _states_or_none(
            "token mixer",
            token_mixer_states,
            len(self.blocks),
        )
        channel_states = _states_or_none(
            "channel mixer",
            channel_mixer_states,
            len(self.blocks),
        )
        if layer_mixer_kwargs is None:
            per_layer_kwargs: Sequence[Mapping[str, Any] | None] = (None,) * len(self.blocks)
        elif len(layer_mixer_kwargs) != len(self.blocks):
            raise ValueError(
                f"expected {len(self.blocks)} layer mixer mappings, got {len(layer_mixer_kwargs)}"
            )
        else:
            per_layer_kwargs = layer_mixer_kwargs

        hidden_states = self.embeddings(input_ids, layer_index=0)
        new_token_states: list[Any] = []
        new_channel_states: list[Any] = []
        auxiliary_loss: Tensor | None = None
        layer_metrics: list[Mapping[str, Any]] = []
        layer_inputs = zip(
            self.blocks,
            token_states,
            channel_states,
            per_layer_kwargs,
            strict=True,
        )
        for layer_index, (block, token_state, channel_state, layer_kwargs) in enumerate(
            layer_inputs
        ):
            if layer_index:
                hidden_states = self.embeddings.inject(hidden_states, input_ids, layer_index)
            block_output = block(
                hidden_states,
                token_mixer_state=token_state,
                channel_mixer_state=channel_state,
                execution_mode=execution_mode,
                attention_mask=attention_mask,
                state_reset_mask=state_reset_mask,
                mixer_kwargs=layer_kwargs,
            )
            hidden_states = block_output.hidden_states
            new_token_states.append(block_output.token_mixer_state)
            new_channel_states.append(block_output.channel_mixer_state)
            layer_metrics.append(block_output.metrics)
            if block_output.auxiliary_loss is not None:
                if auxiliary_loss is None:
                    auxiliary_loss = block_output.auxiliary_loss
                else:
                    auxiliary_loss = auxiliary_loss + block_output.auxiliary_loss

        return BackboneOutput(
            hidden_states=self.final_norm(hidden_states),
            token_mixer_states=tuple(new_token_states),
            channel_mixer_states=tuple(new_channel_states),
            auxiliary_loss=auxiliary_loss,
            layer_metrics=tuple(layer_metrics),
        )


class SPALMERCausalLM(nn.Module):
    """Untied vocabulary projection and ordinary next-token objective."""

    def __init__(
        self,
        config: SPALMERConfig,
        backbone: SPALMERBackbone,
        *,
        potentiation_controller: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if backbone.config != config:
            raise ValueError(
                "backbone configuration does not match the language-model configuration"
            )
        self.config = config
        self.backbone = backbone
        self.potentiation_controller = potentiation_controller
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=config.initializer_range)
        # Shared "average surprise" telemetry (ledger C08/C13): the EMA of
        # realized next-token NLL, consumed by the inference residency controller.
        self.register_buffer("surprise_ema", torch.zeros((), dtype=torch.float32))
        self.register_buffer("surprise_observations", torch.zeros((), dtype=torch.long))

    def _apply(self, fn):
        super()._apply(fn)
        # Module-wide low-precision casts are for weights, not for a slowly
        # moving average that a bf16 update step would silently round away.
        self._buffers["surprise_ema"] = self._buffers["surprise_ema"].float()
        return self

    @property
    def average_surprise(self) -> float:
        """Average realized surprise in nats per token (``0.0`` before any observation)."""

        return float(self.surprise_ema)

    @torch.no_grad()
    def observe_surprise(self, token_nll: Tensor, valid: Tensor | None = None) -> float:
        """Fold realized next-token NLL into the shared average-surprise signal."""

        values = token_nll.detach().float()
        if valid is not None:
            values = values[valid]
        if values.numel() == 0:
            return self.average_surprise
        batch_mean = values.mean().to(self.surprise_ema)
        if int(self.surprise_observations) == 0:
            self.surprise_ema.copy_(batch_mean)
        else:
            decay = self.config.surprise_ema_decay
            self.surprise_ema.mul_(decay).add_(batch_mean, alpha=1.0 - decay)
        self.surprise_observations.add_(1)
        return self.average_surprise

    @property
    def active_experts(self) -> int | None:
        """Experts executed per token by the routed channel mixers, if any."""

        for block in self.backbone.blocks:
            count = getattr(block.channel_mixer, "active_experts", None)
            if isinstance(count, int):
                return count
        return None

    @property
    def active_experts_override(self) -> int | None:
        """Residency override applied to the routed mixers, or ``None``."""

        for block in self.backbone.blocks:
            mixer = block.channel_mixer
            if hasattr(mixer, "active_experts_override"):
                return mixer.active_experts_override
        return None

    def set_active_experts(self, count: int | None) -> None:
        """Apply one inference residency decision to every routed channel mixer."""

        for block in self.backbone.blocks:
            setter = getattr(block.channel_mixer, "set_active_experts", None)
            if callable(setter):
                setter(count)

    def forward(
        self,
        input_ids: Tensor,
        *,
        labels: Tensor | None = None,
        token_mixer_states: Sequence[Any] | None = None,
        channel_mixer_states: Sequence[Any] | None = None,
        execution_mode: Literal["prefill", "decode"] = "prefill",
        attention_mask: Tensor | None = None,
        state_reset_mask: Tensor | None = None,
        layer_mixer_kwargs: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> CausalLMOutput:
        backbone_output = self.backbone(
            input_ids,
            token_mixer_states=token_mixer_states,
            channel_mixer_states=channel_mixer_states,
            execution_mode=execution_mode,
            attention_mask=attention_mask,
            state_reset_mask=state_reset_mask,
            layer_mixer_kwargs=layer_mixer_kwargs,
        )
        logits = self.lm_head(backbone_output.hidden_states)
        loss = None
        token_nll = None
        surprise_calibration = None
        layer_metrics = backbone_output.layer_metrics
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            if input_ids.shape[1] < 2:
                raise ValueError("next-token loss requires a sequence length of at least two")
            shift_labels = labels[:, 1:].contiguous()
            token_nll = F.cross_entropy(
                logits[:, :-1].transpose(1, 2),
                shift_labels,
                ignore_index=-100,
                reduction="none",
            )
            valid = shift_labels != -100
            if not bool(valid.any()):
                raise ValueError("next-token loss requires at least one non-ignored target")
            loss = token_nll[valid].mean()
            layer_metrics, surprise_calibration = _attach_surprise_telemetry(
                layer_metrics,
                token_nll,
                valid,
            )

        with torch.no_grad():
            final_log_probabilities = F.log_softmax(logits[:, -1].detach().float(), dim=-1)
            predictive_entropy = -(final_log_probabilities.exp() * final_log_probabilities).sum(
                dim=-1
            )
        return CausalLMOutput(
            logits=logits,
            token_mixer_states=backbone_output.token_mixer_states,
            channel_mixer_states=backbone_output.channel_mixer_states,
            loss=loss,
            auxiliary_loss=backbone_output.auxiliary_loss,
            token_nll=token_nll,
            surprise_calibration_loss=surprise_calibration,
            predictive_entropy=predictive_entropy,
            layer_metrics=layer_metrics,
        )

    @torch.no_grad()
    def update_potentiation(
        self,
        layer_metrics: Sequence[Mapping[str, Any]],
    ) -> tuple[int, ...]:
        """Apply one shared expert-group promotion decision across all layers."""

        controller = self.potentiation_controller
        if controller is None:
            return self.promoted_expert_ids()
        telemetry = [
            (
                metric.get("potentiation_utilization"),
                metric.get("expert_attributed_nll"),
                metric.get("expert_quantization_error"),
            )
            for metric in layer_metrics
        ]
        complete = [
            values for values in telemetry if all(isinstance(value, Tensor) for value in values)
        ]
        if not complete:
            return self.promoted_expert_ids()
        utilization_by_layer = torch.stack([values[0].float() for values in complete])
        attributed_nll_by_layer = torch.stack([values[1].float() for values in complete])
        quantization_error_by_layer = torch.stack([values[2].float() for values in complete])
        utilization = utilization_by_layer.mean(dim=0)
        responsibility = utilization_by_layer.sum(dim=0)
        attributed_nll = (attributed_nll_by_layer * utilization_by_layer).sum(
            dim=0
        ) / responsibility.clamp_min(1e-8)
        attributed_nll = torch.where(
            responsibility > 0,
            attributed_nll,
            torch.zeros_like(attributed_nll),
        )
        measured = quantization_error_by_layer > 0
        quantization_error = quantization_error_by_layer.sum(dim=0) / measured.sum(dim=0).clamp_min(
            1
        )
        observe = getattr(controller, "observe", None)
        if not callable(observe):
            raise TypeError("potentiation_controller must implement observe(...)")
        observe(utilization, attributed_nll, quantization_error)
        return self.promoted_expert_ids()

    def promoted_expert_ids(self) -> tuple[int, ...]:
        """Return the coherent promoted identity set from the first expert bank."""

        controller = self.potentiation_controller
        if controller is None:
            return ()
        promoted_ids = getattr(controller, "promoted_ids", None)
        if not callable(promoted_ids):
            raise TypeError("potentiation_controller must implement promoted_ids()")
        return promoted_ids()


def _require_same_shape(name: str, update: Any, hidden_states: Tensor) -> None:
    if not isinstance(update, Tensor):
        raise TypeError(f"{name} must return a tensor update")
    if update.shape != hidden_states.shape:
        raise ValueError(
            f"{name} update must match hidden-state shape; "
            f"got {tuple(update.shape)} and {tuple(hidden_states.shape)}"
        )


def _states_or_none(name: str, states: Sequence[Any] | None, count: int) -> Sequence[Any]:
    if states is None:
        return (None,) * count
    if len(states) != count:
        raise ValueError(f"expected {count} {name} states, got {len(states)}")
    return states


def _validate_mask(name: str, mask: Tensor | None, input_ids: Tensor) -> None:
    if mask is not None and mask.shape != input_ids.shape:
        raise ValueError(
            f"{name} must have the same [batch, sequence] shape as input_ids; "
            f"got {tuple(mask.shape)} and {tuple(input_ids.shape)}"
        )


def _attach_surprise_telemetry(
    layer_metrics: Sequence[Mapping[str, Any]],
    token_nll: Tensor,
    valid: Tensor,
) -> tuple[tuple[Mapping[str, Any], ...], Tensor | None]:
    enriched: list[Mapping[str, Any]] = []
    calibration_losses: list[Tensor] = []
    for metrics in layer_metrics:
        scores = metrics.get("router_scores")
        expert_ids = metrics.get("expert_ids")
        routing_weights = metrics.get("routing_weights")
        if not all(isinstance(value, Tensor) for value in (scores, expert_ids, routing_weights)):
            enriched.append(metrics)
            continue

        selected_surprise = scores.gather(-1, expert_ids)[:, :-1]
        # This v0 target calibrates the selected mixture's scalar estimate. The
        # downstream LM NLL is not a counterfactual measurement for each expert.
        predicted_mixture_surprise = (selected_surprise * routing_weights[:, :-1]).sum(dim=-1)
        calibration = F.smooth_l1_loss(
            predicted_mixture_surprise,
            token_nll.detach(),
            reduction="none",
        )
        calibration_losses.append(calibration[valid].mean())

        num_experts = scores.shape[-1]
        attributed_nll, responsibility_utilization = _attribute_nll_to_experts(
            expert_ids[:, :-1],
            routing_weights[:, :-1],
            token_nll.detach(),
            valid,
            num_experts,
        )
        updated = dict(metrics)
        updated["expert_attributed_nll"] = attributed_nll
        updated["potentiation_utilization"] = responsibility_utilization
        enriched.append(updated)

    total_calibration = None
    if calibration_losses:
        total_calibration = torch.stack(calibration_losses).mean()
    return tuple(enriched), total_calibration


def _attribute_nll_to_experts(
    expert_ids: Tensor,
    routing_weights: Tensor,
    token_nll: Tensor,
    valid: Tensor,
    num_experts: int,
) -> tuple[Tensor, Tensor]:
    ids = expert_ids.reshape(-1)
    weights = routing_weights.float() * valid.unsqueeze(-1)
    flat_weights = weights.reshape(-1)
    weighted_nll = (weights * token_nll.float().unsqueeze(-1)).reshape(-1)
    totals = torch.zeros(num_experts, device=ids.device, dtype=torch.float32)
    counts = torch.zeros_like(totals)
    totals.scatter_add_(0, ids, weighted_nll)
    counts.scatter_add_(0, ids, flat_weights)
    attributed_nll = torch.where(
        counts > 0,
        totals / counts.clamp_min(1e-8),
        torch.zeros_like(totals),
    )
    utilization = counts / valid.sum().float().clamp_min(1.0)
    return attributed_nll, utilization
