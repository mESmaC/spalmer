"""Composable decoder shell for SPALMER research components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

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

    def __init__(self, config: SPALMERConfig, backbone: SPALMERBackbone) -> None:
        super().__init__()
        if backbone.config != config:
            raise ValueError(
                "backbone configuration does not match the language-model configuration"
            )
        self.config = config
        self.backbone = backbone
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=config.initializer_range)

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
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            if input_ids.shape[1] < 2:
                raise ValueError("next-token loss requires a sequence length of at least two")
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                logits[:, :-1].transpose(1, 2),
                shift_labels,
                ignore_index=-100,
            )
        return CausalLMOutput(
            logits=logits,
            token_mixer_states=backbone_output.token_mixer_states,
            channel_mixer_states=backbone_output.channel_mixer_states,
            loss=loss,
            auxiliary_loss=backbone_output.auxiliary_loss,
            layer_metrics=backbone_output.layer_metrics,
        )


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
