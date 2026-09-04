"""Composable decoder shell for SPALMER research components."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from spalmer.config import SPALMERConfig
from spalmer.embeddings import AlternatingPLE
from spalmer.nn import RMSNorm

if TYPE_CHECKING:
    from spalmer.experts.accounting import ParameterAccounting
    from spalmer.experts.offload import ExpertOffloadManager, ExpertOffloadTelemetry
    from spalmer.experts.residency import ExpertResidency
    from spalmer.memory import ATXYInjection, ATXYRequest


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


@dataclass(frozen=True, slots=True)
class RecurrentMixerStates:
    """Per-iteration mixer states of ONE core block.

    ``iterations[i]`` is the state after recurrence iteration ``i``. Elements
    may be ``None`` (stateless mixers and channel-mixer doubles); only an empty
    container is rejected. The dataclass is type-distinguishable from ordinary
    mixer states so state containers can be validated fail-closed, and it
    carries ``.to()`` for device/dtype moves.
    """

    iterations: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.iterations, tuple):
            object.__setattr__(self, "iterations", tuple(self.iterations))
        if not self.iterations:
            raise ValueError("recurrent mixer states need at least one iteration")

    def __len__(self) -> int:
        return len(self.iterations)

    def __getitem__(self, index: int) -> Any:
        return self.iterations[index]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.iterations)

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> RecurrentMixerStates:
        moved = []
        for element in self.iterations:
            mover = getattr(element, "to", None)
            moved.append(mover(device=device, dtype=dtype) if callable(mover) else element)
        return RecurrentMixerStates(tuple(moved))


DEFAULT_EXIT_THRESHOLDS: dict[str, float] = {"latent_diff": 0.03, "kl": 1e-3}


@dataclass(frozen=True, slots=True)
class AdaptiveExit:
    """Inference-only early-exit policy for the depth-recurrent core.

    The criteria compare successive iterations, so ``min_steps`` below two is
    clamped to two. The criterion is evaluated at iteration counts
    ``it >= min_steps`` with ``(it - min_steps) % check_every == 0``.
    """

    criterion: Literal["latent_diff", "kl"]
    threshold: float | None = None
    min_steps: int = 2
    check_every: int = 1
    state_policy: Literal["fill", "skip"] = "fill"
    apply_in_prefill: bool = False

    def __post_init__(self) -> None:
        if self.criterion not in DEFAULT_EXIT_THRESHOLDS:
            raise ValueError(
                f"unsupported adaptive exit criterion: {self.criterion!r}; "
                f"expected one of {sorted(DEFAULT_EXIT_THRESHOLDS)}"
            )
        if self.threshold is not None and (
            self.threshold <= 0 or not math.isfinite(self.threshold)
        ):
            raise ValueError("adaptive exit threshold must be finite and positive")
        if self.min_steps < 1:
            raise ValueError("adaptive exit min_steps must be positive")
        if self.min_steps < 2:
            object.__setattr__(self, "min_steps", 2)
        if self.check_every < 1:
            raise ValueError("adaptive exit check_every must be positive")
        if self.state_policy not in {"fill", "skip"}:
            raise ValueError(f"unsupported adaptive exit state policy: {self.state_policy!r}")

    @property
    def resolved_threshold(self) -> float:
        if self.threshold is None:
            return DEFAULT_EXIT_THRESHOLDS[self.criterion]
        return float(self.threshold)


@dataclass(slots=True)
class BackboneOutput:
    hidden_states: Tensor
    token_mixer_states: tuple[Any, ...]
    channel_mixer_states: tuple[Any, ...]
    auxiliary_loss: Tensor | None
    layer_metrics: tuple[Mapping[str, Any], ...]
    recurrence_steps: int | None = None
    latent_states: Tensor | None = None
    latent_deltas: Tensor | None = None
    latent_position_correlation: Tensor | None = None


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
    recurrence_steps: int | None = None
    latent_states: Tensor | None = None
    latent_deltas: Tensor | None = None
    latent_position_correlation: Tensor | None = None


class LatentRecurrence(nn.Module):
    """Huginn-style latent injection for a depth-recurrent block core.

    ``injection_norm`` normalizes the prelude output ONCE per forward, the
    ``adapter`` mixes the fp32 latent with that normalized injection, and
    ``latent_norm`` bounds the latent after the last core block of every
    iteration. The module name deliberately contains neither ``norm`` nor
    ``gate`` so the 2-D adapter weight decays, while the two sub-module names
    contain ``norm`` so their weights do not (``training/optim.py``).
    """

    def __init__(
        self,
        d_model: int,
        *,
        norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        adapter_init: Literal["identity_mix", "random"] = "identity_mix",
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if adapter_init not in {"identity_mix", "random"}:
            raise ValueError(f"unsupported recurrence adapter_init: {adapter_init!r}")
        self.d_model = d_model
        self.injection_norm = RMSNorm(d_model, eps=norm_eps)
        self.adapter = nn.Linear(2 * d_model, d_model, bias=False)
        self.latent_norm = RMSNorm(d_model, eps=norm_eps)
        with torch.no_grad():
            if adapter_init == "identity_mix":
                identity = torch.eye(d_model, dtype=self.adapter.weight.dtype)
                self.adapter.weight.copy_(
                    torch.cat([identity, identity], dim=1) / math.sqrt(2.0)
                )
                self.adapter.weight.add_(
                    torch.randn_like(self.adapter.weight) * initializer_range
                )
            else:
                self.adapter.weight.normal_(mean=0.0, std=1.0 / math.sqrt(2 * d_model))

    def normalize_injection(self, prelude_output: Tensor) -> Tensor:
        """Unit-RMS view of the prelude output, computed once per forward."""

        return self.injection_norm(prelude_output)

    def inject(self, latent: Tensor, normalized_injection: Tensor) -> Tensor:
        """Mix the fp32 latent with the normalized injection into block input."""

        working = normalized_injection.dtype
        return self.adapter(torch.cat([latent.to(working), normalized_injection], dim=-1))

    def normalize_latent(self, hidden_states: Tensor) -> Tensor:
        """Bound the residual stream at the end of one core iteration."""

        return self.latent_norm(hidden_states)


class SPALMERBlock(nn.Module):
    """Conventional pre-norm residual shell around replaceable research modules.

    Token mixers must accept ``state=...`` and return either an update tensor or
    ``(update, new_state)``. Stateful decode uses an explicit ``step`` method.
    Channel mixers may return a same-shaped tensor or :class:`ChannelMixerOutput`.
    An optional directional mixer (ledger C16) runs as a third pre-norm residual
    branch after the channel mixer; when it is absent the block behaves exactly
    as before.
    """

    def __init__(
        self,
        d_model: int,
        token_mixer: nn.Module,
        channel_mixer: nn.Module,
        *,
        directional_mixer: nn.Module | None = None,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.token_norm = RMSNorm(d_model, eps=norm_eps)
        self.token_mixer = token_mixer
        self.channel_norm = RMSNorm(d_model, eps=norm_eps)
        self.channel_mixer = channel_mixer
        self.directional_norm = (
            None if directional_mixer is None else RMSNorm(d_model, eps=norm_eps)
        )
        self.directional_mixer = directional_mixer

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

        if self.directional_mixer is not None:
            directional_update = self.directional_mixer(self.directional_norm(hidden_states))
            _require_same_shape("directional mixer", directional_update, hidden_states)
            hidden_states = hidden_states + directional_update
            directional_metrics = getattr(self.directional_mixer, "last_metrics", {})
            if directional_metrics:
                metrics = {**metrics, "directional": dict(directional_metrics)}

        return BlockOutput(
            hidden_states=hidden_states,
            token_mixer_state=new_state,
            channel_mixer_state=new_channel_state,
            auxiliary_loss=auxiliary_loss,
            metrics=metrics,
        )

    def advance_token_mixer(
        self,
        hidden_states: Tensor,
        *,
        token_mixer_state: Any = None,
        execution_mode: Literal["prefill", "decode"] = "prefill",
        attention_mask: Tensor | None = None,
        state_reset_mask: Tensor | None = None,
        mixer_kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        """Run only the token-mixer branch to advance its state.

        The mixer's residual update is discarded; only the new state is
        returned. This is what the ``"fill"`` adaptive-exit policy uses to keep
        a deeper iteration's KDA/MLA memory aligned with the token stream after
        a sequence has exited.
        """

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
            return mixed[1]
        return token_mixer_state


class SPALMERBackbone(nn.Module):
    """Inject per-layer lexical identity and execute composable decoder blocks.

    An optional :class:`~spalmer.memory.ATXYInjection` (ledger C03/C04) adds
    the factorized address embedding at the input, where every KDA/MLA
    projection can see it, and performs the exact value lookup after its
    configured block boundary. Both happen only when an
    :class:`~spalmer.memory.ATXYRequest` is supplied; without the module or
    without a request the backbone is a plain stack of blocks, so ATXY is
    never a required data dependency.

    Per-layer auxiliary losses are averaged over the layers that report one, so
    the total does not scale with layer count.
    """

    def __init__(
        self,
        config: SPALMERConfig,
        blocks: Sequence[SPALMERBlock],
        *,
        embeddings: AlternatingPLE | None = None,
        atxy: ATXYInjection | None = None,
    ) -> None:
        super().__init__()
        if len(blocks) != config.n_layers:
            raise ValueError(f"expected {config.n_layers} blocks, got {len(blocks)}")
        recurrence_config = config.recurrence
        if atxy is not None:
            if atxy.config.d_model != config.d_model:
                raise ValueError(
                    f"ATXY d_model={atxy.config.d_model} does not match d_model={config.d_model}"
                )
            if not 0 <= atxy.config.injection_layer < config.n_layers:
                raise ValueError(
                    f"ATXY injection_layer={atxy.config.injection_layer} is outside "
                    f"[0, {config.n_layers})"
                )
            if (
                recurrence_config is not None
                and atxy.config.injection_layer in recurrence_config.core_range()
            ):
                core = recurrence_config.core_range()
                raise ValueError(
                    f"ATXY injection_layer={atxy.config.injection_layer} lies inside the "
                    f"recurrent core [{core.start}, {core.stop})"
                )
        self.config = config
        self.embeddings = embeddings or AlternatingPLE(config.ple_config())
        if self.embeddings.config != config.ple_config():
            raise ValueError("embedding configuration does not match the model configuration")
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.atxy = atxy
        self.recurrence = (
            None
            if recurrence_config is None
            else LatentRecurrence(
                config.d_model,
                norm_eps=config.norm_eps,
                initializer_range=config.initializer_range,
                adapter_init=recurrence_config.adapter_init,
            )
        )
        self._core_range = (
            range(0) if recurrence_config is None else recurrence_config.core_range()
        )
        # Model-level depth policy (see SPALMERCausalLM.set_recurrence_defaults).
        # Plain attributes, deliberately not thread-safe: the yyLab engine
        # serialises generation.
        self.recurrence_steps_override: int | None = None
        self.adaptive_exit_override: AdaptiveExit | None = None

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
        atxy: ATXYRequest | None = None,
        recurrence_steps: int | None = None,
        backprop_steps: int | None = None,
        latent_init: Tensor | None = None,
        adaptive_exit: AdaptiveExit | None = None,
        latent_generator: torch.Generator | None = None,
        exit_readout: Callable[[Tensor], Tensor] | None = None,
    ) -> BackboneOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, sequence], got {input_ids.shape}")
        if execution_mode == "decode" and input_ids.shape[1] != 1:
            raise ValueError("decode mode requires exactly one token per sequence")
        _validate_mask("attention_mask", attention_mask, input_ids)
        _validate_mask("state_reset_mask", state_reset_mask, input_ids)
        if atxy is not None:
            if self.atxy is None:
                raise ValueError("an ATXY request was supplied but the backbone has no ATXY module")
            _validate_mask("atxy.mask", atxy.mask, input_ids)
            if atxy.addresses.shape[:2] != input_ids.shape:
                raise ValueError(
                    f"atxy.addresses must start with the input_ids shape {tuple(input_ids.shape)}, "
                    f"got {tuple(atxy.addresses.shape)}"
                )

        token_states = _validate_states(
            "token mixer",
            token_mixer_states,
            self.config,
            len(self.blocks),
        )
        channel_states = _validate_states(
            "channel mixer",
            channel_mixer_states,
            self.config,
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
        if atxy is not None:
            # C03: the semantic coordinate enters the residual stream at the
            # input so the KDA/MLA projections of every block can read it.
            hidden_states = self.atxy.embed_addresses(hidden_states, atxy.addresses, atxy.mask)

        if self.recurrence is None:
            offending = sorted(
                name
                for name, value in (
                    ("recurrence_steps", recurrence_steps),
                    ("backprop_steps", backprop_steps),
                    ("latent_init", latent_init),
                    ("adaptive_exit", adaptive_exit),
                    ("latent_generator", latent_generator),
                )
                if value is not None
            )
            if offending:
                raise ValueError(
                    "model has no recurrent core; recurrence arguments are not accepted: "
                    + ", ".join(offending)
                )
            (
                hidden_states,
                new_token_states,
                new_channel_states,
                layer_metrics,
                auxiliary_losses,
            ) = self._run_block_range(
                hidden_states,
                input_ids,
                range(len(self.blocks)),
                token_states=token_states,
                channel_states=channel_states,
                per_layer_kwargs=per_layer_kwargs,
                execution_mode=execution_mode,
                attention_mask=attention_mask,
                state_reset_mask=state_reset_mask,
                atxy=atxy,
            )
            auxiliary_loss = None
            if auxiliary_losses:
                auxiliary_loss = torch.stack(auxiliary_losses).mean()
            return BackboneOutput(
                hidden_states=self.final_norm(hidden_states),
                token_mixer_states=tuple(new_token_states),
                channel_mixer_states=tuple(new_channel_states),
                auxiliary_loss=auxiliary_loss,
                layer_metrics=tuple(layer_metrics),
            )

        return self._recurrent_forward(
            hidden_states,
            input_ids,
            token_states=token_states,
            channel_states=channel_states,
            per_layer_kwargs=per_layer_kwargs,
            execution_mode=execution_mode,
            attention_mask=attention_mask,
            state_reset_mask=state_reset_mask,
            atxy=atxy,
            recurrence_steps=recurrence_steps,
            backprop_steps=backprop_steps,
            latent_init=latent_init,
            adaptive_exit=adaptive_exit,
            latent_generator=latent_generator,
            exit_readout=exit_readout,
        )

    def _run_block_range(
        self,
        hidden_states: Tensor,
        input_ids: Tensor,
        indices: Sequence[int],
        *,
        token_states: Sequence[Any],
        channel_states: Sequence[Any],
        per_layer_kwargs: Sequence[Mapping[str, Any] | None],
        execution_mode: Literal["prefill", "decode"],
        attention_mask: Tensor | None,
        state_reset_mask: Tensor | None,
        atxy: ATXYRequest | None,
        collect: bool = True,
    ) -> tuple[Tensor, list[Any], list[Any], list[Mapping[str, Any]], list[Tensor]]:
        """Run one contiguous span of physical blocks, exactly as a flat stack.

        With ``collect=False`` nothing is appended and no ATXY value injection
        fires, so the depth-recurrent KL exit probe can read the coda's logits
        without disturbing the states, metrics, or auxiliary losses the real
        coda pass will produce.
        """

        new_token_states: list[Any] = []
        new_channel_states: list[Any] = []
        layer_metrics: list[Mapping[str, Any]] = []
        auxiliary_losses: list[Tensor] = []
        for layer_index in indices:
            if layer_index:
                hidden_states = self.embeddings.inject(hidden_states, input_ids, layer_index)
            block_output = self.blocks[layer_index](
                hidden_states,
                token_mixer_state=token_states[layer_index],
                channel_mixer_state=channel_states[layer_index],
                execution_mode=execution_mode,
                attention_mask=attention_mask,
                state_reset_mask=state_reset_mask,
                mixer_kwargs=per_layer_kwargs[layer_index],
            )
            hidden_states = block_output.hidden_states
            if atxy is not None and layer_index == self.atxy.config.injection_layer:
                # C04: exact lookup and gated value injection at one boundary.
                hidden_states = self.atxy.inject_values(
                    hidden_states,
                    atxy.addresses,
                    atxy.mask,
                    atxy.store,
                    atxy.expected_store_version,
                )
            if collect:
                new_token_states.append(block_output.token_mixer_state)
                new_channel_states.append(block_output.channel_mixer_state)
                layer_metrics.append(block_output.metrics)
                if block_output.auxiliary_loss is not None:
                    auxiliary_losses.append(block_output.auxiliary_loss)
        return (
            hidden_states,
            new_token_states,
            new_channel_states,
            layer_metrics,
            auxiliary_losses,
        )

    def _recurrent_forward(
        self,
        hidden_states: Tensor,
        input_ids: Tensor,
        *,
        token_states: Sequence[Any],
        channel_states: Sequence[Any],
        per_layer_kwargs: Sequence[Mapping[str, Any] | None],
        execution_mode: Literal["prefill", "decode"],
        attention_mask: Tensor | None,
        state_reset_mask: Tensor | None,
        atxy: ATXYRequest | None,
        recurrence_steps: int | None,
        backprop_steps: int | None,
        latent_init: Tensor | None,
        adaptive_exit: AdaptiveExit | None,
        latent_generator: torch.Generator | None,
        exit_readout: Callable[[Tensor], Tensor] | None,
    ) -> BackboneOutput:
        """Prelude, ``r`` iterations of the latent core, then coda."""

        config = self.config
        recurrence_config = config.recurrence
        recurrence = self.recurrence
        prelude = recurrence_config.prelude_layers
        core = recurrence_config.core_layers
        total = len(self.blocks)

        steps = recurrence_steps
        if steps is None:
            steps = self.recurrence_steps_override
        if steps is None:
            steps = recurrence_config.default_steps
        steps = int(steps)
        if steps < 1:
            raise ValueError("recurrence_steps must be at least 1")
        retained = steps if backprop_steps is None else int(backprop_steps)
        if retained < 1:
            raise ValueError("backprop_steps must be at least 1")
        retained = min(retained, steps)
        grad_start = steps - retained

        exit_policy = adaptive_exit
        if exit_policy is None and execution_mode == "decode":
            exit_policy = self.adaptive_exit_override
        if exit_policy is not None:
            if torch.is_grad_enabled():
                raise ValueError("adaptive exit is an inference-only policy")
            if execution_mode != "decode" and not exit_policy.apply_in_prefill:
                raise ValueError(
                    "adaptive exit is only available in decode mode; "
                    "prefill runs every requested iteration"
                )
            if exit_policy.criterion == "kl" and exit_readout is None:
                raise ValueError("the kl adaptive exit criterion requires an exit_readout")

        batch, sequence = input_ids.shape
        d_model = config.d_model
        if latent_init is not None and tuple(latent_init.shape) != (batch, sequence, d_model):
            raise ValueError(
                f"latent_init must have shape {(batch, sequence, d_model)}, "
                f"got {tuple(latent_init.shape)}"
            )

        core_indices = range(prelude, prelude + core)
        budget = _core_state_budget(token_states, channel_states, core_indices, steps)

        # ---- prelude: exactly the flat stack's loop body.
        (
            hidden_states,
            prelude_token,
            prelude_channel,
            prelude_metrics,
            prelude_aux,
        ) = self._run_block_range(
            hidden_states,
            input_ids,
            range(prelude),
            token_states=token_states,
            channel_states=channel_states,
            per_layer_kwargs=per_layer_kwargs,
            execution_mode=execution_mode,
            attention_mask=attention_mask,
            state_reset_mask=state_reset_mask,
            atxy=atxy,
        )
        injection = hidden_states
        working_dtype = injection.dtype
        normalized_injection = recurrence.normalize_injection(injection)
        # One stochastic-rounding draw and one unique() sync per core block,
        # re-added identically at every iteration.
        core_ple = [
            self.embeddings(input_ids, layer_index=prelude + offset) for offset in range(core)
        ]

        if latent_init is not None:
            latent = latent_init.to(device=injection.device, dtype=torch.float32)
        elif recurrence_config.latent_init_std == 0.0:
            latent = torch.zeros(
                batch, sequence, d_model, device=injection.device, dtype=torch.float32
            )
        else:
            latent = (
                torch.randn(
                    batch,
                    sequence,
                    d_model,
                    device=injection.device,
                    dtype=torch.float32,
                    generator=latent_generator,
                )
                * recurrence_config.latent_init_std
            )

        new_core_token: list[list[Any]] = [[] for _ in range(core)]
        new_core_channel: list[list[Any]] = [[] for _ in range(core)]
        core_aux: list[list[Tensor]] = [[] for _ in range(core)]
        core_metrics: list[Mapping[str, Any]] = [{} for _ in range(core)]
        deltas: list[Tensor] = []
        exited = torch.zeros(batch, dtype=torch.bool, device=injection.device)
        any_exited = False
        iteration_inputs: list[Tensor | None] = [None] * core
        frozen_inputs: list[Tensor | None] = [None] * core
        previous_log_probabilities: Tensor | None = None
        iterations_run = 0

        for iteration in range(steps):
            grad_iteration = iteration >= grad_start
            with nullcontext() if grad_iteration else torch.no_grad():
                if grad_iteration and iteration == grad_start:
                    # The no_grad prefix is a constant; make the boundary explicit.
                    latent = latent.detach()
                previous_latent = latent
                block_input = recurrence.inject(latent, normalized_injection)
                for offset in range(core):
                    index = prelude + offset
                    block_input = block_input + core_ple[offset]
                    if any_exited:
                        block_input = torch.where(
                            exited[:, None, None], frozen_inputs[offset], block_input
                        )
                    iteration_inputs[offset] = block_input
                    token_slot = token_states[index]
                    channel_slot = channel_states[index]
                    block_output = self.blocks[index](
                        block_input,
                        token_mixer_state=(
                            None if token_slot is None else token_slot.iterations[iteration]
                        ),
                        channel_mixer_state=(
                            None if channel_slot is None else channel_slot.iterations[iteration]
                        ),
                        execution_mode=execution_mode,
                        attention_mask=attention_mask,
                        state_reset_mask=state_reset_mask,
                        mixer_kwargs=per_layer_kwargs[index],
                    )
                    block_input = block_output.hidden_states
                    new_core_token[offset].append(block_output.token_mixer_state)
                    new_core_channel[offset].append(block_output.channel_mixer_state)
                    # Overwritten every iteration, so the LAST one survives.
                    core_metrics[offset] = block_output.metrics
                    if grad_iteration and block_output.auxiliary_loss is not None:
                        core_aux[offset].append(block_output.auxiliary_loss)
                latent = recurrence.normalize_latent(block_input).float()
                if any_exited:
                    latent = torch.where(exited[:, None, None], previous_latent, latent)
            iterations_run = iteration + 1
            with torch.no_grad():
                per_sequence_delta = (
                    (latent - previous_latent).norm(dim=-1)
                    / latent.norm(dim=-1).clamp_min(1e-6)
                ).mean(dim=1)
                deltas.append(per_sequence_delta.mean())
            if exit_policy is None or iterations_run >= steps:
                continue
            if iterations_run < exit_policy.min_steps:
                continue
            if (iterations_run - exit_policy.min_steps) % exit_policy.check_every:
                continue
            if exit_policy.criterion == "latent_diff":
                signal = per_sequence_delta
            else:
                probe = self._run_block_range(
                    latent.to(working_dtype),
                    input_ids,
                    range(prelude + core, total),
                    token_states=token_states,
                    channel_states=channel_states,
                    per_layer_kwargs=per_layer_kwargs,
                    execution_mode=execution_mode,
                    attention_mask=attention_mask,
                    state_reset_mask=state_reset_mask,
                    atxy=None,
                    collect=False,
                )[0]
                log_probabilities = F.log_softmax(
                    exit_readout(self.final_norm(probe)[:, -1:, :]).float(), dim=-1
                )
                if previous_log_probabilities is None:
                    signal = torch.full(
                        (batch,), float("inf"), device=injection.device, dtype=torch.float32
                    )
                else:
                    signal = (
                        (
                            log_probabilities.exp()
                            * (log_probabilities - previous_log_probabilities)
                        )
                        .sum(-1)
                        .squeeze(1)
                    )
                previous_log_probabilities = log_probabilities
            newly_exited = (signal < exit_policy.resolved_threshold) & ~exited
            if bool(newly_exited.any()):
                for offset in range(core):
                    frozen = frozen_inputs[offset]
                    frozen_inputs[offset] = (
                        iteration_inputs[offset]
                        if frozen is None
                        else torch.where(
                            newly_exited[:, None, None], iteration_inputs[offset], frozen
                        )
                    )
                exited = exited | newly_exited
                any_exited = True
                if bool(exited.all()):
                    break

        # ---- deeper iteration states for exited sequences and short decodes.
        fill = exit_policy is None or exit_policy.state_policy == "fill"
        for extra in range(iterations_run, budget):
            for offset in range(core):
                index = prelude + offset
                token_slot = token_states[index]
                channel_slot = channel_states[index]
                carried = None if token_slot is None else token_slot.iterations[extra]
                fill_input = frozen_inputs[offset] if any_exited else iteration_inputs[offset]
                if fill and fill_input is not None:
                    new_core_token[offset].append(
                        self.blocks[index].advance_token_mixer(
                            fill_input,
                            token_mixer_state=carried,
                            execution_mode=execution_mode,
                            attention_mask=attention_mask,
                            state_reset_mask=state_reset_mask,
                            mixer_kwargs=per_layer_kwargs[index],
                        )
                    )
                else:
                    new_core_token[offset].append(carried)
                new_core_channel[offset].append(
                    None if channel_slot is None else channel_slot.iterations[extra]
                )

        # ---- coda: today's loop body on the final latent.
        (
            hidden_states,
            coda_token,
            coda_channel,
            coda_metrics,
            coda_aux,
        ) = self._run_block_range(
            latent.to(working_dtype),
            input_ids,
            range(prelude + core, total),
            token_states=token_states,
            channel_states=channel_states,
            per_layer_kwargs=per_layer_kwargs,
            execution_mode=execution_mode,
            attention_mask=attention_mask,
            state_reset_mask=state_reset_mask,
            atxy=atxy,
        )

        auxiliary_losses = list(prelude_aux)
        auxiliary_losses.extend(torch.stack(values).mean() for values in core_aux if values)
        auxiliary_losses.extend(coda_aux)
        token_out = (
            list(prelude_token)
            + [RecurrentMixerStates(tuple(values)) for values in new_core_token]
            + list(coda_token)
        )
        channel_out = (
            list(prelude_channel)
            + [
                None
                if all(value is None for value in values)
                else RecurrentMixerStates(tuple(values))
                for values in new_core_channel
            ]
            + list(coda_channel)
        )
        return BackboneOutput(
            hidden_states=self.final_norm(hidden_states),
            token_mixer_states=tuple(token_out),
            channel_mixer_states=tuple(channel_out),
            auxiliary_loss=torch.stack(auxiliary_losses).mean() if auxiliary_losses else None,
            layer_metrics=tuple(list(prelude_metrics) + core_metrics + list(coda_metrics)),
            recurrence_steps=iterations_run,
            latent_states=latent.detach(),
            latent_deltas=torch.stack(deltas),
            latent_position_correlation=_latent_position_correlation(latent),
        )


class SPALMERCausalLM(nn.Module):
    """Untied vocabulary projection and ordinary next-token objective."""

    def __init__(
        self,
        config: SPALMERConfig,
        backbone: SPALMERBackbone,
        *,
        potentiation_controller: nn.Module | None = None,
        residency: ExpertResidency | None = None,
    ) -> None:
        super().__init__()
        if backbone.config != config:
            raise ValueError(
                "backbone configuration does not match the language-model configuration"
            )
        self.config = config
        self.backbone = backbone
        self.potentiation_controller = potentiation_controller
        # The one resident identity set shared by every routed channel mixer
        # (ledger C13). Registered here so it moves with the model; its buffers
        # are non-persistent because residency is request-level state.
        self.residency = residency
        # Plain runtime state: physical cache tensors are deliberately absent
        # from the module tree and checkpoint schema.
        self._expert_offload_manager: ExpertOffloadManager | None = None
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=config.initializer_range)
        # Shared "average surprise" telemetry (ledger C08/C13): the EMA of
        # realized next-token NLL, consumed by the inference residency controller.
        self.register_buffer("surprise_ema", torch.zeros((), dtype=torch.float32))
        self.register_buffer("surprise_observations", torch.zeros((), dtype=torch.long))

    def _apply(self, fn):
        manager = self._expert_offload_manager
        if manager is not None and not manager.allow_model_apply:
            raise RuntimeError(
                "model.to/cpu/cuda/dtype casts are disabled while expert offload is active; "
                "call disable_expert_offload(...) first"
            )
        super()._apply(fn)
        # Module-wide low-precision casts are for weights, not for a slowly
        # moving average that a bf16 update step would silently round away.
        self._buffers["surprise_ema"] = self._buffers["surprise_ema"].float()
        return self

    def train(self, mode: bool = True):
        """Keep detached inference caches out of all training paths."""

        if mode and self._expert_offload_manager is not None:
            raise RuntimeError(
                "expert offload is inference-only; disable it before entering training mode"
            )
        return super().train(mode)

    @property
    def expert_offload_enabled(self) -> bool:
        """Whether complete expert banks are CPU-backed with resident device caches."""

        return self._expert_offload_manager is not None

    @property
    def execution_device(self) -> torch.device:
        """Authoritative device for input tensors and non-expert execution."""

        manager = self._expert_offload_manager
        return self.lm_head.weight.device if manager is None else manager.target_device

    def enable_expert_offload(
        self,
        device: torch.device | str,
        *,
        cache_size: int | None = None,
        resident_ids: Sequence[int] | None = None,
        non_blocking: bool = True,
        pin_memory: bool | None = None,
        paging: bool = True,
    ) -> ExpertOffloadTelemetry:
        """Selectively place this eval model for bounded expert inference.

        Call this instead of ``model.to(device)``.  Shared weights move to the
        inference device, complete expert masters stay on CPU, and only a
        bounded set of expert rows is staged on the device. Paged full-pool
        routing is the default; pass ``paging=False`` for legacy
        request-global resident routing.

        With a depth-recurrent core each core block re-routes once per
        iteration, so the core banks' ``stage_operations``/``transferred_rows``/
        ``evicted_rows`` counters grow up to ``recurrence_steps`` times per
        forward (``layer_count`` stays the physical bank count). Size
        ``cache_size`` to cover the union of experts selected across iterations,
        not just one pass.
        """

        from spalmer.experts.offload import enable_expert_offload

        return enable_expert_offload(
            self,
            device,
            cache_size=cache_size,
            resident_ids=resident_ids,
            non_blocking=non_blocking,
            pin_memory=pin_memory,
            paging=paging,
        )

    def disable_expert_offload(self, *, device: torch.device | str = "cpu") -> None:
        """Drop inference caches and restore ordinary full-bank placement."""

        from spalmer.experts.offload import disable_expert_offload

        disable_expert_offload(self, device=device)

    def expert_offload_telemetry(self) -> ExpertOffloadTelemetry | None:
        """Return current physical placement/traffic telemetry, if enabled."""

        manager = self._expert_offload_manager
        return None if manager is None else manager.telemetry()

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

        if self.residency is not None:
            return self.residency.active_experts
        for block in self.backbone.blocks:
            count = getattr(block.channel_mixer, "active_experts", None)
            if isinstance(count, int):
                return count
        return None

    @property
    def active_experts_override(self) -> int | None:
        """Per-token top-``k`` override applied to the routed mixers, or ``None``."""

        if self.residency is not None:
            return self.residency.active_experts_override
        for block in self.backbone.blocks:
            mixer = block.channel_mixer
            if hasattr(mixer, "active_experts_override"):
                return mixer.active_experts_override
        return None

    def set_active_experts(self, count: int | None) -> None:
        """Override the per-token top-``k`` on every routed channel mixer.

        This changes execution capacity only. It deliberately does not change
        resident identities or end a controller-managed request; callers must
        use :meth:`end_residency_request` (or :meth:`residency_session`) when
        they intend to restore request state.
        """

        if self.residency is not None:
            self.residency.set_active_experts(count)
            return
        for block in self.backbone.blocks:
            setter = getattr(block.channel_mixer, "set_active_experts", None)
            if callable(setter):
                setter(count)

    def end_residency_request(self) -> None:
        """End a dynamic expert request, restoring its prior ids and top-``k``."""

        if self.residency is not None:
            self.residency.end_request()

    @property
    def resident_expert_ids(self) -> tuple[int, ...]:
        """Exact resident expert ids shared by every layer (all ids when unrouted)."""

        if self.residency is not None:
            return self.residency.ids
        for block in self.backbone.blocks:
            bank = getattr(block.channel_mixer, "experts", None)
            if bank is not None:
                return tuple(range(bank.num_experts))
        return ()

    @contextmanager
    def residency_session(self, ids: Sequence[int] | None = None) -> Iterator[SPALMERCausalLM]:
        """Scope request residency; prior identities and top-``k`` return on exit."""

        if self.residency is None:
            yield self
            return
        with self.residency.session(ids):
            yield self

    @property
    def is_recurrent(self) -> bool:
        """Whether this model has a depth-recurrent core."""

        return self.config.recurrence is not None

    @property
    def recurrence_default_steps(self) -> int | None:
        """The checkpoint's own inference depth, or ``None`` when not recurrent."""

        recurrence = self.config.recurrence
        return None if recurrence is None else recurrence.default_steps

    def set_recurrence_defaults(
        self,
        steps: int | None = None,
        adaptive_exit: AdaptiveExit | None = None,
    ) -> None:
        """Apply a depth policy to every call that passes no explicit kwargs.

        Explicit ``recurrence_steps``/``adaptive_exit`` forward arguments still
        win. The adaptive-exit default applies to decode calls only, so a bare
        ``model(ids)`` prefill keeps running the full step budget. Deliberately
        not thread-safe: the yyLab engine serialises generation.
        """

        if not self.is_recurrent:
            raise ValueError("model has no recurrent core; recurrence defaults are not accepted")
        if steps is not None and int(steps) < 1:
            raise ValueError("recurrence steps must be at least 1")
        self.backbone.recurrence_steps_override = None if steps is None else int(steps)
        self.backbone.adaptive_exit_override = adaptive_exit

    def clear_recurrence_defaults(self) -> None:
        """Fall back to ``config.recurrence.default_steps`` and fixed depth."""

        self.backbone.recurrence_steps_override = None
        self.backbone.adaptive_exit_override = None

    def parameter_accounting(self) -> ParameterAccounting:
        """Exact per-component, resident, and per-token parameter counts."""

        from spalmer.experts.accounting import account_parameters

        return account_parameters(self)

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
        atxy: ATXYRequest | None = None,
        recurrence_steps: int | None = None,
        backprop_steps: int | None = None,
        latent_init: Tensor | None = None,
        adaptive_exit: AdaptiveExit | None = None,
        latent_generator: torch.Generator | None = None,
    ) -> CausalLMOutput:
        resolved_exit = adaptive_exit
        if resolved_exit is None and execution_mode == "decode":
            resolved_exit = getattr(self.backbone, "adaptive_exit_override", None)
        exit_readout = (
            (lambda hidden: self.lm_head(hidden))
            if resolved_exit is not None and resolved_exit.criterion == "kl"
            else None
        )
        backbone_output = self.backbone(
            input_ids,
            token_mixer_states=token_mixer_states,
            channel_mixer_states=channel_mixer_states,
            execution_mode=execution_mode,
            attention_mask=attention_mask,
            state_reset_mask=state_reset_mask,
            layer_mixer_kwargs=layer_mixer_kwargs,
            atxy=atxy,
            recurrence_steps=recurrence_steps,
            backprop_steps=backprop_steps,
            latent_init=latent_init,
            adaptive_exit=adaptive_exit,
            latent_generator=latent_generator,
            exit_readout=exit_readout,
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
            recurrence_steps=backbone_output.recurrence_steps,
            latent_states=backbone_output.latent_states,
            latent_deltas=backbone_output.latent_deltas,
            latent_position_correlation=backbone_output.latent_position_correlation,
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


def _validate_states(
    name: str,
    states: Sequence[Any] | None,
    config: SPALMERConfig,
    count: int,
) -> Sequence[Any]:
    """Length-check a mixer state container and fail closed on core/flat mixups."""

    if states is None:
        return (None,) * count
    if len(states) != count:
        raise ValueError(f"expected {count} {name} states, got {len(states)}")
    core_indices = set(config.core_layer_indices)
    for index, slot in enumerate(states):
        if index in core_indices:
            if slot is not None and not isinstance(slot, RecurrentMixerStates):
                raise TypeError(
                    f"core block {index} {name} state must be RecurrentMixerStates, "
                    f"got {type(slot).__name__}"
                )
        elif isinstance(slot, RecurrentMixerStates):
            raise TypeError(
                f"block {index} is outside the recurrent core and cannot carry "
                f"RecurrentMixerStates {name} states"
            )
    return states


def _core_state_budget(
    token_states: Sequence[Any],
    channel_states: Sequence[Any],
    core_indices: Sequence[int],
    steps: int,
) -> int:
    """Output slot length of every core block: the supplied budget, else ``steps``."""

    budget: int | None = None
    for index in core_indices:
        for slot in (token_states[index], channel_states[index]):
            if slot is None:
                continue
            length = len(slot)
            if length < steps:
                raise ValueError(
                    f"core block {index} carries {length} iteration states but {steps} "
                    "recurrence steps were requested; prefill deeper or reduce "
                    "recurrence_steps"
                )
            if budget is None:
                budget = length
            elif length != budget:
                raise ValueError(
                    "recurrent core state slots must share one iteration budget; "
                    f"core block {index} carries {length}, expected {budget}"
                )
    return steps if budget is None else budget


def _latent_position_correlation(latent: Tensor) -> Tensor | None:
    """Mean off-diagonal cosine similarity between sampled latent positions.

    A high value means the recurrent core has collapsed the per-position
    latents onto one shared direction. ``None`` for single-token forwards,
    where there is no off-diagonal pair.
    """

    sequence = latent.shape[1]
    if sequence < 2:
        return None
    # Explicitly float32 even under a bf16 autocast region: this is a
    # diagnostic, and a bf16 matmul of near-parallel vectors is not one.
    with (
        torch.no_grad(),
        torch.amp.autocast(device_type=latent.device.type, enabled=False),
    ):
        count = min(64, sequence)
        positions = (
            torch.linspace(0, sequence - 1, count, device=latent.device).round().long()
        )
        sampled = F.normalize(latent.detach().float()[:, positions, :], dim=-1)
        similarity = sampled @ sampled.transpose(1, 2)
        off_diagonal = ~torch.eye(count, dtype=torch.bool, device=latent.device)
        return similarity[:, off_diagonal].mean(dim=1).mean()


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
