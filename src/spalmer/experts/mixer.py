"""The micro-expert channel mixer (SPALMER ledger C07/C08/C13).

Each layer's channel update is the sum of two paths:

- an always-on **shared** SwiGLU path (general capacity that does not depend
  on which experts are resident), and
- the **routed** path: every token is sent to the ``active_experts``
  least-surprised experts among the currently *resident* identities, and the
  weighted combination of their updates is returned.

The result is a :class:`spalmer.modeling.ChannelMixerOutput`, ready to drop
into a ``SPALMERBlock`` channel-mixer slot; the residual addition stays with
the caller. The shared/routed parameter split is explicit:
``config.shared_parameters_per_layer`` and
``config.expert_pool_parameters_per_layer``.
"""

from __future__ import annotations

from typing import Any
from weakref import ref

import torch
from torch import Tensor, nn

from spalmer.experts.bank import MicroExpertBank
from spalmer.experts.config import MicroExpertsConfig
from spalmer.experts.losses import expert_utilization, load_balance_loss
from spalmer.experts.potentiation import ExpertPotentiationController
from spalmer.experts.residency import ExpertResidency
from spalmer.experts.router import SurpriseRouter, select_least_surprised_experts
from spalmer.modeling import ChannelMixerOutput
from spalmer.nn import SwiGLU


class MicroExpertChannelMixer(nn.Module):
    """Shared SwiGLU path plus a routed bank of small gated-MLP experts.

    Args:
        config: Bank, shared-path, and routing configuration.
        router: Optional shared router. Passing the same router instance to
            several layer-local mixers preserves expert identity across
            layers (one shared router, many banks) and requires every bank
            to agree on ``num_experts``.
        potentiation_controller: Optional shared expert-wide precision
            controller; owned locally when omitted.
        residency: Optional shared :class:`ExpertResidency`. Every layer of a
            model must share one instance so that the resident identity set
            is coherent across depth; owned locally when omitted (tests).
    """

    def __init__(
        self,
        config: MicroExpertsConfig,
        *,
        router: SurpriseRouter | None = None,
        potentiation_controller: ExpertPotentiationController | None = None,
        residency: ExpertResidency | None = None,
    ) -> None:
        super().__init__()
        if router is not None:
            if router.num_experts != config.num_experts:
                raise ValueError(
                    f"shared router serves {router.num_experts} experts; "
                    f"this bank has {config.num_experts}"
                )
            if router.num_features != config.d_model:
                raise ValueError(
                    f"shared router expects width {router.num_features}; "
                    f"this layer has {config.d_model}"
                )
        self.config = config
        self.router = router if router is not None else SurpriseRouter(config)
        self.experts = MicroExpertBank(config)
        shared_width = config.resolved_shared_inter_dim
        self.shared = (
            SwiGLU(config.d_model, shared_width, initializer_range=config.initializer_range)
            if shared_width > 0
            else None
        )
        if potentiation_controller is None:
            self._owned_potentiation_controller = ExpertPotentiationController(config)
            potentiation_controller = self._owned_potentiation_controller
        elif potentiation_controller.config != config:
            raise ValueError("potentiation controller configuration does not match this bank")
        if residency is None:
            self._owned_residency = ExpertResidency(config)
            residency = self._owned_residency
        elif residency.config != config:
            raise ValueError("residency configuration does not match this bank")
        # A factory-built model owns one authoritative controller and one
        # residency set at the LM boundary. Weak references prevent registering
        # the same state under every layer-local bank.
        object.__setattr__(self, "_potentiation_controller_ref", ref(potentiation_controller))
        object.__setattr__(self, "_residency_ref", ref(residency))
        self._active_experts_override: int | None = None

    @property
    def active_experts(self) -> int:
        """Per-token top-``k``: the override if one is set, else the config value."""

        if self._active_experts_override is not None:
            return self._active_experts_override
        return self.config.active_experts

    @property
    def active_experts_override(self) -> int | None:
        """Per-token top-``k`` override currently applied, or ``None``."""

        return self._active_experts_override

    @property
    def max_active_experts(self) -> int:
        """Upper bound of the per-token top-``k`` (bounded by the bank size)."""

        return min(self.config.max_active_experts, self.config.num_experts)

    def set_active_experts(self, count: int | None) -> None:
        """Override the per-token top-``k``; ``None`` restores the configured value.

        This does not change residency. Growing capacity for a request is the
        job of :class:`ExpertResidency` (explicit ids), never of this knob.
        """

        if count is not None and not (
            self.config.min_active_experts <= count <= self.max_active_experts
        ):
            raise ValueError(
                f"active expert count must be in [{self.config.min_active_experts}, "
                f"{self.max_active_experts}]; got {count}"
            )
        self._active_experts_override = count

    @property
    def potentiation_controller(self) -> ExpertPotentiationController:
        controller = self._potentiation_controller_ref()
        if controller is None:
            raise RuntimeError("the shared potentiation controller no longer exists")
        return controller

    @property
    def residency(self) -> ExpertResidency:
        residency = self._residency_ref()
        if residency is None:
            raise RuntimeError("the shared expert residency no longer exists")
        return residency

    @property
    def has_shared_path(self) -> bool:
        return self.shared is not None

    def forward(self, hidden_states: Tensor, *, state: Any = None) -> ChannelMixerOutput:
        """Run the shared path, route to resident experts, and combine.

        Args:
            hidden_states: ``[batch, seq, d_model]`` layer inputs.
            state: Accepted for shell compatibility; this mixer is stateless.

        Returns:
            :class:`ChannelMixerOutput` whose ``update`` is the shared update
            plus the weighted combination of the selected expert updates,
            ``auxiliary_loss`` is the resident-normalized load-balancing loss,
            and ``metrics`` carries ``expert_ids``, ``routing_weights``,
            ``router_scores`` (predicted surprise), ``expert_utilization``,
            ``expert_quantization_error``, ``promoted_experts``,
            ``resident_experts``, and ``num_active_experts``.
        """

        if hidden_states.ndim != 3:
            raise ValueError(
                f"hidden_states must have shape [batch, seq, d_model]; "
                f"got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[-1] != self.config.d_model:
            raise ValueError(
                f"expected trailing dimension {self.config.d_model}; got {hidden_states.shape[-1]}"
            )
        batch, seq_len, d_model = hidden_states.shape
        num_active = self.active_experts
        residency = self.residency
        resident_mask = residency.resident_mask
        full_pool = residency.is_full
        if not full_pool and residency.size < num_active:
            raise ValueError(
                f"{residency.size} resident experts cannot serve a per-token top-{num_active}"
            )
        resident_ids = None if full_pool else residency.resident_ids

        scores = self.router(hidden_states)
        expert_ids, routing_weights = select_least_surprised_experts(
            scores, num_active, None if full_pool else resident_mask
        )

        flat_hidden = hidden_states.reshape(batch * seq_len, d_model)
        token_index = torch.arange(batch * seq_len, device=hidden_states.device).repeat_interleave(
            num_active
        )
        expert_index = expert_ids.reshape(-1)
        flat_weights = routing_weights.reshape(-1)
        controller = self.potentiation_controller
        routed = self.experts.execute_routing(
            flat_hidden,
            token_index,
            expert_index,
            flat_weights,
            promoted_mask=controller.promoted_mask,
            resident_ids=resident_ids,
        ).reshape(batch, seq_len, d_model)
        update = routed if self.shared is None else routed + self.shared(hidden_states)

        utilization = expert_utilization(expert_ids, self.config.num_experts)
        quantization_error = self.experts.quantization_error(expert_ids, resident_ids)
        auxiliary_loss = load_balance_loss(scores, expert_ids, None if full_pool else resident_mask)
        metrics: dict[str, Any] = {
            "expert_ids": expert_ids,
            "routing_weights": routing_weights,
            "router_scores": scores,
            "expert_utilization": utilization,
            "expert_quantization_error": quantization_error,
            "promoted_experts": controller.promoted_mask.detach().clone(),
            "resident_experts": resident_mask.detach().clone(),
            # Distinct experts executed this pass, kept on device (no sync).
            "num_active_experts": (utilization > 0).sum(),
        }
        return ChannelMixerOutput(
            update=update,
            state=None,
            auxiliary_loss=auxiliary_loss,
            metrics=metrics,
        )
