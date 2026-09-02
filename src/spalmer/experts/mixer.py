"""The micro-expert channel mixer (SPALMER ledger C07/C08/C13).

Routes every token to the ``active_experts`` least-surprised experts and
returns the weighted combination of their updates as a
:class:`spalmer.modeling.ChannelMixerOutput`, ready to drop into a
``SPALMERBlock`` channel-mixer slot. The residual addition stays with the
caller.
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
from spalmer.experts.router import SurpriseRouter, select_least_surprised_experts
from spalmer.modeling import ChannelMixerOutput


class MicroExpertChannelMixer(nn.Module):
    """Routed bank of small gated-MLP experts serving one layer.

    Args:
        config: Bank and routing configuration.
        router: Optional shared router. Passing the same router instance to
            several layer-local mixers preserves expert identity across
            layers (one shared router, many banks) and requires every bank
            to agree on ``num_experts``.
    """

    def __init__(
        self,
        config: MicroExpertsConfig,
        *,
        router: SurpriseRouter | None = None,
        potentiation_controller: ExpertPotentiationController | None = None,
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
        if potentiation_controller is None:
            self._owned_potentiation_controller = ExpertPotentiationController(config)
            potentiation_controller = self._owned_potentiation_controller
        elif potentiation_controller.config != config:
            raise ValueError("potentiation controller configuration does not match this bank")
        # A factory-built model owns one authoritative controller at the LM
        # boundary. The weak reference prevents registering the same state under
        # every layer-local bank.
        object.__setattr__(self, "_potentiation_controller_ref", ref(potentiation_controller))
        self._active_experts_override: int | None = None

    @property
    def active_experts(self) -> int:
        """Experts executed per token: the residency override, else the config value."""

        if self._active_experts_override is not None:
            return self._active_experts_override
        return self.config.active_experts

    @property
    def active_experts_override(self) -> int | None:
        """Residency override currently applied, or ``None`` for the configured count."""

        return self._active_experts_override

    @property
    def max_active_experts(self) -> int:
        """Soft residency cap (ledger C13), bounded by the bank size."""

        return min(self.config.max_active_experts, self.config.num_experts)

    def set_active_experts(self, count: int | None) -> None:
        """Change inference residency; ``None`` restores the configured count.

        This is the C13 dynamic-rounding knob: the bank keeps every expert, and
        only the number executed per token moves within
        ``[min_active_experts, max_active_experts]``.
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

    def forward(self, hidden_states: Tensor, *, state: Any = None) -> ChannelMixerOutput:
        """Route, execute the selected experts, and combine their updates.

        Args:
            hidden_states: ``[batch, seq, d_model]`` layer inputs.
            state: Accepted for shell compatibility; this mixer is stateless.

        Returns:
            :class:`ChannelMixerOutput` whose ``update`` is the weighted
            combination of the selected expert updates, ``auxiliary_loss``
            is the load-balancing loss, and ``metrics`` carries
            ``expert_ids``, ``routing_weights``, ``router_scores``
            (predicted surprise), ``expert_utilization``, and
            ``num_active_experts``.
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

        scores = self.router(hidden_states)
        expert_ids, routing_weights = select_least_surprised_experts(scores, num_active)

        flat_hidden = hidden_states.reshape(batch * seq_len, d_model)
        token_index = torch.arange(batch * seq_len, device=hidden_states.device).repeat_interleave(
            num_active
        )
        expert_index = expert_ids.reshape(-1)
        flat_weights = routing_weights.reshape(-1)
        controller = self.potentiation_controller
        update = self.experts.execute_routing(
            flat_hidden,
            token_index,
            expert_index,
            flat_weights,
            promoted_mask=controller.promoted_mask,
        )

        utilization = expert_utilization(expert_ids, self.config.num_experts)
        quantization_error = self.experts.quantization_error(expert_ids)
        auxiliary_loss = load_balance_loss(scores, expert_ids)
        metrics: dict[str, Any] = {
            "expert_ids": expert_ids,
            "routing_weights": routing_weights,
            "router_scores": scores,
            "expert_utilization": utilization,
            "expert_quantization_error": quantization_error,
            "promoted_experts": controller.promoted_mask.detach().clone(),
            "num_active_experts": int(expert_index.unique().numel()),
        }
        return ChannelMixerOutput(
            update=update.reshape(batch, seq_len, d_model),
            state=None,
            auxiliary_loss=auxiliary_loss,
            metrics=metrics,
        )
