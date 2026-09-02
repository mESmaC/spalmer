"""The micro-expert channel mixer (SPALMER ledger C07/C08/C13).

Routes every token to the ``active_experts`` least-surprised experts and
returns the weighted combination of their updates as a
:class:`spalmer.modeling.ChannelMixerOutput`, ready to drop into a
``SPALMERBlock`` channel-mixer slot. The residual addition stays with the
caller.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from spalmer.experts.bank import MicroExpertBank
from spalmer.experts.config import MicroExpertsConfig
from spalmer.experts.losses import expert_utilization, load_balance_loss
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

    def __init__(self, config: MicroExpertsConfig, *, router: SurpriseRouter | None = None):
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

    @property
    def active_experts(self) -> int:
        return self.config.active_experts

    def forward(
        self, hidden_states: Tensor, *, state: Any = None
    ) -> ChannelMixerOutput:
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
                f"expected trailing dimension {self.config.d_model}; "
                f"got {hidden_states.shape[-1]}"
            )
        batch, seq_len, d_model = hidden_states.shape
        num_active = self.config.active_experts

        scores = self.router(hidden_states)
        expert_ids, routing_weights = select_least_surprised_experts(scores, num_active)

        flat_hidden = hidden_states.reshape(batch * seq_len, d_model)
        token_index = (
            torch.arange(batch * seq_len, device=hidden_states.device)
            .repeat_interleave(num_active)
        )
        expert_index = expert_ids.reshape(-1)
        flat_weights = routing_weights.reshape(-1)
        update = self.experts.execute_routing(flat_hidden, token_index, expert_index, flat_weights)

        utilization = expert_utilization(expert_ids, self.config.num_experts)
        auxiliary_loss = load_balance_loss(scores, expert_ids)
        metrics: dict[str, Any] = {
            "expert_ids": expert_ids,
            "routing_weights": routing_weights,
            "router_scores": scores,
            "expert_utilization": utilization,
            "num_active_experts": int(expert_index.unique().numel()),
        }
        return ChannelMixerOutput(
            update=update.reshape(batch, seq_len, d_model),
            state=None,
            auxiliary_loss=auxiliary_loss,
            metrics=metrics,
        )
