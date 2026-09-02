"""Tensorized bank of small gated-MLP experts (SPALMER ledger C07).

All experts live in three stacked parameters (gate, up, down), one per
projection. Execution runs only the experts actually selected by the
router: one small matmul triple per distinct selected expert, with
scatter-add accumulation back onto tokens.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from spalmer.experts.config import MicroExpertsConfig


class MicroExpertBank(nn.Module):
    """A layer-local bank of ``num_experts`` small gated-MLP experts.

    Expert ``e`` computes ``down(silu(x @ gate_e) * (x @ up_e))``. Expert
    identity (the index ``e``) is coherent across layers; the weights are
    local to the owning layer.
    """

    def __init__(self, config: MicroExpertsConfig) -> None:
        super().__init__()
        self.config = config
        num_experts = config.num_experts
        d_model = config.d_model
        inter_dim = config.resolved_inter_dim
        std = config.initializer_range
        self.gate_proj = nn.Parameter(
            torch.randn(num_experts, d_model, inter_dim) * std
        )
        self.up_proj = nn.Parameter(torch.randn(num_experts, d_model, inter_dim) * std)
        self.down_proj = nn.Parameter(torch.randn(num_experts, inter_dim, d_model) * std)

    @property
    def num_experts(self) -> int:
        return self.config.num_experts

    def expert_forward(self, hidden_states: Tensor, expert_index: int) -> Tensor:
        """Apply a single expert to ``[tokens, d_model]`` hidden states."""

        gate = F.silu(hidden_states @ self.gate_proj[expert_index])
        up = hidden_states @ self.up_proj[expert_index]
        return (gate * up) @ self.down_proj[expert_index]

    def execute_routing(
        self,
        hidden_states: Tensor,
        token_index: Tensor,
        expert_index: Tensor,
        routing_weights: Tensor,
    ) -> Tensor:
        """Execute only the selected (token, expert) routing pairs.

        Args:
            hidden_states: ``[num_tokens, d_model]`` flattened inputs.
            token_index: ``[num_pairs]`` token row of each routing pair.
            expert_index: ``[num_pairs]`` expert of each routing pair.
            routing_weights: ``[num_pairs]`` combination weight of each pair.

        Returns:
            ``[num_tokens, d_model]`` weighted combination of the selected
            expert updates (zero for tokens is impossible: every token owns
            exactly ``k`` pairs).
        """

        num_tokens = hidden_states.shape[0]
        update = hidden_states.new_zeros(num_tokens, hidden_states.shape[-1])
        for expert in torch.unique(expert_index):
            slot_mask = expert_index == expert
            rows = token_index[slot_mask]
            outputs = self.expert_forward(hidden_states[rows], int(expert))
            weighted = routing_weights[slot_mask].unsqueeze(-1) * outputs
            update = update.index_add(0, rows, weighted)
        return update
