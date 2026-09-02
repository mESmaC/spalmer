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
from spalmer.nn import fake_quantize_low_bit


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

    def expert_forward(
        self,
        hidden_states: Tensor,
        expert_index: int,
        promoted_mask: Tensor | None = None,
    ) -> Tensor:
        """Apply a single expert to ``[tokens, d_model]`` hidden states."""

        promoted = None if promoted_mask is None else promoted_mask[expert_index]
        gate_weight = self._effective_weight(self.gate_proj[expert_index], promoted)
        up_weight = self._effective_weight(self.up_proj[expert_index], promoted)
        down_weight = self._effective_weight(self.down_proj[expert_index], promoted)
        gate = F.silu(hidden_states @ gate_weight)
        up = hidden_states @ up_weight
        return (gate * up) @ down_weight

    def _effective_weight(self, shadow: Tensor, promoted: Tensor | None) -> Tensor:
        """Use the low-bit substrate or its reversible shadow-precision residual."""

        if not self.config.expert_fake_quantization:
            return shadow
        low = fake_quantize_low_bit(
            shadow,
            bits=self.config.expert_quant_bits,
            stochastic=self.training and self.config.expert_stochastic_rounding,
            straight_through=self.training,
        )
        if promoted is None:
            return low
        # One scalar Boolean controls the complete expert. Expressing the
        # reversible residual algebraically avoids a GPU synchronization per
        # selected expert while preserving the same shadow-weight gradient.
        enabled = promoted.to(device=shadow.device, dtype=shadow.dtype)
        return low + enabled * (shadow - low)

    @torch.no_grad()
    def quantization_error(self, expert_ids: Tensor) -> Tensor:
        """Return deterministic reconstruction MSE for selected expert identities."""

        errors = torch.zeros(
            self.num_experts,
            dtype=torch.float32,
            device=self.gate_proj.device,
        )
        if not self.config.expert_fake_quantization:
            return errors
        for expert in torch.unique(expert_ids):
            index = int(expert)
            squared_error = torch.zeros((), device=self.gate_proj.device, dtype=torch.float32)
            squared_weight = torch.zeros_like(squared_error)
            for parameter in (self.gate_proj, self.up_proj, self.down_proj):
                shadow = parameter[index]
                low = fake_quantize_low_bit(
                    shadow,
                    bits=self.config.expert_quant_bits,
                    stochastic=False,
                    straight_through=False,
                )
                squared_error.add_((shadow.float() - low.float()).square().sum())
                squared_weight.add_(shadow.float().square().sum())
            errors[index] = squared_error / squared_weight.clamp_min(1e-12)
        return errors

    def execute_routing(
        self,
        hidden_states: Tensor,
        token_index: Tensor,
        expert_index: Tensor,
        routing_weights: Tensor,
        promoted_mask: Tensor | None = None,
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
            outputs = self.expert_forward(
                hidden_states[rows],
                int(expert),
                promoted_mask,
            )
            weighted = routing_weights[slot_mask].unsqueeze(-1) * outputs
            update = update.index_add(0, rows, weighted)
        return update
