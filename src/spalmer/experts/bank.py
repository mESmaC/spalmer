"""Tensorized bank of small gated-MLP experts (SPALMER ledger C07).

All experts live in three stacked parameters (gate, up, down), one per
projection. Two execution paths produce the same routed update:

- ``grouped`` (default): pairs are sorted by expert, the tokens of every
  expert that received at least one pair are gathered into one padded
  ``[groups, capacity, d_model]`` block and executed with three batched
  matmuls. There is no per-expert Python loop; the host synchronizes twice per
  call (which groups are non-empty, and the padded capacity). Dropping empty
  groups keeps the block proportional to the executed pairs even when routing
  collapses onto a few experts.
- ``loop``: the per-expert reference path (one small matmul triple per
  distinct selected expert) used for equivalence checks and tiny CPU runs.
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
        self.gate_proj = nn.Parameter(torch.randn(num_experts, d_model, inter_dim) * std)
        self.up_proj = nn.Parameter(torch.randn(num_experts, d_model, inter_dim) * std)
        self.down_proj = nn.Parameter(torch.randn(num_experts, inter_dim, d_model) * std)

    @property
    def num_experts(self) -> int:
        return self.config.num_experts

    @property
    def parameters_per_expert(self) -> int:
        """Exact parameter count of one expert identity in this layer."""

        return self.gate_proj[0].numel() + self.up_proj[0].numel() + self.down_proj[0].numel()

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
        """Use the low-bit substrate or its reversible shadow-precision residual.

        ``shadow`` may be one expert's matrix or a stack ``[m, ...]`` of them;
        ``promoted`` is then a Boolean scalar or a ``[m]`` vector. The
        quantization scale is taken along the last dimension, so a stacked
        call quantizes each expert exactly as the single-expert call does.
        """

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
        # One Boolean per expert controls the complete expert. Expressing the
        # reversible residual algebraically avoids a GPU synchronization per
        # expert while preserving the same shadow-weight gradient.
        enabled = promoted.to(device=shadow.device, dtype=shadow.dtype)
        enabled = enabled.reshape(-1, *([1] * (shadow.dim() - 1))) if enabled.dim() else enabled
        return low + enabled * (shadow - low)

    @torch.no_grad()
    def quantization_error(
        self,
        expert_ids: Tensor,
        resident_ids: Tensor | None = None,
    ) -> Tensor:
        """Deterministic reconstruction MSE of every expert selected this pass.

        Vectorized over the resident experts (all experts when ``resident_ids``
        is ``None``); entries of experts that were not selected are zero, which
        the potentiation controller treats as "not measured".
        """

        errors = torch.zeros(self.num_experts, dtype=torch.float32, device=self.gate_proj.device)
        if not self.config.expert_fake_quantization:
            return errors
        squared_error: Tensor | None = None
        squared_weight: Tensor | None = None
        for parameter in (self.gate_proj, self.up_proj, self.down_proj):
            shadow = parameter if resident_ids is None else parameter[resident_ids]
            low = fake_quantize_low_bit(
                shadow,
                bits=self.config.expert_quant_bits,
                stochastic=False,
                straight_through=False,
            )
            error = (shadow.float() - low.float()).square().sum(dim=(1, 2))
            weight = shadow.float().square().sum(dim=(1, 2))
            squared_error = error if squared_error is None else squared_error + error
            squared_weight = weight if squared_weight is None else squared_weight + weight
        assert squared_error is not None and squared_weight is not None
        measured = squared_error / squared_weight.clamp_min(1e-12)
        if resident_ids is None:
            errors = measured
        else:
            errors[resident_ids] = measured
        selected = _pair_counts(expert_ids.reshape(-1), self.num_experts) > 0
        return errors * selected.to(errors.dtype)

    def execute_routing(
        self,
        hidden_states: Tensor,
        token_index: Tensor,
        expert_index: Tensor,
        routing_weights: Tensor,
        promoted_mask: Tensor | None = None,
        resident_ids: Tensor | None = None,
    ) -> Tensor:
        """Execute only the selected (token, expert) routing pairs.

        Args:
            hidden_states: ``[num_tokens, d_model]`` flattened inputs.
            token_index: ``[num_pairs]`` token row of each routing pair.
            expert_index: ``[num_pairs]`` expert of each routing pair.
            routing_weights: ``[num_pairs]`` combination weight of each pair.
            promoted_mask: Optional ``[num_experts]`` Boolean potentiation mask.
            resident_ids: Optional ``[m]`` resident expert ids. The grouped
                path executes exactly these groups; ``None`` means the full
                pool is resident.

        Returns:
            ``[num_tokens, d_model]`` weighted combination of the selected
            expert updates.
        """

        if self.config.expert_execution == "loop":
            return self._execute_loop(
                hidden_states, token_index, expert_index, routing_weights, promoted_mask
            )
        return self._execute_grouped(
            hidden_states, token_index, expert_index, routing_weights, promoted_mask, resident_ids
        )

    def _execute_loop(
        self,
        hidden_states: Tensor,
        token_index: Tensor,
        expert_index: Tensor,
        routing_weights: Tensor,
        promoted_mask: Tensor | None,
    ) -> Tensor:
        num_tokens = hidden_states.shape[0]
        update = hidden_states.new_zeros(num_tokens, hidden_states.shape[-1])
        for expert in torch.unique(expert_index):
            slot_mask = expert_index == expert
            rows = token_index[slot_mask]
            outputs = self.expert_forward(hidden_states[rows], int(expert), promoted_mask)
            weighted = routing_weights[slot_mask].unsqueeze(-1) * outputs
            update = update.index_add(0, rows, weighted)
        return update

    def _execute_grouped(
        self,
        hidden_states: Tensor,
        token_index: Tensor,
        expert_index: Tensor,
        routing_weights: Tensor,
        promoted_mask: Tensor | None,
        resident_ids: Tensor | None,
    ) -> Tensor:
        num_tokens, d_model = hidden_states.shape
        num_pairs = expert_index.numel()
        device = hidden_states.device
        update = hidden_states.new_zeros(num_tokens, d_model)
        if num_pairs == 0:
            return update

        counts = _pair_counts(expert_index, self.num_experts)
        starts = torch.cumsum(counts, dim=0) - counts
        order = torch.argsort(expert_index, stable=True)
        candidates = (
            torch.arange(self.num_experts, device=device) if resident_ids is None else resident_ids
        )
        # Host synchronization 1: which candidate experts received any pair.
        group_ids = candidates[counts[candidates] > 0]
        if group_ids.numel() == 0:
            return update
        group_counts = counts[group_ids]
        group_starts = starts[group_ids]
        # Host synchronization 2: the padded capacity of the largest group.
        capacity = int(group_counts.max())

        slot = torch.arange(capacity, device=device)
        valid = slot[None, :] < group_counts[:, None]
        gather = (group_starts[:, None] + slot[None, :]).clamp_max(num_pairs - 1)
        pair_rows = order[gather]
        token_rows = token_index[pair_rows]
        inputs = hidden_states[token_rows]

        gate_weight, up_weight, down_weight = self._stacked_effective_weights(
            group_ids, promoted_mask
        )
        hidden = F.silu(torch.bmm(inputs, gate_weight)) * torch.bmm(inputs, up_weight)
        outputs = torch.bmm(hidden, down_weight)
        weights = (routing_weights[pair_rows] * valid.to(routing_weights.dtype)).unsqueeze(-1)
        return update.index_add(0, token_rows.reshape(-1), (outputs * weights).reshape(-1, d_model))

    def _stacked_effective_weights(
        self,
        group_ids: Tensor | None,
        promoted_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Effective (quantized / promoted) weights of the given experts, stacked."""

        if group_ids is None:
            promoted = promoted_mask
            stacks = (self.gate_proj, self.up_proj, self.down_proj)
        else:
            promoted = None if promoted_mask is None else promoted_mask[group_ids]
            stacks = (
                self.gate_proj[group_ids],
                self.up_proj[group_ids],
                self.down_proj[group_ids],
            )
        gate, up, down = (self._effective_weight(stack, promoted) for stack in stacks)
        return gate, up, down


def _pair_counts(expert_index: Tensor, num_experts: int) -> Tensor:
    """Per-expert pair counts without the host synchronization ``bincount`` needs."""

    counts = torch.zeros(num_experts, dtype=torch.long, device=expert_index.device)
    return counts.scatter_add_(0, expert_index, torch.ones_like(expert_index))
