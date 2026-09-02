"""Routing telemetry and the load-balancing auxiliary loss.

Normalization contract: every quantity here is a per-layer average over
tokens and is scale-free in the number of eligible experts, so its magnitude
does not drift with sequence length, batch size, or resident count. The
backbone averages (never sums) per-layer auxiliary losses, so layer count is
neutral as well.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def expert_utilization(expert_ids: Tensor, num_experts: int) -> Tensor:
    """Fraction of routed (token, slot) pairs served by each expert.

    Args:
        expert_ids: ``[..., k]`` selected expert ids.
        num_experts: Total number of experts in the bank.

    Returns:
        ``[num_experts]`` detached float32 utilization vector summing to one.
    """

    flat_ids = expert_ids.reshape(-1)
    if flat_ids.numel() == 0:
        raise ValueError("expert_ids cannot be empty")
    counts = torch.zeros(num_experts, dtype=torch.float32, device=flat_ids.device)
    counts = counts.scatter_add(0, flat_ids, torch.ones_like(flat_ids, dtype=counts.dtype))
    return counts.detach() / flat_ids.numel()


def load_balance_loss(
    scores: Tensor,
    expert_ids: Tensor,
    resident_mask: Tensor | None = None,
) -> Tensor:
    """Switch-style load-balancing auxiliary loss over the resident experts.

    ``loss = m * sum_i f_i * P_i`` where ``m`` is the number of eligible
    (resident) experts, ``f_i`` is the (detached) fraction of routed slots
    served by expert ``i`` and ``P_i`` is the mean routing probability of
    expert ``i`` under ``softmax(-scores)`` restricted to residents (low
    predicted surprise means high routing probability). The loss equals one
    exactly at uniform utilization for any ``m``, so its magnitude is
    independent of the configured resident count; gradients flow through
    ``P`` only, the standard differentiable-through-probabilities form.

    Args:
        scores: ``[..., E]`` predicted surprise scores.
        expert_ids: ``[..., k]`` selected expert ids.
        resident_mask: Optional ``[E]`` boolean mask of resident experts.
    """

    num_experts = scores.shape[-1]
    flat_scores = scores.reshape(-1, num_experts)
    logits = -flat_scores
    if resident_mask is not None:
        mask = resident_mask.to(scores.device)
        logits = logits.masked_fill(~mask, -torch.inf)
        eligible = mask.to(logits.dtype).sum()
    else:
        eligible = torch.tensor(float(num_experts), device=scores.device, dtype=logits.dtype)
    probabilities = F.softmax(logits, dim=-1)
    mean_probability = probabilities.mean(dim=0)
    fractions = expert_utilization(expert_ids, num_experts).to(mean_probability.dtype)
    return eligible * (fractions * mean_probability).sum()
