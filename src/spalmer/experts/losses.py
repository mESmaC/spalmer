"""Routing telemetry and the load-balancing auxiliary loss."""

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
    ones = torch.ones_like(flat_ids, dtype=counts.dtype)
    counts = counts.scatter_add(0, flat_ids, ones)
    return counts.detach() / flat_ids.numel()


def load_balance_loss(scores: Tensor, expert_ids: Tensor) -> Tensor:
    """Switch-style load-balancing auxiliary loss.

    ``loss = E * sum_i f_i * P_i`` where ``f_i`` is the (detached) fraction
    of routed slots served by expert ``i`` and ``P_i`` is the mean routing
    probability of expert ``i`` under ``softmax(-scores)`` (low predicted
    surprise means high routing probability). The loss reaches its minimum
    of one exactly at uniform utilization; gradients flow through ``P``
    only, which is the standard differentiable-through-probabilities form.

    Args:
        scores: ``[..., E]`` predicted surprise scores.
        expert_ids: ``[..., k]`` selected expert ids.
    """

    num_experts = scores.shape[-1]
    flat_scores = scores.reshape(-1, num_experts)
    probabilities = F.softmax(-flat_scores, dim=-1)
    mean_probability = probabilities.mean(dim=0)
    fractions = expert_utilization(expert_ids, num_experts)
    return num_experts * (fractions * mean_probability).sum()
