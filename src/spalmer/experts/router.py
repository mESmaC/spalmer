"""Surprise router (SPALMER ledger C08).

The router treats its per-token scores as *predicted expert surprise*.
Selection prefers the lowest predicted surprise among the currently resident
experts (ledger C13), and combination weights are a softmax over negative
surprise among the selected experts, so the least surprised expert receives
the largest weight.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from spalmer.experts.config import MicroExpertsConfig


class SurpriseRouter(nn.Module):
    """Learned per-token router emitting predicted surprise for every expert.

    A single router instance may be shared by several layer-local expert
    banks (one bank per layer, expert identity preserved across layers), as
    long as every bank has the same ``num_experts``.
    """

    def __init__(self, config: MicroExpertsConfig) -> None:
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.proj = nn.Linear(config.d_model, config.num_experts, bias=config.router_bias)
        nn.init.normal_(self.proj.weight, mean=0.0, std=config.initializer_range)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    @property
    def num_features(self) -> int:
        return self.config.d_model

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Return predicted surprise scores of shape ``[batch, seq, experts]``.

        Scores are produced for every expert, resident or not: the residency
        controller ranks non-resident candidates with them.
        """

        if hidden_states.shape[-1] != self.config.d_model:
            raise ValueError(
                f"expected trailing dimension {self.config.d_model}, got {hidden_states.shape[-1]}"
            )
        scores = self.proj(hidden_states)
        if self.config.router_score_transform == "identity":
            return scores
        # Surprise is an NLL-like non-negative quantity. Calibration against
        # realized next-token NLL happens at the language-model boundary.
        return F.softplus(scores)


def select_least_surprised_experts(
    scores: Tensor,
    num_active: int,
    resident_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Select the ``num_active`` least-surprised *resident* experts per token.

    Args:
        scores: ``[..., experts]`` predicted surprise scores.
        num_active: Number of experts to select per token.
        resident_mask: Optional ``[experts]`` boolean mask of resident experts.
            Non-resident experts are never selected; ``None`` means every
            expert is resident.

    Returns:
        A pair ``(expert_ids, routing_weights)`` of shapes ``[..., k]``.
        Ids are ordered by ascending predicted surprise (most preferred
        first), are unique per token, and weights are a softmax over
        negative surprise among the selected experts, summing to one.
    """

    num_experts = scores.shape[-1]
    if not 1 <= num_active <= num_experts:
        raise ValueError(f"num_active must be in [1, {num_experts}]; got {num_active}")
    eligible = scores
    if resident_mask is not None:
        if resident_mask.shape != (num_experts,):
            raise ValueError(
                f"resident_mask must have shape ({num_experts},); got {tuple(resident_mask.shape)}"
            )
        # ExpertResidency guarantees at least num_active residents, so no
        # device synchronization is needed here on the hot path.
        eligible = scores.masked_fill(~resident_mask.to(scores.device), torch.inf)
    expert_ids = eligible.topk(num_active, dim=-1, largest=False).indices
    selected_scores = scores.gather(-1, expert_ids)
    routing_weights = F.softmax(-selected_scores, dim=-1)
    return expert_ids, routing_weights
