"""Checkpointable expert-wide precision-promotion controller."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from spalmer.experts.config import MicroExpertsConfig


class ExpertPotentiationController(nn.Module):
    """Promote coherent expert identities using auditable EMA telemetry.

    This reference controller promotes an expert from deterministic/stochastic
    fake-low-bit execution to its floating shadow parameter. It establishes the
    expert-wide control semantics; packed FP4 plus FP8 residual storage remains a
    later numerical backend.
    """

    def __init__(self, config: MicroExpertsConfig) -> None:
        super().__init__()
        self.config = config
        shape = (config.num_experts,)
        self.register_buffer("utilization_ema", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("attributed_nll_ema", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("quantization_error_ema", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("promoted_mask", torch.zeros(shape, dtype=torch.bool))
        self.register_buffer("observations", torch.zeros((), dtype=torch.long))
        self.register_buffer("last_change", torch.zeros((), dtype=torch.long))

    @property
    def scores(self) -> Tensor:
        return self.utilization_ema * self.attributed_nll_ema * self.quantization_error_ema

    def _apply(self, fn):
        super()._apply(fn)
        # Module-wide BF16 conversion is appropriate for model weights but not
        # for the slowly accumulated controller telemetry.
        for name in ("utilization_ema", "attributed_nll_ema", "quantization_error_ema"):
            self._buffers[name] = self._buffers[name].float()
        return self

    @torch.no_grad()
    def observe(
        self,
        utilization: Tensor,
        attributed_nll: Tensor,
        quantization_error: Tensor,
    ) -> tuple[int, ...]:
        """Update the three signal roles and, when stable, the promoted set."""

        for name, value in (
            ("utilization", utilization),
            ("attributed_nll", attributed_nll),
            ("quantization_error", quantization_error),
        ):
            if value.shape != self.utilization_ema.shape:
                raise ValueError(
                    f"{name} must have shape {tuple(self.utilization_ema.shape)}, "
                    f"got {tuple(value.shape)}"
                )

        decay = self.config.potentiation_ema_decay
        self.utilization_ema.mul_(decay).add_(
            utilization.to(self.utilization_ema), alpha=1.0 - decay
        )
        self.attributed_nll_ema.mul_(decay).add_(
            attributed_nll.to(self.attributed_nll_ema), alpha=1.0 - decay
        )
        self.quantization_error_ema.mul_(decay).add_(
            quantization_error.to(self.quantization_error_ema), alpha=1.0 - decay
        )
        self.observations.add_(1)
        self._maybe_update_mask()
        return self.promoted_ids()

    def promoted_ids(self) -> tuple[int, ...]:
        return tuple(self.promoted_mask.nonzero(as_tuple=False).flatten().tolist())

    @torch.no_grad()
    def _maybe_update_mask(self) -> None:
        budget = self.config.potentiation_budget
        step = int(self.observations)
        if budget == 0:
            self.promoted_mask.zero_()
            return
        if step < self.config.potentiation_warmup_steps:
            return
        if self.promoted_mask.any() and (
            step - int(self.last_change) < self.config.potentiation_hold_steps
        ):
            return

        scores = self.scores
        eligible = (self.utilization_ema > 0) & (scores > 0) & torch.isfinite(scores)
        eligible_ids = eligible.nonzero(as_tuple=False).flatten()
        if eligible_ids.numel() == 0:
            return
        count = min(budget, int(eligible_ids.numel()))
        candidate_ids = eligible_ids[scores[eligible_ids].topk(count).indices]
        candidate_mask = torch.zeros_like(self.promoted_mask)
        candidate_mask[candidate_ids] = True

        current_ids = self.promoted_mask.nonzero(as_tuple=False).flatten()
        if current_ids.numel() == count and current_ids.numel() > 0:
            incoming = candidate_ids[~self.promoted_mask[candidate_ids]]
            if incoming.numel() > 0:
                weakest_current = scores[current_ids].min()
                strongest_incoming = scores[incoming].max()
                required = weakest_current * (1.0 + self.config.potentiation_hysteresis)
                if strongest_incoming <= required:
                    return

        if not torch.equal(candidate_mask, self.promoted_mask):
            self.promoted_mask.copy_(candidate_mask)
            self.last_change.copy_(self.observations)


__all__ = ["ExpertPotentiationController"]
