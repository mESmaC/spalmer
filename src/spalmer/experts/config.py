"""Configuration for the micro-expert channel-mixer slice."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class MicroExpertsConfig:
    """Configuration of one layer-local micro-expert bank and its router.

    Args:
        d_model: Residual-stream width shared with the surrounding block.
        num_experts: Number of experts in the bank. The nominal C07 value is
            about 200; tiny tests may use far fewer.
        expert_inter_dim: Hidden width of each expert. ``None`` means
            ``d_model // 2`` (deliberately small experts).
        active_experts: Experts executed per token (C13 starts at 2).
        min_active_experts: Lower bound enforced on ``active_experts``.
        max_active_experts: Nominal upper bound (the soft 10% cap of ~20
            experts for a 200-expert pool).
        expert_quant_bits: Fake-quantized reference substrate width.
        expert_fake_quantization: Enable the reference low-bit expert substrate.
        expert_stochastic_rounding: Apply stochastic rounding to the low-bit
            substrate during training.
        potentiation_budget: Number of complete expert identities promoted to
            shadow precision across every layer. Zero disables promotion.
        potentiation_ema_decay: Smoothing applied to the expert-wide precision
            pressure used to choose the promoted set.
        potentiation_warmup_steps: Observations required before the first
            promotion decision.
        potentiation_hold_steps: Minimum observations between set changes.
        potentiation_hysteresis: Fractional score advantage required to evict
            an already-promoted expert.
        router_score_transform: ``softplus`` makes new surprise estimates
            non-negative; ``identity`` exists only to reproduce legacy
            checkpoints trained with raw router logits.
        router_bias: Learnable bias on the router projection.
        residency_increment: Experts added per expansion step of the C13
            inference residency controller.
        residency_min_gain: Smallest drop in effective NLL (nats per token)
            that an expansion step must buy to be retained; otherwise the
            controller rolls back to the previous active count.
        initializer_range: Standard deviation of the normal initializations.
    """

    d_model: int
    num_experts: int = 200
    expert_inter_dim: int | None = None
    active_experts: int = 2
    min_active_experts: int = 2
    max_active_experts: int = 20
    expert_quant_bits: int = 4
    expert_fake_quantization: bool = True
    expert_stochastic_rounding: bool = True
    potentiation_budget: int = 2
    potentiation_ema_decay: float = 0.95
    potentiation_warmup_steps: int = 4
    potentiation_hold_steps: int = 8
    potentiation_hysteresis: float = 0.05
    router_score_transform: Literal["softplus", "identity"] = "softplus"
    router_bias: bool = False
    residency_increment: int = 2
    residency_min_gain: float = 0.02
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        _require_positive("d_model", self.d_model)
        _require_positive("num_experts", self.num_experts)
        _require_positive("min_active_experts", self.min_active_experts)
        _require_positive("max_active_experts", self.max_active_experts)
        if self.expert_inter_dim is not None and self.expert_inter_dim <= 0:
            raise ValueError("expert_inter_dim must be positive when provided")
        if not 2 <= self.expert_quant_bits <= 8:
            raise ValueError("expert_quant_bits must be between 2 and 8")
        if not 0 <= self.potentiation_budget <= self.num_experts:
            raise ValueError(
                "potentiation_budget must be between zero and num_experts; "
                f"got {self.potentiation_budget}"
            )
        if not 0 <= self.potentiation_ema_decay < 1:
            raise ValueError("potentiation_ema_decay must be in [0, 1)")
        _require_positive("potentiation_warmup_steps", self.potentiation_warmup_steps)
        _require_positive("potentiation_hold_steps", self.potentiation_hold_steps)
        if self.potentiation_hysteresis < 0:
            raise ValueError("potentiation_hysteresis must be non-negative")
        if self.router_score_transform not in {"softplus", "identity"}:
            raise ValueError("router_score_transform must be 'softplus' or 'identity'")
        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be positive")
        _require_positive("residency_increment", self.residency_increment)
        if not math.isfinite(self.residency_min_gain):
            raise ValueError("residency_min_gain must be finite")
        if self.min_active_experts > self.max_active_experts:
            raise ValueError(
                f"min_active_experts ({self.min_active_experts}) cannot exceed "
                f"max_active_experts ({self.max_active_experts})"
            )
        if self.num_experts < self.min_active_experts:
            raise ValueError(
                f"num_experts ({self.num_experts}) cannot be below the minimum "
                f"active expert count ({self.min_active_experts})"
            )
        effective_max = min(self.max_active_experts, self.num_experts)
        if not self.min_active_experts <= self.active_experts <= effective_max:
            raise ValueError(
                f"active_experts must be in [{self.min_active_experts}, "
                f"{effective_max}]; got {self.active_experts}"
            )

    @property
    def resolved_inter_dim(self) -> int:
        """Effective expert hidden width."""

        if self.expert_inter_dim is not None:
            return self.expert_inter_dim
        return max(1, self.d_model // 2)


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
