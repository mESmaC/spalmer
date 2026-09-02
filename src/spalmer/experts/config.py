"""Configuration for the micro-expert channel-mixer slice."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class MicroExpertsConfig:
    """Configuration of one layer-local micro-expert bank and its router.

    Two counts govern expert execution and must not be confused:

    - ``active_experts`` is the per-token top-``k``: how many experts one token
      executes, chosen among the currently *resident* experts.
    - the resident set (ledger C13) is one request-level identity set shared by
      every layer; its size moves between ``min_resident_experts`` and
      ``max_resident_experts`` by adding explicit expert ids, never by changing
      ``k``. Outside a residency session every expert is resident.

    Args:
        d_model: Residual-stream width shared with the surrounding block.
        num_experts: Number of experts in the bank. The nominal C07 value is
            about 200; tiny tests may use far fewer.
        expert_inter_dim: Hidden width of each expert. ``None`` means
            ``d_model // 2`` (deliberately small experts).
        shared_inter_dim: Hidden width of the always-on shared SwiGLU channel
            path that runs beside the routed experts. ``None`` means
            ``2 * d_model``; ``0`` removes the shared path (routed only).
        active_experts: Experts executed per token (top-``k`` among residents).
        min_active_experts: Lower bound enforced on ``active_experts``.
        max_active_experts: Upper bound enforced on ``active_experts`` and on
            its runtime override.
        min_resident_experts: Residents a request starts with (C13 begins with
            two). ``None`` resolves to ``max(2, active_experts)``; an explicit
            value must be at least ``active_experts``.
        max_resident_experts: Soft residency cap for the expansion controller
            (the ledger's ~10% of a 200-expert pool). Bounded by
            ``num_experts``.
        expert_execution: ``grouped`` executes every resident expert's tokens
            with padded batched matmuls (no per-expert Python loop);
            ``loop`` is the per-expert reference path.
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
        residency_increment: Expert ids added per expansion step of the C13
            inference residency controller.
        residency_min_gain: Smallest drop in effective NLL (nats per token)
            that an expansion step must buy to be retained; otherwise the
            controller rolls back to the previous resident set.
        initializer_range: Standard deviation of the normal initializations.
    """

    d_model: int
    num_experts: int = 200
    expert_inter_dim: int | None = None
    shared_inter_dim: int | None = None
    active_experts: int = 2
    min_active_experts: int = 2
    max_active_experts: int = 20
    min_resident_experts: int | None = None
    max_resident_experts: int = 20
    expert_execution: Literal["grouped", "loop"] = "grouped"
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
        if self.min_resident_experts is None:
            object.__setattr__(self, "min_resident_experts", max(2, self.active_experts))
        _require_positive("min_resident_experts", self.min_resident_experts)
        _require_positive("max_resident_experts", self.max_resident_experts)
        if self.expert_inter_dim is not None and self.expert_inter_dim <= 0:
            raise ValueError("expert_inter_dim must be positive when provided")
        if self.shared_inter_dim is not None and self.shared_inter_dim < 0:
            raise ValueError("shared_inter_dim must be non-negative when provided")
        if self.expert_execution not in {"grouped", "loop"}:
            raise ValueError("expert_execution must be 'grouped' or 'loop'")
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
        if self.min_resident_experts < self.active_experts:
            raise ValueError(
                f"min_resident_experts ({self.min_resident_experts}) cannot be below "
                f"active_experts ({self.active_experts}): every token needs k residents"
            )
        if self.min_resident_experts > self.num_experts:
            raise ValueError(
                f"min_resident_experts ({self.min_resident_experts}) cannot exceed "
                f"num_experts ({self.num_experts})"
            )
        if self.min_resident_experts > self.resident_cap:
            raise ValueError(
                f"min_resident_experts ({self.min_resident_experts}) cannot exceed the "
                f"resident cap ({self.resident_cap})"
            )

    @property
    def resolved_inter_dim(self) -> int:
        """Effective expert hidden width."""

        if self.expert_inter_dim is not None:
            return self.expert_inter_dim
        return max(1, self.d_model // 2)

    @property
    def resolved_shared_inter_dim(self) -> int:
        """Effective shared-path hidden width (``0`` disables the shared path)."""

        if self.shared_inter_dim is not None:
            return self.shared_inter_dim
        return 2 * self.d_model

    @property
    def resident_cap(self) -> int:
        """Largest resident set the expansion controller may reach."""

        return min(self.max_resident_experts, self.num_experts)

    @property
    def expert_parameters_per_layer(self) -> int:
        """Exact parameter count of one expert in one layer (gate, up, down)."""

        return 3 * self.d_model * self.resolved_inter_dim

    @property
    def shared_parameters_per_layer(self) -> int:
        """Exact parameter count of the shared SwiGLU path in one layer."""

        return 3 * self.d_model * self.resolved_shared_inter_dim

    @property
    def expert_pool_parameters_per_layer(self) -> int:
        """Exact parameter count of the whole expert bank in one layer."""

        return self.num_experts * self.expert_parameters_per_layer


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
