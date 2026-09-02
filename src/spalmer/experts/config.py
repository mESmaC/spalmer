"""Configuration for the micro-expert channel-mixer slice."""

from __future__ import annotations

from dataclasses import dataclass


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
        router_bias: Learnable bias on the router projection.
        initializer_range: Standard deviation of the normal initializations.
    """

    d_model: int
    num_experts: int = 200
    expert_inter_dim: int | None = None
    active_experts: int = 2
    min_active_experts: int = 2
    max_active_experts: int = 20
    router_bias: bool = False
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        _require_positive("d_model", self.d_model)
        _require_positive("num_experts", self.num_experts)
        _require_positive("min_active_experts", self.min_active_experts)
        _require_positive("max_active_experts", self.max_active_experts)
        if self.expert_inter_dim is not None and self.expert_inter_dim <= 0:
            raise ValueError("expert_inter_dim must be positive when provided")
        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be positive")
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
