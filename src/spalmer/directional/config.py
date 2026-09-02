"""Configuration for the directional subnetwork slice (ledger C16)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DirectionalConfig:
    """Configuration of the Feed Laterally / Lateral Active Silencing mixer.

    Args:
        d_model: Residual-stream width shared with the surrounding block.
        num_feature_groups: Number of same-depth feature-group peers. Must
            divide ``d_model`` evenly.
        lateral_rank: Rank of the groups-by-groups lateral mixing matrix.
        enabled: Feature gate. Disabled configs construct no mixer.
        residual_gate_init: Initial value of the learned ``residual_gate``.
            The default ``0.0`` makes the branch an exact no-op at
            initialization (ledger C06: gated branches start as identity).
        initializer_range: Standard deviation of the normal initializations.
    """

    d_model: int
    num_feature_groups: int = 4
    lateral_rank: int = 16
    enabled: bool = False
    residual_gate_init: float = 0.0
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        _require_positive("d_model", self.d_model)
        if self.num_feature_groups < 2:
            raise ValueError("num_feature_groups must be at least 2 to have peers")
        if self.d_model % self.num_feature_groups != 0:
            raise ValueError(
                f"num_feature_groups ({self.num_feature_groups}) must divide "
                f"d_model ({self.d_model}) evenly"
            )
        _require_positive("lateral_rank", self.lateral_rank)
        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be positive")

    @property
    def group_width(self) -> int:
        """Features per group."""

        return self.d_model // self.num_feature_groups

    @property
    def parameters_per_layer(self) -> int:
        """Exact parameter count of one mixer (rank factors, silencing net, gate)."""

        groups, rank, width = self.num_feature_groups, self.lateral_rank, self.group_width
        return 2 * groups * rank + (2 * width + 1) + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
