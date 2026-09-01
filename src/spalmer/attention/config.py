"""Configuration for the SPALMER KDA token mixer (ledger C04)."""

from __future__ import annotations

from dataclasses import dataclass

_BACKENDS = ("auto", "reference", "fla")


@dataclass
class KDAConfig:
    """Configuration of one KDA token-mixer layer.

    Shapes per head: key ``K = head_k_dim``, value ``V = head_v_dim``.
    The recurrent state is ``[batch, num_heads, K, V]``.

    Args:
        hidden_size: Residual-stream width ``d_model``.
        num_heads: Number of heads ``H`` (no GVA in this slice; value heads
            equal key heads).
        head_k_dim: Per-head key dimension ``K``.
        head_v_dim: Per-head value dimension ``V``. ``None`` means ``K``.
        conv_width: Kernel width of the causal depthwise q/k/v convolutions.
        conv_bias: Learnable bias on the short convolutions.
        use_short_conv: Disable to fall back to plain SiLU on q/k/v.
        norm_eps: Epsilon for the per-head gated output RMSNorm.
        allow_neg_eigval: Multiply beta by 2 (KDA negative-eigenvalue option).
        gate_lower_bound: Optional clamp of the log forget gate g to
            ``[gate_lower_bound, 0)``; ``None`` disables the clamp.
        backend: ``"auto"`` (fla on CUDA when importable, else reference),
            ``"reference"``, or ``"fla"`` (raises if fla is unavailable).
    """

    hidden_size: int = 512
    num_heads: int = 8
    head_k_dim: int = 64
    head_v_dim: int | None = None
    conv_width: int = 4
    conv_bias: bool = False
    use_short_conv: bool = True
    norm_eps: float = 1e-5
    allow_neg_eigval: bool = False
    gate_lower_bound: float | None = None
    backend: str = "auto"

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.head_k_dim <= 0:
            raise ValueError(f"head_k_dim must be positive, got {self.head_k_dim}")
        if self.head_v_dim is None:
            self.head_v_dim = self.head_k_dim
        if self.head_v_dim <= 0:
            raise ValueError(f"head_v_dim must be positive, got {self.head_v_dim}")
        if not self.use_short_conv:
            raise NotImplementedError(
                "use_short_conv=False is not wired up in this slice; the KDA "
                "baseline always uses the short convolutions."
            )
        if self.conv_width < 2:
            raise ValueError(f"conv_width must be >= 2, got {self.conv_width}")
        if self.backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {_BACKENDS}, got {self.backend!r}")
        if self.gate_lower_bound is not None and self.gate_lower_bound >= 0:
            raise ValueError(
                f"gate_lower_bound must be negative (log-space bound), got {self.gate_lower_bound}"
            )

    @property
    def key_dim(self) -> int:
        return self.num_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_heads * self.head_v_dim
