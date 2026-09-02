"""Small fake-quantization primitives shared by reference backends."""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor

_MIN_QUANT_BITS: Final = 2
_MAX_QUANT_BITS: Final = 8


def fake_quantize_low_bit(
    values: Tensor,
    *,
    bits: int,
    stochastic: bool,
    straight_through: bool,
    eps: float = 1e-8,
) -> Tensor:
    """Symmetrically quantize the final dimension with optional stochastic rounding.

    This is a correctness backend: floating shadow parameters remain the source of
    truth for autograd and checkpointing while the forward path sees low-bit values.
    """

    if not _MIN_QUANT_BITS <= bits <= _MAX_QUANT_BITS:
        raise ValueError(f"bits must be between {_MIN_QUANT_BITS} and {_MAX_QUANT_BITS}")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if not values.is_floating_point():
        raise TypeError("values must use a floating-point dtype")

    work = values.float()
    qmax = (1 << (bits - 1)) - 1
    scale = (work.detach().abs().amax(dim=-1, keepdim=True) / qmax).clamp_min(eps)
    scaled = (work / scale).clamp(-qmax, qmax)

    if stochastic:
        lower = torch.floor(scaled)
        rounded = lower + (torch.rand_like(scaled) < (scaled - lower)).to(scaled.dtype)
    else:
        rounded = torch.round(scaled)

    dequantized = (rounded * scale).to(values.dtype)
    if straight_through:
        return values + (dequantized - values).detach()
    return dequantized


__all__ = ["fake_quantize_low_bit"]
