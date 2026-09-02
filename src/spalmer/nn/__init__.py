"""Small neural-network primitives shared across SPALMER components."""

from spalmer.nn.norm import RMSNorm
from spalmer.nn.quantization import fake_quantize_low_bit

__all__ = ["RMSNorm", "fake_quantize_low_bit"]
