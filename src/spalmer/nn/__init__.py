"""Small neural-network primitives shared across SPALMER components."""

from spalmer.nn.ffn import SwiGLU
from spalmer.nn.norm import RMSNorm
from spalmer.nn.quantization import fake_quantize_low_bit

__all__ = ["RMSNorm", "SwiGLU", "fake_quantize_low_bit"]
