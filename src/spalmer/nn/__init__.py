"""Small neural-network primitives shared across SPALMER components."""

from spalmer.nn.ffn import SwiGLU
from spalmer.nn.norm import RMSNorm

__all__ = ["RMSNorm", "SwiGLU"]
