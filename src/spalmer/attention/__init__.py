"""SPALMER attention components (ledger C04).

Self-contained KDA token-mixer slice with a plain-PyTorch reference backend
and a guarded fla-core backend adapter."""

from spalmer.attention.config import KDAConfig
from spalmer.attention.mixer import KDATokenMixer
from spalmer.attention.mla import MLAConfig, MLAState, MLATokenMixer
from spalmer.attention.state import KDAState

__all__ = [
    "KDAConfig",
    "KDAState",
    "KDATokenMixer",
    "MLAConfig",
    "MLAState",
    "MLATokenMixer",
]
