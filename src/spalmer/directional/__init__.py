"""Directional neural subnetworks (SPALMER ledger C16).

First concrete slice, ported from ``agent/directional-v0`` (09a04ec): Feed
Laterally NN plus Lateral Active Silencing NN. Feed Forward remains the
existing shared/routed channel path; Feed Backward is deferred."""

from spalmer.directional.config import DirectionalConfig
from spalmer.directional.mixer import LateralSilencingMixer

__all__ = ["DirectionalConfig", "LateralSilencingMixer"]
