"""SPALMER research model components."""

from spalmer.config import PLEConfig, SPALMERConfig
from spalmer.modeling import (
    BackboneOutput,
    BlockOutput,
    CausalLMOutput,
    ChannelMixerOutput,
    SPALMERBackbone,
    SPALMERBlock,
    SPALMERCausalLM,
)

__all__ = [
    "BackboneOutput",
    "BlockOutput",
    "CausalLMOutput",
    "ChannelMixerOutput",
    "PLEConfig",
    "SPALMERBackbone",
    "SPALMERBlock",
    "SPALMERCausalLM",
    "SPALMERConfig",
]

__version__ = "0.1.0"
