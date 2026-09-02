"""SPALMER research model components."""

from spalmer.config import PLEConfig, SPALMERConfig
from spalmer.factory import DenseSwiGLU, build_kda_bootstrap_model
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
    "DenseSwiGLU",
    "PLEConfig",
    "SPALMERBackbone",
    "SPALMERBlock",
    "SPALMERCausalLM",
    "SPALMERConfig",
    "build_kda_bootstrap_model",
]

__version__ = "0.1.0"
