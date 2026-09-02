"""SPALMER research model components."""

from spalmer.checkpoint import load_checkpoint, save_checkpoint
from spalmer.config import PLEConfig, SPALMERConfig
from spalmer.experts.offload import (
    ExpertOffloadTelemetry,
    disable_expert_offload,
    enable_expert_offload,
)
from spalmer.factory import DenseSwiGLU, build_kda_bootstrap_model, build_spalmer_model
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
    "ExpertOffloadTelemetry",
    "PLEConfig",
    "SPALMERBackbone",
    "SPALMERBlock",
    "SPALMERCausalLM",
    "SPALMERConfig",
    "build_kda_bootstrap_model",
    "build_spalmer_model",
    "disable_expert_offload",
    "enable_expert_offload",
    "load_checkpoint",
    "save_checkpoint",
]

__version__ = "0.1.0"
