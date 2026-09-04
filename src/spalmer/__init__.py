"""SPALMER research model components."""

from spalmer.checkpoint import load_checkpoint, save_checkpoint
from spalmer.config import PLEConfig, RecurrenceConfig, SPALMERConfig
from spalmer.experts.offload import (
    ExpertOffloadTelemetry,
    disable_expert_offload,
    enable_expert_offload,
)
from spalmer.factory import DenseSwiGLU, build_kda_bootstrap_model, build_spalmer_model
from spalmer.modeling import (
    AdaptiveExit,
    BackboneOutput,
    BlockOutput,
    CausalLMOutput,
    ChannelMixerOutput,
    LatentRecurrence,
    RecurrentMixerStates,
    SPALMERBackbone,
    SPALMERBlock,
    SPALMERCausalLM,
)
from spalmer.runtime import RecurrenceTrace

__all__ = [
    "AdaptiveExit",
    "BackboneOutput",
    "BlockOutput",
    "CausalLMOutput",
    "ChannelMixerOutput",
    "DenseSwiGLU",
    "ExpertOffloadTelemetry",
    "LatentRecurrence",
    "PLEConfig",
    "RecurrenceConfig",
    "RecurrenceTrace",
    "RecurrentMixerStates",
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
