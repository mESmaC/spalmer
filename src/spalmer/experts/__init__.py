"""Micro-expert channel mixing (SPALMER ledger C07/C08/C13).

A tensorized bank of small gated-MLP experts with learned per-token routing,
realized-NLL calibration, an expert-wide potentiation controller, and a
request-level inference residency controller that expands the active set while
effective surprise stays above average. Routed experts execute only through a
verified native BF16, NVFP4, or MXFP8 provider selected for the current GPU;
low-precision kernels derive packed operands from one BF16 master payload. The
inference offload backend keeps complete expert masters on CPU and stages only
request-resident rows on the execution device.
"""

from spalmer.experts.accounting import ParameterAccounting, account_parameters
from spalmer.experts.bank import MicroExpertBank
from spalmer.experts.config import MicroExpertsConfig
from spalmer.experts.losses import expert_utilization, load_balance_loss
from spalmer.experts.mixer import MicroExpertChannelMixer
from spalmer.experts.offload import (
    ExpertOffloadManager,
    ExpertOffloadTelemetry,
    disable_expert_offload,
    enable_expert_offload,
)
from spalmer.experts.potentiation import ExpertPotentiationController
from spalmer.experts.residency import (
    ExpertResidency,
    ResidencyDecision,
    choose_inference_residency,
    default_resident_ids,
    rank_nonresident_experts,
)
from spalmer.experts.router import SurpriseRouter, select_least_surprised_experts

__all__ = [
    "ExpertPotentiationController",
    "ExpertOffloadManager",
    "ExpertOffloadTelemetry",
    "ExpertResidency",
    "MicroExpertBank",
    "MicroExpertChannelMixer",
    "MicroExpertsConfig",
    "ParameterAccounting",
    "ResidencyDecision",
    "SurpriseRouter",
    "account_parameters",
    "choose_inference_residency",
    "default_resident_ids",
    "disable_expert_offload",
    "enable_expert_offload",
    "expert_utilization",
    "load_balance_loss",
    "rank_nonresident_experts",
    "select_least_surprised_experts",
]
