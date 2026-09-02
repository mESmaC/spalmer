"""Micro-expert channel mixing (SPALMER ledger C07/C08/C13).

A tensorized bank of small gated-MLP experts with learned per-token routing,
realized-NLL calibration, an expert-wide potentiation controller, and a
request-level inference residency controller that expands the active set while
effective surprise stays above average. Routed-expert QAT derives selectable
MXFP4/NVFP4 forward weights and MXFP8 activations from one BF16 master payload;
native mixed W4A8 kernels, packed deployment, residency transfer, and CPU
offload remain later backends.
"""

from spalmer.experts.accounting import ParameterAccounting, account_parameters
from spalmer.experts.bank import MicroExpertBank
from spalmer.experts.config import MicroExpertsConfig
from spalmer.experts.losses import expert_utilization, load_balance_loss
from spalmer.experts.mixer import MicroExpertChannelMixer
from spalmer.experts.potentiation import ExpertPotentiationController
from spalmer.experts.qat import (
    ExpertQATBackendStatus,
    ExpertQATConfig,
    expert_qat_backend_status,
    require_expert_qat_backend,
)
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
    "ExpertQATBackendStatus",
    "ExpertQATConfig",
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
    "expert_utilization",
    "expert_qat_backend_status",
    "load_balance_loss",
    "rank_nonresident_experts",
    "require_expert_qat_backend",
    "select_least_surprised_experts",
]
