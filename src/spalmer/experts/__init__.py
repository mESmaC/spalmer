"""Micro-expert channel mixing (SPALMER ledger C07/C08/C13).

A tensorized bank of small gated-MLP experts with learned per-token routing,
realized-NLL calibration, an expert-wide reference potentiation controller, and
a request-level inference residency controller that expands the active set
while effective surprise stays above average. The numerical lane is fake
low-bit plus shadow precision; packed FP4/FP8, residency transfer, and CPU
offload remain later backends.
"""

from spalmer.experts.accounting import ParameterAccounting, account_parameters
from spalmer.experts.bank import MicroExpertBank
from spalmer.experts.config import MicroExpertsConfig
from spalmer.experts.losses import expert_utilization, load_balance_loss
from spalmer.experts.mixer import MicroExpertChannelMixer
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
    "load_balance_loss",
    "rank_nonresident_experts",
    "select_least_surprised_experts",
]
