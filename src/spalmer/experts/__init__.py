"""Micro-expert channel mixing (SPALMER ledger C07/C08/C13).

A tensorized bank of small gated-MLP experts with learned per-token routing,
realized-NLL calibration, and an expert-wide reference potentiation controller.
The numerical lane is fake low-bit plus shadow precision; packed FP4/FP8,
residency transfer, CPU offload, and iterative NLL recomputation remain later
backends.
"""

from spalmer.experts.bank import MicroExpertBank
from spalmer.experts.config import MicroExpertsConfig
from spalmer.experts.losses import expert_utilization, load_balance_loss
from spalmer.experts.mixer import MicroExpertChannelMixer
from spalmer.experts.potentiation import ExpertPotentiationController
from spalmer.experts.router import SurpriseRouter, select_least_surprised_experts

__all__ = [
    "MicroExpertBank",
    "MicroExpertChannelMixer",
    "MicroExpertsConfig",
    "ExpertPotentiationController",
    "SurpriseRouter",
    "expert_utilization",
    "load_balance_loss",
    "select_least_surprised_experts",
]
