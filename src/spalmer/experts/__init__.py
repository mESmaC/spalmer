"""Micro-expert channel mixing (SPALMER ledger C07/C08/C13).

A tensorized bank of small gated-MLP experts with learned per-token routing.
Router scores are interpreted as predicted expert surprise; lower predicted
surprise is preferred. This is the smallest trainable slice: no FP4, no
residency transfer, no CPU offload, no iterative NLL recomputation, no
potentiation, no distributed training, and no generalized routing framework.
"""

from spalmer.experts.bank import MicroExpertBank
from spalmer.experts.config import MicroExpertsConfig
from spalmer.experts.losses import expert_utilization, load_balance_loss
from spalmer.experts.mixer import MicroExpertChannelMixer
from spalmer.experts.router import SurpriseRouter, select_least_surprised_experts

__all__ = [
    "MicroExpertBank",
    "MicroExpertChannelMixer",
    "MicroExpertsConfig",
    "SurpriseRouter",
    "expert_utilization",
    "load_balance_loss",
    "select_least_surprised_experts",
]
