"""Exact parameter and byte accounting (ledger C07/C13 and completion idea 3).

The ledger separates total weights, stored weights, resident weights, active
weights, and active compute so they cannot be conflated::

    P_expert_pool = P_total - P_shared_base - P_vocab - P_router
    P_resident    = P_shared_base + P_vocab + P_router + m * P_expert
    P_per_token   = P_shared_base + P_vocab + P_router + k * P_expert

where ``m`` is the resident expert count and ``k`` the per-token top-``k``.
Every number here is measured from the model's actual parameter tensors, not
estimated from a formula, so it stays exact under any configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import nn

_COMPONENT_ORDER = (
    "embeddings",
    "attention",
    "norms",
    "shared_channel",
    "router",
    "expert_pool",
    "directional",
    "atxy",
    "vocab_head",
    "other",
)


@dataclass(frozen=True)
class ParameterAccounting:
    """Exact parameter accounting of one model at one residency state."""

    components: dict[str, int]
    num_experts: int
    resident_expert_ids: tuple[int, ...]
    active_experts_per_token: int
    parameters_per_expert: int
    nominal_bits: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.components.values())

    @property
    def expert_pool(self) -> int:
        return self.components["expert_pool"]

    @property
    def vocab_head(self) -> int:
        return self.components["vocab_head"]

    @property
    def router(self) -> int:
        return self.components["router"]

    @property
    def shared_base(self) -> int:
        """Everything outside the expert pool, vocabulary head, and router."""

        return self.total - self.expert_pool - self.vocab_head - self.router

    @property
    def shared_channel(self) -> int:
        return self.components["shared_channel"]

    @property
    def resident_experts(self) -> int:
        return len(self.resident_expert_ids)

    @property
    def resident_expert_parameters(self) -> int:
        return self.resident_experts * self.parameters_per_expert

    @property
    def resident_parameters(self) -> int:
        """Parameters that must be resident to serve the current request."""

        return self.total - self.expert_pool + self.resident_expert_parameters

    @property
    def per_token_active_parameters(self) -> int:
        """Parameters one token touches: shared base, vocab, router, top-``k`` experts."""

        return (
            self.total
            - self.expert_pool
            + self.active_experts_per_token * self.parameters_per_expert
        )

    @property
    def expert_fraction_resident(self) -> float:
        return self.resident_experts / self.num_experts if self.num_experts else 0.0

    def nominal_bytes(self) -> dict[str, float]:
        """Storage at each component's nominal precision (bits from ``nominal_bits``)."""

        return {
            name: count * self.nominal_bits.get(name, 16) / 8
            for name, count in self.components.items()
        }

    def resident_bytes(self) -> float:
        """Nominal bytes that must be resident for the current request."""

        per_expert_bytes = self.parameters_per_expert * self.nominal_bits.get("expert_pool", 16) / 8
        pool_bytes = self.nominal_bytes()["expert_pool"]
        return (
            sum(self.nominal_bytes().values())
            - pool_bytes
            + (self.resident_experts * per_expert_bytes)
        )

    def summary(self) -> str:
        lines = [f"total={self.total:,}"]
        for name in _COMPONENT_ORDER:
            count = self.components.get(name, 0)
            if count:
                lines.append(f"  {name}={count:,}")
        lines.append(
            f"resident={self.resident_parameters:,} "
            f"({self.resident_experts}/{self.num_experts} experts, "
            f"{self.parameters_per_expert:,} per expert)"
        )
        lines.append(
            f"per_token_active={self.per_token_active_parameters:,} "
            f"(top-{self.active_experts_per_token})"
        )
        return "\n".join(lines)


def classify_parameter(name: str) -> str:
    """Map a ``named_parameters`` path of a SPALMER model to an accounting component."""

    if name.startswith("lm_head."):
        return "vocab_head"
    if name.startswith("backbone.embeddings."):
        return "embeddings"
    if name.startswith("backbone.atxy") or ".atxy." in name:
        return "atxy"
    if ".directional_mixer." in name:
        return "directional"
    if ".channel_mixer.experts." in name:
        return "expert_pool"
    if ".channel_mixer.router." in name:
        return "router"
    if ".channel_mixer.shared." in name:
        return "shared_channel"
    if ".token_mixer." in name:
        return "attention"
    if name.endswith("norm.weight") or "_norm." in name:
        return "norms"
    return "other"


def account_parameters(
    model: nn.Module,
    *,
    nominal_bits: dict[str, int] | None = None,
) -> ParameterAccounting:
    """Measure a model's parameters by component at its current residency state.

    Shared modules (the router, the residency set) are registered once and
    therefore counted once, exactly as ``named_parameters`` deduplicates them.
    """

    components = {name: 0 for name in _COMPONENT_ORDER}
    for name, parameter in model.named_parameters():
        components[classify_parameter(name)] += parameter.numel()

    residency = getattr(model, "residency", None)
    num_experts = 0
    resident_ids: tuple[int, ...] = ()
    per_token = 0
    per_expert = 0
    banks = [
        block.channel_mixer.experts
        for block in getattr(getattr(model, "backbone", None), "blocks", [])
        if hasattr(getattr(block, "channel_mixer", None), "experts")
    ]
    if banks:
        num_experts = banks[0].num_experts
        per_expert = sum(bank.parameters_per_expert for bank in banks)
        per_token = getattr(model.backbone.blocks[0].channel_mixer, "active_experts", 0)
        resident_ids = residency.ids if residency is not None else tuple(range(num_experts))
    if per_expert * num_experts != components["expert_pool"]:
        raise RuntimeError(
            "expert pool accounting mismatch: "
            f"{per_expert} per expert x {num_experts} != {components['expert_pool']}"
        )
    bits = dict(nominal_bits or {})
    if not bits:
        config = getattr(model, "config", None)
        ple_bits = getattr(config, "ple_quant_bits", None)
        if ple_bits is not None:
            bits["embeddings"] = int(ple_bits)
        if banks and banks[0].config.expert_fake_quantization:
            bits["expert_pool"] = int(banks[0].config.expert_quant_bits)
    return ParameterAccounting(
        components=components,
        num_experts=num_experts,
        resident_expert_ids=tuple(resident_ids),
        active_experts_per_token=int(per_token),
        parameters_per_expert=per_expert,
        nominal_bits=bits,
    )


__all__ = ["ParameterAccounting", "account_parameters", "classify_parameter"]
