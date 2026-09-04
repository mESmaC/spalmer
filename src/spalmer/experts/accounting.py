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
    "recurrence",
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
    # Persistent/master storage width. This intentionally does not describe
    # fake-quantized forward execution.
    nominal_bits: dict[str, int] = field(default_factory=dict)
    # Forward execution widths where they differ from persistent storage
    # (currently PLE lookup weights and routed-expert weights).
    execution_bits: dict[str, int] = field(default_factory=dict)
    # Bytes occupied by the model's current PyTorch parameter tensors. This is
    # useful when a newly constructed reference model has not yet been cast to
    # its configured persistent dtype.
    parameter_bytes: dict[str, int] = field(default_factory=dict)
    # Block passes one token makes: ``n_layers`` for a flat stack, and
    # ``prelude + steps * core + coda`` for a depth-recurrent model at its
    # checkpointed default depth.
    effective_depth: int | None = None
    per_token_block_passes: int | None = None

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
        """Parameters one token touches: shared base, vocab, router, top-``k`` experts.

        This counts DISTINCT parameters, so a depth-recurrent core contributes
        its core weights once no matter how many iterations run. Compute is
        ``per_token_block_passes`` times the per-pass cost, not this number.
        """

        return (
            self.total
            - self.expert_pool
            + self.active_experts_per_token * self.parameters_per_expert
        )

    @property
    def expert_fraction_resident(self) -> float:
        return self.resident_experts / self.num_experts if self.num_experts else 0.0

    def nominal_bytes(self) -> dict[str, float]:
        """Persistent storage at each component's master precision."""

        return {
            name: (
                count * self.nominal_bits[name] / 8
                if name in self.nominal_bits
                else float(self.parameter_bytes.get(name, count * 2))
            )
            for name, count in self.components.items()
        }

    @property
    def actual_parameter_bytes(self) -> int:
        """Bytes occupied by all current (possibly not-yet-cast) parameter tensors."""

        return sum(self.parameter_bytes.values())

    def resident_bytes(self) -> float:
        """Nominal bytes that must be resident for the current request."""

        per_expert_bytes = self.parameters_per_expert * self.nominal_bits.get("expert_pool", 16) / 8
        pool_bytes = self.nominal_bytes()["expert_pool"]
        return (
            sum(self.nominal_bytes().values())
            - pool_bytes
            + (self.resident_experts * per_expert_bytes)
        )

    @property
    def is_recurrent(self) -> bool:
        """Whether one token makes more block passes than there are blocks."""

        return (
            self.per_token_block_passes is not None
            and self.effective_depth is not None
            and self.components.get("recurrence", 0) > 0
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
        if self.is_recurrent:
            lines.append(
                f"effective_depth={self.effective_depth} "
                f"(block passes per token {self.per_token_block_passes})"
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
    # Must precede the directional and norm rules: the recurrence module owns
    # two RMSNorm weights whose names would otherwise land in "norms".
    if ".recurrence." in name:
        return "recurrence"
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
    parameter_bytes = {name: 0 for name in _COMPONENT_ORDER}
    parameter_widths: dict[str, set[int]] = {name: set() for name in _COMPONENT_ORDER}
    for name, parameter in model.named_parameters():
        component = classify_parameter(name)
        components[component] += parameter.numel()
        parameter_bytes[component] += parameter.numel() * parameter.element_size()
        parameter_widths[component].add(parameter.element_size() * 8)

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
        resident_ids = residency.ids if residency is not None else tuple(range(num_experts))
        per_token = (
            residency.active_experts
            if residency is not None
            else getattr(model.backbone.blocks[0].channel_mixer, "active_experts", 0)
        )
        if per_token > len(resident_ids):
            raise RuntimeError(
                f"cannot account for top-{per_token} execution with only "
                f"{len(resident_ids)} resident experts"
            )
    if per_expert * num_experts != components["expert_pool"]:
        raise RuntimeError(
            "expert pool accounting mismatch: "
            f"{per_expert} per expert x {num_experts} != {components['expert_pool']}"
        )
    bits = {
        name: next(iter(widths))
        for name, widths in parameter_widths.items()
        if len(widths) == 1
    }
    execution_bits: dict[str, int] = {}
    config = getattr(model, "config", None)
    ple_bits = getattr(config, "ple_quant_bits", None)
    if ple_bits is not None and getattr(config, "ple_backend", "fake_qat") == "fake_qat":
        # Only the legacy fake-QAT lookup derives low-bit forward operands;
        # QR-PLE codebooks execute at their BF16 storage width.
        execution_bits["embeddings"] = int(ple_bits)
    if banks:
        expert_config = banks[0].config
        bits["expert_pool"] = int(expert_config.expert_master_bits)
        execution_bits["expert_pool"] = int(expert_config.expert_forward_weight_bits)
    bits.update(nominal_bits or {})
    effective_depth = None
    if config is not None and callable(getattr(config, "effective_depth", None)):
        effective_depth = int(config.effective_depth())
    return ParameterAccounting(
        components=components,
        num_experts=num_experts,
        resident_expert_ids=tuple(resident_ids),
        active_experts_per_token=int(per_token),
        parameters_per_expert=per_expert,
        nominal_bits=bits,
        execution_bits=execution_bits,
        parameter_bytes=parameter_bytes,
        effective_depth=effective_depth,
        per_token_block_passes=effective_depth,
    )


__all__ = ["ParameterAccounting", "account_parameters", "classify_parameter"]
