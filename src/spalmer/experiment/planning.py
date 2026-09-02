"""Scale-aware, construction-free planning for SPALMER checkpoints.

The estimates mirror the currently assembled 3 KDA : 1 MLA prototype, but
this module intentionally imports no model code.  It therefore cannot allocate
weights, execute a forward pass, or start training.  Counts are planning
estimates rather than a substitute for a final ``named_parameters`` audit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelScaleConfig:
    """One countable SPALMER shape.

    ``vocab_size`` is mandatory.  It should come from the intended tokenizer
    artifact or a caller-owned scale policy; this module has no fixed vocabulary
    assumption.  The latent widths follow the CLI defaults when omitted.
    """

    vocab_size: int
    d_model: int
    n_layers: int
    num_heads: int
    num_experts: int
    expert_width: int
    ple_expansion: int
    q_latent_dim: int | None = None
    kv_latent_dim: int | None = None
    conv_width: int = 4
    conv_bias: bool = False
    router_bias: bool = False

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "d_model",
            "n_layers",
            "num_heads",
            "num_experts",
            "expert_width",
            "ple_expansion",
            "conv_width",
        ):
            _require_positive(name, int(getattr(self, name)))
        if self.n_layers % 4:
            raise ValueError("n_layers must be a multiple of 4 for the 3:1 KDA/MLA cycle")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.conv_width < 2:
            raise ValueError("conv_width must be at least 2")
        if self.q_latent_dim is not None:
            _require_positive("q_latent_dim", self.q_latent_dim)
        if self.kv_latent_dim is not None:
            _require_positive("kv_latent_dim", self.kv_latent_dim)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def resolved_q_latent_dim(self) -> int:
        return self.q_latent_dim or max(self.head_dim, self.d_model // 2)

    @property
    def resolved_kv_latent_dim(self) -> int:
        return self.kv_latent_dim or max(self.head_dim, self.d_model // 4)

    @property
    def kda_layers(self) -> int:
        return self.n_layers * 3 // 4

    @property
    def mla_layers(self) -> int:
        return self.n_layers // 4

    def to_dict(self) -> dict[str, int | bool | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ParameterBreakdown:
    """Transparent parameter ownership for a planned model shape."""

    ple_lookup: int
    ple_controls: int
    kda_mixers: int
    mla_mixers: int
    block_norms: int
    router: int
    expert_banks: int
    lm_head: int

    @property
    def total(self) -> int:
        return sum(
            (
                self.ple_lookup,
                self.ple_controls,
                self.kda_mixers,
                self.mla_mixers,
                self.block_norms,
                self.router,
                self.expert_banks,
                self.lm_head,
            )
        )

    @property
    def low_bit_candidates(self) -> int:
        """Parameters currently represented by fake-QAT PLE/expert shadows."""

        return self.ple_lookup + self.expert_banks

    @property
    def dense_parameters(self) -> int:
        return self.total - self.low_bit_candidates

    def to_dict(self) -> dict[str, int]:
        values = asdict(self)
        values.update(
            total=self.total,
            low_bit_candidates=self.low_bit_candidates,
            dense_parameters=self.dense_parameters,
        )
        return values


def count_parameters(config: ModelScaleConfig) -> ParameterBreakdown:
    """Count trainable parameters for the current core-v0 component formulas."""

    d = config.d_model
    heads = config.num_heads
    head_k = config.head_dim
    head_v = head_k
    key_dim = heads * head_k
    value_dim = heads * head_v

    ple_lookup = config.n_layers * config.vocab_size * config.ple_expansion * d
    ple_controls = config.n_layers * (config.ple_expansion + 1)

    conv_channels = 2 * key_dim + value_dim
    one_kda = (
        d * key_dim  # q projection
        + d * key_dim  # k projection
        + d * value_dim  # v projection
        + conv_channels * config.conv_width
        + (conv_channels if config.conv_bias else 0)
        + d * head_v  # forget-gate bottleneck in
        + head_v * key_dim  # forget-gate bottleneck out
        + d * heads  # beta projection
        + heads  # A_log
        + key_dim  # dt_bias
        + d * head_v  # output-gate bottleneck in
        + head_v * value_dim  # output-gate bottleneck out
        + value_dim  # output-gate bias
        + value_dim * d  # output projection
    )
    kda_mixers = config.kda_layers * one_kda

    q_latent = config.resolved_q_latent_dim
    kv_latent = config.resolved_kv_latent_dim
    one_mla = (
        d * q_latent
        + q_latent  # q latent RMSNorm
        + q_latent * key_dim
        + d * kv_latent
        + kv_latent  # kv latent RMSNorm
        + kv_latent * (key_dim + value_dim)
        + value_dim * d
    )
    mla_mixers = config.mla_layers * one_mla

    # Two pre-norm weights per block and one final RMSNorm weight.
    block_norms = (2 * config.n_layers + 1) * d
    router = d * config.num_experts + (config.num_experts if config.router_bias else 0)
    expert_banks = (
        config.n_layers * config.num_experts * 3 * d * config.expert_width
    )
    lm_head = d * config.vocab_size
    return ParameterBreakdown(
        ple_lookup=ple_lookup,
        ple_controls=ple_controls,
        kda_mixers=kda_mixers,
        mla_mixers=mla_mixers,
        block_norms=block_norms,
        router=router,
        expert_banks=expert_banks,
        lm_head=lm_head,
    )


@dataclass(frozen=True, slots=True)
class MemoryAssumptions:
    """Explicit byte-width assumptions for a rough persistent-state estimate.

    Activations, logits, caches, input batches, allocator fragmentation, and
    distributed communication buffers are intentionally excluded.  The
    reference defaults mirror the current engine: FP32 persistent parameters,
    FP32 gradients, two FP32 optimizer slots, and no separate master copy.

    The ``hypothetical_*`` fields describe a future packed-weight training lane,
    not the current fake-QAT implementation.  That estimate retains a separate
    FP32 master copy because packed weights would not themselves be directly
    updateable by the optimizer.
    """

    reference_weight_bits: int = 32
    reference_gradient_bits: int = 32
    optimizer_state_bits: int = 32
    optimizer_slots: int = 2
    hypothetical_packed_base_weight_bits: int = 16
    hypothetical_packed_low_bit_bits: int = 4
    hypothetical_master_weight_bits: int = 32
    overhead_fraction: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "reference_weight_bits",
            "reference_gradient_bits",
            "optimizer_state_bits",
            "hypothetical_packed_base_weight_bits",
            "hypothetical_packed_low_bit_bits",
        ):
            value = getattr(self, name)
            _require_positive(name, value)
            if value % 2:
                raise ValueError(f"{name} must be divisible by 2")
        if (
            self.hypothetical_master_weight_bits < 0
            or self.hypothetical_master_weight_bits % 2
        ):
            raise ValueError(
                "hypothetical_master_weight_bits must be zero or divisible by 2"
            )
        if self.optimizer_slots < 0:
            raise ValueError("optimizer_slots cannot be negative")
        if not 0 <= self.overhead_fraction <= 1:
            raise ValueError("overhead_fraction must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    """Persistent-state estimates; activation memory is not included.

    ``reference_*`` is the current FP32 parameter/gradient lane.
    ``hypothetical_*`` assumes future packed execution weights plus a distinct
    master copy; the fake-QAT implementation does not realize those weights.
    """

    reference_weights_bytes: int
    reference_gradients_bytes: int
    optimizer_bytes: int
    hypothetical_packed_weights_bytes: int
    hypothetical_master_weights_bytes: int
    reference_training_bytes: int
    hypothetical_packed_training_bytes: int
    overhead_fraction: float

    @property
    def packed_weights_bytes(self) -> int:
        """Compatibility alias for the hypothetical packed-weight estimate."""

        return self.hypothetical_packed_weights_bytes

    @property
    def gradients_bytes(self) -> int:
        """Compatibility alias for current FP32 reference gradients."""

        return self.reference_gradients_bytes

    @property
    def master_weights_bytes(self) -> int:
        """Compatibility alias for the hypothetical packed master copy."""

        return self.hypothetical_master_weights_bytes

    @property
    def packed_training_bytes(self) -> int:
        """Compatibility alias for the hypothetical packed training lane."""

        return self.hypothetical_packed_training_bytes

    @staticmethod
    def gib(byte_count: int) -> float:
        return byte_count / (1024**3)

    def to_dict(self) -> dict[str, int | float]:
        values: dict[str, int | float] = asdict(self)
        for name in (
            "reference_weights_bytes",
            "reference_gradients_bytes",
            "optimizer_bytes",
            "hypothetical_packed_weights_bytes",
            "hypothetical_master_weights_bytes",
            "reference_training_bytes",
            "hypothetical_packed_training_bytes",
        ):
            values[f"{name.removesuffix('_bytes')}_gib"] = self.gib(int(values[name]))
        return values


def _bits_to_bytes(parameters: int, bits: int) -> int:
    return (parameters * bits + 7) // 8


def estimate_memory(
    parameters: ParameterBreakdown,
    assumptions: MemoryAssumptions = MemoryAssumptions(),
) -> MemoryEstimate:
    """Estimate current FP32 fake-QAT and hypothetical packed persistent memory."""

    reference_weights = _bits_to_bytes(parameters.total, assumptions.reference_weight_bits)
    packed_weights = _bits_to_bytes(
        parameters.dense_parameters, assumptions.hypothetical_packed_base_weight_bits
    ) + _bits_to_bytes(
        parameters.low_bit_candidates, assumptions.hypothetical_packed_low_bit_bits
    )
    gradients = _bits_to_bytes(parameters.total, assumptions.reference_gradient_bits)
    optimizer = _bits_to_bytes(
        parameters.total,
        assumptions.optimizer_state_bits * assumptions.optimizer_slots,
    )
    hypothetical_master = _bits_to_bytes(
        parameters.total, assumptions.hypothetical_master_weight_bits
    )
    multiplier = 1.0 + assumptions.overhead_fraction
    reference_total = int((reference_weights + gradients + optimizer) * multiplier)
    packed_total = int(
        (packed_weights + gradients + optimizer + hypothetical_master) * multiplier
    )
    return MemoryEstimate(
        reference_weights_bytes=reference_weights,
        reference_gradients_bytes=gradients,
        optimizer_bytes=optimizer,
        hypothetical_packed_weights_bytes=packed_weights,
        hypothetical_master_weights_bytes=hypothetical_master,
        reference_training_bytes=reference_total,
        hypothetical_packed_training_bytes=packed_total,
        overhead_fraction=assumptions.overhead_fraction,
    )


@runtime_checkable
class VocabularySizePolicy(Protocol):
    """Caller-owned policy for resolving vocabulary size at a model scale."""

    def vocab_size_for(self, target_parameters: int) -> int: ...


@dataclass(frozen=True, slots=True)
class ExplicitVocabularyPolicy:
    """An explicit target-to-vocabulary table with no hidden fallback."""

    sizes: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.sizes:
            raise ValueError("sizes cannot be empty")
        targets = [target for target, _ in self.sizes]
        if targets != sorted(targets) or len(set(targets)) != len(targets):
            raise ValueError("vocabulary targets must be unique and sorted")
        for target, size in self.sizes:
            _require_positive("target_parameters", target)
            _require_positive("vocab_size", size)

    def vocab_size_for(self, target_parameters: int) -> int:
        _require_positive("target_parameters", target_parameters)
        for target, size in self.sizes:
            if target == target_parameters:
                return size
        raise KeyError(f"no vocabulary size is defined for target {target_parameters:,}")


@dataclass(frozen=True, slots=True)
class ScaleSearchSpace:
    """Finite, deterministic set of shapes considered by the planner.

    Expert identity count is a fixed architecture policy, not a parameter-fit
    knob.  Width can become very small at the 10M rung while preserving the
    nominal 200 coherent expert identities across the ladder.
    """

    d_models: tuple[int, ...] = (64, 96, 128, 160, 192, 224, 256, 320, 384)
    n_layers: tuple[int, ...] = (4, 8, 12, 16)
    num_heads: tuple[int, ...] = (2, 4, 8)
    num_experts: int = 200
    expert_width_divisors: tuple[int, ...] = (2, 4, 8, 16)
    ple_expansions: tuple[int, ...] = (2, 4)

    def __post_init__(self) -> None:
        for name in (
            "d_models",
            "n_layers",
            "num_heads",
            "expert_width_divisors",
            "ple_expansions",
        ):
            values = getattr(self, name)
            if not values:
                raise ValueError(f"{name} cannot be empty")
            if any(value <= 0 for value in values):
                raise ValueError(f"{name} must contain only positive values")
        _require_positive("num_experts", self.num_experts)


DEFAULT_SEARCH_SPACE = ScaleSearchSpace()


@dataclass(frozen=True, slots=True)
class ScalePlan:
    target_parameters: int
    config: ModelScaleConfig
    parameters: ParameterBreakdown
    memory: MemoryEstimate

    @property
    def parameter_error(self) -> int:
        return self.parameters.total - self.target_parameters

    @property
    def relative_error(self) -> float:
        return self.parameter_error / self.target_parameters

    @property
    def fingerprint(self) -> str:
        return _canonical_digest(
            {
                "target_parameters": self.target_parameters,
                "config": self.config.to_dict(),
                "parameters": self.parameters.to_dict(),
                "memory": self.memory.to_dict(),
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "target_parameters": self.target_parameters,
            "parameter_error": self.parameter_error,
            "relative_error": self.relative_error,
            "config": self.config.to_dict(),
            "parameters": self.parameters.to_dict(),
            "memory": self.memory.to_dict(),
            "fingerprint": self.fingerprint,
        }


def plan_configuration(
    target_parameters: int,
    vocab_size: int,
    *,
    search_space: ScaleSearchSpace = DEFAULT_SEARCH_SPACE,
    memory_assumptions: MemoryAssumptions = MemoryAssumptions(),
) -> ScalePlan:
    """Choose the enumerated shape nearest an explicit total parameter target."""

    _require_positive("target_parameters", target_parameters)
    _require_positive("vocab_size", vocab_size)
    candidates: list[tuple[tuple[int, ...], ModelScaleConfig, ParameterBreakdown]] = []
    for d_model in sorted(set(search_space.d_models)):
        for n_layers in sorted(set(search_space.n_layers)):
            if n_layers % 4:
                continue
            for num_heads in sorted(set(search_space.num_heads)):
                if d_model % num_heads:
                    continue
                for divisor in sorted(set(search_space.expert_width_divisors)):
                    if d_model % divisor:
                        continue
                    for expansion in sorted(set(search_space.ple_expansions)):
                        config = ModelScaleConfig(
                            vocab_size=vocab_size,
                            d_model=d_model,
                            n_layers=n_layers,
                            num_heads=num_heads,
                            num_experts=search_space.num_experts,
                            expert_width=d_model // divisor,
                            ple_expansion=expansion,
                        )
                        breakdown = count_parameters(config)
                        # Target distance dominates; stable tie-breakers prefer
                        # fewer shadow weights, then the simpler shape.
                        key = (
                            abs(breakdown.total - target_parameters),
                            breakdown.low_bit_candidates,
                            n_layers,
                            d_model,
                            num_heads,
                            divisor,
                            expansion,
                        )
                        candidates.append((key, config, breakdown))
    if not candidates:
        raise ValueError("search_space contains no valid 3:1 KDA/MLA configurations")
    _, config, breakdown = min(candidates, key=lambda candidate: candidate[0])
    return ScalePlan(
        target_parameters=target_parameters,
        config=config,
        parameters=breakdown,
        memory=estimate_memory(breakdown, memory_assumptions),
    )


def plan_ladder(
    targets: tuple[int, ...],
    vocabulary_policy: VocabularySizePolicy,
    *,
    search_space: ScaleSearchSpace = DEFAULT_SEARCH_SPACE,
    memory_assumptions: MemoryAssumptions = MemoryAssumptions(),
) -> tuple[ScalePlan, ...]:
    """Plan several scales, resolving vocabulary size independently per rung."""

    if not targets:
        raise ValueError("targets cannot be empty")
    plans = []
    for target in targets:
        vocab_size = vocabulary_policy.vocab_size_for(target)
        plans.append(
            plan_configuration(
                target,
                vocab_size,
                search_space=search_space,
                memory_assumptions=memory_assumptions,
            )
        )
    return tuple(plans)


__all__ = [
    "DEFAULT_SEARCH_SPACE",
    "ExplicitVocabularyPolicy",
    "MemoryAssumptions",
    "MemoryEstimate",
    "ModelScaleConfig",
    "ParameterBreakdown",
    "ScalePlan",
    "ScaleSearchSpace",
    "VocabularySizePolicy",
    "count_parameters",
    "estimate_memory",
    "plan_configuration",
    "plan_ladder",
]
