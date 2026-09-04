"""Scale-aware, construction-free planning for SPALMER checkpoints.

The estimates mirror the currently assembled 3 KDA : 1 MLA prototype, but
this module intentionally imports no model code (only the pure-arithmetic QR
lane partition helpers shared with the QR-PLE module).  It therefore cannot
allocate weights, execute a forward pass, or start training.  Counts are
planning estimates rather than a substitute for a final ``named_parameters``
audit.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Literal, Protocol, runtime_checkable

from spalmer.precision import (
    EXPERT_ACTIVATION_FORMATS,
    EXPERT_WEIGHT_FORMATS,
    ExpertActivationFormat,
    ExpertWeightFormat,
)
from spalmer.qr import qr_codebook_rows, qr_lane_moduli

# Plans use the BF16 quotient/remainder compositional backend exclusively.
PLE_BACKENDS: tuple[str, ...] = ("qr",)


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class DirectionalScaleConfig:
    """Countable configuration for the optional lateral/silencing branch."""

    num_feature_groups: int = 4
    lateral_rank: int = 16

    def __post_init__(self) -> None:
        if self.num_feature_groups < 2:
            raise ValueError("num_feature_groups must be at least 2")
        _require_positive("lateral_rank", self.lateral_rank)

    def parameters_per_layer(self, d_model: int) -> int:
        if d_model % self.num_feature_groups:
            raise ValueError("num_feature_groups must divide d_model")
        group_width = d_model // self.num_feature_groups
        return 2 * self.num_feature_groups * self.lateral_rank + 2 * group_width + 2


@dataclass(frozen=True, slots=True)
class ATXYScaleConfig:
    """Countable configuration for the optional exact-memory path."""

    value_dim: int
    a_cardinality: int
    t_cardinality: int
    x_cardinality: int
    y_cardinality: int
    injection_layer: int = 0

    def __post_init__(self) -> None:
        for name in (
            "value_dim",
            "a_cardinality",
            "t_cardinality",
            "x_cardinality",
            "y_cardinality",
        ):
            _require_positive(name, int(getattr(self, name)))
        if self.injection_layer < 0:
            raise ValueError("injection_layer must be non-negative")

    @property
    def cardinalities(self) -> tuple[int, int, int, int]:
        return (
            self.a_cardinality,
            self.t_cardinality,
            self.x_cardinality,
            self.y_cardinality,
        )

    def parameter_count(self, d_model: int) -> int:
        return (
            sum(self.cardinalities) * d_model
            + d_model * d_model
            + self.value_dim * d_model
            + 1
        )


@dataclass(frozen=True, slots=True)
class RecurrenceScaleConfig:
    """Countable configuration for the optional depth-recurrent latent core.

    The core is the block span between ``prelude_layers`` leading blocks and
    ``coda_layers`` trailing ones; it is executed ``default_steps`` times at
    inference unless a caller asks for another depth.
    """

    prelude_layers: int = 1
    coda_layers: int = 1
    default_steps: int = 8
    latent_init_std: float = 1.0

    def __post_init__(self) -> None:
        for name in ("prelude_layers", "coda_layers", "default_steps"):
            _require_positive(name, int(getattr(self, name)))
        if not math.isfinite(self.latent_init_std) or self.latent_init_std < 0:
            raise ValueError("latent_init_std must be a finite, non-negative value")

    def core_layers(self, n_layers: int) -> int:
        core = n_layers - self.prelude_layers - self.coda_layers
        if core <= 0:
            raise ValueError("recurrence needs at least one core layer")
        return core

    def parameter_count(self, d_model: int) -> int:
        """Adapter projection over ``cat[s, e]`` plus the two learned RMSNorms."""

        return 2 * d_model * d_model + 2 * d_model

    def effective_depth(self, n_layers: int, steps: int | None = None) -> int:
        resolved = self.default_steps if steps is None else steps
        _require_positive("steps", int(resolved))
        return self.prelude_layers + resolved * self.core_layers(n_layers) + self.coda_layers


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
    shared_width: int | None = None
    q_latent_dim: int | None = None
    kv_latent_dim: int | None = None
    conv_width: int = 4
    conv_bias: bool = False
    router_bias: bool = False
    active_experts: int = 2
    min_resident_experts: int = 2
    max_resident_experts: int = 20
    expert_weight_format: ExpertWeightFormat = "bfloat16"
    expert_activation_format: ExpertActivationFormat = "bfloat16"
    expert_master_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    directional: DirectionalScaleConfig | None = None
    atxy: ATXYScaleConfig | None = None
    recurrence: RecurrenceScaleConfig | None = None
    # ``ple_expansion`` is the QR lane count on refresh layers ``1..L-1``.
    ple_backend: Literal["qr"] = "qr"

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
            "active_experts",
            "min_resident_experts",
            "max_resident_experts",
        ):
            _require_positive(name, int(getattr(self, name)))
        if self.shared_width is not None:
            _require_positive("shared_width", self.shared_width)
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
        if self.active_experts < 2:
            raise ValueError("active_experts must preserve the architecture's two-expert floor")
        if not self.active_experts <= self.min_resident_experts <= self.max_resident_experts:
            raise ValueError(
                "expert counts must satisfy active_experts <= min_resident_experts "
                "<= max_resident_experts"
            )
        if self.min_resident_experts > self.num_experts:
            raise ValueError("min_resident_experts cannot exceed num_experts")
        if self.expert_weight_format not in EXPERT_WEIGHT_FORMATS:
            raise ValueError(f"expert_weight_format must be one of {EXPERT_WEIGHT_FORMATS}")
        if self.expert_activation_format not in EXPERT_ACTIVATION_FORMATS:
            raise ValueError(
                f"expert_activation_format must be one of {EXPERT_ACTIVATION_FORMATS}"
            )
        if self.expert_weight_format in {"bfloat16", "float32"}:
            if self.expert_activation_format != self.expert_weight_format:
                raise ValueError(
                    f"{self.expert_weight_format.upper()} expert weights require matching "
                    "activations"
                )
        elif self.expert_weight_format == "nvfp4":
            if self.expert_activation_format not in {"nvfp4", "mxfp8"}:
                raise ValueError("NVFP4 expert weights require NVFP4 or MXFP8 activations")
        elif self.expert_activation_format != "mxfp8":
            raise ValueError(
                f"{self.expert_weight_format.upper()} expert weights require MXFP8 activations"
            )
        expected_master = (
            "float32" if self.expert_weight_format == "float32" else "bfloat16"
        )
        if self.expert_master_dtype != expected_master:
            raise ValueError(
                f"{self.expert_weight_format.upper()} expert planning requires "
                f"expert_master_dtype={expected_master!r}"
            )
        if self.ple_backend not in PLE_BACKENDS:
            raise ValueError("ple_backend must be 'qr'; fake-QAT PLE is retired")
        if self.directional is not None:
            self.directional.parameters_per_layer(self.d_model)
        if self.atxy is not None and self.atxy.injection_layer >= self.n_layers:
            raise ValueError("ATXY injection_layer must be below n_layers")
        if self.recurrence is not None:
            if self.n_layers <= self.recurrence.prelude_layers + self.recurrence.coda_layers:
                raise ValueError("recurrence needs at least one core layer")
            if self.atxy is not None and (
                self.recurrence.prelude_layers
                <= self.atxy.injection_layer
                < self.n_layers - self.recurrence.coda_layers
            ):
                raise ValueError("ATXY injection_layer must lie outside the recurrent core")

    @property
    def core_layers(self) -> int:
        """Physical blocks inside the recurrent core (0 when not recurrent)."""

        if self.recurrence is None:
            return 0
        return self.recurrence.core_layers(self.n_layers)

    def effective_depth(self, steps: int | None = None) -> int:
        """Block passes per token: ``n_layers`` unless a core is iterated."""

        if self.recurrence is None:
            return self.n_layers
        return self.recurrence.effective_depth(self.n_layers, steps)

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
    def resolved_shared_width(self) -> int:
        return self.shared_width or 2 * self.d_model

    @property
    def kda_layers(self) -> int:
        return self.n_layers * 3 // 4

    @property
    def mla_layers(self) -> int:
        return self.n_layers // 4

    @property
    def ple_lane_moduli(self) -> tuple[int, ...]:
        """Deterministic QR lane moduli."""

        return qr_lane_moduli(self.vocab_size, self.ple_expansion)

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        if self.recurrence is None:
            # Non-recurrent shapes keep the exact dict every stored plan
            # fingerprint was computed from.
            values.pop("recurrence")
        return values


@dataclass(frozen=True, slots=True)
class ParameterBreakdown:
    """Transparent parameter ownership for a planned model shape."""

    ple_lookup: int
    ple_controls: int
    kda_mixers: int
    mla_mixers: int
    block_norms: int
    shared_channel: int
    router: int
    expert_banks: int
    directional: int
    atxy: int
    lm_head: int
    recurrence: int = 0
    # QR-PLE decomposition of ``ple_lookup``: the exact BF16 input table of
    # layer 0 and the quotient/remainder codebooks of the refresh layers.
    ple_exact_input: int = 0
    ple_qr_refresh: int = 0
    # Expert parameters entering a real native sub-BF16 kernel. This is zero
    # for the dense BF16 default.
    expert_low_bit: int = 0

    @property
    def total(self) -> int:
        return sum(
            (
                self.ple_lookup,
                self.ple_controls,
                self.kda_mixers,
                self.mla_mixers,
                self.block_norms,
                self.shared_channel,
                self.router,
                self.expert_banks,
                self.directional,
                self.atxy,
                self.recurrence,
                self.lm_head,
            )
        )

    @property
    def embeddings(self) -> int:
        """All PLE tensors, matching ``account_parameters(...).components``."""

        return self.ple_lookup + self.ple_controls

    @property
    def attention(self) -> int:
        """All KDA and MLA tensors."""

        return self.kda_mixers + self.mla_mixers

    @property
    def norms(self) -> int:
        return self.block_norms

    @property
    def expert_pool(self) -> int:
        return self.expert_banks

    @property
    def components(self) -> dict[str, int]:
        """Component map with the same ownership keys as measured accounting."""

        return {
            "embeddings": self.embeddings,
            "attention": self.attention,
            "norms": self.norms,
            "shared_channel": self.shared_channel,
            "router": self.router,
            "expert_pool": self.expert_pool,
            "directional": self.directional,
            "atxy": self.atxy,
            "recurrence": self.recurrence,
            "vocab_head": self.lm_head,
            "other": 0,
        }

    @property
    def ple_total(self) -> int:
        """Every PLE parameter: lookup tables or codebooks plus scalar controls."""

        return self.ple_lookup + self.ple_controls

    @property
    def low_bit_candidates(self) -> int:
        """Parameters whose forward representations are derived at low precision.

        QR-PLE codebooks execute at their BF16 storage width and are excluded.
        """

        return self.expert_low_bit

    @property
    def dense_parameters(self) -> int:
        return self.total - self.low_bit_candidates

    def to_dict(self) -> dict[str, int]:
        values = asdict(self)
        if not self.recurrence:
            # Keep every non-recurrent breakdown byte-identical to the dicts
            # existing plan fingerprints were hashed from.
            values.pop("recurrence")
        if not self.ple_exact_input and not self.ple_qr_refresh:
            values.pop("ple_exact_input")
            values.pop("ple_qr_refresh")
        if not self.expert_low_bit:
            values.pop("expert_low_bit")
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

    ple_exact_input = 0
    ple_qr_refresh = 0
    if config.ple_backend == "qr":
        # Layer 0 is one exact BF16 [vocab, d] table without lanes; every later
        # physical layer composes ``ple_expansion`` quotient/remainder lanes.
        ple_exact_input = config.vocab_size * d
        remainder_rows, quotient_rows = qr_codebook_rows(
            config.vocab_size,
            config.ple_expansion,
        )
        ple_qr_refresh = (
            (config.n_layers - 1) * (remainder_rows + quotient_rows) * d
        )
        ple_lookup = ple_exact_input + ple_qr_refresh
        ple_controls = (config.n_layers - 1) * (config.ple_expansion + 1)
    else:
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

    # Two ordinary pre-norm weights per block, an additional pre-norm when the
    # directional branch is enabled, and one final RMSNorm weight.
    norms_per_block = 3 if config.directional is not None else 2
    block_norms = (norms_per_block * config.n_layers + 1) * d
    shared_channel = config.n_layers * 3 * d * config.resolved_shared_width
    router = d * config.num_experts + (config.num_experts if config.router_bias else 0)
    expert_banks = (
        config.n_layers * config.num_experts * 3 * d * config.expert_width
    )
    directional = (
        0
        if config.directional is None
        else config.n_layers * config.directional.parameters_per_layer(d)
    )
    atxy = 0 if config.atxy is None else config.atxy.parameter_count(d)
    recurrence = 0 if config.recurrence is None else config.recurrence.parameter_count(d)
    lm_head = d * config.vocab_size
    return ParameterBreakdown(
        ple_lookup=ple_lookup,
        ple_controls=ple_controls,
        kda_mixers=kda_mixers,
        mla_mixers=mla_mixers,
        block_norms=block_norms,
        shared_channel=shared_channel,
        router=router,
        expert_banks=expert_banks,
        directional=directional,
        atxy=atxy,
        recurrence=recurrence,
        lm_head=lm_head,
        ple_exact_input=ple_exact_input,
        ple_qr_refresh=ple_qr_refresh,
        expert_low_bit=(
            expert_banks
            if config.expert_weight_format not in {"bfloat16", "float32"}
            else 0
        ),
    )


@dataclass(frozen=True, slots=True)
class MemoryAssumptions:
    """Explicit byte-width assumptions for a rough persistent-state estimate.

    Activations, logits, caches, input batches, allocator fragmentation, and
    distributed communication buffers are intentionally excluded.  The
    reference defaults mirror the selected base-pretraining lane: one BF16
    persistent/master parameter payload, BF16 gradients, two FP32 optimizer
    slots, and no duplicate FP32 weight copy.

    The ``hypothetical_*`` fields describe optionally caching a packed forward
    payload beside the BF16 master. The current QAT correctness lane derives
    quantized operands per call and does not retain that extra packed copy.
    """

    reference_weight_bits: int = 16
    reference_gradient_bits: int = 16
    optimizer_state_bits: int = 32
    optimizer_slots: int = 2
    hypothetical_packed_base_weight_bits: int = 16
    # Four data bits plus a conservative half-bit/value allowance for block
    # scales (NVFP4 is 8 scale bits per 16 values; MXFP4 needs less).
    # Tensor-level scales and backend-specific alignment padding remain outside
    # this rough estimate and are called out in the result documentation.
    hypothetical_packed_low_bit_bits: float = 4.5
    hypothetical_master_weight_bits: int = 16
    overhead_fraction: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "reference_weight_bits",
            "reference_gradient_bits",
            "optimizer_state_bits",
            "hypothetical_packed_base_weight_bits",
        ):
            value = getattr(self, name)
            _require_positive(name, value)
            if value % 2:
                raise ValueError(f"{name} must be divisible by 2")
        if self.hypothetical_packed_low_bit_bits <= 0:
            raise ValueError("hypothetical_packed_low_bit_bits must be positive")
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

    ``reference_*`` is the current BF16 master/gradient lane. ``hypothetical_*``
    adds cached packed execution weights beside the same BF16 master; the QAT
    correctness backend does not retain that cache. Packed estimates include
    a conservative block-scale allowance but exclude tensor-level metadata and
    backend alignment padding.
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
        """Compatibility alias for current BF16 reference gradients."""

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


def _bits_to_bytes(parameters: int, bits: int | float) -> int:
    return math.ceil(parameters * bits / 8)


def estimate_memory(
    parameters: ParameterBreakdown,
    assumptions: MemoryAssumptions = MemoryAssumptions(),
) -> MemoryEstimate:
    """Estimate BF16-master QAT and an optional cached packed-weight lane."""

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
    shared_width_multipliers: tuple[int, ...] = (1, 2, 4)
    ple_expansions: tuple[int, ...] = (2, 4)

    def __post_init__(self) -> None:
        for name in (
            "d_models",
            "n_layers",
            "num_heads",
            "expert_width_divisors",
            "shared_width_multipliers",
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
    def effective_depth(self) -> int:
        """Physical block passes per token at the plan's default depth."""

        return self.config.effective_depth()

    @property
    def block_passes_per_token(self) -> int:
        """Alias of :attr:`effective_depth`; one pass per physical block."""

        return self.effective_depth

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
        payload: dict[str, object] = {
            "target_parameters": self.target_parameters,
            "parameter_error": self.parameter_error,
            "relative_error": self.relative_error,
            "config": self.config.to_dict(),
            "parameters": self.parameters.to_dict(),
            "memory": self.memory.to_dict(),
            "fingerprint": self.fingerprint,
        }
        recurrence = self.config.recurrence
        if recurrence is not None:
            payload["recurrence"] = {
                "prelude_layers": recurrence.prelude_layers,
                "core_layers": self.config.core_layers,
                "coda_layers": recurrence.coda_layers,
                "default_steps": recurrence.default_steps,
                "effective_depth": self.effective_depth,
                "core_mixers": _core_mixer_counts(self.config),
            }
        if self.config.ple_backend == "qr":
            payload["ple"] = self.ple_summary()
        return payload

    def ple_summary(self) -> dict[str, object]:
        """Backend-aware PLE ownership: exact input, QR refresh, controls, total."""

        config = self.config
        parameters = self.parameters
        return {
            "backend": config.ple_backend,
            "lanes": config.ple_expansion,
            "moduli": list(config.ple_lane_moduli),
            "exact_input_parameters": parameters.ple_exact_input,
            "qr_refresh_parameters": parameters.ple_qr_refresh,
            "lookup_parameters": parameters.ple_lookup,
            "control_parameters": parameters.ple_controls,
            "total_parameters": parameters.ple_total,
        }


def _core_mixer_counts(config: ModelScaleConfig) -> dict[str, int]:
    """Count the 3:1 KDA/MLA cycle inside the recurrent core, by physical index."""

    recurrence = config.recurrence
    if recurrence is None:
        return {"kda": 0, "mla": 0}
    start = recurrence.prelude_layers
    stop = config.n_layers - recurrence.coda_layers
    mla = sum(1 for index in range(start, stop) if index % 4 == 3)
    return {"kda": (stop - start) - mla, "mla": mla}


def plan_configuration(
    target_parameters: int,
    vocab_size: int,
    *,
    search_space: ScaleSearchSpace = DEFAULT_SEARCH_SPACE,
    memory_assumptions: MemoryAssumptions = MemoryAssumptions(),
    expert_weight_format: ExpertWeightFormat = "bfloat16",
    expert_activation_format: ExpertActivationFormat | None = None,
    directional: DirectionalScaleConfig | None = None,
    atxy: ATXYScaleConfig | None = None,
    recurrence: RecurrenceScaleConfig | None = None,
    ple_backend: Literal["qr"] = "qr",
) -> ScalePlan:
    """Choose the enumerated shape nearest an explicit total parameter target.

    The same finite grid serves both PLE backends, so a QR plan redistributes
    the capacity its smaller codebooks free into width, depth, the shared
    channel, and the expert banks while landing as close to ``target_parameters``
    as the grid allows.
    """

    _require_positive("target_parameters", target_parameters)
    _require_positive("vocab_size", vocab_size)
    if ple_backend not in PLE_BACKENDS:
        raise ValueError("ple_backend must be 'qr'; fake-QAT PLE is retired")
    if expert_weight_format not in EXPERT_WEIGHT_FORMATS:
        raise ValueError(f"expert_weight_format must be one of {EXPERT_WEIGHT_FORMATS}")
    resolved_activation = expert_activation_format or {
        "bfloat16": "bfloat16",
        "float32": "float32",
        "nvfp4": "nvfp4",
    }.get(expert_weight_format, "mxfp8")
    candidates: list[tuple[tuple[int, ...], ModelScaleConfig, ParameterBreakdown]] = []
    for d_model in sorted(set(search_space.d_models)):
        if directional is not None and d_model % directional.num_feature_groups:
            continue
        for n_layers in sorted(set(search_space.n_layers)):
            if n_layers % 4:
                continue
            if atxy is not None and atxy.injection_layer >= n_layers:
                continue
            if recurrence is not None:
                if n_layers <= recurrence.prelude_layers + recurrence.coda_layers:
                    continue
                if atxy is not None and (
                    recurrence.prelude_layers
                    <= atxy.injection_layer
                    < n_layers - recurrence.coda_layers
                ):
                    continue
            for num_heads in sorted(set(search_space.num_heads)):
                if d_model % num_heads:
                    continue
                for divisor in sorted(set(search_space.expert_width_divisors)):
                    if d_model % divisor:
                        continue
                    for shared_multiplier in sorted(
                        set(search_space.shared_width_multipliers)
                    ):
                        for expansion in sorted(set(search_space.ple_expansions)):
                            config = ModelScaleConfig(
                                vocab_size=vocab_size,
                                d_model=d_model,
                                n_layers=n_layers,
                                num_heads=num_heads,
                                num_experts=search_space.num_experts,
                                expert_width=d_model // divisor,
                                shared_width=d_model * shared_multiplier,
                                ple_expansion=expansion,
                                max_resident_experts=min(20, search_space.num_experts),
                                expert_weight_format=expert_weight_format,
                                expert_activation_format=resolved_activation,
                                expert_master_dtype=(
                                    "float32"
                                    if expert_weight_format == "float32"
                                    else "bfloat16"
                                ),
                                directional=directional,
                                atxy=atxy,
                                recurrence=recurrence,
                                ple_backend=ple_backend,
                            )
                            breakdown = count_parameters(config)
                            # Target distance dominates; stable tie-breakers prefer
                            # the smaller low-bit footprint and simpler shape.
                            key = (
                                abs(breakdown.total - target_parameters),
                                breakdown.low_bit_candidates,
                                n_layers,
                                d_model,
                                num_heads,
                                divisor,
                                shared_multiplier,
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
    expert_weight_format: ExpertWeightFormat = "bfloat16",
    expert_activation_format: ExpertActivationFormat | None = None,
    directional: DirectionalScaleConfig | None = None,
    atxy: ATXYScaleConfig | None = None,
    recurrence: RecurrenceScaleConfig | None = None,
    ple_backend: Literal["qr"] = "qr",
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
                expert_weight_format=expert_weight_format,
                expert_activation_format=expert_activation_format,
                directional=directional,
                atxy=atxy,
                recurrence=recurrence,
                ple_backend=ple_backend,
            )
        )
    return tuple(plans)


__all__ = [
    "ATXYScaleConfig",
    "DEFAULT_SEARCH_SPACE",
    "DirectionalScaleConfig",
    "ExplicitVocabularyPolicy",
    "MemoryAssumptions",
    "MemoryEstimate",
    "ModelScaleConfig",
    "PLE_BACKENDS",
    "ParameterBreakdown",
    "RecurrenceScaleConfig",
    "ScalePlan",
    "ScaleSearchSpace",
    "VocabularySizePolicy",
    "count_parameters",
    "estimate_memory",
    "plan_configuration",
    "plan_ladder",
]
