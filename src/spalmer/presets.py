"""Architecture-only shape hooks for parameter classes and variable vocabularies.

This module answers one question for the data/training half: *given a
parameter class and a vocabulary size, which configuration objects build the
model, and exactly how many parameters land in each component?* It does not
plan experiments, ingest data, or train.

Parameter classes (``10M``, ``50M``, ``100M``) are defined on the
**non-embedding** parameters: attention, norms, the shared channel path, the
router, the expert pool, and the optional directional/ATXY branches. The
vocabulary-dependent parts — the per-layer low-bit lookup tables (PLE, ledger
C02) and the untied vocabulary head (C11) — scale with ``vocab_size`` and are
reported separately, in parameters and in nominal storage bytes, because the
ledger intends vocabulary to be scale-dependent rather than fixed. PLE tables
cost ``n_layers * vocab_size * expansion_factor * d_model`` parameters, so a
large vocabulary is a deliberate budget decision that this module makes
visible instead of hiding.

All estimates here are analytic and are checked against the measured
:func:`spalmer.experts.account_parameters` in the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from spalmer.attention import KDAConfig, MLAConfig
from spalmer.config import SPALMERConfig
from spalmer.directional import DirectionalConfig
from spalmer.experts import MicroExpertsConfig
from spalmer.memory import ATXYConfig
from spalmer.modeling import SPALMERCausalLM

HYBRID_CYCLE: tuple[str, ...] = ("kda", "kda", "kda", "mla")


@dataclass(frozen=True)
class ArchitectureShape:
    """Every architecture knob needed to build one SPALMER model."""

    name: str
    d_model: int
    n_layers: int
    num_heads: int
    head_dim: int
    q_latent_dim: int
    kv_latent_dim: int
    num_experts: int
    expert_inter_dim: int
    shared_inter_dim: int
    active_experts: int = 2
    min_resident_experts: int = 2
    max_resident_experts: int = 20
    ple_expansion_factor: int = 1
    ple_quant_bits: int = 4
    expert_quant_bits: int = 4
    conv_width: int = 4
    directional_feature_groups: int = 4
    directional_lateral_rank: int = 16
    token_mixer_pattern: tuple[str, ...] = HYBRID_CYCLE

    def __post_init__(self) -> None:
        if self.n_layers % len(self.token_mixer_pattern):
            raise ValueError(
                f"n_layers={self.n_layers} must be a multiple of the "
                f"{len(self.token_mixer_pattern)}-layer token mixer pattern"
            )
        if self.num_heads * self.head_dim != self.d_model:
            raise ValueError("num_heads * head_dim must equal d_model")

    @property
    def key_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kda_layers(self) -> int:
        cycles = self.n_layers // len(self.token_mixer_pattern)
        return cycles * self.token_mixer_pattern.count("kda")

    @property
    def mla_layers(self) -> int:
        return self.n_layers - self.kda_layers

    def with_overrides(self, **overrides: Any) -> ArchitectureShape:
        return replace(self, **overrides)


@dataclass(frozen=True)
class ParameterEstimate:
    """Analytic parameter counts of a shape at one vocabulary size."""

    attention: int
    norms: int
    shared_channel: int
    router: int
    expert_pool: int
    directional: int
    atxy: int
    embeddings: int
    vocab_head: int
    parameters_per_expert: int
    nominal_bits: dict[str, int]

    @property
    def non_embedding(self) -> int:
        return (
            self.attention
            + self.norms
            + self.shared_channel
            + self.router
            + self.expert_pool
            + self.directional
            + self.atxy
        )

    @property
    def total(self) -> int:
        return self.non_embedding + self.embeddings + self.vocab_head

    def resident(self, resident_experts: int) -> int:
        return self.total - self.expert_pool + resident_experts * self.parameters_per_expert

    def per_token_active(self, active_experts: int) -> int:
        return self.total - self.expert_pool + active_experts * self.parameters_per_expert

    def nominal_bytes(self) -> dict[str, float]:
        counts = {
            "attention": self.attention,
            "norms": self.norms,
            "shared_channel": self.shared_channel,
            "router": self.router,
            "expert_pool": self.expert_pool,
            "directional": self.directional,
            "atxy": self.atxy,
            "embeddings": self.embeddings,
            "vocab_head": self.vocab_head,
        }
        return {name: count * self.nominal_bits.get(name, 16) / 8 for name, count in counts.items()}

    def summary(self) -> str:
        return (
            f"non_embedding={self.non_embedding:,} total={self.total:,} "
            f"(attention={self.attention:,} shared={self.shared_channel:,} "
            f"experts={self.expert_pool:,} embeddings={self.embeddings:,} "
            f"vocab_head={self.vocab_head:,} per_expert={self.parameters_per_expert:,})"
        )


def kda_parameters(d_model: int, num_heads: int, head_dim: int, conv_width: int = 4) -> int:
    """Exact parameter count of one :class:`KDATokenMixer` (head_v_dim == head_dim)."""

    key_dim = num_heads * head_dim
    return (
        3 * d_model * key_dim  # q, k, v projections
        + 3 * key_dim * conv_width  # short convolutions
        + (d_model * head_dim + head_dim * key_dim)  # low-rank forget gate
        + d_model * num_heads  # beta
        + num_heads  # A_log
        + num_heads * head_dim  # dt_bias
        + (d_model * head_dim + head_dim * key_dim + key_dim)  # output gate
        + key_dim * d_model  # output projection
    )


def mla_parameters(
    d_model: int, num_heads: int, head_dim: int, q_latent_dim: int, kv_latent_dim: int
) -> int:
    """Exact parameter count of one :class:`MLATokenMixer` (head_v_dim == head_dim)."""

    key_dim = num_heads * head_dim
    return (
        d_model * q_latent_dim
        + q_latent_dim  # q norm
        + q_latent_dim * key_dim
        + d_model * kv_latent_dim
        + kv_latent_dim  # kv norm
        + kv_latent_dim * num_heads * (head_dim + head_dim)
        + key_dim * d_model
    )


def estimate_parameters(
    shape: ArchitectureShape,
    vocab_size: int,
    *,
    directional: bool = False,
    atxy: ATXYConfig | None = None,
) -> ParameterEstimate:
    """Analytic per-component parameter counts of ``shape`` at ``vocab_size``."""

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    d = shape.d_model
    attention = shape.kda_layers * kda_parameters(
        d, shape.num_heads, shape.head_dim, shape.conv_width
    ) + shape.mla_layers * mla_parameters(
        d, shape.num_heads, shape.head_dim, shape.q_latent_dim, shape.kv_latent_dim
    )
    norms_per_layer = 3 * d if directional else 2 * d
    norms = shape.n_layers * norms_per_layer + d
    shared_channel = shape.n_layers * 3 * d * shape.shared_inter_dim
    router = d * shape.num_experts
    per_expert_per_layer = 3 * d * shape.expert_inter_dim
    parameters_per_expert = shape.n_layers * per_expert_per_layer
    expert_pool = shape.num_experts * parameters_per_expert
    directional_count = 0
    if directional:
        directional_config = DirectionalConfig(
            d_model=d,
            num_feature_groups=shape.directional_feature_groups,
            lateral_rank=shape.directional_lateral_rank,
            enabled=True,
        )
        directional_count = shape.n_layers * directional_config.parameters_per_layer
    atxy_count = 0 if atxy is None else atxy.parameter_count
    # PLE: one wide low-bit table per layer plus k lane logits and a gate.
    embeddings = shape.n_layers * (vocab_size * shape.ple_expansion_factor * d) + shape.n_layers * (
        shape.ple_expansion_factor + 1
    )
    vocab_head = vocab_size * d
    return ParameterEstimate(
        attention=attention,
        norms=norms,
        shared_channel=shared_channel,
        router=router,
        expert_pool=expert_pool,
        directional=directional_count,
        atxy=atxy_count,
        embeddings=embeddings,
        vocab_head=vocab_head,
        parameters_per_expert=parameters_per_expert,
        nominal_bits={"embeddings": shape.ple_quant_bits, "expert_pool": shape.expert_quant_bits},
    )


# Shapes sized on non-embedding parameters (see module docstring). Expert
# widths are deliberately small (ledger C07: many micro-experts); the shared
# channel path carries the general capacity; the resident cap follows the
# ledger's ~10% of the pool.
PARAMETER_CLASSES: dict[str, ArchitectureShape] = {
    "10M": ArchitectureShape(
        name="10M",
        d_model=256,
        n_layers=8,
        num_heads=4,
        head_dim=64,
        q_latent_dim=128,
        kv_latent_dim=64,
        num_experts=64,
        expert_inter_dim=12,
        shared_inter_dim=512,
        max_resident_experts=8,
    ),
    "50M": ArchitectureShape(
        name="50M",
        d_model=512,
        n_layers=12,
        num_heads=8,
        head_dim=64,
        q_latent_dim=256,
        kv_latent_dim=128,
        num_experts=128,
        expert_inter_dim=8,
        shared_inter_dim=1024,
        max_resident_experts=12,
    ),
    "100M": ArchitectureShape(
        name="100M",
        d_model=768,
        n_layers=16,
        num_heads=12,
        head_dim=64,
        q_latent_dim=384,
        kv_latent_dim=192,
        num_experts=200,
        expert_inter_dim=5,
        shared_inter_dim=768,
        max_resident_experts=20,
    ),
}


def shape_for(parameter_class: str, **overrides: Any) -> ArchitectureShape:
    """Look up a parameter-class shape, optionally overriding any knob."""

    try:
        shape = PARAMETER_CLASSES[parameter_class]
    except KeyError as error:
        raise KeyError(
            f"unknown parameter class {parameter_class!r}; choose from {sorted(PARAMETER_CLASSES)}"
        ) from error
    return shape.with_overrides(**overrides) if overrides else shape


@dataclass(frozen=True)
class ConfigBundle:
    """Every configuration object :func:`spalmer.factory.build_spalmer_model` takes."""

    model: SPALMERConfig
    kda: KDAConfig
    mla: MLAConfig
    experts: MicroExpertsConfig
    directional: DirectionalConfig | None = None
    atxy: ATXYConfig | None = None

    def build(self) -> SPALMERCausalLM:
        from spalmer.factory import build_spalmer_model

        return build_spalmer_model(
            self.model,
            self.kda,
            self.mla,
            self.experts,
            directional_config=self.directional,
            atxy_config=self.atxy,
        )


def build_configs(
    shape: ArchitectureShape,
    *,
    vocab_size: int,
    tokenizer_version: int,
    tokenizer_fingerprint: str,
    attention_backend: str = "auto",
    directional: bool = False,
    atxy: ATXYConfig | None = None,
    experts_overrides: dict[str, Any] | None = None,
    model_overrides: dict[str, Any] | None = None,
) -> ConfigBundle:
    """Turn a shape plus the tokenizer's identity into buildable configuration objects.

    ``vocab_size``, ``tokenizer_version`` and ``tokenizer_fingerprint`` come
    from the tokenizer artifact; nothing here assumes a fixed vocabulary.
    ``attention_backend`` is passed to the KDA mixer (``auto`` uses the
    optimized fla-core kernels on CUDA when installed and the reference path
    otherwise). ``experts_overrides`` / ``model_overrides`` patch individual
    fields of the expert and model configs (for example potentiation knobs).
    """

    model = SPALMERConfig(
        vocab_size=vocab_size,
        d_model=shape.d_model,
        n_layers=shape.n_layers,
        tokenizer_version=tokenizer_version,
        tokenizer_fingerprint=tokenizer_fingerprint,
        ple_expansion_factor=shape.ple_expansion_factor,
        ple_quant_bits=shape.ple_quant_bits,
        token_mixer_pattern=shape.token_mixer_pattern,
        **(model_overrides or {}),
    )
    kda = KDAConfig(
        hidden_size=shape.d_model,
        num_heads=shape.num_heads,
        head_k_dim=shape.head_dim,
        head_v_dim=shape.head_dim,
        conv_width=shape.conv_width,
        backend=attention_backend,
    )
    mla = MLAConfig(
        hidden_size=shape.d_model,
        num_heads=shape.num_heads,
        head_k_dim=shape.head_dim,
        head_v_dim=shape.head_dim,
        q_latent_dim=shape.q_latent_dim,
        kv_latent_dim=shape.kv_latent_dim,
    )
    expert_fields: dict[str, Any] = {
        "d_model": shape.d_model,
        "num_experts": shape.num_experts,
        "expert_inter_dim": shape.expert_inter_dim,
        "shared_inter_dim": shape.shared_inter_dim,
        "active_experts": shape.active_experts,
        "max_active_experts": min(shape.max_resident_experts, shape.num_experts),
        "min_resident_experts": shape.min_resident_experts,
        "max_resident_experts": shape.max_resident_experts,
        "expert_quant_bits": shape.expert_quant_bits,
    }
    expert_fields.update(experts_overrides or {})
    experts = MicroExpertsConfig(**expert_fields)
    directional_config = None
    if directional:
        directional_config = DirectionalConfig(
            d_model=shape.d_model,
            num_feature_groups=shape.directional_feature_groups,
            lateral_rank=shape.directional_lateral_rank,
            enabled=True,
        )
    if atxy is not None and atxy.d_model != shape.d_model:
        raise ValueError(
            f"ATXY d_model={atxy.d_model} does not match shape d_model={shape.d_model}"
        )
    return ConfigBundle(
        model=model,
        kda=kda,
        mla=mla,
        experts=experts,
        directional=directional_config,
        atxy=atxy,
    )


__all__ = [
    "HYBRID_CYCLE",
    "PARAMETER_CLASSES",
    "ArchitectureShape",
    "ConfigBundle",
    "ParameterEstimate",
    "build_configs",
    "estimate_parameters",
    "kda_parameters",
    "mla_parameters",
    "shape_for",
]
