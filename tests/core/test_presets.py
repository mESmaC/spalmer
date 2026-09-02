"""Architecture shape hooks: analytic accounting must match built models (no training)."""

from __future__ import annotations

import pytest
import torch

from spalmer.experts import account_parameters
from spalmer.memory import ATXYConfig
from spalmer.presets import (
    PARAMETER_CLASSES,
    ArchitectureShape,
    build_configs,
    estimate_parameters,
    shape_for,
)

TINY = ArchitectureShape(
    name="tiny",
    d_model=32,
    n_layers=4,
    num_heads=2,
    head_dim=16,
    q_latent_dim=16,
    kv_latent_dim=8,
    num_experts=8,
    expert_inter_dim=4,
    shared_inter_dim=16,
    max_resident_experts=4,
)


@pytest.mark.parametrize("vocab_size", [37, 300])
@pytest.mark.parametrize("directional", [False, True])
@pytest.mark.parametrize("with_atxy", [False, True])
def test_analytic_estimate_matches_measured_parameters(vocab_size, directional, with_atxy) -> None:
    atxy = (
        ATXYConfig(
            d_model=32,
            value_dim=6,
            a_cardinality=3,
            t_cardinality=4,
            x_cardinality=5,
            y_cardinality=6,
            injection_layer=1,
        )
        if with_atxy
        else None
    )
    bundle = build_configs(
        TINY,
        vocab_size=vocab_size,
        tokenizer_version=1,
        tokenizer_fingerprint="presets",
        attention_backend="reference",
        directional=directional,
        atxy=atxy,
    )
    torch.manual_seed(0)
    model = bundle.build()
    measured = account_parameters(model)
    estimate = estimate_parameters(TINY, vocab_size, directional=directional, atxy=atxy)
    for name in (
        "attention",
        "norms",
        "shared_channel",
        "router",
        "expert_pool",
        "directional",
        "atxy",
        "embeddings",
        "vocab_head",
    ):
        assert measured.components[name] == getattr(estimate, name), name
    assert measured.total == estimate.total
    assert measured.parameters_per_expert == estimate.parameters_per_expert
    assert estimate.resident(TINY.min_resident_experts) == (
        measured.total - measured.expert_pool + 2 * measured.parameters_per_expert
    )
    output = model(torch.randint(0, vocab_size, (1, 5)))
    assert output.logits.shape == (1, 5, vocab_size)


@pytest.mark.parametrize(
    "name,target", [("10M", 10_000_000), ("50M", 50_000_000), ("100M", 100_000_000)]
)
def test_parameter_classes_land_near_their_non_embedding_targets(name, target) -> None:
    shape = shape_for(name)
    estimate = estimate_parameters(shape, vocab_size=4096)
    assert abs(estimate.non_embedding - target) / target < 0.06
    assert estimate.non_embedding == estimate_parameters(shape, vocab_size=65536).non_embedding
    # The ledger's residency numbers: start at two, cap at roughly 10%.
    assert shape.min_resident_experts == 2
    assert shape.max_resident_experts <= max(2, round(0.125 * shape.num_experts))
    assert (
        3 * shape.d_model * shape.expert_inter_dim * shape.n_layers
        == estimate.parameters_per_expert
    )


def test_vocabulary_scales_only_the_embedding_and_head_components() -> None:
    shape = shape_for("10M")
    small = estimate_parameters(shape, vocab_size=4096)
    large = estimate_parameters(shape, vocab_size=32768)
    assert large.non_embedding == small.non_embedding
    assert large.vocab_head == 8 * small.vocab_head
    assert large.embeddings - small.embeddings == shape.n_layers * (32768 - 4096) * shape.d_model
    bytes_small = small.nominal_bytes()
    assert bytes_small["embeddings"] == small.embeddings * shape.ple_quant_bits / 8
    assert bytes_small["attention"] == small.attention * 2


def test_shape_overrides_and_bundle_contents() -> None:
    shape = shape_for("50M", num_experts=200, expert_inter_dim=6)
    assert shape.num_experts == 200 and shape.name == "50M"
    with pytest.raises(KeyError, match="unknown parameter class"):
        shape_for("7B")
    with pytest.raises(ValueError, match="multiple"):
        shape.with_overrides(n_layers=6)
    bundle = build_configs(
        shape,
        vocab_size=12345,
        tokenizer_version=3,
        tokenizer_fingerprint="abc",
        experts_overrides={"potentiation_budget": 0},
        model_overrides={"surprise_ema_decay": 0.9},
    )
    assert bundle.model.vocab_size == 12345
    assert bundle.model.tokenizer_version == 3
    assert bundle.model.surprise_ema_decay == 0.9
    assert bundle.kda.backend == "auto"
    assert bundle.mla.kv_latent_dim == shape.kv_latent_dim
    assert bundle.experts.num_experts == 200
    assert bundle.experts.potentiation_budget == 0
    assert bundle.experts.max_resident_experts == shape.max_resident_experts
    assert bundle.directional is None and bundle.atxy is None
    assert bundle.model.token_mixer_pattern == ("kda", "kda", "kda", "mla")
    assert set(PARAMETER_CLASSES) == {"10M", "50M", "100M"}
