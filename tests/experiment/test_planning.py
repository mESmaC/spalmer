from __future__ import annotations

import math

import pytest

from spalmer.experiment import (
    ATXYScaleConfig,
    DirectionalScaleConfig,
    ExplicitVocabularyPolicy,
    MemoryAssumptions,
    ModelScaleConfig,
    ScaleSearchSpace,
    count_parameters,
    estimate_memory,
    plan_configuration,
    plan_ladder,
)


def test_parameter_breakdown_matches_known_core_shape() -> None:
    config = ModelScaleConfig(
        vocab_size=1_024,
        d_model=128,
        n_layers=8,
        num_heads=4,
        num_experts=32,
        expert_width=64,
        ple_expansion=2,
    )
    breakdown = count_parameters(config)

    assert breakdown.total == 9_908_080
    assert breakdown.ple_lookup == 2_097_152
    assert breakdown.shared_channel == 786_432
    assert breakdown.expert_banks == 6_291_456
    assert breakdown.low_bit_candidates == breakdown.ple_lookup + breakdown.expert_banks
    assert sum(breakdown.components.values()) == breakdown.total
    assert breakdown.components["shared_channel"] == breakdown.shared_channel


def test_optional_directional_and_atxy_capacity_is_inside_total() -> None:
    base = ModelScaleConfig(
        vocab_size=1_024,
        d_model=128,
        n_layers=8,
        num_heads=4,
        num_experts=32,
        expert_width=64,
        ple_expansion=2,
    )
    extended = ModelScaleConfig(
        vocab_size=1_024,
        d_model=128,
        n_layers=8,
        num_heads=4,
        num_experts=32,
        expert_width=64,
        ple_expansion=2,
        directional=DirectionalScaleConfig(num_feature_groups=4, lateral_rank=8),
        atxy=ATXYScaleConfig(
            value_dim=16,
            a_cardinality=2,
            t_cardinality=3,
            x_cardinality=4,
            y_cardinality=5,
            injection_layer=3,
        ),
    )

    base_count = count_parameters(base)
    extended_count = count_parameters(extended)
    assert extended_count.directional == 1_040
    assert extended_count.atxy == 20_225
    assert extended_count.block_norms - base_count.block_norms == 8 * 128
    assert extended_count.total - base_count.total == 1_040 + 20_225 + 8 * 128


def test_memory_estimate_names_bf16_master_and_cached_packed_footprints_separately() -> None:
    config = ModelScaleConfig(
        vocab_size=512,
        d_model=64,
        n_layers=4,
        num_heads=4,
        num_experts=16,
        expert_width=32,
        ple_expansion=2,
    )
    breakdown = count_parameters(config)
    assumptions = MemoryAssumptions(overhead_fraction=0)
    memory = estimate_memory(breakdown, assumptions)

    assert memory.reference_weights_bytes == breakdown.total * 2
    assert memory.reference_gradients_bytes == breakdown.total * 2
    assert memory.hypothetical_packed_weights_bytes == (
        breakdown.dense_parameters * 2
        + math.ceil(breakdown.low_bit_candidates * 4.5 / 8)
    )
    assert memory.reference_training_bytes == (
        memory.reference_weights_bytes
        + memory.reference_gradients_bytes
        + memory.optimizer_bytes
    )
    assert memory.hypothetical_master_weights_bytes == breakdown.total * 2
    assert memory.hypothetical_packed_training_bytes == (
        memory.hypothetical_packed_weights_bytes
        + memory.reference_gradients_bytes
        + memory.optimizer_bytes
        + memory.hypothetical_master_weights_bytes
    )
    assert memory.hypothetical_packed_training_bytes > memory.reference_training_bytes
    assert memory.to_dict()["reference_training_gib"] > 0
    assert memory.to_dict()["hypothetical_packed_training_gib"] > 0


def test_ladder_resolves_vocabulary_per_target_and_is_deterministic() -> None:
    targets = (10_000_000, 50_000_000, 100_000_000)
    policy = ExplicitVocabularyPolicy(
        ((10_000_000, 1_024), (50_000_000, 2_048), (100_000_000, 8_192))
    )

    first = plan_ladder(targets, policy)
    second = plan_ladder(targets, policy)

    assert [plan.config.vocab_size for plan in first] == [1_024, 2_048, 8_192]
    assert {plan.config.num_experts for plan in first} == {200}
    assert [plan.fingerprint for plan in first] == [plan.fingerprint for plan in second]
    assert all(abs(plan.relative_error) < 0.01 for plan in first)


def test_default_checkpoint_ladder_keeps_200_experts_with_small_widths() -> None:
    targets = (10_000_000, 50_000_000, 100_000_000)
    policy = ExplicitVocabularyPolicy(
        ((10_000_000, 4_096), (50_000_000, 8_192), (100_000_000, 16_384))
    )

    plans = plan_ladder(targets, policy)

    assert [plan.config.vocab_size for plan in plans] == [4_096, 8_192, 16_384]
    assert [plan.config.num_experts for plan in plans] == [200, 200, 200]
    assert all(plan.config.expert_width > 0 for plan in plans)
    assert all(plan.config.resolved_shared_width > 0 for plan in plans)
    assert {plan.config.ple_expansion for plan in plans}.issubset({2, 4})
    assert all(abs(plan.relative_error) < 0.01 for plan in plans)


def test_search_space_constraints_are_honored() -> None:
    search = ScaleSearchSpace(
        d_models=(128,),
        n_layers=(8,),
        num_heads=(4,),
        num_experts=32,
        expert_width_divisors=(2,),
        shared_width_multipliers=(2,),
        ple_expansions=(2,),
    )
    plan = plan_configuration(10_000_000, 1_024, search_space=search)

    assert plan.config == ModelScaleConfig(
        vocab_size=1_024,
        d_model=128,
        n_layers=8,
        num_heads=4,
        num_experts=32,
        expert_width=64,
        shared_width=256,
        ple_expansion=2,
    )
    assert plan.parameters.total == 9_908_080


@pytest.mark.parametrize(
    "config, message",
    [
        (
            dict(
                vocab_size=1_024,
                d_model=127,
                n_layers=8,
                num_heads=4,
                num_experts=32,
                expert_width=64,
                ple_expansion=2,
            ),
            "divisible",
        ),
        (
            dict(
                vocab_size=1_024,
                d_model=128,
                n_layers=6,
                num_heads=4,
                num_experts=32,
                expert_width=64,
                ple_expansion=2,
            ),
            "multiple of 4",
        ),
        (
            dict(
                vocab_size=1_024,
                d_model=128,
                n_layers=8,
                num_heads=4,
                num_experts=32,
                expert_width=64,
                ple_expansion=2,
                active_experts=1,
                min_resident_experts=1,
            ),
            "two-expert floor",
        ),
    ],
)
def test_invalid_model_scale_config_fails_closed(config: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ModelScaleConfig(**config)


def test_explicit_vocabulary_policy_has_no_implicit_fallback() -> None:
    with pytest.raises(ValueError, match="sorted"):
        ExplicitVocabularyPolicy(((50_000_000, 2_048), (10_000_000, 1_024)))

    policy = ExplicitVocabularyPolicy(((10_000_000, 1_024),))
    with pytest.raises(KeyError, match="50,000,000"):
        policy.vocab_size_for(50_000_000)
