from __future__ import annotations

import math

import pytest

from spalmer.experiment import (
    ATXYScaleConfig,
    DirectionalScaleConfig,
    ExplicitVocabularyPolicy,
    MemoryAssumptions,
    ModelScaleConfig,
    RecurrenceScaleConfig,
    ScaleSearchSpace,
    count_parameters,
    estimate_memory,
    plan_configuration,
    plan_ladder,
)
from spalmer.qr import qr_codebook_rows, qr_lane_moduli


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

    assert breakdown.total == 8_059_373
    assert breakdown.ple_lookup == 248_448
    assert breakdown.shared_channel == 786_432
    assert breakdown.expert_banks == 6_291_456
    assert breakdown.low_bit_candidates == 0
    assert sum(breakdown.components.values()) == breakdown.total
    assert breakdown.components["shared_channel"] == breakdown.shared_channel


def test_qr_ple_breakdown_uses_exact_input_and_sqrt_vocab_refreshes() -> None:
    config = ModelScaleConfig(
        vocab_size=1_024,
        d_model=128,
        n_layers=8,
        num_heads=4,
        num_experts=32,
        expert_width=64,
        ple_expansion=2,
        ple_backend="qr",
    )
    breakdown = count_parameters(config)
    remainder_rows, quotient_rows = qr_codebook_rows(1_024, 2)

    assert config.ple_lane_moduli == qr_lane_moduli(1_024, 2) == (37, 41)
    assert breakdown.ple_exact_input == 1_024 * 128
    assert breakdown.ple_qr_refresh == 7 * (remainder_rows + quotient_rows) * 128
    assert breakdown.ple_lookup == breakdown.ple_exact_input + breakdown.ple_qr_refresh
    assert breakdown.ple_controls == 7 * 3
    assert breakdown.low_bit_candidates == 0
    assert breakdown.total == sum(breakdown.components.values())
    assert config.to_dict()["ple_backend"] == "qr"

    plan = plan_configuration(10_000_000, 1_024, ple_backend="qr")
    payload = plan.to_dict()
    assert payload["ple"]["backend"] == "qr"
    assert payload["ple"]["moduli"] == list(plan.config.ple_lane_moduli)
    assert payload["ple"]["exact_input_parameters"] == plan.parameters.ple_exact_input
    assert payload["ple"]["qr_refresh_parameters"] == plan.parameters.ple_qr_refresh


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
    assert plan.parameters.total == 8_059_373


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


def _breakdown_shape(**overrides: object) -> ModelScaleConfig:
    values: dict[str, object] = {
        "vocab_size": 1_024,
        "d_model": 128,
        "n_layers": 8,
        "num_heads": 4,
        "num_experts": 32,
        "expert_width": 64,
        "ple_expansion": 2,
    }
    values.update(overrides)
    return ModelScaleConfig(**values)  # type: ignore[arg-type]


def test_recurrence_adds_adapter_and_keeps_fingerprints() -> None:
    flat = _breakdown_shape()
    flat_counts = count_parameters(flat)
    assert flat_counts.total == 8_059_373
    assert flat_counts.recurrence == 0
    assert "recurrence" not in flat_counts.to_dict()
    assert "recurrence" not in flat.to_dict()
    assert flat_counts.components["recurrence"] == 0
    assert flat.effective_depth() == 8
    assert flat.core_layers == 0

    recurrent = _breakdown_shape(
        recurrence=RecurrenceScaleConfig(prelude_layers=1, coda_layers=1, default_steps=4)
    )
    counts = count_parameters(recurrent)

    adapter = 2 * 128 * 128 + 2 * 128
    assert counts.recurrence == adapter
    assert counts.total == flat_counts.total + adapter
    assert counts.to_dict()["recurrence"] == adapter
    assert sum(counts.components.values()) == counts.total
    assert recurrent.core_layers == 6
    assert recurrent.effective_depth() == 1 + 4 * 6 + 1
    assert recurrent.effective_depth(2) == 1 + 2 * 6 + 1
    assert recurrent.to_dict()["recurrence"] == {
        "prelude_layers": 1,
        "coda_layers": 1,
        "default_steps": 4,
        "latent_init_std": 1.0,
    }


def test_recurrent_scale_plan_publishes_a_recurrence_block() -> None:
    flat = plan_configuration(2_000_000, 512)
    assert "recurrence" not in flat.to_dict()
    assert flat.effective_depth == flat.config.n_layers
    assert flat.block_passes_per_token == flat.config.n_layers

    plan = plan_configuration(
        2_000_000,
        512,
        recurrence=RecurrenceScaleConfig(prelude_layers=1, coda_layers=1, default_steps=3),
    )
    block = plan.to_dict()["recurrence"]

    assert block["prelude_layers"] == 1
    assert block["coda_layers"] == 1
    assert block["core_layers"] == plan.config.n_layers - 2
    assert block["default_steps"] == 3
    assert block["effective_depth"] == plan.effective_depth
    assert sum(block["core_mixers"].values()) == block["core_layers"]
    assert plan.fingerprint != flat.fingerprint


def test_recurrent_shapes_reject_impossible_cores() -> None:
    with pytest.raises(ValueError, match="at least one core layer"):
        _breakdown_shape(
            n_layers=4,
            recurrence=RecurrenceScaleConfig(prelude_layers=2, coda_layers=2),
        )
    with pytest.raises(ValueError, match="outside the recurrent core"):
        _breakdown_shape(
            atxy=ATXYScaleConfig(
                value_dim=16,
                a_cardinality=2,
                t_cardinality=2,
                x_cardinality=2,
                y_cardinality=2,
                injection_layer=3,
            ),
            recurrence=RecurrenceScaleConfig(prelude_layers=1, coda_layers=1),
        )
    with pytest.raises(ValueError, match="latent_init_std"):
        RecurrenceScaleConfig(latent_init_std=-1.0)
    with pytest.raises(ValueError, match="prelude_layers must be positive"):
        RecurrenceScaleConfig(prelude_layers=0)


def test_planner_skips_grid_points_that_cannot_hold_a_core() -> None:
    space = ScaleSearchSpace(d_models=(64,), n_layers=(4, 8), num_heads=(4,))
    plan = plan_configuration(
        50_000_000,
        1_024,
        search_space=space,
        recurrence=RecurrenceScaleConfig(prelude_layers=3, coda_layers=2),
    )

    assert plan.config.n_layers == 8

    with pytest.raises(ValueError, match="no valid"):
        plan_configuration(
            50_000_000,
            1_024,
            search_space=ScaleSearchSpace(d_models=(64,), n_layers=(4,), num_heads=(4,)),
            recurrence=RecurrenceScaleConfig(prelude_layers=3, coda_layers=2),
        )


def test_plan_ladder_threads_recurrence_to_every_rung() -> None:
    policy = ExplicitVocabularyPolicy(((2_000_000, 512), (4_000_000, 1_024)))
    plans = plan_ladder(
        (2_000_000, 4_000_000),
        policy,
        recurrence=RecurrenceScaleConfig(prelude_layers=1, coda_layers=1, default_steps=2),
    )

    assert all(plan.config.recurrence is not None for plan in plans)
    assert all("recurrence" in plan.to_dict() for plan in plans)
