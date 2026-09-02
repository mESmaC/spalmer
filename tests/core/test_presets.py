"""Static checks for the ScalePlan-to-configuration construction boundary."""

from __future__ import annotations

import pytest

from spalmer.experiment import (
    ATXYScaleConfig,
    DirectionalScaleConfig,
    ScaleSearchSpace,
    plan_configuration,
)
from spalmer.experts.accounting import ParameterAccounting
from spalmer.presets import (
    TOTAL_PARAMETER_TARGETS,
    assert_accounting_matches_plan,
    build_configs,
    plan_named_scale,
)


def _one_shape_search() -> ScaleSearchSpace:
    return ScaleSearchSpace(
        d_models=(32,),
        n_layers=(4,),
        num_heads=(2,),
        num_experts=8,
        expert_width_divisors=(8,),
        shared_width_multipliers=(1,),
        ple_expansions=(2,),
    )


def _tiny_plan(*, extensions: bool = False, expert_weight_format: str = "mxfp4"):
    directional = (
        DirectionalScaleConfig(num_feature_groups=4, lateral_rank=4)
        if extensions
        else None
    )
    atxy = (
        ATXYScaleConfig(
            value_dim=6,
            a_cardinality=3,
            t_cardinality=4,
            x_cardinality=5,
            y_cardinality=6,
            injection_layer=1,
        )
        if extensions
        else None
    )
    return plan_configuration(
        1_000_000,
        300,
        search_space=_one_shape_search(),
        expert_weight_format=expert_weight_format,
        directional=directional,
        atxy=atxy,
    )


@pytest.mark.parametrize("extensions", [False, True])
@pytest.mark.parametrize("expert_weight_format", ["mxfp4", "nvfp4"])
def test_scale_plan_builds_matching_static_configs(extensions, expert_weight_format) -> None:
    plan = _tiny_plan(
        extensions=extensions,
        expert_weight_format=expert_weight_format,
    )
    bundle = build_configs(
        plan,
        tokenizer_version=3,
        tokenizer_fingerprint="presets",
        attention_backend="reference",
        experts_overrides={"potentiation_budget": 0},
        model_overrides={"surprise_ema_decay": 0.9},
    )
    config = plan.config

    assert bundle.plan is plan
    assert bundle.model.vocab_size == config.vocab_size
    assert bundle.model.d_model == config.d_model
    assert bundle.model.n_layers == config.n_layers
    assert bundle.model.ple_expansion_factor == config.ple_expansion
    assert bundle.model.surprise_ema_decay == 0.9
    assert bundle.kda.hidden_size == config.d_model
    assert bundle.kda.num_heads == config.num_heads
    assert bundle.mla.q_latent_dim == config.resolved_q_latent_dim
    assert bundle.mla.kv_latent_dim == config.resolved_kv_latent_dim
    assert bundle.experts.num_experts == config.num_experts
    assert bundle.experts.resolved_inter_dim == config.expert_width
    assert bundle.experts.resolved_shared_inter_dim == config.resolved_shared_width
    assert bundle.experts.expert_weight_format == expert_weight_format
    assert bundle.experts.expert_activation_format == "mxfp8"
    assert bundle.experts.expert_master_dtype == "bfloat16"
    assert bundle.experts.potentiation_budget == 0

    assert plan.parameters.shared_channel == (
        config.n_layers * bundle.experts.shared_parameters_per_layer
    )
    assert plan.parameters.expert_pool == (
        config.n_layers * bundle.experts.expert_pool_parameters_per_layer
    )
    assert plan.parameters.router == config.d_model * config.num_experts
    assert (bundle.directional is not None) is extensions
    assert (bundle.atxy is not None) is extensions
    if extensions:
        assert plan.parameters.directional == (
            config.n_layers * bundle.directional.parameters_per_layer
        )
        assert plan.parameters.atxy == bundle.atxy.parameter_count
    else:
        assert plan.parameters.directional == 0
        assert plan.parameters.atxy == 0


def test_measured_accounting_contract_uses_the_same_component_ownership() -> None:
    plan = _tiny_plan(extensions=True)
    per_expert = (
        plan.config.n_layers
        * 3
        * plan.config.d_model
        * plan.config.expert_width
    )
    accounting = ParameterAccounting(
        components=dict(plan.parameters.components),
        num_experts=plan.config.num_experts,
        resident_expert_ids=tuple(range(plan.config.num_experts)),
        active_experts_per_token=plan.config.active_experts,
        parameters_per_expert=per_expert,
    )

    assert_accounting_matches_plan(plan, accounting)
    wrong_components = dict(accounting.components)
    wrong_components["shared_channel"] += 1
    wrong = ParameterAccounting(
        components=wrong_components,
        num_experts=accounting.num_experts,
        resident_expert_ids=accounting.resident_expert_ids,
        active_experts_per_token=accounting.active_experts_per_token,
        parameters_per_expert=accounting.parameters_per_expert,
    )
    with pytest.raises(ValueError, match="shared_channel"):
        assert_accounting_matches_plan(plan, wrong)


@pytest.mark.parametrize("name", ["10M", "50M", "100M"])
def test_named_scales_are_total_targets_with_scale_owned_vocab(name) -> None:
    plan = plan_named_scale(name, vocab_size=4_096)

    assert plan.target_parameters == TOTAL_PARAMETER_TARGETS[name]
    assert plan.config.vocab_size == 4_096
    assert plan.config.num_experts == 200
    assert plan.config.ple_expansion in {2, 4}
    assert plan.parameters.total == sum(plan.parameters.components.values())
    assert abs(plan.relative_error) < 0.01


def test_plan_fields_cannot_be_silently_overridden() -> None:
    plan = _tiny_plan()

    with pytest.raises(ValueError, match="ScalePlan fields"):
        build_configs(
            plan,
            tokenizer_version=1,
            tokenizer_fingerprint="presets",
            model_overrides={"d_model": 64},
        )
    with pytest.raises(ValueError, match="ScalePlan fields"):
        build_configs(
            plan,
            tokenizer_version=1,
            tokenizer_fingerprint="presets",
            experts_overrides={"num_experts": 200},
        )
    with pytest.raises(ValueError, match="ScalePlan fields"):
        build_configs(
            plan,
            tokenizer_version=1,
            tokenizer_fingerprint="presets",
            experts_overrides={"expert_fake_quantization": False},
        )
    with pytest.raises(KeyError, match="unknown scale"):
        plan_named_scale("7B", vocab_size=32_000)


def test_named_scale_target_table_contains_only_total_budget_labels() -> None:
    assert TOTAL_PARAMETER_TARGETS == {
        "10M": 10_000_000,
        "50M": 50_000_000,
        "100M": 100_000_000,
    }
