"""QR-PLE-only planning, preset, and CLI surfaces."""

from __future__ import annotations

import json
import math

import pytest

from spalmer.experiment import (
    ExplicitVocabularyPolicy,
    ModelScaleConfig,
    ScaleSearchSpace,
    count_parameters,
    plan_configuration,
    plan_ladder,
)
from spalmer.experiment.planning import PLE_BACKENDS
from spalmer.experts.accounting import account_parameters
from spalmer.presets import assert_accounting_matches_plan, build_configs, plan_named_scale
from spalmer.qr import qr_codebook_rows, qr_lane_moduli


def _shape(**overrides: object) -> ModelScaleConfig:
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


def test_default_plan_is_qr_ple_with_dense_bf16_experts() -> None:
    shape = _shape()
    counts = count_parameters(shape)

    assert PLE_BACKENDS == ("qr",)
    assert shape.ple_backend == "qr"
    assert shape.expert_weight_format == "bfloat16"
    assert shape.expert_activation_format == "bfloat16"
    assert counts.ple_exact_input > 0
    assert counts.ple_qr_refresh > 0
    assert counts.low_bit_candidates == 0
    assert shape.ple_lane_moduli
    assert shape.to_dict()["ple_backend"] == "qr"
    assert "ple" in plan_configuration(2_000_000, 512).to_dict()


def test_qr_counts_exact_input_and_refresh_lanes() -> None:
    qr = _shape(ple_backend="qr")
    counts = count_parameters(qr)
    moduli = qr_lane_moduli(1_024, 2)
    remainder_rows, quotient_rows = qr_codebook_rows(1_024, 2)
    per_layer = sum((modulus + math.ceil(1_024 / modulus)) * 128 for modulus in moduli)

    assert qr.ple_lane_moduli == moduli
    assert len(set(moduli)) == 2
    assert all(math.gcd(left, right) == 1 for left in moduli for right in moduli if left != right)
    assert per_layer == (remainder_rows + quotient_rows) * 128
    assert counts.ple_exact_input == 1_024 * 128
    assert counts.ple_qr_refresh == 7 * per_layer
    assert counts.ple_lookup == counts.ple_exact_input + counts.ple_qr_refresh
    assert counts.ple_controls == 7 * (2 + 1)
    assert counts.ple_total == counts.ple_lookup + counts.ple_controls
    assert counts.low_bit_candidates == 0
    assert counts.dense_parameters == counts.total
    assert sum(counts.components.values()) == counts.total

    payload = counts.to_dict()
    assert payload["ple_exact_input"] == counts.ple_exact_input
    assert payload["ple_qr_refresh"] == counts.ple_qr_refresh
    assert payload["low_bit_candidates"] == 0
    assert qr.to_dict()["ple_backend"] == "qr"


def test_invalid_ple_backend_fails_closed() -> None:
    with pytest.raises(ValueError, match="ple_backend"):
        _shape(ple_backend="int4")
    with pytest.raises(ValueError, match="ple_backend"):
        plan_configuration(1_000_000, 300, ple_backend="int4")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fake-QAT PLE is retired"):
        _shape(ple_backend="fake_qat")
    with pytest.raises(ValueError, match="fake-QAT PLE is retired"):
        plan_configuration(1_000_000, 300, ple_backend="fake_qat")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "target, vocab_size",
    [(10_000_000, 8_192), (50_000_000, 8_192), (100_000_000, 32_768)],
)
def test_qr_plan_lands_near_target(target: int, vocab_size: int) -> None:
    qr = plan_configuration(target, vocab_size)

    assert qr.config.ple_backend == "qr"
    assert qr.parameters == count_parameters(qr.config)
    assert abs(qr.relative_error) <= 0.01

    block = qr.to_dict()["ple"]
    assert block["backend"] == "qr"
    assert block["lanes"] == qr.config.ple_expansion
    assert block["moduli"] == list(qr_lane_moduli(vocab_size, qr.config.ple_expansion))
    assert block["exact_input_parameters"] == vocab_size * qr.config.d_model
    assert block["qr_refresh_parameters"] == qr.parameters.ple_qr_refresh
    assert block["lookup_parameters"] == qr.parameters.ple_lookup
    assert block["control_parameters"] == qr.parameters.ple_controls
    assert block["total_parameters"] == qr.parameters.ple_total
    assert qr.to_dict()["config"]["ple_backend"] == "qr"


def test_plan_ladder_threads_ple_backend_to_every_rung() -> None:
    policy = ExplicitVocabularyPolicy(((1_000_000, 300), (2_000_000, 512)))
    plans = plan_ladder((1_000_000, 2_000_000), policy, ple_backend="qr")

    assert [plan.config.ple_backend for plan in plans] == ["qr", "qr"]
    assert all("ple" in plan.to_dict() for plan in plans)
    legacy = plan_ladder((1_000_000, 2_000_000), policy)
    assert all(plan.config.ple_backend == "qr" for plan in legacy)


def test_build_configs_passes_qr_backend_and_protects_it() -> None:
    qr_plan = plan_configuration(
        1_000_000, 300, search_space=_one_shape_search(), ple_backend="qr"
    )
    bundle = build_configs(qr_plan, tokenizer_version=1, tokenizer_fingerprint="presets")
    assert bundle.model.ple_backend == "qr"
    assert bundle.model.ple_expansion_factor == qr_plan.config.ple_expansion

    default = build_configs(
        plan_configuration(1_000_000, 300, search_space=_one_shape_search()),
        tokenizer_version=1,
        tokenizer_fingerprint="presets",
    )
    assert default.model.ple_backend == "qr"

    with pytest.raises(ValueError, match="ScalePlan fields"):
        build_configs(
            qr_plan,
            tokenizer_version=1,
            tokenizer_fingerprint="presets",
            model_overrides={"ple_backend": "fake_qat"},
        )
    assert plan_named_scale("10M", vocab_size=8_192, ple_backend="qr").config.ple_backend == "qr"


def test_qr_accounting_matches_the_plan_component_by_component() -> None:
    plan = plan_configuration(
        1_000_000, 300, search_space=_one_shape_search(), ple_backend="qr"
    )
    bundle = build_configs(
        plan,
        tokenizer_version=1,
        tokenizer_fingerprint="presets",
        attention_backend="reference",
    )
    model = bundle.build()

    accounting = account_parameters(model)
    assert accounting.components["embeddings"] == plan.parameters.embeddings
    assert_accounting_matches_plan(plan, accounting)
    # BF16 codebooks execute at their storage width: no low-bit forward lane.
    assert "embeddings" not in accounting.execution_bits


def test_plan_cli_reports_the_ple_backend(capsys: pytest.CaptureFixture[str]) -> None:
    from spalmer.__main__ import _plan_parser, _run_plan

    args = _plan_parser().parse_args(
        ["1m", "--vocab-size", "300", "--ple-backend", "qr", "--json"]
    )
    _run_plan(args)
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["config"]["ple_backend"] == "qr"
    assert payload[0]["ple"]["backend"] == "qr"
    assert payload[0]["ple"]["moduli"] == list(qr_lane_moduli(300, payload[0]["ple"]["lanes"]))

    _run_plan(_plan_parser().parse_args(["1m", "--vocab-size", "300", "--ple-backend", "qr"]))
    table = capsys.readouterr().out
    assert "backend" in table
    assert "qr" in table

    _run_plan(_plan_parser().parse_args(["1m", "--vocab-size", "300", "--json"]))
    default = json.loads(capsys.readouterr().out)
    assert default[0]["config"]["ple_backend"] == "qr"
    assert default[0]["ple"]["backend"] == "qr"
