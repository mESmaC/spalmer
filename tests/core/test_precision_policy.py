from __future__ import annotations

import json
import sys
import types

import pytest

import spalmer.precision as precision_module
from spalmer.__main__ import _plan_parser, _precision_parser, _run_precision, _training_parser
from spalmer.config import PLEConfig, SPALMERConfig
from spalmer.experts import MicroExpertsConfig
from spalmer.precision import (
    ExpertPrecisionCapability,
    detect_precision_capabilities,
)


def test_new_configs_default_to_qr_ple_and_real_bf16_experts() -> None:
    model = SPALMERConfig(
        vocab_size=128,
        d_model=16,
        n_layers=4,
        tokenizer_version=1,
        tokenizer_fingerprint="precision-policy",
    )
    experts = MicroExpertsConfig(
        d_model=16,
        num_experts=4,
        max_active_experts=4,
        max_resident_experts=4,
    )

    assert model.ple_backend == "qr"
    assert model.ple_config().backend == "qr"
    assert experts.expert_weight_format == "bfloat16"
    assert experts.expert_activation_format == "bfloat16"
    assert experts.expert_promotion_format == "bfloat16"
    assert experts.potentiation_budget == 0


def test_new_configs_reject_retired_emulation_paths() -> None:
    with pytest.raises(ValueError, match="fake-QAT PLE is retired"):
        PLEConfig(vocab_size=32, d_model=8, n_layers=4, backend="fake_qat")
    with pytest.raises(ValueError, match="fake-QAT PLE is retired"):
        SPALMERConfig(
            vocab_size=32,
            d_model=8,
            n_layers=4,
            tokenizer_version=1,
            tokenizer_fingerprint="retired",
            ple_backend="fake_qat",
        )
    for override, message in (
        ({"expert_qat_backend": "reference"}, "expert_qat_backend"),
        (
            {
                "expert_weight_format": "legacy_int",
                "expert_activation_format": "bfloat16",
                "expert_master_dtype": "float32",
            },
            "expert_weight_format",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            MicroExpertsConfig(
                d_model=8,
                num_experts=4,
                max_active_experts=4,
                max_resident_experts=4,
                **override,
            )

    for retired_field in (
        "expert_quant_bits",
        "expert_fake_quantization",
        "expert_stochastic_rounding",
    ):
        with pytest.raises(TypeError, match=retired_field):
            MicroExpertsConfig(
                d_model=8,
                num_experts=4,
                max_active_experts=4,
                max_resident_experts=4,
                **{retired_field: False},
            )


def test_fake_ple_has_no_private_construction_bypass() -> None:
    with pytest.raises(TypeError, match="_allow_legacy_fake_qat"):
        PLEConfig(  # type: ignore[call-arg]
            vocab_size=32,
            d_model=8,
            n_layers=4,
            _allow_legacy_fake_qat=True,
        )


def test_expert_emulation_has_no_private_construction_bypass() -> None:
    with pytest.raises(TypeError, match="_allow_legacy_emulation"):
        MicroExpertsConfig(  # type: ignore[call-arg]
            d_model=8,
            num_experts=4,
            max_active_experts=4,
            max_resident_experts=4,
            _allow_legacy_emulation=True,
        )


def test_capability_report_exposes_only_real_selectable_pairs_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        precision_module.torch.cpu,
        "_is_avx512_bf16_supported",
        lambda: False,
        raising=False,
    )
    report = detect_precision_capabilities("cpu")

    assert not report.supports("bfloat16", "bfloat16")
    assert report.supports("float32", "float32")
    assert not report.supports("mxfp4", "mxfp8")
    assert report.selectable_weight_formats == ("float32",)
    fp32 = report.require("float32", "float32")
    assert fp32.provider_id == "fp32_torch_cpu"
    assert fp32.selectable
    assert report.to_dict()["selectable_pairs"] == [
        {"weight_format": "float32", "activation_format": "float32"}
    ]


def test_native_hook_must_verify_forward_and_backward(monkeypatch: pytest.MonkeyPatch) -> None:
    native = types.ModuleType("spalmer.experts.native")

    def capabilities(*, device):
        assert str(device) == "cpu"
        return [
            {
                "provider_id": "torchao_nvfp4_dense",
                "weight_format": "nvfp4",
                "activation_format": "nvfp4",
                "forward_available": True,
                "backward_available": True,
                "grouped_available": False,
                "verified": True,
                "detail": "test provider",
            },
            {
                "provider_id": "cutlass_mxfp6_dense",
                "weight_format": "mxfp6",
                "activation_format": "mxfp8",
                "forward_available": True,
                "backward_available": False,
                "grouped_available": False,
                "verified": True,
                "detail": "forward only",
            },
        ]

    native.native_provider_capabilities = capabilities
    monkeypatch.setitem(sys.modules, "spalmer.experts.native", native)

    report = detect_precision_capabilities("cpu")
    assert report.supports("nvfp4", "nvfp4")
    assert not report.supports("mxfp6", "mxfp8")
    assert report.require("nvfp4", "nvfp4").provider_id == "torchao_nvfp4_dense"
    with pytest.raises(RuntimeError, match="no verified native"):
        report.require("mxfp6", "mxfp8")


def test_emulation_cannot_register_as_a_precision_provider() -> None:
    with pytest.raises(ValueError, match="providers must execute real kernels"):
        ExpertPrecisionCapability(
            provider_id="reference_fake_qat",
            weight_format="mxfp4",
            activation_format="mxfp8",
            forward_available=True,
            backward_available=True,
            grouped_available=False,
            verified=True,
        )


def test_cli_surfaces_have_no_fake_or_reference_choices(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        precision_module.torch.cpu,
        "_is_avx512_bf16_supported",
        lambda: False,
        raising=False,
    )
    plan = _plan_parser(device="cpu")
    training = _training_parser(device="cpu")
    assert "fake_qat" not in plan.format_help()
    assert "fake_qat" not in training.format_help()
    assert "reference" not in training.format_help()
    assert "mxfp4" not in plan.format_help()
    assert "mxfp6" not in training.format_help()
    with pytest.raises(SystemExit):
        plan.parse_args(
            ["1m", "--vocab-size", "256", "--expert-weight-format", "mxfp4"]
        )
    with pytest.raises(SystemExit):
        training.parse_args(["--smoke", "--expert-weight-format", "mxfp6"])
    with pytest.raises(SystemExit):
        training.parse_args(["--smoke", "--expert-promotion-format", "mxfp8"])

    args = _precision_parser().parse_args(["--device", "cpu", "--json"])
    _run_precision(args)
    payload = json.loads(capsys.readouterr().out)
    assert payload["selectable_pairs"] == [
        {"weight_format": "float32", "activation_format": "float32"}
    ]
