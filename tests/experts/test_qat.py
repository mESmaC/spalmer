"""Static numerical checks for routed-expert QAT formats.

These tests do not construct a model, run backward, or take optimizer steps.
"""

from __future__ import annotations

import pytest
import torch

from spalmer.experts import MicroExpertsConfig
from spalmer.experts.qat import (
    ExpertQATConfig,
    expert_qat_backend_status,
    fake_quantize_expert_activation,
    fake_quantize_expert_weight,
    fake_quantize_mxfp4,
    fake_quantize_mxfp8,
    fake_quantize_nvfp4,
    require_expert_qat_backend,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "quantizer",
    [fake_quantize_mxfp4, fake_quantize_mxfp8, fake_quantize_nvfp4],
)
def test_quantizers_preserve_shape_dtype_and_non_multiple_final_dimension(
    dtype: torch.dtype,
    quantizer,
) -> None:
    values = torch.linspace(-7.0, 7.0, 2 * 3 * 35, dtype=dtype).reshape(2, 3, 35)
    result = quantizer(values, straight_through=False)
    assert result.shape == values.shape
    assert result.dtype == values.dtype
    assert torch.isfinite(result).all()


def test_mxfp4_uses_independent_32_value_e8m0_blocks() -> None:
    first_block = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 4,
        dtype=torch.float32,
    )
    values = torch.cat((first_block, torch.tensor([12.0])))
    result = fake_quantize_mxfp4(values, straight_through=False)
    torch.testing.assert_close(result[:32], first_block)
    torch.testing.assert_close(result[32:], torch.tensor([12.0]))


def test_mxfp8_preserves_representable_e4m3_values_in_one_block() -> None:
    values = torch.tensor(
        [0.0, 2.0**-9, 2.0**-6, 0.25, 1.0, 1.125, 16.0, 448.0],
        dtype=torch.float32,
    )
    result = fake_quantize_mxfp8(values, straight_through=False)
    torch.testing.assert_close(result, values)


def test_nvfp4_uses_outer_fp32_and_16_value_e4m3_scales() -> None:
    # With an amax of six, the two-level decoding scales multiply back to one
    # for the largest block.  Every value here is exactly representable in E2M1.
    values = torch.tensor(
        [[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 2 + [0.0]],
        dtype=torch.float32,
    )
    result = fake_quantize_nvfp4(values, straight_through=False)
    torch.testing.assert_close(result, values)


def test_nvfp4_stacked_experts_have_independent_outer_scales() -> None:
    first = torch.linspace(-1.0, 1.0, 32).repeat(3, 1)
    second = 1_000.0 * first
    alone = fake_quantize_nvfp4(first, straight_through=False)
    stacked = fake_quantize_nvfp4(
        torch.stack((first, second)),
        straight_through=False,
    )
    torch.testing.assert_close(stacked[0], alone)


@pytest.mark.parametrize(
    "quantizer",
    [fake_quantize_mxfp4, fake_quantize_mxfp8, fake_quantize_nvfp4],
)
def test_zero_and_empty_tensors_are_preserved(quantizer) -> None:
    zeros = torch.zeros(2, 17, dtype=torch.bfloat16)
    torch.testing.assert_close(quantizer(zeros, straight_through=False), zeros)
    empty = torch.empty(2, 0, dtype=torch.bfloat16)
    result = quantizer(empty, straight_through=False)
    assert result.shape == empty.shape
    assert result.dtype == empty.dtype


def test_ste_result_keeps_the_source_tensor_as_its_autograd_leaf() -> None:
    values = torch.tensor([[0.3, 0.7, 1.1]], requires_grad=True)
    result = fake_quantize_mxfp4(values)
    assert result.requires_grad
    assert result.grad_fn is not None
    assert result.shape == values.shape


@pytest.mark.parametrize("weight_format", ["mxfp4", "nvfp4"])
def test_contract_dispatches_w4_weights_and_mxfp8_activations(weight_format: str) -> None:
    config = ExpertQATConfig(
        weight_format=weight_format,
        activation_format="mxfp8",
        backend="reference",
        stochastic_rounding=False,
    )
    values = torch.tensor([[0.1, 0.5, 1.0, 2.0]], dtype=torch.bfloat16)
    weight = fake_quantize_expert_weight(values, config)
    activation = fake_quantize_expert_activation(values, config)
    assert weight.shape == activation.shape == values.shape
    assert weight.dtype == activation.dtype == torch.bfloat16


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"weight_format": "int4"}, "weight_format"),
        ({"activation_format": "nvfp4"}, "activation_format"),
        ({"backend": "torchao"}, "backend"),
    ],
)
def test_config_rejects_formats_outside_the_contract(overrides, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ExpertQATConfig(**overrides)


def test_backend_status_is_honest_about_emulation_and_optional_torchao() -> None:
    status = expert_qat_backend_status(ExpertQATConfig(backend="auto"))
    assert status.selected_backend == "reference"
    assert not status.native_w4a8_available
    assert status.emulated
    assert isinstance(status.torchao_available, bool)
    assert "not implemented" in status.detail


def test_strict_native_request_fails_instead_of_silently_emulating() -> None:
    config = ExpertQATConfig(weight_format="nvfp4", backend="native")
    status = expert_qat_backend_status(config)
    assert status.selected_backend is None
    assert not status.emulated
    with pytest.raises(RuntimeError, match="strict native QAT requested"):
        require_expert_qat_backend(config)
    with pytest.raises(RuntimeError, match="strict native QAT requested"):
        fake_quantize_expert_weight(torch.ones(1, 16), config)


def test_request_start_residency_cannot_exceed_dynamic_execution_cap() -> None:
    with pytest.raises(ValueError, match="maximum active experts"):
        MicroExpertsConfig(
            d_model=16,
            num_experts=20,
            min_resident_experts=10,
            max_resident_experts=20,
            max_active_experts=6,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"expert_master_dtype": "float32"},
        {"expert_activation_format": "bfloat16"},
        {"expert_fake_quantization": False},
    ],
)
def test_fp4_qat_recipe_is_fail_closed(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="FP4 expert"):
        MicroExpertsConfig(d_model=16, num_experts=4, **overrides)


@pytest.mark.parametrize(
    "values",
    [torch.tensor(1.0), torch.ones(4, dtype=torch.int64)],
)
def test_quantizers_reject_inputs_without_a_floating_final_dimension(values: torch.Tensor) -> None:
    expected = ValueError if values.ndim == 0 else TypeError
    with pytest.raises(expected):
        fake_quantize_mxfp4(values)
