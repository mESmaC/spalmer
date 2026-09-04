from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import spalmer.experts.bank as bank_module
from spalmer.experts import MicroExpertBank, MicroExpertsConfig
from spalmer.experts.native import (
    NativeExpertPrecisionError,
    native_expert_linear,
    native_expert_matmul,
)
from spalmer.precision import detect_precision_capabilities


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        (
            (actual.detach().float() - expected.detach().float()).norm()
            / expected.detach().float().norm().clamp_min(1e-12)
        ).item()
    )


@pytest.mark.parametrize(
    ("weight_format", "activation_format", "forward_tolerance"),
    [
        ("nvfp4", "nvfp4", 0.30),
        ("mxfp8", "mxfp8", 0.15),
    ],
)
def test_native_expert_kernel_forward_and_backward_are_numerically_valid(
    weight_format: str,
    activation_format: str,
    forward_tolerance: float,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    capabilities = detect_precision_capabilities("cuda")
    if not capabilities.supports(weight_format, activation_format):
        pytest.skip(f"native {weight_format}/{activation_format} is unavailable")

    generator = torch.Generator(device="cpu").manual_seed(1204)
    input = (
        torch.randn((7, 37), generator=generator)
        .to(device="cuda", dtype=torch.bfloat16)
        .requires_grad_(True)
    )
    weight = (
        torch.randn((23, 37), generator=generator)
        .to(device="cuda", dtype=torch.bfloat16)
        .requires_grad_(True)
    )
    grad_output = torch.randn((7, 23), generator=generator).to(
        device="cuda", dtype=torch.bfloat16
    )

    output = native_expert_linear(
        input,
        weight,
        weight_format=weight_format,
        activation_format=activation_format,
    )
    output.backward(grad_output)
    assert input.grad is not None
    assert weight.grad is not None

    reference_output = input.detach().float() @ weight.detach().float().t()
    reference_input_grad = grad_output.float() @ weight.detach().float()
    reference_weight_grad = grad_output.float().t() @ input.detach().float()
    forward_error = _relative_error(output, reference_output)
    input_grad_error = _relative_error(input.grad, reference_input_grad)
    weight_grad_error = _relative_error(weight.grad, reference_weight_grad)

    assert forward_error < forward_tolerance, forward_error
    assert input_grad_error < 0.15, input_grad_error
    assert weight_grad_error < 0.15, weight_grad_error


def test_native_expert_dispatch_never_falls_back_for_unknown_pair() -> None:
    values = torch.ones((2, 4), dtype=torch.bfloat16)
    with pytest.raises(NativeExpertPrecisionError, match="no integrated native expert kernel"):
        native_expert_linear(
            values,
            values,
            weight_format="mxfp4",
            activation_format="mxfp8",
        )


@pytest.mark.parametrize(
    ("weight_format", "activation_format", "forward_tolerance"),
    [
        ("nvfp4", "nvfp4", 0.30),
        ("mxfp8", "mxfp8", 0.15),
    ],
)
def test_native_expert_io_weight_layout_preserves_forward_and_autograd(
    weight_format: str,
    activation_format: str,
    forward_tolerance: float,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if not detect_precision_capabilities("cuda").supports(weight_format, activation_format):
        pytest.skip(f"native {weight_format}/{activation_format} is unavailable")

    generator = torch.Generator(device="cpu").manual_seed(812)
    input = (
        torch.randn((11, 37), generator=generator)
        .to(device="cuda", dtype=torch.bfloat16)
        .requires_grad_(True)
    )
    weight = (
        torch.randn((37, 23), generator=generator)
        .to(device="cuda", dtype=torch.bfloat16)
        .requires_grad_(True)
    )
    grad_output = torch.randn((11, 23), generator=generator).to(
        device="cuda", dtype=torch.bfloat16
    )
    output = native_expert_matmul(
        input,
        weight,
        weight_format=weight_format,
        activation_format=activation_format,
    )
    output.backward(grad_output)
    assert input.grad is not None
    assert weight.grad is not None
    assert (
        _relative_error(output, input.detach().float() @ weight.detach().float())
        < forward_tolerance
    )
    assert _relative_error(input.grad, grad_output.float() @ weight.detach().float().t()) < 0.15
    assert _relative_error(weight.grad, input.detach().float().t() @ grad_output.float()) < 0.15


def test_expert_bank_validates_and_caches_the_actual_execution_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices: list[str] = []
    capability = SimpleNamespace(provider_id="verified_cpu", grouped_available=True)

    class Report:
        def require(self, weight_format: str, activation_format: str):
            assert (weight_format, activation_format) == ("bfloat16", "bfloat16")
            return capability

    def detect(device: torch.device):
        devices.append(str(device))
        return Report()

    monkeypatch.setattr(bank_module, "detect_precision_capabilities", detect)
    config = MicroExpertsConfig(
        d_model=4,
        num_experts=2,
        expert_inter_dim=2,
        active_experts=1,
        min_active_experts=1,
        max_active_experts=2,
        min_resident_experts=1,
        max_resident_experts=2,
        expert_execution="loop",
    )
    bank = MicroExpertBank(config)
    assert devices == []
    hidden = torch.randn(2, 4)
    token_index = torch.tensor([0, 1])
    expert_index = torch.tensor([0, 1])
    routing_weights = torch.ones(2)
    first = bank.execute_routing(hidden, token_index, expert_index, routing_weights)
    second = bank.execute_routing(hidden, token_index, expert_index, routing_weights)
    assert first.shape == second.shape == hidden.shape
    assert devices == ["cpu"]
    assert bank.native_provider_id == "verified_cpu"
