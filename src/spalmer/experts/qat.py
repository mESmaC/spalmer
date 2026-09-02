"""Format-faithful fake quantization for routed-expert QAT.

This module is deliberately independent from the expert-bank implementation.  It
provides a pure-PyTorch numerical reference for the W4A8 contract used by the
routed experts:

* MXFP4 weights use E2M1 elements and one E8M0 scale per 32 values.
* MXFP8 activations use E4M3 elements and one E8M0 scale per 32 values.
* NVFP4 weights use E2M1 elements, one E4M3 scale per 16 values, and one
  tensor-wide FP32 outer scale.

The returned tensors are dequantized back into the input dtype and use a
straight-through estimator by default.  They are therefore QAT operands, not a
packed checkpoint representation and not evidence of a native W4A8 GEMM.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F

WeightFormat = Literal["mxfp4", "nvfp4"]
ActivationFormat = Literal["mxfp8"]
BackendPreference = Literal["auto", "reference", "native"]

_E2M1_POSITIVE_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MAX = 6.0
_E4M3_MAX = 448.0
_E4M3_MIN_NORMAL = 2.0**-6
_E8M0_MIN_EXPONENT = -127
_E8M0_MAX_EXPONENT = 127


def _e4m3_positive_values() -> tuple[float, ...]:
    """Return every non-negative finite E4M3FN value in encoding order."""

    values = [mantissa * 2.0**-9 for mantissa in range(8)]
    for encoded_exponent in range(1, 15):
        exponent = encoded_exponent - 7
        values.extend((1.0 + mantissa / 8.0) * 2.0**exponent for mantissa in range(8))
    # E4M3FN uses six finite mantissas in the all-ones exponent field; the
    # final encoding is NaN.  This makes 448 the largest finite value.
    values.extend((1.0 + mantissa / 8.0) * 2.0**8 for mantissa in range(7))
    return tuple(values)


_E4M3_POSITIVE_VALUES = _e4m3_positive_values()


@dataclass(frozen=True, slots=True)
class ExpertQATConfig:
    """Numerical contract for routed-expert W4A8 quantization-aware training.

    ``auto`` currently resolves to the format-faithful reference path.  A
    ``native`` request is strict and fails until this module contains a real
    mixed W4A8 backend; it never silently substitutes dequantized matmuls.
    """

    weight_format: WeightFormat = "mxfp4"
    activation_format: ActivationFormat = "mxfp8"
    backend: BackendPreference = "auto"
    stochastic_rounding: bool = True

    def __post_init__(self) -> None:
        if self.weight_format not in {"mxfp4", "nvfp4"}:
            raise ValueError("weight_format must be 'mxfp4' or 'nvfp4'")
        if self.activation_format != "mxfp8":
            raise ValueError("activation_format must be 'mxfp8'")
        if self.backend not in {"auto", "reference", "native"}:
            raise ValueError("backend must be 'auto', 'reference', or 'native'")


@dataclass(frozen=True, slots=True)
class ExpertQATBackendStatus:
    """Read-only description of the backend selected for one QAT contract."""

    requested_backend: BackendPreference
    selected_backend: Literal["reference"] | None
    native_w4a8_available: bool
    emulated: bool
    torchao_available: bool
    detail: str


def expert_qat_backend_status(
    config: ExpertQATConfig | None = None,
) -> ExpertQATBackendStatus:
    """Report capability without importing optional runtimes or changing state."""

    config = config or ExpertQATConfig()
    torchao_available = importlib.util.find_spec("torchao") is not None
    native_detail = (
        f"native {config.weight_format.upper()}/{config.activation_format.upper()} "
        "routed-expert GEMM is not implemented in SPALMER"
    )
    if config.backend == "native":
        return ExpertQATBackendStatus(
            requested_backend="native",
            selected_backend=None,
            native_w4a8_available=False,
            emulated=False,
            torchao_available=torchao_available,
            detail=native_detail,
        )
    return ExpertQATBackendStatus(
        requested_backend=config.backend,
        selected_backend="reference",
        native_w4a8_available=False,
        emulated=True,
        torchao_available=torchao_available,
        detail=f"{native_detail}; using pure-Torch fake quantize/dequantize",
    )


def require_expert_qat_backend(config: ExpertQATConfig) -> ExpertQATBackendStatus:
    """Resolve a backend, raising when a strict native request cannot be met."""

    status = expert_qat_backend_status(config)
    if config.backend == "native" and not status.native_w4a8_available:
        raise RuntimeError(
            f"strict native QAT requested, but {status.detail}; "
            "choose backend='auto' or backend='reference' to use emulation"
        )
    return status


def fake_quantize_expert_weight(values: Tensor, config: ExpertQATConfig) -> Tensor:
    """Fake-quantize routed-expert weights according to ``config``."""

    require_expert_qat_backend(config)
    if config.weight_format == "mxfp4":
        return fake_quantize_mxfp4(
            values,
            stochastic=config.stochastic_rounding,
            straight_through=True,
        )
    return fake_quantize_nvfp4(
        values,
        stochastic=config.stochastic_rounding,
        straight_through=True,
    )


def fake_quantize_expert_activation(values: Tensor, config: ExpertQATConfig) -> Tensor:
    """Fake-quantize routed-expert inputs to MXFP8 according to ``config``."""

    require_expert_qat_backend(config)
    return fake_quantize_mxfp8(
        values,
        stochastic=config.stochastic_rounding,
        straight_through=True,
    )


def fake_quantize_mxfp4(
    values: Tensor,
    *,
    stochastic: bool = False,
    straight_through: bool = True,
) -> Tensor:
    """Apply E2M1 elements with 32-value E8M0 block scaling."""

    return _fake_quantize_mx(
        values,
        element_values=_E2M1_POSITIVE_VALUES,
        element_max=_E2M1_MAX,
        block_size=32,
        stochastic=stochastic,
        straight_through=straight_through,
    )


def fake_quantize_mxfp8(
    values: Tensor,
    *,
    stochastic: bool = False,
    straight_through: bool = True,
) -> Tensor:
    """Apply E4M3 elements with 32-value E8M0 block scaling."""

    return _fake_quantize_mx(
        values,
        element_values=_E4M3_POSITIVE_VALUES,
        element_max=_E4M3_MAX,
        block_size=32,
        stochastic=stochastic,
        straight_through=straight_through,
    )


def fake_quantize_nvfp4(
    values: Tensor,
    *,
    stochastic: bool = False,
    straight_through: bool = True,
) -> Tensor:
    """Apply NVFP4's FP32 outer, E4M3 block-scale, and E2M1 hierarchy."""

    _validate_values(values)
    if values.shape[-1] == 0:
        return values.clone()

    work, original_length = _pad_final_dimension(values.float(), block_size=16)
    blocks = work.reshape(*work.shape[:-1], work.shape[-1] // 16, 16)

    # This is the decoding outer scale.  Dividing by both format maxima lets
    # the largest block use the full E4M3 scale range and the full E2M1 range.
    # A stacked expert bank is ``[experts, rows, K]``. Give each expert its
    # own outer scale so its quantization does not change with the other
    # resident/selected expert identities in the batch.
    tensor_amax = (
        work.abs().amax(dim=(-2, -1), keepdim=True)
        if work.ndim >= 3
        else work.abs().amax()
    )
    outer_scale = torch.where(
        tensor_amax > 0,
        tensor_amax / (_E4M3_MAX * _E2M1_MAX),
        torch.ones_like(tensor_amax),
    ).float()
    block_outer_scale = outer_scale.unsqueeze(-1) if outer_scale.ndim else outer_scale

    unquantized_block_scale = blocks.abs().amax(dim=-1, keepdim=True) / _E2M1_MAX
    normalized_block_scale = unquantized_block_scale / block_outer_scale
    normalized_block_scale = normalized_block_scale.clamp(
        min=_E4M3_MIN_NORMAL,
        max=_E4M3_MAX,
    )
    block_scale = _round_positive_to_format(
        normalized_block_scale,
        _E4M3_POSITIVE_VALUES,
        stochastic=False,
    )

    combined_scale = block_outer_scale * block_scale
    normalized = (blocks / combined_scale).clamp(-_E2M1_MAX, _E2M1_MAX)
    quantized = _round_signed_to_format(
        normalized,
        _E2M1_POSITIVE_VALUES,
        stochastic=stochastic,
    )
    dequantized = (quantized * combined_scale).reshape_as(work)
    dequantized = dequantized[..., :original_length].to(values.dtype)
    return _apply_ste(values, dequantized, straight_through)


def _fake_quantize_mx(
    values: Tensor,
    *,
    element_values: tuple[float, ...],
    element_max: float,
    block_size: int,
    stochastic: bool,
    straight_through: bool,
) -> Tensor:
    _validate_values(values)
    if values.shape[-1] == 0:
        return values.clone()

    work, original_length = _pad_final_dimension(values.float(), block_size=block_size)
    blocks = work.reshape(*work.shape[:-1], work.shape[-1] // block_size, block_size)
    scales = _e8m0_rceil_scales(blocks, element_max=element_max)
    normalized = (blocks / scales).clamp(-element_max, element_max)
    quantized = _round_signed_to_format(
        normalized,
        element_values,
        stochastic=stochastic,
    )
    dequantized = (quantized * scales).reshape_as(work)
    dequantized = dequantized[..., :original_length].to(values.dtype)
    return _apply_ste(values, dequantized, straight_through)


def _e8m0_rceil_scales(blocks: Tensor, *, element_max: float) -> Tensor:
    """Choose the smallest E8M0 power-of-two scale that avoids overflow."""

    max_abs = blocks.abs().amax(dim=-1, keepdim=True)
    raw_scale = max_abs / element_max
    exponent = torch.ceil(torch.log2(raw_scale.clamp_min(2.0**_E8M0_MIN_EXPONENT)))
    exponent = exponent.clamp(_E8M0_MIN_EXPONENT, _E8M0_MAX_EXPONENT)
    scale = torch.exp2(exponent)
    return torch.where(max_abs > 0, scale, torch.ones_like(scale))


def _round_signed_to_format(
    values: Tensor,
    positive_values: tuple[float, ...],
    *,
    stochastic: bool,
) -> Tensor:
    magnitude = _round_positive_to_format(values.abs(), positive_values, stochastic=stochastic)
    return torch.copysign(magnitude, values)


def _round_positive_to_format(
    values: Tensor,
    positive_values: tuple[float, ...],
    *,
    stochastic: bool,
) -> Tensor:
    levels = values.new_tensor(positive_values)
    bounded = values.clamp(0.0, positive_values[-1])
    upper_index = torch.bucketize(bounded.contiguous(), levels).clamp_max(len(positive_values) - 1)
    lower_index = (upper_index - 1).clamp_min(0)
    lower = levels[lower_index]
    upper = levels[upper_index]

    if stochastic:
        width = upper - lower
        upper_probability = torch.where(
            width > 0,
            (bounded - lower) / width,
            torch.zeros_like(width),
        )
        rounded = torch.where(torch.rand_like(bounded) < upper_probability, upper, lower)
    else:
        lower_distance = bounded - lower
        upper_distance = upper - bounded
        ties = lower_distance == upper_distance
        # Positive format levels are in encoding order, so an even index is
        # the round-to-nearest-even choice at an exact midpoint.
        tie_chooses_upper = torch.remainder(upper_index, 2) == 0
        choose_upper = (upper_distance < lower_distance) | (ties & tie_chooses_upper)
        rounded = torch.where(choose_upper, upper, lower)
    return rounded


def _pad_final_dimension(values: Tensor, *, block_size: int) -> tuple[Tensor, int]:
    original_length = values.shape[-1]
    padded_length = math.ceil(original_length / block_size) * block_size
    padding = padded_length - original_length
    if padding:
        values = F.pad(values, (0, padding))
    return values, original_length


def _validate_values(values: Tensor) -> None:
    if not isinstance(values, Tensor):
        raise TypeError("values must be a torch.Tensor")
    if not values.is_floating_point():
        raise TypeError("values must use a floating-point dtype")
    if values.ndim == 0:
        raise ValueError("values must have a final dimension")


def _apply_ste(values: Tensor, dequantized: Tensor, straight_through: bool) -> Tensor:
    if straight_through:
        return values + (dequantized - values).detach()
    return dequantized


__all__ = [
    "ActivationFormat",
    "BackendPreference",
    "ExpertQATBackendStatus",
    "ExpertQATConfig",
    "WeightFormat",
    "expert_qat_backend_status",
    "fake_quantize_expert_activation",
    "fake_quantize_expert_weight",
    "fake_quantize_mxfp4",
    "fake_quantize_mxfp8",
    "fake_quantize_nvfp4",
    "require_expert_qat_backend",
]
