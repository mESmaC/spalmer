"""Native routed-expert precision providers.

This module contains no numerical fallback.  Low-precision operands are packed
to their real hardware formats and every matrix multiply is dispatched through
PyTorch/TorchAO's scaled Tensor Core operators.  A format is reported as
selectable only after a forward-and-backward smoke test succeeds on the exact
CUDA device.

The currently integrated CUDA providers are deliberately honest about their
operand pairs:

* NVFP4/NVFP4 forward GEMMs with MXFP8 dgrad/wgrad GEMMs.
* MXFP8/MXFP8 forward, dgrad, and wgrad GEMMs.

They are dense per-expert providers.  SPALMER must not route them through its
ordinary grouped ``torch.bmm`` executor; a grouped low-precision capability is
advertised only when such a kernel is actually integrated.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


class NativeExpertPrecisionError(RuntimeError):
    """Raised when a requested native expert kernel cannot execute."""


def _ceil_multiple(value: int, multiple: int) -> int:
    return ((int(value) + multiple - 1) // multiple) * multiple


def _require_cuda_bf16(left: Tensor, right: Tensor, *, operation: str) -> None:
    if not left.is_cuda or not right.is_cuda:
        raise NativeExpertPrecisionError(f"{operation} requires CUDA operands")
    if left.device != right.device:
        raise NativeExpertPrecisionError(f"{operation} operands must share one device")
    if left.dtype != torch.bfloat16 or right.dtype != torch.bfloat16:
        raise NativeExpertPrecisionError(
            f"{operation} requires BF16 source activations and master weights"
        )


def _mx_quantize_rows(value: Tensor):
    """Pack matrix rows to E4M3 elements with block-32 E8M0 scales."""

    from torchao.prototype.mx_formats.config import (
        MXFP8Dim0CastKernelChoice,
        ScaleCalculationMode,
    )
    from torchao.prototype.mx_formats.mx_tensor import MXTensor
    from torchao.quantization.quantize_.common.kernel_preference import (
        KernelPreference,
    )

    return MXTensor.to_mx(
        value.contiguous(),
        torch.float8_e4m3fn,
        32,
        ScaleCalculationMode.RCEIL,
        KernelPreference.AUTO,
        mxfp8_dim0_cast_kernel_choice=MXFP8Dim0CastKernelChoice.TRITON,
    )


def _mxfp8_mm_nt(left: Tensor, right_rows: Tensor) -> Tensor:
    """Compute ``left @ right_rows.T`` with a real MXFP8 GEMM."""

    if left.ndim != 2 or right_rows.ndim != 2:
        raise ValueError("native MXFP8 operands must be rank-2 matrices")
    if left.shape[1] != right_rows.shape[1]:
        raise ValueError("native MXFP8 operands must share a contraction dimension")
    _require_cuda_bf16(left, right_rows, operation="native MXFP8 GEMM")
    if left.numel() == 0 or right_rows.numel() == 0:
        return left.new_zeros((left.shape[0], right_rows.shape[0]))

    m, contraction = left.shape
    n = right_rows.shape[0]
    # The SM120 scaled-MX path requires aligned M/N/K.  Padding is part of the
    # real packed operation and is sliced away after the Tensor Core GEMM.
    alignment = 128
    padded_m = _ceil_multiple(m, alignment)
    padded_n = _ceil_multiple(n, alignment)
    padded_k = _ceil_multiple(contraction, alignment)
    left_padded = F.pad(left, (0, padded_k - contraction, 0, padded_m - m))
    right_padded = F.pad(
        right_rows,
        (0, padded_k - contraction, 0, padded_n - n),
    )
    left_mx = _mx_quantize_rows(left_padded)
    right_mx = _mx_quantize_rows(right_padded)
    output = torch.mm(left_mx, right_mx.t())
    return output[:m, :n].to(dtype=left.dtype).contiguous()


class _NativeMXFP8Linear(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        weight_is_io: bool,
    ) -> Tensor:
        original_shape = input.shape
        flat_input = input.reshape(-1, input.shape[-1]).contiguous()
        right_rows = weight.t() if weight_is_io else weight
        output = _mxfp8_mm_nt(flat_input, right_rows)
        ctx.save_for_backward(input, weight)
        ctx.has_bias = bias is not None
        ctx.weight_is_io = weight_is_io
        if bias is not None:
            output = output + bias
        return output.reshape(*original_shape[:-1], right_rows.shape[0])

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: Any, grad_output: Tensor):
        input, weight = ctx.saved_tensors
        input_flat = input.reshape(-1, input.shape[-1]).contiguous()
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        if ctx.weight_is_io:
            # The expert bank stores projections as [input, output]. Keeping
            # that original tensor in autograd avoids retaining a full BF16
            # transposed copy for every routed projection.
            grad_input = _mxfp8_mm_nt(grad_flat, weight).reshape_as(input)
            grad_weight = _mxfp8_mm_nt(
                input_flat.t().contiguous(),
                grad_flat.t().contiguous(),
            )
        else:
            grad_input = _mxfp8_mm_nt(
                grad_flat,
                weight.t().contiguous(),
            ).reshape_as(input)
            grad_weight = _mxfp8_mm_nt(
                grad_flat.t().contiguous(),
                input_flat.t().contiguous(),
            )
        grad_bias = grad_flat.sum(dim=0) if ctx.has_bias else None
        return grad_input, grad_weight, grad_bias, None


def native_mxfp8_linear(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    """Linear operation whose forward and backward GEMMs execute in MXFP8."""

    if weight.ndim != 2 or input.ndim < 1 or input.shape[-1] != weight.shape[1]:
        raise ValueError("native MXFP8 linear received incompatible input/weight shapes")
    _require_cuda_bf16(input, weight, operation="native MXFP8 linear")
    return _NativeMXFP8Linear.apply(input, weight, bias, False)


def _pack_nvfp4(value: Tensor):
    """Pack BF16 rows into the actual E2M1/E4M3 NVFP4 representation."""

    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        per_tensor_amax_to_scale,
    )

    scale = per_tensor_amax_to_scale(value.detach().abs().amax())
    return NVFP4Tensor.to_nvfp4(
        value.contiguous(),
        per_tensor_scale=scale,
        use_triton_kernel=False,
    )


class _NativeNVFP4Linear(torch.autograd.Function):
    """Real NVFP4 forward with real MXFP8 dgrad/wgrad."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        weight_is_io: bool,
    ) -> Tensor:
        from torchao.prototype.mx_formats.nvfp4_tensor import _addmm_nvfp4_dispatch

        original_shape = input.shape
        flat_input = input.reshape(-1, input.shape[-1]).contiguous()
        m, contraction = flat_input.shape
        right_rows = weight.t() if weight_is_io else weight
        out_features = right_rows.shape[0]
        # PyTorch's SM120 NVFP4 GEMM accepts arbitrary M but requires N and K
        # to be divisible by 16.  Padding keeps small SPALMER experts eligible.
        # E2M1 stores two logical values per byte.  The scaled-matmul contract
        # requires the packed trailing dimension to be divisible by 16, hence
        # 32 logical contraction values rather than merely 16.
        padded_k = _ceil_multiple(contraction, 32)
        padded_n = _ceil_multiple(out_features, 16)
        input_padded = F.pad(flat_input, (0, padded_k - contraction))
        weight_padded = F.pad(
            right_rows,
            (0, padded_k - contraction, 0, padded_n - out_features),
        )
        packed_input = _pack_nvfp4(input_padded)
        packed_weight = _pack_nvfp4(weight_padded)
        output = _addmm_nvfp4_dispatch(
            packed_input,
            packed_weight.t(),
            None,
            None,
        )[:m, :out_features]
        ctx.save_for_backward(input, weight)
        ctx.has_bias = bias is not None
        ctx.weight_is_io = weight_is_io
        if bias is not None:
            output = output + bias
        return output.reshape(*original_shape[:-1], out_features)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: Any, grad_output: Tensor):
        input, weight = ctx.saved_tensors
        input_flat = input.reshape(-1, input.shape[-1]).contiguous()
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        if ctx.weight_is_io:
            grad_input = _mxfp8_mm_nt(grad_flat, weight).reshape_as(input)
            grad_weight = _mxfp8_mm_nt(
                input_flat.t().contiguous(),
                grad_flat.t().contiguous(),
            )
        else:
            grad_input = _mxfp8_mm_nt(
                grad_flat,
                weight.t().contiguous(),
            ).reshape_as(input)
            grad_weight = _mxfp8_mm_nt(
                grad_flat.t().contiguous(),
                input_flat.t().contiguous(),
            )
        grad_bias = grad_flat.sum(dim=0) if ctx.has_bias else None
        return grad_input, grad_weight, grad_bias, None


def native_nvfp4_linear(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    """Linear with genuine NVFP4 forward operands and MXFP8 backward GEMMs."""

    if weight.ndim != 2 or input.ndim < 1 or input.shape[-1] != weight.shape[1]:
        raise ValueError("native NVFP4 linear received incompatible input/weight shapes")
    _require_cuda_bf16(input, weight, operation="native NVFP4 linear")
    return _NativeNVFP4Linear.apply(input, weight, bias, False)


def native_expert_matmul(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    weight_format: str,
    activation_format: str,
) -> Tensor:
    """Dispatch ``input @ weight`` for an expert's native ``[in, out]`` weight.

    SPALMER stores expert projections in the layout consumed by its grouped
    BF16 executor. Low-precision packing still needs an output-major temporary,
    but autograd now retains the original parameter rather than a separately
    allocated BF16 transpose for every active expert and projection.
    """

    if weight.ndim != 2 or input.ndim < 1 or input.shape[-1] != weight.shape[0]:
        raise ValueError("native expert matmul received incompatible input/weight shapes")
    pair = (str(weight_format), str(activation_format))
    if pair == ("nvfp4", "nvfp4"):
        _require_cuda_bf16(input, weight, operation="native NVFP4 expert matmul")
        return _NativeNVFP4Linear.apply(input, weight, bias, True)
    if pair == ("mxfp8", "mxfp8"):
        _require_cuda_bf16(input, weight, operation="native MXFP8 expert matmul")
        return _NativeMXFP8Linear.apply(input, weight, bias, True)
    if pair == ("bfloat16", "bfloat16"):
        output = torch.matmul(input, weight)
        return output if bias is None else output + bias
    raise NativeExpertPrecisionError(
        "no integrated native expert kernel for "
        f"{weight_format}/{activation_format}; emulation is prohibited"
    )


def native_expert_linear(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    weight_format: str,
    activation_format: str,
) -> Tensor:
    """Dispatch one exact expert precision pair without any fallback."""

    pair = (str(weight_format), str(activation_format))
    if pair == ("nvfp4", "nvfp4"):
        return native_nvfp4_linear(input, weight, bias)
    if pair == ("mxfp8", "mxfp8"):
        return native_mxfp8_linear(input, weight, bias)
    if pair == ("bfloat16", "bfloat16"):
        return F.linear(input, weight, bias)
    raise NativeExpertPrecisionError(
        "no integrated native expert kernel for "
        f"{weight_format}/{activation_format}; emulation is prohibited"
    )


def native_weight_reconstruction_error(weight: Tensor, *, weight_format: str) -> Tensor:
    """Measure error from an actual packed representation, never fake quantization."""

    if weight.ndim != 2:
        raise ValueError("native weight error requires one rank-2 expert matrix")
    if weight_format == "nvfp4":
        padded_k = _ceil_multiple(weight.shape[1], 32)
        padded_n = _ceil_multiple(weight.shape[0], 16)
        padded = F.pad(
            weight,
            (0, padded_k - weight.shape[1], 0, padded_n - weight.shape[0]),
        )
        packed = _pack_nvfp4(padded)
        restored = packed.dequantize(weight.dtype)[: weight.shape[0], : weight.shape[1]]
    elif weight_format == "mxfp8":
        padded_k = _ceil_multiple(weight.shape[1], 32)
        padded = F.pad(weight, (0, padded_k - weight.shape[1]))
        packed = _mx_quantize_rows(padded)
        restored = packed.dequantize(weight.dtype)[:, : weight.shape[1]]
    elif weight_format == "bfloat16":
        return weight.new_zeros((), dtype=torch.float32)
    else:
        raise NativeExpertPrecisionError(
            f"no native packed representation for {weight_format!r}"
        )
    denominator = weight.float().square().sum().clamp_min(1e-12)
    return (weight.float() - restored.float()).square().sum() / denominator


def _native_prerequisites(device: torch.device) -> tuple[bool, str]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False, "CUDA is unavailable"
    index = torch.cuda.current_device() if device.index is None else device.index
    capability = torch.cuda.get_device_capability(index)
    required = (
        hasattr(torch, "float4_e2m1fn_x2")
        and hasattr(torch, "float8_e4m3fn")
        and hasattr(torch, "float8_e8m0fnu")
        and hasattr(torch.ops.aten, "_scaled_mm")
    )
    if not required:
        return False, "PyTorch does not expose the required FP4/FP8 scaled operators"
    try:
        import torchao  # noqa: F401
    except (ImportError, OSError) as error:
        return False, f"TorchAO failed to import: {error}"
    return True, f"SM{capability[0]}{capability[1]} PyTorch/TorchAO scaled Tensor Cores"


@lru_cache(maxsize=8)
def _smoke_native_pair(
    device_index: int,
    weight_format: str,
    activation_format: str,
) -> tuple[bool, str]:
    device = torch.device("cuda", device_index)
    available, note = _native_prerequisites(device)
    if not available:
        return False, note
    try:
        generator = torch.Generator(device="cpu").manual_seed(1204)
        input_cpu = torch.randn((7, 37), generator=generator)
        weight_cpu = torch.randn((23, 37), generator=generator)
        grad_cpu = torch.randn((7, 23), generator=generator)
        input = input_cpu.to(device=device, dtype=torch.bfloat16).requires_grad_(True)
        weight = weight_cpu.to(device=device, dtype=torch.bfloat16).requires_grad_(True)
        grad_output = grad_cpu.to(device=device, dtype=torch.bfloat16)
        output = native_expert_linear(
            input,
            weight,
            weight_format=weight_format,
            activation_format=activation_format,
        )
        output.backward(grad_output)
        torch.cuda.synchronize(device)
        if not (
            bool(torch.isfinite(output).all())
            and input.grad is not None
            and bool(torch.isfinite(input.grad).all())
            and weight.grad is not None
            and bool(torch.isfinite(weight.grad).all())
        ):
            raise NativeExpertPrecisionError("kernel returned a non-finite output or gradient")

        def relative_error(actual: Tensor, expected: Tensor) -> float:
            return float(
                (
                    (actual.detach().float() - expected.detach().float()).norm()
                    / expected.detach().float().norm().clamp_min(1e-12)
                ).item()
            )

        forward_error = relative_error(output, input.detach().float() @ weight.detach().float().t())
        input_grad_error = relative_error(input.grad, grad_output.float() @ weight.detach().float())
        weight_grad_error = relative_error(
            weight.grad,
            grad_output.float().t() @ input.detach().float(),
        )
        forward_tolerance = 0.30 if weight_format == "nvfp4" else 0.15
        if (
            forward_error >= forward_tolerance
            or input_grad_error >= 0.15
            or weight_grad_error >= 0.15
        ):
            raise NativeExpertPrecisionError(
                "kernel error exceeded tolerance: "
                f"forward={forward_error:.4f}, dgrad={input_grad_error:.4f}, "
                f"wgrad={weight_grad_error:.4f}"
            )
        return (
            True,
            f"{note}; native forward/backward smoke passed "
            f"(relative error {forward_error:.4f}/{input_grad_error:.4f}/"
            f"{weight_grad_error:.4f})",
        )
    except Exception as error:
        return False, f"native forward/backward smoke failed: {error}"


def native_provider_capabilities(
    device: str | torch.device | None = None,
) -> list[dict[str, str | bool]]:
    """Return device-tested native provider claims for precision discovery."""

    resolved = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if resolved.type == "cuda" and resolved.index is None and torch.cuda.is_available():
        resolved = torch.device("cuda", torch.cuda.current_device())
    if resolved.type != "cuda" or resolved.index is None:
        available, detail = _native_prerequisites(resolved)
        return [
            {
                "provider_id": provider,
                "weight_format": weight,
                "activation_format": activation,
                "forward_available": available,
                "backward_available": available,
                "grouped_available": False,
                "verified": False,
                "detail": detail,
            }
            for provider, weight, activation in (
                ("torchao_nvfp4_dense", "nvfp4", "nvfp4"),
                ("torchao_mxfp8_dense", "mxfp8", "mxfp8"),
            )
        ]

    claims: list[dict[str, str | bool]] = []
    for provider, weight, activation in (
        ("torchao_nvfp4_dense", "nvfp4", "nvfp4"),
        ("torchao_mxfp8_dense", "mxfp8", "mxfp8"),
    ):
        available, detail = _smoke_native_pair(resolved.index, weight, activation)
        claims.append(
            {
                "provider_id": provider,
                "weight_format": weight,
                "activation_format": activation,
                "forward_available": available,
                "backward_available": available,
                "grouped_available": False,
                "verified": available,
                "detail": detail,
            }
        )
    return claims


__all__ = [
    "NativeExpertPrecisionError",
    "native_expert_matmul",
    "native_expert_linear",
    "native_mxfp8_linear",
    "native_nvfp4_linear",
    "native_provider_capabilities",
    "native_weight_reconstruction_error",
]
