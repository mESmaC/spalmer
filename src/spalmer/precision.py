"""Read-only discovery and selection of real expert compute kernels.

Precision names in a model configuration describe the operands that enter the
matrix multiply.  A package being importable, or a GPU advertising a dtype, is
not enough: a low-precision pair is selectable only when a SPALMER-integrated
provider reports a verified forward *and* backward implementation.  This keeps
``auto`` fail-closed and prevents quantize/dequantize emulation from quietly
standing in for a Tensor Core kernel.

Optional native integrations expose a state-preserving hook at
``spalmer.experts.native.native_provider_capabilities(device=None)``.  The hook
returns an iterable of :class:`ExpertPrecisionCapability` objects or mappings
with the same constructor fields.  Missing hooks simply leave the corresponding
formats unavailable.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

ExpertWeightFormat = Literal[
    "mxfp4", "nvfp4", "mxfp6", "mxfp8", "bfloat16", "float32"
]
ExpertActivationFormat = Literal["mxfp8", "nvfp4", "bfloat16", "float32"]

EXPERT_WEIGHT_FORMATS: tuple[ExpertWeightFormat, ...] = (
    "mxfp4",
    "nvfp4",
    "mxfp6",
    "mxfp8",
    "bfloat16",
    "float32",
)
EXPERT_ACTIVATION_FORMATS: tuple[ExpertActivationFormat, ...] = (
    "mxfp8",
    "nvfp4",
    "bfloat16",
    "float32",
)

_FORBIDDEN_PROVIDER_MARKERS = ("fake", "emulat", "reference")


@dataclass(frozen=True, slots=True, order=True)
class ExpertPrecisionPair:
    """One exact routed-expert weight/activation operand pair."""

    weight_format: ExpertWeightFormat
    activation_format: ExpertActivationFormat

    def __post_init__(self) -> None:
        if self.weight_format not in EXPERT_WEIGHT_FORMATS:
            raise ValueError(f"unsupported expert weight format: {self.weight_format!r}")
        if self.activation_format not in EXPERT_ACTIVATION_FORMATS:
            raise ValueError(
                f"unsupported expert activation format: {self.activation_format!r}"
            )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExpertPrecisionCapability:
    """Availability claim from one real expert-kernel provider.

    ``verified`` means the provider has checked its complete integration rather
    than merely detecting a library or GPU feature.  A training selection also
    requires both forward and backward support.  Dense per-expert kernels remain
    selectable when ``grouped_available`` is false; the flag lets callers prefer
    a grouped MoE implementation when one is actually integrated.
    """

    provider_id: str
    weight_format: ExpertWeightFormat
    activation_format: ExpertActivationFormat
    forward_available: bool
    backward_available: bool
    grouped_available: bool
    verified: bool
    detail: str = ""

    def __post_init__(self) -> None:
        provider = self.provider_id.strip()
        if not provider:
            raise ValueError("provider_id cannot be empty")
        lowered = provider.lower()
        if any(marker in lowered for marker in _FORBIDDEN_PROVIDER_MARKERS):
            raise ValueError(
                "precision providers must execute real kernels; fake, emulated, and "
                "reference providers are forbidden"
            )
        object.__setattr__(self, "provider_id", provider)
        ExpertPrecisionPair(self.weight_format, self.activation_format)
        for name in (
            "forward_available",
            "backward_available",
            "grouped_available",
            "verified",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")

    @property
    def available(self) -> bool:
        """Whether the provider exposes any executable forward kernel."""

        return self.verified and self.forward_available

    @property
    def selectable(self) -> bool:
        """Whether this pair can be selected for training."""

        return self.available and self.backward_available

    @property
    def pair(self) -> ExpertPrecisionPair:
        return ExpertPrecisionPair(self.weight_format, self.activation_format)

    def to_dict(self) -> dict[str, str | bool]:
        values: dict[str, str | bool] = asdict(self)
        values["available"] = self.available
        values["selectable"] = self.selectable
        return values


@dataclass(frozen=True, slots=True)
class PrecisionCapabilities:
    """Immutable precision report for one concrete execution device."""

    device: str
    device_type: str
    device_index: int | None
    device_name: str | None
    compute_capability: tuple[int, int] | None
    expert_precisions: tuple[ExpertPrecisionCapability, ...]
    diagnostics: tuple[str, ...] = ()

    @property
    def selectable_expert_precisions(self) -> tuple[ExpertPrecisionCapability, ...]:
        return tuple(capability for capability in self.expert_precisions if capability.selectable)

    @property
    def selectable_pairs(self) -> tuple[ExpertPrecisionPair, ...]:
        return tuple(
            sorted({capability.pair for capability in self.selectable_expert_precisions})
        )

    @property
    def selectable_weight_formats(self) -> tuple[ExpertWeightFormat, ...]:
        selected = {pair.weight_format for pair in self.selectable_pairs}
        return tuple(value for value in EXPERT_WEIGHT_FORMATS if value in selected)

    @property
    def selectable_activation_formats(self) -> tuple[ExpertActivationFormat, ...]:
        selected = {pair.activation_format for pair in self.selectable_pairs}
        return tuple(value for value in EXPERT_ACTIVATION_FORMATS if value in selected)

    def supports(self, weight_format: str, activation_format: str) -> bool:
        return any(
            pair.weight_format == weight_format and pair.activation_format == activation_format
            for pair in self.selectable_pairs
        )

    def require(
        self,
        weight_format: ExpertWeightFormat,
        activation_format: ExpertActivationFormat,
    ) -> ExpertPrecisionCapability:
        """Return the first selectable provider for a pair, or fail closed."""

        for capability in self.selectable_expert_precisions:
            if (
                capability.weight_format == weight_format
                and capability.activation_format == activation_format
            ):
                return capability
        available = ", ".join(
            f"{pair.weight_format}/{pair.activation_format}" for pair in self.selectable_pairs
        ) or "none"
        raise RuntimeError(
            "no verified native SPALMER training kernel for "
            f"{weight_format}/{activation_format} on {self.device}; "
            f"selectable pairs: {available}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "device_type": self.device_type,
            "device_index": self.device_index,
            "device_name": self.device_name,
            "compute_capability": (
                None if self.compute_capability is None else list(self.compute_capability)
            ),
            "expert_precisions": [item.to_dict() for item in self.expert_precisions],
            "selectable_pairs": [pair.to_dict() for pair in self.selectable_pairs],
            "selectable_weight_formats": list(self.selectable_weight_formats),
            "selectable_activation_formats": list(self.selectable_activation_formats),
            "diagnostics": list(self.diagnostics),
        }


def detect_precision_capabilities(
    device: str | torch.device | None = None,
) -> PrecisionCapabilities:
    """Inspect one device without mutating a model or persistent system state.

    A provider may run and cache a bounded forward/backward smoke operation so
    ``verified`` means more than package presence. The smoke owns only temporary
    tensors and performs no optimizer step or model training.
    """

    resolved = _resolve_device(device)
    device_name, compute_capability = _device_metadata(resolved)
    capabilities = [_bf16_capability(resolved), _float32_capability(resolved)]
    native, diagnostics = _native_capabilities(resolved)
    capabilities.extend(native)

    unique: dict[tuple[str, str, str], ExpertPrecisionCapability] = {}
    for capability in capabilities:
        key = (
            capability.provider_id,
            capability.weight_format,
            capability.activation_format,
        )
        if key in unique:
            diagnostics.append(
                "native precision provider returned duplicate capability "
                f"{capability.provider_id}:{capability.weight_format}/"
                f"{capability.activation_format}"
            )
            continue
        unique[key] = capability
    ordered = tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                EXPERT_WEIGHT_FORMATS.index(item.weight_format),
                EXPERT_ACTIVATION_FORMATS.index(item.activation_format),
                item.provider_id,
            ),
        )
    )
    return PrecisionCapabilities(
        device=str(resolved),
        device_type=resolved.type,
        device_index=resolved.index,
        device_name=device_name,
        compute_capability=compute_capability,
        expert_precisions=ordered,
        diagnostics=tuple(diagnostics),
    )


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None and torch.cuda.is_available():
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def _device_metadata(
    device: torch.device,
) -> tuple[str | None, tuple[int, int] | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return ("CPU" if device.type == "cpu" else None), None
    try:
        index = torch.cuda.current_device() if device.index is None else device.index
        return torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index)
    except (AssertionError, RuntimeError, ValueError):
        return None, None


def _bf16_capability(device: torch.device) -> ExpertPrecisionCapability:
    if device.type == "cpu":
        try:
            hardware_probe = getattr(torch.cpu, "_is_avx512_bf16_supported", None)
            if not callable(hardware_probe) or not bool(hardware_probe()):
                raise RuntimeError("CPU does not report AVX512_BF16 support")
            # A feature bit alone is insufficient. Verify the same BF16
            # forward/backward family used by the expert bank while explicitly
            # escaping a possible inference-mode caller.
            with torch.inference_mode(False), torch.enable_grad():
                generator = torch.Generator(device="cpu").manual_seed(1616)
                left = torch.randn((8, 16), generator=generator).to(
                    dtype=torch.bfloat16
                ).requires_grad_(True)
                right = torch.randn((16, 8), generator=generator).to(
                    dtype=torch.bfloat16
                ).requires_grad_(True)
                output = left @ right
                gradients = torch.autograd.grad(
                    output.float().square().mean(),
                    (left, right),
                )
            available = bool(torch.isfinite(output).all()) and all(
                bool(torch.isfinite(gradient).all()) for gradient in gradients
            )
            if not available:
                raise RuntimeError("CPU BF16 smoke returned non-finite values")
            detail = "native AVX512_BF16 PyTorch forward/backward smoke passed"
        except (AssertionError, AttributeError, RuntimeError, TypeError, ValueError):
            available = False
            capability = getattr(torch.backends.cpu, "get_cpu_capability", lambda: "unknown")()
            detail = (
                "CPU has no verified native AVX512_BF16 forward/backward path "
                f"(reported capability {capability})"
            )
    elif device.type == "cuda" and torch.cuda.is_available():
        try:
            index = torch.cuda.current_device() if device.index is None else int(device.index)
            if index < 0 or index >= torch.cuda.device_count():
                raise ValueError(f"CUDA device index {index} is unavailable")
            # Capability discovery can be reached from an inference-mode
            # model call.  The probe must create ordinary autograd tensors or
            # it would falsely reject a working backward kernel.
            with torch.inference_mode(False), torch.enable_grad(), torch.cuda.device(index):
                # PyTorch defaults ``including_emulation`` to true. That is
                # explicitly unsuitable for SPALMER's native-only contract.
                available = bool(torch.cuda.is_bf16_supported(including_emulation=False))
                if available:
                    generator = torch.Generator(device="cpu").manual_seed(1616)
                    left = torch.randn((8, 16), generator=generator).to(
                        device=device, dtype=torch.bfloat16
                    ).requires_grad_(True)
                    right = torch.randn((16, 8), generator=generator).to(
                        device=device, dtype=torch.bfloat16
                    ).requires_grad_(True)
                    output = left @ right
                    torch.autograd.grad(output.float().square().mean(), (left, right))
                    torch.cuda.synchronize(device)
        except (AssertionError, RuntimeError, ValueError):
            available = False
        detail = (
            "native PyTorch CUDA BF16 forward/backward smoke passed"
            if available
            else "CUDA device has no verified native BF16 forward/backward path"
        )
    else:
        available = False
        detail = f"PyTorch BF16 expert execution is not verified on device type {device.type!r}"
    return ExpertPrecisionCapability(
        provider_id="bf16_torch",
        weight_format="bfloat16",
        activation_format="bfloat16",
        forward_available=available,
        backward_available=available,
        grouped_available=available,
        verified=available,
        detail=detail,
    )


def _float32_capability(device: torch.device) -> ExpertPrecisionCapability:
    """Expose an honest CPU/dev lane without relabelling FP32 as BF16."""

    available = False
    if device.type == "cpu":
        try:
            with torch.inference_mode(False), torch.enable_grad():
                generator = torch.Generator(device="cpu").manual_seed(3232)
                left = torch.randn((8, 16), generator=generator, requires_grad=True)
                right = torch.randn((16, 8), generator=generator, requires_grad=True)
                output = left @ right
                gradients = torch.autograd.grad(output.square().mean(), (left, right))
            available = bool(torch.isfinite(output).all()) and all(
                bool(torch.isfinite(gradient).all()) for gradient in gradients
            )
        except (AssertionError, RuntimeError, TypeError, ValueError):
            available = False
    detail = (
        "native PyTorch CPU FP32 forward/backward smoke passed"
        if available
        else "FP32 dev expert execution is exposed only on a verified CPU path"
    )
    return ExpertPrecisionCapability(
        provider_id="fp32_torch_cpu",
        weight_format="float32",
        activation_format="float32",
        forward_available=available,
        backward_available=available,
        grouped_available=available,
        verified=available,
        detail=detail,
    )


def _native_capabilities(
    device: torch.device,
) -> tuple[list[ExpertPrecisionCapability], list[str]]:
    try:
        module = importlib.import_module("spalmer.experts.native")
    except ModuleNotFoundError as error:
        if error.name == "spalmer.experts.native":
            return [], []
        return [], [f"native precision provider import failed: {error}"]
    except Exception as error:  # pragma: no cover - defensive optional-provider boundary
        return [], [f"native precision provider import failed: {error}"]

    hook = getattr(module, "native_provider_capabilities", None)
    if not callable(hook):
        return [], ["spalmer.experts.native has no native_provider_capabilities hook"]
    try:
        claims = hook(device=device)
        return _coerce_native_claims(claims), []
    except Exception as error:  # pragma: no cover - provider-specific failure boundary
        return [], [f"native precision capability probe failed: {error}"]


def _coerce_native_claims(claims: object) -> list[ExpertPrecisionCapability]:
    if not isinstance(claims, Iterable) or isinstance(claims, str | bytes | Mapping):
        raise TypeError("native_provider_capabilities must return an iterable of claims")
    capabilities = []
    for claim in claims:
        if isinstance(claim, ExpertPrecisionCapability):
            capability = claim
        elif isinstance(claim, Mapping):
            capability = ExpertPrecisionCapability(**dict(claim))
        else:
            raise TypeError("native precision capability claims must be dataclasses or mappings")
        capabilities.append(capability)
    return capabilities


__all__ = [
    "EXPERT_ACTIVATION_FORMATS",
    "EXPERT_WEIGHT_FORMATS",
    "ExpertActivationFormat",
    "ExpertPrecisionCapability",
    "ExpertPrecisionPair",
    "ExpertWeightFormat",
    "PrecisionCapabilities",
    "detect_precision_capabilities",
]
