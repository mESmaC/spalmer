"""Accelerator selection and deterministic seed controls."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from spalmer.training.config import TrainingConfig


@dataclass(frozen=True, slots=True)
class RuntimeDevice:
    device: torch.device
    compute_dtype: torch.dtype
    autocast_enabled: bool
    cuda_name: str | None
    compute_capability: tuple[int, int] | None


def resolve_runtime_device(config: TrainingConfig) -> RuntimeDevice:
    """Resolve the requested runtime and fail closed for a missing GPU lane."""

    device = torch.device(config.device)
    if config.require_cuda and device.type != "cuda":
        raise RuntimeError("this experiment requires CUDA; choose device='cuda'")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but this Python environment has no CUDA-enabled Torch runtime"
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        capability = torch.cuda.get_device_capability(index)
        name = torch.cuda.get_device_name(index)
        if config.compute_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                f"BF16 compute was requested but {name} does not report BF16 support"
            )
    else:
        capability = None
        name = None

    dtype = torch.bfloat16 if config.compute_dtype == "bfloat16" else torch.float32
    return RuntimeDevice(
        device=device,
        compute_dtype=dtype,
        autocast_enabled=device.type == "cuda" and dtype != torch.float32,
        cuda_name=name,
        compute_capability=capability,
    )


def seed_everything(seed: int, *, deterministic_algorithms: bool = False) -> None:
    """Seed model initialization and stochastic runtime operations."""

    if seed < 0:
        raise ValueError("seed cannot be negative")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)


__all__ = ["RuntimeDevice", "resolve_runtime_device", "seed_everything"]
