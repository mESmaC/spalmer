"""Configuration for reproducible, GPU-first SPALMER experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Runtime controls independent of the model and corpus definitions.

    Parameters remain FP32 while BF16 autocast supplies the accelerator compute
    lane.  This gives AdamW FP32 parameters and moment state without maintaining
    a second hand-written shadow copy.
    """

    max_steps: int
    micro_batch_size: int
    sequence_length: int
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    min_learning_rate_ratio: float = 0.1
    warmup_steps: int = 0
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    gradient_clip: float | None = 1.0
    auxiliary_loss_weight: float = 0.01
    surprise_calibration_weight: float = 0.05
    device: str = "cuda"
    require_cuda: bool = True
    compute_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    parameter_dtype: Literal["float32"] = "float32"
    fused_adamw: Literal["auto", "on", "off"] = "auto"
    activation_checkpointing: bool = False
    deterministic_algorithms: bool = False
    seed: int = 0
    validation_interval_steps: int = 100
    checkpoint_interval_steps: int = 100
    telemetry_interval_steps: int = 10

    def __post_init__(self) -> None:
        for name in (
            "max_steps",
            "micro_batch_size",
            "sequence_length",
            "gradient_accumulation_steps",
            "validation_interval_steps",
            "checkpoint_interval_steps",
            "telemetry_interval_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError("warmup_steps must be in [0, max_steps)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.min_learning_rate_ratio <= 1:
            raise ValueError("min_learning_rate_ratio must be in [0, 1]")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if not 0 <= self.adam_beta1 < 1 or not 0 <= self.adam_beta2 < 1:
            raise ValueError("Adam betas must be in [0, 1)")
        if self.adam_epsilon <= 0:
            raise ValueError("adam_epsilon must be positive")
        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive or None")
        if self.auxiliary_loss_weight < 0 or self.surprise_calibration_weight < 0:
            raise ValueError("auxiliary loss weights cannot be negative")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if self.compute_dtype not in {"bfloat16", "float32"}:
            raise ValueError(f"unsupported compute_dtype: {self.compute_dtype!r}")
        if self.parameter_dtype != "float32":
            raise ValueError("the current persistent parameter lane must remain float32")
        if self.fused_adamw not in {"auto", "on", "off"}:
            raise ValueError(f"unsupported fused_adamw policy: {self.fused_adamw!r}")

    @property
    def tokens_per_optimizer_step(self) -> int:
        return self.micro_batch_size * self.sequence_length * self.gradient_accumulation_steps

    @property
    def token_budget(self) -> int:
        return self.max_steps * self.tokens_per_optimizer_step

    def learning_rate_multiplier(self, completed_steps: int) -> float:
        """Linear warmup followed by cosine decay to the configured floor."""

        if completed_steps < 0:
            raise ValueError("completed_steps cannot be negative")
        if self.warmup_steps and completed_steps < self.warmup_steps:
            return (completed_steps + 1) / self.warmup_steps
        if self.max_steps <= self.warmup_steps + 1:
            return self.min_learning_rate_ratio
        progress = min(
            1.0,
            (completed_steps - self.warmup_steps)
            / (self.max_steps - self.warmup_steps - 1),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = self.min_learning_rate_ratio
        return floor + (1.0 - floor) * cosine

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["TrainingConfig"]
