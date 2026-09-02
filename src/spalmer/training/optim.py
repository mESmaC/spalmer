"""Optimizer construction with explicit dense, sparse, and decay policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from spalmer.training.config import TrainingConfig


@dataclass(frozen=True, slots=True)
class ParameterGroups:
    decay: tuple[nn.Parameter, ...]
    no_decay: tuple[nn.Parameter, ...]
    sparse: tuple[nn.Parameter, ...]
    names: dict[int, str]

    @property
    def trainable_count(self) -> int:
        return sum(parameter.numel() for parameter in (*self.decay, *self.no_decay, *self.sparse))


def classify_parameters(model: nn.Module) -> ParameterGroups:
    """Partition every trainable parameter exactly once.

    Fake-QAT PLE rows requested with sparse gradients must use SparseAdam;
    applying AdamW to those gradients fails at the first optimizer step.  Bias,
    scalar, vector, normalization, and explicitly marked parameters do not decay.
    """

    sparse_ids: set[int] = set()
    for module in model.modules():
        module_config = getattr(module, "config", None)
        if not getattr(module_config, "sparse_gradients", False):
            continue
        weight = getattr(module, "weight", None)
        if isinstance(weight, nn.Parameter) and weight.requires_grad:
            sparse_ids.add(id(weight))

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    sparse: list[nn.Parameter] = []
    names: dict[int, str] = {}
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        names[parameter_id] = name
        if parameter_id in sparse_ids:
            sparse.append(parameter)
        elif _no_weight_decay(name, parameter):
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    return ParameterGroups(tuple(decay), tuple(no_decay), tuple(sparse), names)


def _no_weight_decay(name: str, parameter: nn.Parameter) -> bool:
    if bool(getattr(parameter, "_no_weight_decay", False)):
        return True
    lowered = name.lower()
    return (
        parameter.ndim < 2
        or lowered.endswith(".bias")
        or "norm" in lowered
        or lowered.endswith("lane_logits")
        or lowered.endswith(".gate")
    )


class OptimizerBundle:
    """One stateful façade for dense AdamW and optional SparseAdam."""

    def __init__(
        self,
        *,
        dense: torch.optim.Optimizer | None,
        sparse: torch.optim.Optimizer | None,
    ) -> None:
        if dense is None and sparse is None:
            raise ValueError("at least one optimizer is required")
        self.dense = dense
        self.sparse = sparse

    @property
    def optimizers(self) -> tuple[torch.optim.Optimizer, ...]:
        return tuple(item for item in (self.dense, self.sparse) if item is not None)

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "dense": None if self.dense is None else self.dense.state_dict(),
            "sparse": None if self.sparse is None else self.sparse.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        _load_optional_optimizer("dense", self.dense, state.get("dense"))
        _load_optional_optimizer("sparse", self.sparse, state.get("sparse"))

    def set_learning_rate(self, value: float) -> None:
        if value < 0:
            raise ValueError("learning rate cannot be negative")
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                group["lr"] = value


def build_optimizers(model: nn.Module, config: TrainingConfig) -> OptimizerBundle:
    groups = classify_parameters(model)
    dense_optimizer: torch.optim.Optimizer | None = None
    sparse_optimizer: torch.optim.Optimizer | None = None
    dense_groups: list[dict[str, Any]] = []
    if groups.decay:
        dense_groups.append({"params": groups.decay, "weight_decay": config.weight_decay})
    if groups.no_decay:
        dense_groups.append({"params": groups.no_decay, "weight_decay": 0.0})
    if dense_groups:
        requested_fused = config.fused_adamw == "on" or (
            config.fused_adamw == "auto" and next(model.parameters()).device.type == "cuda"
        )
        kwargs: dict[str, Any] = {
            "lr": config.learning_rate,
            "betas": (config.adam_beta1, config.adam_beta2),
            "eps": config.adam_epsilon,
        }
        if requested_fused:
            kwargs["fused"] = True
        try:
            dense_optimizer = torch.optim.AdamW(dense_groups, **kwargs)
        except (RuntimeError, TypeError):
            if config.fused_adamw == "on":
                raise
            kwargs.pop("fused", None)
            dense_optimizer = torch.optim.AdamW(dense_groups, **kwargs)
    if groups.sparse:
        sparse_optimizer = torch.optim.SparseAdam(
            groups.sparse,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
        )
    return OptimizerBundle(dense=dense_optimizer, sparse=sparse_optimizer)


def gradients_are_finite(parameters: tuple[nn.Parameter, ...] | list[nn.Parameter]) -> bool:
    """Check dense and sparse gradients without synchronizing each tensor separately."""

    checks: list[Tensor] = []
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        values = gradient.coalesce().values() if gradient.is_sparse else gradient
        checks.append(torch.isfinite(values).all())
    if not checks:
        return True
    return bool(torch.stack(checks).all())


def _load_optional_optimizer(
    name: str,
    optimizer: torch.optim.Optimizer | None,
    state: dict[str, Any] | None,
) -> None:
    if (optimizer is None) != (state is None):
        raise ValueError(f"checkpoint {name} optimizer topology does not match this run")
    if optimizer is not None and state is not None:
        optimizer.load_state_dict(state)


__all__ = [
    "OptimizerBundle",
    "ParameterGroups",
    "build_optimizers",
    "classify_parameters",
    "gradients_are_finite",
]
