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


class BF16MasterAdamW(torch.optim.Optimizer):
    """AdamW with BF16 parameters, FP32 moments, and no FP32 weight copy.

    Updates are formed a bounded chunk at a time in FP32, then written directly
    to the sole BF16 parameter payload. Optional stochastic rounding preserves
    sub-ULP updates in expectation without retaining a full-sized FP32 master.
    When ``state_offload='cpu'``, moment tensors remain in CPU storage and only
    one bounded pair of moment chunks visits the parameter device at a time.
    """

    def __init__(
        self,
        params,
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        stochastic_rounding: bool,
        update_chunk_size: int,
        state_offload: str = "none",
    ) -> None:
        if lr < 0:
            raise ValueError("learning rate cannot be negative")
        if eps <= 0:
            raise ValueError("epsilon must be positive")
        if update_chunk_size <= 0:
            raise ValueError("update_chunk_size must be positive")
        if state_offload not in {"none", "cpu"}:
            raise ValueError("state_offload must be 'none' or 'cpu'")
        beta1, beta2 = betas
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Adam betas must be in [0, 1)")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "stochastic_rounding": stochastic_rounding,
            "update_chunk_size": update_chunk_size,
            "state_offload": state_offload,
        }
        super().__init__(params, defaults)
        self._state_offload = state_offload

    @property
    def optimizer_state_offload(self) -> str:
        """Configured moment placement policy (``none`` or ``cpu``)."""

        return self._state_offload

    def state_placement(self) -> dict[str, Any]:
        """Return observed optimizer-state placement for receipts and UI telemetry."""

        parameters = [
            parameter
            for group in self.param_groups
            for parameter in group["params"]
        ]
        moments = [
            value
            for state in self.state.values()
            for name in ("exp_avg", "exp_avg_sq")
            if isinstance((value := state.get(name)), Tensor)
        ]
        cpu_moments = [value for value in moments if value.device.type == "cpu"]
        pinned = [value.is_pinned() for value in cpu_moments]
        return {
            "policy": self._state_offload,
            "configured_device": "cpu" if self._state_offload == "cpu" else "parameter",
            "moment_dtype": "float32",
            "initialized": bool(moments),
            "moment_tensors": len(moments),
            "moment_bytes": sum(value.numel() * value.element_size() for value in moments),
            "devices": sorted({str(value.device) for value in moments}),
            "cpu_pinned": all(pinned) if pinned else None,
            "parameter_devices": sorted({str(value.device) for value in parameters}),
            "parameter_dtypes": sorted(
                {str(value.dtype).removeprefix("torch.") for value in parameters}
            ),
            "update_chunk_sizes": sorted(
                {int(group["update_chunk_size"]) for group in self.param_groups}
            ),
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        pending_cuda_devices: set[torch.device] = set()
        try:
            for group in self.param_groups:
                if group.get("state_offload", "none") != self._state_offload:
                    raise RuntimeError(
                        "optimizer parameter groups disagree on state offload policy"
                    )
                lr = float(group["lr"])
                beta1, beta2 = group["betas"]
                eps = float(group["eps"])
                weight_decay = float(group.get("weight_decay", 0.0))
                stochastic = bool(group["stochastic_rounding"])
                chunk_size = int(group["update_chunk_size"])
                for parameter in group["params"]:
                    gradient = parameter.grad
                    if gradient is None:
                        continue
                    if gradient.is_sparse:
                        raise RuntimeError("BF16MasterAdamW does not accept sparse gradients")
                    if parameter.dtype != torch.bfloat16:
                        raise TypeError(
                            "BF16MasterAdamW requires BF16 parameters; "
                            f"received {parameter.dtype}"
                        )
                    state = self.state[parameter]
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = _new_moment_tensor(
                            parameter,
                            state_offload=self._state_offload,
                        )
                        state["exp_avg_sq"] = _new_moment_tensor(
                            parameter,
                            state_offload=self._state_offload,
                        )
                    _validate_moment_state(
                        parameter,
                        state,
                        state_offload=self._state_offload,
                    )
                    state["step"] += 1
                    step = int(state["step"])
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    bias_correction1 = 1.0 - beta1**step
                    bias_correction2 = 1.0 - beta2**step
                    step_size = lr / bias_correction1
                    correction2_sqrt = bias_correction2**0.5

                    parameter_flat = parameter.view(-1)
                    gradient_flat = gradient.contiguous().view(-1)
                    exp_avg_flat = exp_avg.view(-1)
                    exp_avg_sq_flat = exp_avg_sq.view(-1)
                    for start in range(0, parameter.numel(), chunk_size):
                        stop = min(parameter.numel(), start + chunk_size)
                        grad = gradient_flat[start:stop].float()
                        mean_storage = exp_avg_flat[start:stop]
                        variance_storage = exp_avg_sq_flat[start:stop]
                        moments_staged = mean_storage.device != parameter.device
                        mean, non_blocking = _stage_moment_chunk(
                            mean_storage,
                            parameter.device,
                        )
                        variance, variance_non_blocking = _stage_moment_chunk(
                            variance_storage,
                            parameter.device,
                        )
                        non_blocking = non_blocking and variance_non_blocking
                        if non_blocking:
                            pending_cuda_devices.add(parameter.device)
                        mean.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                        variance.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                        updated = parameter_flat[start:stop].float()
                        if weight_decay:
                            updated.mul_(1.0 - lr * weight_decay)
                        denominator = variance.sqrt().div_(correction2_sqrt).add_(eps)
                        updated.addcdiv_(mean, denominator, value=-step_size)
                        rounded = (
                            _stochastic_round_bfloat16(updated)
                            if stochastic
                            else updated.to(torch.bfloat16)
                        )
                        parameter_flat[start:stop].copy_(rounded)
                        if moments_staged:
                            mean_storage.copy_(mean, non_blocking=non_blocking)
                            variance_storage.copy_(variance, non_blocking=non_blocking)
        finally:
            # A CPU consumer (checkpointing, telemetry, or the next host-side
            # update) cannot observe pinned destinations until D2H copies end.
            for device in pending_cuda_devices:
                torch.cuda.current_stream(device).synchronize()
        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # PyTorch normally casts floating optimizer state to the parameter's
        # dtype/device. Strip the two moments before delegating so resume never
        # creates full-sized BF16 moment copies on the accelerator, then restore
        # their exact FP32 values directly into the configured storage lane.
        saved_groups = state_dict.get("param_groups", [])
        saved_state = state_dict.get("state", {})
        stripped_groups = [dict(group) for group in saved_groups]
        for group in stripped_groups:
            group["state_offload"] = self._state_offload
        stripped_state = {
            key: {
                name: value
                for name, value in value.items()
                if name not in {"exp_avg", "exp_avg_sq"}
            }
            for key, value in saved_state.items()
        }
        super().load_state_dict(
            {
                **state_dict,
                "state": stripped_state,
                "param_groups": stripped_groups,
            }
        )
        saved_moments: list[tuple[nn.Parameter, dict[str, Any]]] = []
        for saved_group, current_group in zip(saved_groups, self.param_groups, strict=True):
            saved_moments.extend(
                (
                    parameter,
                    saved_state.get(saved_id, {}),
                )
                for saved_id, parameter in zip(
                    saved_group.get("params", []),
                    current_group.get("params", []),
                    strict=True,
                )
            )
        for parameter, saved_state in saved_moments:
            state = self.state.get(parameter)
            if state is None:
                continue
            for name in ("exp_avg", "exp_avg_sq"):
                value = saved_state.get(name)
                if isinstance(value, Tensor):
                    state[name] = _restore_moment_tensor(
                        value,
                        parameter,
                        state_offload=self._state_offload,
                    )
            if state:
                _validate_moment_state(
                    parameter,
                    state,
                    state_offload=self._state_offload,
                )


def _new_moment_tensor(parameter: nn.Parameter, *, state_offload: str) -> Tensor:
    if state_offload == "none":
        return torch.zeros_like(
            parameter,
            dtype=torch.float32,
            memory_format=torch.preserve_format,
        )
    prefer_pinned = parameter.device.type == "cuda"
    try:
        return torch.zeros(
            tuple(parameter.shape),
            dtype=torch.float32,
            device="cpu",
            pin_memory=prefer_pinned,
        )
    except (RuntimeError, TypeError):
        # CUDA can be present while the pinned allocator is unavailable or its
        # quota is exhausted. Pageable CPU state remains correct, just slower.
        return torch.zeros(tuple(parameter.shape), dtype=torch.float32, device="cpu")


def _stage_moment_chunk(moment: Tensor, parameter_device: torch.device) -> tuple[Tensor, bool]:
    if moment.device == parameter_device:
        return moment, False
    non_blocking = (
        parameter_device.type == "cuda"
        and moment.device.type == "cpu"
        and moment.is_pinned()
    )
    return moment.to(device=parameter_device, non_blocking=non_blocking), non_blocking


def _restore_moment_tensor(
    value: Tensor,
    parameter: nn.Parameter,
    *,
    state_offload: str,
) -> Tensor:
    if tuple(value.shape) != tuple(parameter.shape):
        raise ValueError("loaded BF16MasterAdamW moment shape does not match its parameter")
    if state_offload == "none":
        return value.detach().to(device=parameter.device, dtype=torch.float32, copy=True)
    restored = _new_moment_tensor(parameter, state_offload="cpu")
    non_blocking = value.device.type == "cuda" and restored.is_pinned()
    restored.copy_(value.detach(), non_blocking=non_blocking)
    if non_blocking:
        torch.cuda.current_stream(value.device).synchronize()
    return restored


def _validate_moment_state(
    parameter: nn.Parameter,
    state: dict[str, Any],
    *,
    state_offload: str,
) -> None:
    expected_device = torch.device("cpu") if state_offload == "cpu" else parameter.device
    for name in ("exp_avg", "exp_avg_sq"):
        value = state.get(name)
        if not isinstance(value, Tensor):
            raise ValueError(f"BF16MasterAdamW state is missing {name}")
        if value.dtype != torch.float32:
            raise ValueError("BF16MasterAdamW moments must remain FP32")
        if value.device != expected_device:
            raise ValueError(
                f"BF16MasterAdamW {name} must reside on {expected_device}, found {value.device}"
            )
        if tuple(value.shape) != tuple(parameter.shape):
            raise ValueError(f"BF16MasterAdamW {name} shape does not match its parameter")


def _stochastic_round_bfloat16(values: Tensor) -> Tensor:
    """Unbiased in-range FP32-to-BF16 rounding with symmetric saturation."""

    maximum = torch.finfo(torch.bfloat16).max
    bounded = values.clamp(min=-maximum, max=maximum)
    nearest = bounded.to(torch.bfloat16)
    nearest_fp32 = nearest.float()
    negative_infinity = torch.full_like(nearest, float("-inf"))
    positive_infinity = torch.full_like(nearest, float("inf"))
    lower = torch.where(
        nearest_fp32 > bounded,
        torch.nextafter(nearest, negative_infinity),
        nearest,
    )
    upper = torch.where(
        nearest_fp32 < bounded,
        torch.nextafter(nearest, positive_infinity),
        nearest,
    )
    lower_fp32 = lower.float()
    upper_fp32 = upper.float()
    span = upper_fp32 - lower_fp32
    probability_upper = torch.where(
        span > 0,
        ((bounded - lower_fp32) / span).clamp(0.0, 1.0),
        torch.zeros_like(bounded),
    )
    rounded = torch.where(torch.rand_like(probability_upper) < probability_upper, upper, lower)
    return torch.where(torch.isfinite(values), rounded, values.to(torch.bfloat16))


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

    def state_placement(self) -> dict[str, dict[str, Any] | None]:
        """Report configured and observed placement for every optimizer lane."""

        return {
            "dense": _optimizer_state_placement(self.dense),
            "sparse": _optimizer_state_placement(self.sparse),
        }

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
        expected_dtype = (
            torch.bfloat16 if config.parameter_dtype == "bfloat16" else torch.float32
        )
        wrong_dtype = next(
            (
                parameter.dtype
                for parameter in (*groups.decay, *groups.no_decay)
                if parameter.dtype != expected_dtype
            ),
            None,
        )
        if wrong_dtype is not None:
            raise TypeError(
                f"optimizer expected {expected_dtype} parameters, found {wrong_dtype}; "
                "initialize or cast the model through the training configuration first"
            )
        if config.parameter_dtype == "bfloat16":
            dense_optimizer = BF16MasterAdamW(
                dense_groups,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon,
                stochastic_rounding=config.stochastic_parameter_rounding,
                update_chunk_size=config.optimizer_update_chunk_size,
                state_offload=config.optimizer_state_offload,
            )
        else:
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
        if config.parameter_dtype == "bfloat16":
            raise ValueError(
                "BF16 master training currently requires dense PLE gradients so its FP32 "
                "moment contract is preserved"
            )
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


def _optimizer_state_placement(
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any] | None:
    if optimizer is None:
        return None
    reporter = getattr(optimizer, "state_placement", None)
    if callable(reporter):
        return dict(reporter())
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    moments = [
        value
        for state in optimizer.state.values()
        for name in ("exp_avg", "exp_avg_sq")
        if isinstance((value := state.get(name)), Tensor)
    ]
    cpu_moments = [value for value in moments if value.device.type == "cpu"]
    pinned = [value.is_pinned() for value in cpu_moments]
    return {
        "policy": "none",
        "configured_device": "parameter",
        "moment_dtype": (
            None if not moments else str(moments[0].dtype).removeprefix("torch.")
        ),
        "initialized": bool(moments),
        "moment_tensors": len(moments),
        "moment_bytes": sum(value.numel() * value.element_size() for value in moments),
        "devices": sorted({str(value.device) for value in moments}),
        "cpu_pinned": all(pinned) if pinned else None,
        "parameter_devices": sorted({str(value.device) for value in parameters}),
        "parameter_dtypes": sorted(
            {str(value.dtype).removeprefix("torch.") for value in parameters}
        ),
        "update_chunk_sizes": None,
    }


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
    "BF16MasterAdamW",
    "OptimizerBundle",
    "ParameterGroups",
    "build_optimizers",
    "classify_parameters",
    "gradients_are_finite",
]
