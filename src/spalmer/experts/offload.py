"""Physical request-resident inference cache for routed expert banks.

The checkpoint-visible expert parameters remain one complete CPU bank per
layer.  A model-owned :class:`ExpertOffloadManager` mirrors exactly the global
expert ids committed by :class:`~spalmer.experts.residency.ExpertResidency`
into small, non-persistent execution tensors on the inference device.  The
same ids occupy the same logical slots in every layer, while weights remain
layer-local.

This is intentionally an inference backend.  It does not create trainable
shadow parameters and it does not alter state-dict names, so ordinary
all-resident training and existing checkpoints keep their original behavior
when the backend is disabled.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from weakref import ref

import torch
from torch import nn

from spalmer.experts.bank import MicroExpertBank

if TYPE_CHECKING:
    from spalmer.modeling import SPALMERCausalLM


@dataclass(frozen=True, slots=True)
class ExpertOffloadTelemetry:
    """Auditable placement, occupancy, and cumulative transfer counters."""

    enabled: bool
    target_device: str | None
    capacity: int
    resident_expert_ids: tuple[int, ...]
    layer_count: int
    master_devices: tuple[str, ...]
    master_pinned_by_layer: tuple[bool, ...]
    cache_devices: tuple[str, ...]
    cached_expert_ids_by_layer: tuple[tuple[int, ...], ...]
    master_bytes: int
    cache_bytes: int
    cache_index_bytes: int
    stage_operations: int
    transferred_expert_rows: int
    evicted_expert_rows: int
    transfer_bytes: int

    @property
    def occupancy(self) -> float:
        """Fraction of the configured identity capacity currently occupied."""

        return len(self.resident_expert_ids) / self.capacity if self.capacity else 0.0

    @property
    def masters_on_cpu(self) -> bool:
        """Whether every complete expert bank is physically on CPU."""

        return bool(self.master_devices) and all(device == "cpu" for device in self.master_devices)

    @property
    def masters_pinned(self) -> bool:
        """Whether every CPU expert bank is page-locked for async transfer."""

        return bool(self.master_pinned_by_layer) and all(self.master_pinned_by_layer)

    @property
    def caches_on_target(self) -> bool:
        """Whether every layer cache is on the reported inference device."""

        return (
            self.target_device is not None
            and bool(self.cache_devices)
            and all(device == self.target_device for device in self.cache_devices)
        )


class ExpertOffloadManager:
    """Synchronize one bounded resident identity set across layer-local banks."""

    def __init__(
        self,
        model: SPALMERCausalLM,
        banks: Sequence[MicroExpertBank],
        target_device: torch.device,
        *,
        capacity: int,
        non_blocking: bool,
    ) -> None:
        if not banks:
            raise ValueError("expert offload requires at least one layer-local expert bank")
        num_experts = banks[0].num_experts
        if any(bank.num_experts != num_experts for bank in banks):
            raise ValueError("every offloaded bank must expose the same global expert identities")
        self._model_ref = ref(model)
        self._banks = tuple(banks)
        self.target_device = target_device
        self.capacity = capacity
        self.non_blocking = non_blocking
        self._resident_ids: tuple[int, ...] = ()
        self._stage_operations = 0
        self._transferred_rows = 0
        self._evicted_rows = 0
        self._transfer_bytes = 0
        self._load_hook: torch.utils.hooks.RemovableHandle | None = None
        self._allow_model_apply = False

    @property
    def resident_ids(self) -> tuple[int, ...]:
        return self._resident_ids

    @property
    def allow_model_apply(self) -> bool:
        """Internal guard used during the manager's selective placement."""

        return self._allow_model_apply

    def resolve_target(self, target_device: torch.device) -> None:
        """Replace an index-less device with the concrete placed device."""

        if self._resident_ids:
            raise RuntimeError("cannot resolve the expert target after cache staging")
        self.target_device = target_device
        for bank in self._banks:
            bank._set_expert_offload_target(target_device)

    def stage(self, expert_ids: Sequence[int], *, force: bool = False) -> None:
        """Stage an exact, sorted identity set in every layer as one operation."""

        raw_ids = tuple(int(expert) for expert in expert_ids)
        desired = tuple(sorted(set(raw_ids)))
        if len(desired) != len(raw_ids):
            raise ValueError("resident expert ids must be unique")
        if not desired:
            raise ValueError("at least one resident expert id is required")
        if len(desired) > self.capacity:
            raise ValueError(
                f"{len(desired)} resident experts exceed physical cache capacity {self.capacity}"
            )
        previous = self._resident_ids
        completed: list[MicroExpertBank] = []
        deltas: list[tuple[int, int, int]] = []
        try:
            for bank in self._banks:
                deltas.append(bank._stage_expert_rows(desired, force=force))
                completed.append(bank)
        except BaseException:
            if previous:
                for bank in completed:
                    bank._stage_expert_rows(previous, force=True)
            raise

        self._resident_ids = desired
        self._stage_operations += 1
        self._transferred_rows += sum(delta[0] for delta in deltas)
        self._evicted_rows += sum(delta[1] for delta in deltas)
        self._transfer_bytes += sum(delta[2] for delta in deltas)

    def install_load_hook(self) -> None:
        """Refresh detached execution caches after master weights are loaded."""

        model = self._require_model()

        def refresh_after_load(module: nn.Module, incompatible_keys: object) -> None:
            del module, incompatible_keys
            if self._resident_ids:
                self.stage(self._resident_ids, force=True)

        self._load_hook = model.register_load_state_dict_post_hook(refresh_after_load)

    def remove_load_hook(self) -> None:
        hook = self._load_hook
        if hook is not None:
            hook.remove()
            self._load_hook = None

    def telemetry(self) -> ExpertOffloadTelemetry:
        """Return a synchronization-free snapshot of physical cache state."""

        master_devices = tuple(str(bank.expert_master_device) for bank in self._banks)
        cache_devices = tuple(
            "" if bank.expert_cache_device is None else str(bank.expert_cache_device)
            for bank in self._banks
        )
        return ExpertOffloadTelemetry(
            enabled=True,
            target_device=str(self.target_device),
            capacity=self.capacity,
            resident_expert_ids=self._resident_ids,
            layer_count=len(self._banks),
            master_devices=master_devices,
            master_pinned_by_layer=tuple(
                bank.expert_masters_pinned for bank in self._banks
            ),
            cache_devices=cache_devices,
            cached_expert_ids_by_layer=tuple(
                bank.cached_expert_ids for bank in self._banks
            ),
            master_bytes=sum(bank.expert_master_bytes for bank in self._banks),
            cache_bytes=sum(bank.expert_cache_bytes for bank in self._banks),
            cache_index_bytes=sum(bank.expert_cache_index_bytes for bank in self._banks),
            stage_operations=self._stage_operations,
            transferred_expert_rows=self._transferred_rows,
            evicted_expert_rows=self._evicted_rows,
            transfer_bytes=self._transfer_bytes,
        )

    def clear(self) -> None:
        """Drop all device caches and unregister request/load callbacks."""

        self.remove_load_hook()
        model = self._model_ref()
        if model is not None and model.residency is not None:
            model.residency._detach_offload_backend(self)
        for bank in self._banks:
            bank._clear_expert_offload()
        self._resident_ids = ()

    def _require_model(self) -> SPALMERCausalLM:
        model = self._model_ref()
        if model is None:
            raise RuntimeError("the model owning this expert cache no longer exists")
        return model


def enable_expert_offload(
    model: SPALMERCausalLM,
    device: torch.device | str,
    *,
    cache_size: int | None = None,
    resident_ids: Sequence[int] | None = None,
    non_blocking: bool = True,
    pin_memory: bool | None = None,
) -> ExpertOffloadTelemetry:
    """Place shared inference weights on ``device`` and cache only resident experts.

    Call this *instead of* ``model.to(device)``.  Complete expert master banks
    stay on CPU; all other model weights and buffers move normally.  The
    default fixed resident set is the configured request minimum (or the
    current bounded set), so inference is immediately usable even without the
    dynamic residency controller.  Dynamic expansion and rollback reuse the
    same cache through the shared residency commit boundary.
    """

    if model.training:
        raise RuntimeError("expert offload is inference-only; call model.eval() first")
    if model.residency is None:
        raise TypeError("expert offload requires a model with shared ExpertResidency")
    existing = getattr(model, "_expert_offload_manager", None)
    if existing is not None:
        raise RuntimeError("expert offload is already enabled")
    target = torch.device(device)
    if target.type == "cpu":
        raise ValueError("expert offload needs a non-CPU inference device")
    should_pin = target.type == "cuda" if pin_memory is None else bool(pin_memory)
    if should_pin and target.type != "cuda":
        raise ValueError("pin_memory is currently supported only for CUDA expert offload")

    banks = _expert_banks(model)
    config = model.residency.config
    capacity = config.resident_cap if cache_size is None else int(cache_size)
    if not model.residency.active_experts <= capacity <= config.num_experts:
        raise ValueError(
            f"cache_size must lie in [{model.residency.active_experts}, "
            f"{config.num_experts}]; got {capacity}"
        )
    if resident_ids is None:
        if model.residency.is_full:
            from spalmer.experts.residency import default_resident_ids

            initial_count = max(config.min_resident_experts, model.residency.active_experts)
            initial = default_resident_ids(model, initial_count)
        else:
            initial = model.residency.ids
    else:
        initial = tuple(resident_ids)
    initial = tuple(sorted({int(expert) for expert in initial}))
    if len(initial) < model.residency.active_experts:
        raise ValueError(
            f"the initial cache needs at least active_experts={model.residency.active_experts} ids"
        )
    if len(initial) > capacity:
        raise ValueError(
            f"{len(initial)} initial residents exceed physical cache capacity {capacity}"
        )
    if model.residency.request_open:
        raise RuntimeError("cannot enable expert offload during an open residency request")

    manager = ExpertOffloadManager(
        model,
        banks,
        target,
        capacity=capacity,
        non_blocking=non_blocking,
    )
    setattr(model, "_expert_offload_manager", manager)
    try:
        for bank in banks:
            bank._prepare_expert_offload(
                target,
                capacity=capacity,
                non_blocking=non_blocking,
                pin_memory=should_pin,
            )
        model.residency._attach_offload_backend(manager)
        manager._allow_model_apply = True
        try:
            model.to(device=target)
        finally:
            manager._allow_model_apply = False
        concrete_target = model.lm_head.weight.device
        manager.resolve_target(concrete_target)
        model.residency.set(initial)
        manager.install_load_hook()
        _validate_physical_placement(model, banks, concrete_target)
    except BaseException:
        manager.clear()
        setattr(model, "_expert_offload_manager", None)
        raise
    return manager.telemetry()


def disable_expert_offload(
    model: SPALMERCausalLM,
    *,
    device: torch.device | str = "cpu",
) -> None:
    """Drop caches, restore full logical residency, and place the whole model."""

    manager = getattr(model, "_expert_offload_manager", None)
    if manager is None:
        return
    if model.residency is not None:
        model.residency.end_request()
    manager.clear()
    setattr(model, "_expert_offload_manager", None)
    if model.residency is not None:
        model.residency.reset()
    model.to(device=torch.device(device))


def _expert_banks(model: SPALMERCausalLM) -> tuple[MicroExpertBank, ...]:
    banks: list[MicroExpertBank] = []
    for block in model.backbone.blocks:
        bank = getattr(block.channel_mixer, "experts", None)
        if isinstance(bank, MicroExpertBank):
            banks.append(bank)
    if not banks:
        raise TypeError("model has no routed MicroExpertBank layers to offload")
    return tuple(banks)


def _validate_physical_placement(
    model: SPALMERCausalLM,
    banks: Sequence[MicroExpertBank],
    target: torch.device,
) -> None:
    expert_parameter_ids = {
        id(parameter)
        for bank in banks
        for parameter in bank.parameters(recurse=False)
    }
    misplaced_masters = [
        str(bank.expert_master_device)
        for bank in banks
        if bank.expert_master_device.type != "cpu"
    ]
    if misplaced_masters:
        raise RuntimeError(f"expert masters escaped CPU placement: {misplaced_masters}")
    misplaced_shared = [
        name
        for name, parameter in model.named_parameters()
        if id(parameter) not in expert_parameter_ids and parameter.device != target
    ]
    if misplaced_shared:
        raise RuntimeError(
            f"non-expert inference parameters did not reach {target}: {misplaced_shared}"
        )
    incoherent = [
        index
        for index, bank in enumerate(banks)
        if bank.cached_expert_ids != model.residency.ids
        or bank.expert_cache_device != target
    ]
    if incoherent:
        raise RuntimeError(f"layer expert caches are incoherent: {incoherent}")


__all__ = [
    "ExpertOffloadManager",
    "ExpertOffloadTelemetry",
    "disable_expert_offload",
    "enable_expert_offload",
]
