"""Bounded physical inference caches for routed expert banks.

The checkpoint-visible expert parameters remain one complete CPU bank per
layer. In the default paged mode, routers remain unrestricted and each layer
stages only the expert rows selected at that layer. Long prefills are tiled
when their distinct selections exceed cache capacity. The former mode remains
available with ``paging=False``: it mirrors one request-global identity set
into every layer and restricts routing to those residents.

The published steady-state cache of each layer contains at most ``capacity``
rows. Replacement is transactional: new tensors are populated before they are
published, so a transfer failure leaves the old cache executable. During that
brief replacement window both old and new tensors may exist (at most roughly
``2 * capacity`` rows per layer).

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
    paging: bool
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
        """Largest per-layer fraction of physical cache capacity occupied."""

        if not self.capacity:
            return 0.0
        if self.paging:
            occupied = max((len(ids) for ids in self.cached_expert_ids_by_layer), default=0)
        else:
            occupied = len(self.resident_expert_ids)
        return occupied / self.capacity

    @property
    def mode(self) -> str:
        """Human-readable cache policy name."""

        return "paged" if self.paging else "resident"

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
    """Own bounded layer caches in paged or request-global resident mode."""

    def __init__(
        self,
        model: SPALMERCausalLM,
        banks: Sequence[MicroExpertBank],
        target_device: torch.device,
        *,
        capacity: int,
        non_blocking: bool,
        paging: bool,
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
        self.paging = paging
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

        if self.paging:
            raise RuntimeError(
                "request-global staging is unavailable in paged expert mode"
            )

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

    def warm(self, expert_ids: Sequence[int]) -> None:
        """Optionally seed every independent layer cache with the same ids."""

        if not self.paging:
            raise RuntimeError("cache warming is only available in paged expert mode")
        desired = tuple(sorted({int(expert) for expert in expert_ids}))
        if not desired:
            return
        if len(desired) > self.capacity:
            raise ValueError(
                f"{len(desired)} warm experts exceed physical cache capacity {self.capacity}"
            )
        for bank in self._banks:
            bank._stage_expert_rows(desired)

    def install_load_hook(self) -> None:
        """Refresh detached execution caches after master weights are loaded."""

        model = self._require_model()

        def refresh_after_load(module: nn.Module, incompatible_keys: object) -> None:
            del module, incompatible_keys
            if self.paging:
                for bank in self._banks:
                    if bank.cached_expert_ids:
                        bank._stage_expert_rows(bank.cached_expert_ids, force=True)
            elif self._resident_ids:
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
        if self.paging:
            counters = tuple(bank.expert_offload_counters for bank in self._banks)
            stage_operations = sum(counter[0] for counter in counters)
            transferred_rows = sum(counter[1] for counter in counters)
            evicted_rows = sum(counter[2] for counter in counters)
            transfer_bytes = sum(counter[3] for counter in counters)
        else:
            stage_operations = self._stage_operations
            transferred_rows = self._transferred_rows
            evicted_rows = self._evicted_rows
            transfer_bytes = self._transfer_bytes
        return ExpertOffloadTelemetry(
            enabled=True,
            paging=self.paging,
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
            stage_operations=stage_operations,
            transferred_expert_rows=transferred_rows,
            evicted_expert_rows=evicted_rows,
            transfer_bytes=transfer_bytes,
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
    paging: bool = True,
) -> ExpertOffloadTelemetry:
    """Place shared weights on ``device`` and bound layer-local expert caches.

    Call this *instead of* ``model.to(device)``.  Complete expert master banks
    stay on CPU; all other model weights and buffers move normally. By default,
    routing scores/selects from the full trained pool and selected layer rows
    are paged on demand. ``resident_ids`` then only warms the physical caches.
    Set ``paging=False`` to retain legacy request-global masked routing.
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
    if model.residency.request_open:
        raise RuntimeError("cannot enable expert offload during an open residency request")
    previous_residency = model.residency.snapshot_state()
    capacity = config.resident_cap if cache_size is None else int(cache_size)
    minimum_capacity = 1 if paging else model.residency.active_experts
    if not minimum_capacity <= capacity <= config.num_experts:
        raise ValueError(
            f"cache_size must lie in [{minimum_capacity}, {config.num_experts}]; "
            f"got {capacity}"
        )
    if paging:
        initial = () if resident_ids is None else tuple(resident_ids)
    elif resident_ids is None:
        if model.residency.is_full:
            from spalmer.experts.residency import default_resident_ids

            initial_count = max(config.min_resident_experts, model.residency.active_experts)
            initial = default_resident_ids(model, initial_count)
        else:
            initial = model.residency.ids
    else:
        initial = tuple(resident_ids)
    initial = tuple(sorted({int(expert) for expert in initial}))
    if not paging and len(initial) < model.residency.active_experts:
        raise ValueError(
            f"the initial cache needs at least active_experts={model.residency.active_experts} ids"
        )
    if len(initial) > capacity:
        raise ValueError(
            f"{len(initial)} initial residents exceed physical cache capacity {capacity}"
        )
    manager = ExpertOffloadManager(
        model,
        banks,
        target,
        capacity=capacity,
        non_blocking=non_blocking,
        paging=paging,
    )
    setattr(model, "_expert_offload_manager", manager)
    try:
        if paging:
            # Physical cache policy must not inherit a stale logical
            # restriction. No residency backend is attached in this mode:
            # router eligibility and cache placement have independent
            # lifecycles.
            model.residency.reset()
        for bank in banks:
            bank._prepare_expert_offload(
                target,
                capacity=capacity,
                non_blocking=non_blocking,
                pin_memory=should_pin,
                paging=paging,
            )
        if not paging:
            model.residency._attach_offload_backend(manager)
        manager._allow_model_apply = True
        try:
            model.to(device=target)
        finally:
            manager._allow_model_apply = False
        concrete_target = model.lm_head.weight.device
        manager.resolve_target(concrete_target)
        if paging:
            manager.warm(initial)
        else:
            model.residency.set(initial)
        manager.install_load_hook()
        _validate_physical_placement(model, banks, concrete_target)
    except BaseException:
        manager.clear()
        setattr(model, "_expert_offload_manager", None)
        model.residency.restore_state(previous_residency)
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
    manager = getattr(model, "_expert_offload_manager", None)
    paging = manager is not None and manager.paging
    incoherent = []
    for index, bank in enumerate(banks):
        wrong_device = bank.expert_cache_device != target
        wrong_policy = paging != bank.expert_paging_enabled
        over_capacity = len(bank.cached_expert_ids) > (bank.expert_cache_capacity or 0)
        wrong_global_set = not paging and bank.cached_expert_ids != model.residency.ids
        if wrong_device or wrong_policy or over_capacity or wrong_global_set:
            incoherent.append(index)
    if incoherent:
        raise RuntimeError(f"layer expert caches are incoherent: {incoherent}")
    if paging and not model.residency.is_full:
        raise RuntimeError("paged expert offload must start with full-pool routing eligibility")


__all__ = [
    "ExpertOffloadManager",
    "ExpertOffloadTelemetry",
    "disable_expert_offload",
    "enable_expert_offload",
]
