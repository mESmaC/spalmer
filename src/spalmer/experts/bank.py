"""Tensorized bank of small gated-MLP experts (SPALMER ledger C07).

All experts live in three stacked parameters (gate, up, down), one per
projection. Two execution paths produce the same routed update:

- ``grouped`` (default): pairs are sorted by expert, then experts are placed in
  power-of-two count buckets. Each bucket executes three batched matmuls over
  ``[bucket_groups, bucket_capacity, d_model]``. There is no per-expert Python
  loop, and the sum of padded slots is strictly less than twice the number of
  real routed pairs. A single heavily used expert therefore cannot force every
  other expert to allocate its global maximum capacity.
- ``loop``: the per-expert path.  BF16 uses ordinary PyTorch GEMMs; native
  low-precision providers use this path until a verified grouped kernel is
  integrated.

Inference paging is an execution detail below routing.  The router may select
any expert in the trained pool; this bank divides the selected identities into
cache-sized tiles, stages only one tile of layer-local rows at a time, and
accumulates every routed contribution.  Consequently cache capacity never
changes the logical top-k decision.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from spalmer.experts.config import MicroExpertsConfig
from spalmer.experts.native import (
    native_expert_matmul,
    native_weight_reconstruction_error,
)
from spalmer.precision import detect_precision_capabilities


class MicroExpertBank(nn.Module):
    """A layer-local bank of ``num_experts`` small gated-MLP experts.

    Expert ``e`` computes ``down(silu(x @ gate_e) * (x @ up_e))``. Expert
    identity (the index ``e``) is coherent across layers; the weights are
    local to the owning layer.
    """

    def __init__(self, config: MicroExpertsConfig) -> None:
        super().__init__()
        self.config = config
        num_experts = config.num_experts
        d_model = config.d_model
        inter_dim = config.resolved_inter_dim
        std = config.initializer_range
        self.gate_proj = nn.Parameter(torch.randn(num_experts, d_model, inter_dim) * std)
        self.up_proj = nn.Parameter(torch.randn(num_experts, d_model, inter_dim) * std)
        self.down_proj = nn.Parameter(torch.randn(num_experts, inter_dim, d_model) * std)
        # Inference offload deliberately keeps the checkpoint-visible master
        # parameters above on CPU.  These attributes are plain tensors rather
        # than buffers: the resident execution copy is request-local cache
        # state and must never appear in ``state_dict`` or be moved by a later
        # recursive ``Module.to`` call.
        self._offload_target: torch.device | None = None
        self._offload_capacity: int | None = None
        self._offload_non_blocking = True
        self._offload_paging = False
        self._cached_expert_ids: tuple[int, ...] = ()
        self._cached_slot_by_id: dict[int, int] = {}
        # Oldest-to-newest identity order used only to retain useful rows when
        # a page does not fill the physical cache.  Identity-to-slot placement
        # remains independent and layer-local.
        self._cached_lru: list[int] = []
        self._cached_slot_map: Tensor | None = None
        self._cached_gate_proj: Tensor | None = None
        self._cached_up_proj: Tensor | None = None
        self._cached_down_proj: Tensor | None = None
        self._offload_stage_operations = 0
        self._offload_transferred_rows = 0
        self._offload_evicted_rows = 0
        self._offload_transfer_bytes = 0
        self._last_paged_quantization_ids: tuple[int, ...] | None = None
        self._last_paged_quantization_error: Tensor | None = None
        # Construction happens before callers move a model to its execution
        # device. Validate lazily against the actual hidden-state device and
        # cache that exact result; process-default CUDA is not authoritative.
        self._native_provider_by_device: dict[str, str] = {}
        self._native_provider_id = "unvalidated"

    @property
    def num_experts(self) -> int:
        return self.config.num_experts

    @property
    def parameters_per_expert(self) -> int:
        """Exact parameter count of one expert identity in this layer."""

        return self.gate_proj[0].numel() + self.up_proj[0].numel() + self.down_proj[0].numel()

    @property
    def expert_offload_enabled(self) -> bool:
        """Whether this bank executes from a request-resident device cache."""

        return self._offload_target is not None

    @property
    def expert_paging_enabled(self) -> bool:
        """Whether selected experts are paged independently in this layer."""

        return self.expert_offload_enabled and self._offload_paging

    @property
    def expert_cache_capacity(self) -> int | None:
        """Maximum number of expert rows in the physical execution cache."""

        return self._offload_capacity

    @property
    def expert_offload_counters(self) -> tuple[int, int, int, int]:
        """Layer-local stages, transferred rows, evictions, and transfer bytes."""

        return (
            self._offload_stage_operations,
            self._offload_transferred_rows,
            self._offload_evicted_rows,
            self._offload_transfer_bytes,
        )

    @property
    def cached_expert_ids(self) -> tuple[int, ...]:
        """Global expert identities currently staged in this layer."""

        return self._cached_expert_ids

    @property
    def expert_cache_device(self) -> torch.device | None:
        """Device holding resident execution rows, or ``None`` when disabled."""

        return self._offload_target

    @property
    def expert_master_device(self) -> torch.device:
        """Device holding the complete checkpoint-visible expert bank."""

        devices = {parameter.device for parameter in self.parameters(recurse=False)}
        if len(devices) != 1:
            raise RuntimeError(
                f"expert master projections span devices: {sorted(map(str, devices))}"
            )
        return next(iter(devices))

    @property
    def expert_master_bytes(self) -> int:
        """Physical bytes in the complete persistent expert bank."""

        return sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.parameters(recurse=False)
        )

    @property
    def expert_masters_pinned(self) -> bool:
        """Whether every CPU master projection uses page-locked storage."""

        parameters = tuple(self.parameters(recurse=False))
        return bool(parameters) and all(parameter.is_pinned() for parameter in parameters)

    @property
    def expert_cache_bytes(self) -> int:
        """Physical bytes in staged expert weights (excluding the tiny id map)."""

        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self._cached_gate_proj,
                self._cached_up_proj,
                self._cached_down_proj,
            )
            if tensor is not None
        )

    @property
    def expert_cache_index_bytes(self) -> int:
        """Physical bytes in the global-id-to-cache-slot lookup."""

        mapping = self._cached_slot_map
        return 0 if mapping is None else mapping.numel() * mapping.element_size()

    def _apply(self, fn, recurse: bool = True):
        # ``SPALMERCausalLM.enable_expert_offload`` uses an ordinary recursive
        # model placement for the shared base.  Once prepared, an expert bank
        # must opt out or that operation would first materialize all expert
        # rows on the accelerator and defeat offload before the cache exists.
        if self.expert_offload_enabled:
            return self
        return super()._apply(fn, recurse=recurse)

    @torch.no_grad()
    def _prepare_expert_offload(
        self,
        target_device: torch.device,
        *,
        capacity: int,
        non_blocking: bool,
        pin_memory: bool,
        paging: bool = False,
    ) -> None:
        """Keep masters on CPU and prepare an empty inference cache."""

        if self.training:
            raise RuntimeError("expert offload is inference-only; put the model in eval mode first")
        if self.expert_offload_enabled:
            raise RuntimeError("expert offload is already enabled for this bank")
        if not 1 <= capacity <= self.num_experts:
            raise ValueError(
                f"expert cache capacity must lie in [1, {self.num_experts}]; got {capacity}"
            )
        # Do this before setting ``_offload_target`` so our _apply override
        # does not intercept the CPU migration.  Parameter identity and names
        # remain unchanged, preserving optimizers/checkpoints after offload is
        # explicitly disabled.
        self.to(device="cpu")
        if pin_memory:
            for parameter in self.parameters(recurse=False):
                try:
                    parameter.data = parameter.detach().pin_memory()
                    if parameter.grad is not None:
                        parameter.grad.data = parameter.grad.detach().pin_memory()
                except RuntimeError:
                    # Page locking is an optimization, not a correctness
                    # requirement. Telemetry reports the resulting mixed or
                    # pageable state truthfully and ``Tensor.to`` falls back
                    # to a synchronous host transfer where necessary.
                    break
        self._offload_target = target_device
        self._offload_capacity = capacity
        self._offload_non_blocking = non_blocking
        self._offload_paging = paging
        self._offload_stage_operations = 0
        self._offload_transferred_rows = 0
        self._offload_evicted_rows = 0
        self._offload_transfer_bytes = 0

    @torch.no_grad()
    def _stage_expert_rows(
        self,
        expert_ids: tuple[int, ...],
        *,
        force: bool = False,
    ) -> tuple[int, int, int]:
        """Atomically stage exact global ids and return rows-in, rows-out, bytes-in."""

        target = self._offload_target
        capacity = self._offload_capacity
        if target is None or capacity is None:
            raise RuntimeError("expert offload has not been prepared for this bank")
        if not expert_ids:
            raise ValueError("at least one expert must be staged")
        if len(expert_ids) > capacity:
            raise ValueError(
                f"{len(expert_ids)} resident experts exceed cache capacity {capacity}"
            )
        if tuple(sorted(set(expert_ids))) != expert_ids:
            raise ValueError("staged expert ids must be unique and sorted")
        if expert_ids[0] < 0 or expert_ids[-1] >= self.num_experts:
            raise ValueError(f"expert ids must lie in [0, {self.num_experts})")
        if any(parameter.device.type != "cpu" for parameter in self.parameters(recurse=False)):
            raise RuntimeError("offloaded expert master parameters must remain on CPU")
        if not force and expert_ids == self._cached_expert_ids:
            return 0, 0, 0

        old_ids = self._cached_expert_ids
        old_slots = self._cached_slot_by_id
        retained = () if force else tuple(expert for expert in expert_ids if expert in old_slots)
        incoming = tuple(expert for expert in expert_ids if expert not in set(retained))
        evicted = tuple(expert for expert in old_ids if expert not in set(expert_ids))
        new_slots = {expert: slot for slot, expert in enumerate(expert_ids)}

        def stage_projection(parameter: Tensor, cached: Tensor | None) -> Tensor:
            shape = (len(expert_ids), *parameter.shape[1:])
            staged = torch.empty(shape, dtype=parameter.dtype, device=target)
            if retained:
                assert cached is not None
                old_index = torch.tensor(
                    [old_slots[expert] for expert in retained],
                    dtype=torch.long,
                    device=target,
                )
                new_index = torch.tensor(
                    [new_slots[expert] for expert in retained],
                    dtype=torch.long,
                    device=target,
                )
                staged.index_copy_(0, new_index, cached.index_select(0, old_index))
            if incoming:
                source_index = torch.tensor(incoming, dtype=torch.long, device=parameter.device)
                transferred = parameter.detach().index_select(0, source_index).to(
                    device=target,
                    non_blocking=self._offload_non_blocking,
                )
                destination_index = torch.tensor(
                    [new_slots[expert] for expert in incoming],
                    dtype=torch.long,
                    device=target,
                )
                staged.index_copy_(0, destination_index, transferred)
            return staged

        # Construct every projection before publishing any of them.  A failed
        # allocation/transfer therefore leaves the prior executable cache
        # intact for manager-level rollback across layers.
        gate = stage_projection(self.gate_proj, self._cached_gate_proj)
        up = stage_projection(self.up_proj, self._cached_up_proj)
        down = stage_projection(self.down_proj, self._cached_down_proj)
        slot_map = torch.full(
            (self.num_experts,),
            -1,
            dtype=torch.long,
            device=target,
        )
        id_tensor = torch.tensor(expert_ids, dtype=torch.long, device=target)
        slot_map[id_tensor] = torch.arange(len(expert_ids), dtype=torch.long, device=target)

        self._cached_gate_proj = gate
        self._cached_up_proj = up
        self._cached_down_proj = down
        self._cached_slot_map = slot_map
        self._cached_expert_ids = expert_ids
        self._cached_slot_by_id = new_slots
        self._last_paged_quantization_ids = None
        self._last_paged_quantization_error = None
        bytes_per_row = self.parameters_per_expert * self.gate_proj.element_size()
        delta = (len(incoming), len(evicted), len(incoming) * bytes_per_row)
        self._cached_lru = list(expert_ids)
        self._record_offload_stage(delta)
        return delta

    def _record_offload_stage(self, delta: tuple[int, int, int]) -> None:
        rows_in, rows_out, bytes_in = delta
        self._offload_stage_operations += 1
        self._offload_transferred_rows += rows_in
        self._offload_evicted_rows += rows_out
        self._offload_transfer_bytes += bytes_in

    def _stage_execution_page(self, required_ids: tuple[int, ...]) -> None:
        """Stage required ids and retain recent rows in any spare slots."""

        capacity = self._offload_capacity
        if capacity is None or not self._offload_paging:
            raise RuntimeError("layer-local expert paging is not enabled")
        required = tuple(sorted(set(required_ids)))
        if not required:
            raise ValueError("an execution page needs at least one expert")
        if len(required) > capacity:
            raise ValueError(
                f"{len(required)} requested experts exceed cache capacity {capacity}"
            )
        keep = set(required)
        for expert in reversed(self._cached_lru):
            if len(keep) >= capacity:
                break
            keep.add(expert)
        desired = tuple(sorted(keep))
        previous_lru = self._cached_lru
        self._stage_expert_rows(desired)
        self._cached_lru = [
            expert for expert in previous_lru if expert in keep and expert not in required
        ]
        self._cached_lru.extend(required)

    def _set_expert_offload_target(self, target_device: torch.device) -> None:
        """Resolve an index-less target before the first cache allocation."""

        if self._cached_expert_ids:
            raise RuntimeError("cannot change an expert cache target after staging")
        if not self.expert_offload_enabled:
            raise RuntimeError("expert offload has not been prepared for this bank")
        self._offload_target = target_device

    def _clear_expert_offload(self) -> None:
        """Drop request-local cache tensors while leaving CPU masters intact."""

        self._cached_gate_proj = None
        self._cached_up_proj = None
        self._cached_down_proj = None
        self._cached_slot_map = None
        self._cached_expert_ids = ()
        self._cached_slot_by_id = {}
        self._cached_lru = []
        self._offload_target = None
        self._offload_capacity = None
        self._offload_paging = False
        self._last_paged_quantization_ids = None
        self._last_paged_quantization_error = None

    def _cached_projection_rows(self, name: str, expert_ids: Tensor) -> Tensor:
        cache = getattr(self, f"_cached_{name}")
        slot_map = self._cached_slot_map
        target = self._offload_target
        if cache is None or slot_map is None or target is None:
            raise RuntimeError("resident expert cache is not populated")
        if expert_ids.device != target:
            raise RuntimeError(
                f"expert ids are on {expert_ids.device}, but the resident cache is on {target}"
            )
        slots = slot_map[expert_ids]
        if target.type != "meta" and bool((slots < 0).any()):
            missing = expert_ids[slots < 0].detach().cpu().tolist()
            raise RuntimeError(f"experts {missing} are not resident in the device cache")
        return cache.index_select(0, slots)

    def expert_forward(
        self,
        hidden_states: Tensor,
        expert_index: int,
        promoted_mask: Tensor | None = None,
        *,
        is_promoted: bool | None = None,
        precision_validated: bool = False,
    ) -> Tensor:
        """Apply a single expert to ``[tokens, d_model]`` hidden states."""

        if not precision_validated:
            self._ensure_native_precision(hidden_states.device)
        if is_promoted is None:
            promoted = None if promoted_mask is None else promoted_mask[expert_index]
            is_promoted = promoted is not None and bool(promoted.detach().item())
        if self.expert_offload_enabled:
            try:
                slot = self._cached_slot_by_id[expert_index]
            except KeyError as error:
                raise RuntimeError(
                    f"expert {expert_index} is not resident in the device cache"
                ) from error
            assert self._cached_gate_proj is not None
            assert self._cached_up_proj is not None
            assert self._cached_down_proj is not None
            gate_shadow = self._cached_gate_proj[slot]
            up_shadow = self._cached_up_proj[slot]
            down_shadow = self._cached_down_proj[slot]
        else:
            gate_shadow = self.gate_proj[expert_index]
            up_shadow = self.up_proj[expert_index]
            down_shadow = self.down_proj[expert_index]
        weight_format, activation_format = self._execution_pair(is_promoted)
        gate = F.silu(
            native_expert_matmul(
                hidden_states,
                gate_shadow,
                weight_format=weight_format,
                activation_format=activation_format,
            )
        )
        up = native_expert_matmul(
            hidden_states,
            up_shadow,
            weight_format=weight_format,
            activation_format=activation_format,
        )
        return native_expert_matmul(
            gate * up,
            down_shadow,
            weight_format=weight_format,
            activation_format=activation_format,
        )

    def _execution_pair(self, is_promoted: bool) -> tuple[str, str]:
        """Choose one real kernel lane for a complete expert identity."""

        if not is_promoted:
            return self.config.expert_weight_format, self.config.expert_activation_format
        if self.config.expert_promotion_format == "bfloat16":
            return "bfloat16", "bfloat16"
        if self.config.expert_promotion_format == "mxfp8":
            return "mxfp8", "mxfp8"
        raise RuntimeError(
            f"no native promotion kernel for {self.config.expert_promotion_format!r}"
        )

    @property
    def native_provider_id(self) -> str:
        """Provider last verified on a real execution device, or ``unvalidated``."""

        return self._native_provider_id

    def _ensure_native_precision(self, device: torch.device) -> str:
        """Verify the configured pair once for this bank and exact device."""

        resolved = torch.device(device)
        key = str(resolved)
        cached = self._native_provider_by_device.get(key)
        if cached is not None:
            self._native_provider_id = cached
            return cached
        capabilities = detect_precision_capabilities(resolved)
        capability = capabilities.require(
            self.config.expert_weight_format,
            self.config.expert_activation_format,
        )
        if self.config.expert_execution == "grouped" and not capability.grouped_available:
            raise RuntimeError(
                f"native provider {capability.provider_id!r} is dense per-expert; "
                "select expert_execution='loop'. SPALMER will not substitute BF16 "
                "torch.bmm for a requested low-precision grouped kernel"
            )
        if (
            self.config.potentiation_budget
            and self.config.expert_promotion_format == "mxfp8"
        ):
            capabilities.require("mxfp8", "mxfp8")
        self._native_provider_by_device[key] = capability.provider_id
        self._native_provider_id = capability.provider_id
        return capability.provider_id

    def _promotion_flags(self, promoted_mask: Tensor | None) -> tuple[bool, ...] | None:
        """Copy one tiny expert mask once, avoiding a device sync per expert."""

        if promoted_mask is None:
            return None
        if promoted_mask.shape != (self.num_experts,):
            raise ValueError(
                f"promoted_mask must have shape {(self.num_experts,)}, "
                f"got {tuple(promoted_mask.shape)}"
            )
        return tuple(bool(value) for value in promoted_mask.detach().cpu().tolist())

    @torch.no_grad()
    def quantization_error(
        self,
        expert_ids: Tensor,
        resident_ids: Tensor | None = None,
    ) -> Tensor:
        """Deterministic reconstruction MSE of every expert selected this pass.

        Vectorized over the resident experts (all experts when ``resident_ids``
        is ``None``); entries of experts that were not selected are zero, which
        the potentiation controller treats as "not measured".
        """

        error_device = (
            self._offload_target if self.expert_offload_enabled else self.gate_proj.device
        )
        assert error_device is not None
        errors = torch.zeros(self.num_experts, dtype=torch.float32, device=error_device)
        if self.config.expert_weight_format == "bfloat16":
            return errors

        if self.expert_paging_enabled:
            selected_ids = tuple(
                sorted(int(expert) for expert in torch.unique(expert_ids).detach().cpu().tolist())
            )
            if (
                selected_ids != self._last_paged_quantization_ids
                or self._last_paged_quantization_error is None
            ):
                raise RuntimeError(
                    "paged quantization error is available only for the most recent "
                    "routing execution"
                )
            return self._last_paged_quantization_error

        # Measure only experts that actually executed.  Repacking the complete
        # 200-expert bank for telemetry would dominate the useful GEMMs and was
        # the principal cost of the retired fake-QAT path.
        selected_ids = torch.unique(expert_ids.reshape(-1))
        measured = self._quantization_error_for_rows(selected_ids)
        errors[selected_ids] = measured
        return errors

    def _quantization_error_for_rows(self, row_ids: Tensor | None) -> Tensor:
        """Reconstruction MSE for ``row_ids`` (or the complete local bank)."""

        projection_errors: list[Tensor] = []
        for name, parameter in (
            ("gate_proj", self.gate_proj),
            ("up_proj", self.up_proj),
            ("down_proj", self.down_proj),
        ):
            if self.expert_offload_enabled:
                if row_ids is None:
                    raise RuntimeError("offloaded execution requires explicit resident expert ids")
                shadow = self._cached_projection_rows(name, row_ids)
            else:
                shadow = parameter if row_ids is None else parameter[row_ids]
            errors = [
                native_weight_reconstruction_error(
                    expert_matrix.t().contiguous(),
                    weight_format=self.config.expert_weight_format,
                )
                for expert_matrix in shadow
            ]
            projection_errors.append(torch.stack(errors))
        return torch.stack(projection_errors).mean(dim=0)

    def execute_routing(
        self,
        hidden_states: Tensor,
        token_index: Tensor,
        expert_index: Tensor,
        routing_weights: Tensor,
        promoted_mask: Tensor | None = None,
        resident_ids: Tensor | None = None,
    ) -> Tensor:
        """Execute only the selected (token, expert) routing pairs.

        Args:
            hidden_states: ``[num_tokens, d_model]`` flattened inputs.
            token_index: ``[num_pairs]`` token row of each routing pair.
            expert_index: ``[num_pairs]`` expert of each routing pair.
            routing_weights: ``[num_pairs]`` combination weight of each pair.
            promoted_mask: Optional ``[num_experts]`` Boolean potentiation mask.
            resident_ids: Optional ``[m]`` resident expert ids. The grouped
                path executes exactly these groups; ``None`` means the full
                pool is resident.

        Returns:
            ``[num_tokens, d_model]`` weighted combination of the selected
            expert updates.
        """

        self._ensure_native_precision(hidden_states.device)
        if self.expert_paging_enabled:
            return self._execute_paged(
                hidden_states,
                token_index,
                expert_index,
                routing_weights,
                promoted_mask,
            )

        if self.config.expert_execution == "loop":
            return self._execute_loop(
                hidden_states, token_index, expert_index, routing_weights, promoted_mask
            )
        return self._execute_grouped(
            hidden_states, token_index, expert_index, routing_weights, promoted_mask, resident_ids
        )

    def _execute_paged(
        self,
        hidden_states: Tensor,
        token_index: Tensor,
        expert_index: Tensor,
        routing_weights: Tensor,
        promoted_mask: Tensor | None,
    ) -> Tensor:
        """Execute unrestricted router choices through cache-sized tiles."""

        if self.config.expert_execution == "loop":
            return self._execute_paged_loop(
                hidden_states,
                token_index,
                expert_index,
                routing_weights,
                promoted_mask,
            )
        return self._execute_paged_grouped(
            hidden_states,
            token_index,
            expert_index,
            routing_weights,
            promoted_mask,
        )

    def _execute_loop(
        self,
        hidden_states: Tensor,
        token_index: Tensor,
        expert_index: Tensor,
        routing_weights: Tensor,
        promoted_mask: Tensor | None,
    ) -> Tensor:
        num_tokens = hidden_states.shape[0]
        update = hidden_states.new_zeros(num_tokens, hidden_states.shape[-1])
        selected_ids = tuple(
            int(expert) for expert in torch.unique(expert_index).detach().cpu().tolist()
        )
        promotion_flags = self._promotion_flags(promoted_mask)
        for expert in selected_ids:
            slot_mask = expert_index == expert
            rows = token_index[slot_mask]
            outputs = self.expert_forward(
                hidden_states[rows],
                expert,
                is_promoted=False if promotion_flags is None else promotion_flags[expert],
                precision_validated=True,
            )
            weighted = _weighted_expert_updates(
                outputs,
                routing_weights[slot_mask],
                target_dtype=update.dtype,
            )
            update = update.index_add(0, rows, weighted)
        return update

    def _execute_grouped(
        self,
        hidden_states: Tensor,
        token_index: Tensor,
        expert_index: Tensor,
        routing_weights: Tensor,
        promoted_mask: Tensor | None,
        resident_ids: Tensor | None,
    ) -> Tensor:
        num_tokens, d_model = hidden_states.shape
        num_pairs = expert_index.numel()
        device = hidden_states.device
        update = hidden_states.new_zeros(num_tokens, d_model)
        if num_pairs == 0:
            return update

        counts = _pair_counts(expert_index, self.num_experts)
        starts = torch.cumsum(counts, dim=0) - counts
        order = torch.argsort(expert_index, stable=True)
        candidates = (
            torch.arange(self.num_experts, device=device) if resident_ids is None else resident_ids
        )
        # Host synchronization 1: which candidate experts received any pair.
        group_ids = candidates[counts[candidates] > 0]
        if group_ids.numel() == 0:
            return update
        group_counts = counts[group_ids]
        group_starts = starts[group_ids]
        # Host synchronization 2: copy at most ``num_experts`` scalar counts.
        # Bucket construction then stays on the host and creates at most
        # ceil(log2(max_count)) batched GEMMs, independent of expert count.
        buckets = _power_of_two_count_buckets(group_counts.detach().cpu().tolist())

        # Quantize the selected weight stacks exactly once, in global expert
        # order. Besides avoiding repeated QAT work across buckets, this keeps
        # stochastic-rounding draws aligned with the former one-block path.
        all_gate, all_up, all_down = self._stacked_weights(
            group_ids, promoted_mask
        )
        routed_rows: list[Tensor] = []
        routed_updates: list[Tensor] = []
        for capacity, positions in buckets:
            group_positions = torch.tensor(positions, dtype=torch.long, device=device)
            bucket_counts = group_counts.index_select(0, group_positions)
            bucket_starts = group_starts.index_select(0, group_positions)
            slot = torch.arange(capacity, device=device)
            valid = slot[None, :] < bucket_counts[:, None]
            gather = (bucket_starts[:, None] + slot[None, :]).clamp_max(num_pairs - 1)
            pair_rows = order[gather]
            token_rows = token_index[pair_rows]
            inputs = hidden_states[token_rows]

            gate_weight = all_gate.index_select(0, group_positions)
            up_weight = all_up.index_select(0, group_positions)
            down_weight = all_down.index_select(0, group_positions)
            hidden = F.silu(torch.bmm(inputs, gate_weight)) * torch.bmm(
                inputs, up_weight
            )
            outputs = torch.bmm(hidden, down_weight)

            # Remove padding before retaining results for the final reduction.
            # Every real routing pair appears exactly once across the buckets.
            real_rows = pair_rows[valid]
            routed_rows.append(real_rows)
            routed_updates.append(
                _weighted_expert_updates(
                    outputs[valid],
                    routing_weights.index_select(0, real_rows),
                    target_dtype=update.dtype,
                )
            )

        pair_rows = torch.cat(routed_rows)
        weighted_updates = torch.cat(routed_updates)
        return update.index_add(0, token_index.index_select(0, pair_rows), weighted_updates)

    def _execute_paged_loop(
        self,
        hidden_states: Tensor,
        token_index: Tensor,
        expert_index: Tensor,
        routing_weights: Tensor,
        promoted_mask: Tensor | None,
    ) -> Tensor:
        """Mirror loop execution order while staging cache-sized expert runs."""

        update = hidden_states.new_zeros(hidden_states.shape)
        selected_ids = tuple(
            sorted(int(expert) for expert in torch.unique(expert_index).detach().cpu().tolist())
        )
        promotion_flags = self._promotion_flags(promoted_mask)
        errors = hidden_states.new_zeros(self.num_experts, dtype=torch.float32)
        capacity = self._require_paging_capacity()
        for offset in range(0, len(selected_ids), capacity):
            page = selected_ids[offset : offset + capacity]
            self._stage_execution_page(page)
            page_ids = torch.tensor(page, dtype=torch.long, device=expert_index.device)
            self._measure_paged_quantization(errors, page_ids)
            for expert in page:
                slot_mask = expert_index == expert
                rows = token_index[slot_mask]
                outputs = self.expert_forward(
                    hidden_states[rows],
                    expert,
                    is_promoted=False if promotion_flags is None else promotion_flags[expert],
                    precision_validated=True,
                )
                weighted = _weighted_expert_updates(
                    outputs,
                    routing_weights[slot_mask],
                    target_dtype=update.dtype,
                )
                update = update.index_add(0, rows, weighted)
        self._publish_paged_quantization(selected_ids, errors)
        return update

    def _execute_paged_grouped(
        self,
        hidden_states: Tensor,
        token_index: Tensor,
        expert_index: Tensor,
        routing_weights: Tensor,
        promoted_mask: Tensor | None,
    ) -> Tensor:
        """Mirror unrestricted grouping/reduction while paging group weights."""

        num_tokens, d_model = hidden_states.shape
        num_pairs = expert_index.numel()
        device = hidden_states.device
        update = hidden_states.new_zeros(num_tokens, d_model)
        if num_pairs == 0:
            self._publish_paged_quantization((), hidden_states.new_zeros(self.num_experts))
            return update

        counts = _pair_counts(expert_index, self.num_experts)
        starts = torch.cumsum(counts, dim=0) - counts
        order = torch.argsort(expert_index, stable=True)
        candidates = torch.arange(self.num_experts, device=device)
        group_ids = candidates[counts > 0]
        group_counts = counts[group_ids]
        group_starts = starts[group_ids]
        buckets = _power_of_two_count_buckets(group_counts.detach().cpu().tolist())
        selected_ids = tuple(int(expert) for expert in group_ids.detach().cpu().tolist())
        errors = hidden_states.new_zeros(self.num_experts, dtype=torch.float32)
        cache_capacity = self._require_paging_capacity()
        routed_rows: list[Tensor] = []
        routed_updates: list[Tensor] = []

        # Preserve the unrestricted executor's bucket and group ordering.
        # Only each bucket's weight batch is split to respect cache capacity.
        for bucket_capacity, positions in buckets:
            for offset in range(0, len(positions), cache_capacity):
                page_positions_tuple = positions[offset : offset + cache_capacity]
                page_positions = torch.tensor(
                    page_positions_tuple,
                    dtype=torch.long,
                    device=device,
                )
                page_ids = group_ids.index_select(0, page_positions)
                page = tuple(selected_ids[position] for position in page_positions_tuple)
                self._stage_execution_page(page)
                self._measure_paged_quantization(errors, page_ids)

                page_counts = group_counts.index_select(0, page_positions)
                page_starts = group_starts.index_select(0, page_positions)
                slot = torch.arange(bucket_capacity, device=device)
                valid = slot[None, :] < page_counts[:, None]
                gather = (page_starts[:, None] + slot[None, :]).clamp_max(num_pairs - 1)
                pair_rows = order[gather]
                token_rows = token_index[pair_rows]
                inputs = hidden_states[token_rows]

                gate_weight, up_weight, down_weight = self._stacked_weights(
                    page_ids,
                    promoted_mask,
                )
                hidden = F.silu(torch.bmm(inputs, gate_weight)) * torch.bmm(
                    inputs,
                    up_weight,
                )
                outputs = torch.bmm(hidden, down_weight)
                real_rows = pair_rows[valid]
                routed_rows.append(real_rows)
                routed_updates.append(
                    _weighted_expert_updates(
                        outputs[valid],
                        routing_weights.index_select(0, real_rows),
                        target_dtype=update.dtype,
                    )
                )

        pair_rows = torch.cat(routed_rows)
        weighted_updates = torch.cat(routed_updates)
        self._publish_paged_quantization(selected_ids, errors)
        return update.index_add(0, token_index.index_select(0, pair_rows), weighted_updates)

    def _require_paging_capacity(self) -> int:
        capacity = self._offload_capacity
        if capacity is None or not self.expert_paging_enabled:
            raise RuntimeError("expert paging has no configured cache capacity")
        return capacity

    @torch.no_grad()
    def _measure_paged_quantization(self, errors: Tensor, page_ids: Tensor) -> None:
        if self.config.expert_weight_format != "bfloat16":
            errors[page_ids] = self._quantization_error_for_rows(page_ids)

    def _publish_paged_quantization(
        self,
        selected_ids: tuple[int, ...],
        errors: Tensor,
    ) -> None:
        self._last_paged_quantization_ids = selected_ids
        self._last_paged_quantization_error = errors

    def _stacked_weights(
        self,
        group_ids: Tensor | None,
        promoted_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """BF16 weights for the ordinary grouped executor.

        Low-precision providers cannot reach this method unless they explicitly
        advertise a true grouped kernel and receive a dedicated executor.
        """

        if self.config.expert_weight_format != "bfloat16":
            raise RuntimeError(
                "the torch.bmm grouped executor is BF16-only; no low-precision "
                "fallback is permitted"
            )
        # BF16 configs are construction-validated with a zero potentiation
        # budget, so no runtime reduction is needed here.  Calling
        # ``promoted_mask.any()`` would force one host/device synchronization
        # per layer on the ordinary grouped path.

        if self.expert_offload_enabled:
            if group_ids is None:
                raise RuntimeError("offloaded execution requires explicit resident expert ids")
            stacks = (
                self._cached_projection_rows("gate_proj", group_ids),
                self._cached_projection_rows("up_proj", group_ids),
                self._cached_projection_rows("down_proj", group_ids),
            )
        elif group_ids is None:
            stacks = (self.gate_proj, self.up_proj, self.down_proj)
        else:
            stacks = (
                self.gate_proj[group_ids],
                self.up_proj[group_ids],
                self.down_proj[group_ids],
            )
        return stacks


def _weighted_expert_updates(
    outputs: Tensor,
    routing_weights: Tensor,
    *,
    target_dtype: torch.dtype,
) -> Tensor:
    """Combine routes in the residual stream's dtype under mixed precision."""

    return outputs.to(dtype=target_dtype) * routing_weights.to(dtype=target_dtype).unsqueeze(-1)


def _pair_counts(expert_index: Tensor, num_experts: int) -> Tensor:
    """Per-expert pair counts without the host synchronization ``bincount`` needs."""

    counts = torch.zeros(num_experts, dtype=torch.long, device=expert_index.device)
    return counts.scatter_add_(0, expert_index, torch.ones_like(expert_index))


def _power_of_two_count_buckets(
    counts: list[int],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Group positive counts by next-power-of-two padded capacity.

    For every positive count ``n``, ``next_power_of_two(n) < 2 * n``. Summing
    that inequality over all groups guarantees aggregate padded routing slots
    remain strictly below twice the real pair count.
    """

    grouped: dict[int, list[int]] = {}
    real_slots = 0
    for position, raw_count in enumerate(counts):
        count = int(raw_count)
        if count < 1:
            raise ValueError("count buckets require positive group counts")
        capacity = 1 << (count - 1).bit_length()
        grouped.setdefault(capacity, []).append(position)
        real_slots += count
    buckets = tuple(
        (capacity, tuple(positions))
        for capacity, positions in sorted(grouped.items())
    )
    padded_slots = sum(capacity * len(positions) for capacity, positions in buckets)
    if padded_slots >= 2 * real_slots:
        raise RuntimeError("power-of-two routing buckets violated their padding bound")
    return buckets
