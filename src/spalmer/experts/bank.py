"""Tensorized bank of small gated-MLP experts (SPALMER ledger C07).

All experts live in three stacked parameters (gate, up, down), one per
projection. Two execution paths produce the same routed update:

- ``grouped`` (default): pairs are sorted by expert, then experts are placed in
  power-of-two count buckets. Each bucket executes three batched matmuls over
  ``[bucket_groups, bucket_capacity, d_model]``. There is no per-expert Python
  loop, and the sum of padded slots is strictly less than twice the number of
  real routed pairs. A single heavily used expert therefore cannot force every
  other expert to allocate its global maximum capacity.
- ``loop``: the per-expert reference path (one small matmul triple per
  distinct selected expert) used for equivalence checks and tiny CPU runs.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from spalmer.experts.config import MicroExpertsConfig
from spalmer.experts.qat import (
    ExpertQATBackendStatus,
    ExpertQATConfig,
    expert_qat_backend_status,
    fake_quantize_expert_activation,
    fake_quantize_expert_weight,
    fake_quantize_mxfp8,
    require_expert_qat_backend,
)
from spalmer.nn import fake_quantize_low_bit


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
        self._cached_expert_ids: tuple[int, ...] = ()
        self._cached_slot_by_id: dict[int, int] = {}
        self._cached_slot_map: Tensor | None = None
        self._cached_gate_proj: Tensor | None = None
        self._cached_up_proj: Tensor | None = None
        self._cached_down_proj: Tensor | None = None
        resolved_weight_format = (
            "mxfp4" if config.expert_weight_format == "legacy_int" else config.expert_weight_format
        )
        self._qat_deterministic_config = ExpertQATConfig(
            weight_format=resolved_weight_format,
            activation_format="mxfp8",
            backend=config.expert_qat_backend,
            stochastic_rounding=False,
        )
        self._qat_stochastic_config = ExpertQATConfig(
            weight_format=resolved_weight_format,
            activation_format="mxfp8",
            backend=config.expert_qat_backend,
            stochastic_rounding=True,
        )
        # Resolve strict-native requests at construction rather than silently
        # reaching a dequantized GEMM on the first batch.
        if (
            config.expert_fake_quantization
            and config.expert_weight_format != "legacy_int"
        ):
            require_expert_qat_backend(self._qat_config(stochastic=False))

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
        bytes_per_row = self.parameters_per_expert * self.gate_proj.element_size()
        return len(incoming), len(evicted), len(incoming) * bytes_per_row

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
        self._offload_target = None
        self._offload_capacity = None

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
    ) -> Tensor:
        """Apply a single expert to ``[tokens, d_model]`` hidden states."""

        promoted = None if promoted_mask is None else promoted_mask[expert_index]
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
        gate_weight = self._effective_weight(gate_shadow, promoted)
        up_weight = self._effective_weight(up_shadow, promoted)
        down_weight = self._effective_weight(down_shadow, promoted)
        expert_input = self._effective_activation(hidden_states)
        gate = F.silu(expert_input @ gate_weight)
        up = expert_input @ up_weight
        down_input = self._effective_activation(gate * up)
        return down_input @ down_weight

    def _effective_weight(self, shadow: Tensor, promoted: Tensor | None) -> Tensor:
        """Derive an FP4 or promoted execution weight from the one master.

        ``shadow`` may be one expert's matrix or a stack ``[m, ...]`` of them;
        ``promoted`` is then a Boolean scalar or a ``[m]`` vector. The
        quantizer sees the contraction dimension by transposing ``[K, N]`` to
        ``[N, K]``. No second persistent high-precision weight is created.
        """

        if not self.config.expert_fake_quantization:
            return shadow
        low = self._quantize_weight(shadow, promoted=False)
        if promoted is None:
            return low
        high = self._quantize_weight(shadow, promoted=True)
        # One Boolean per expert controls the complete expert. Algebraic
        # selection avoids host synchronization and keeps gradients on the
        # same BF16 master parameters.
        enabled = promoted.to(device=shadow.device, dtype=shadow.dtype)
        enabled = enabled.reshape(-1, *([1] * (shadow.dim() - 1))) if enabled.dim() else enabled
        return low + enabled * (high - low)

    def _quantize_weight(self, master: Tensor, *, promoted: bool) -> Tensor:
        if self.config.expert_weight_format == "legacy_int" and not promoted:
            return fake_quantize_low_bit(
                master,
                bits=self.config.expert_quant_bits,
                stochastic=self.training and self.config.expert_stochastic_rounding,
                straight_through=self.training,
            )
        transposed = master.transpose(-2, -1).contiguous()
        if promoted and self.config.expert_promotion_format == "bfloat16":
            effective = transposed.to(torch.bfloat16).to(transposed.dtype)
            if self.training:
                effective = transposed + (effective - transposed).detach()
        elif promoted:
            effective = fake_quantize_mxfp8(
                transposed,
                stochastic=self.training and self.config.expert_stochastic_rounding,
                straight_through=self.training,
            )
        else:
            effective = fake_quantize_expert_weight(
                transposed,
                self._qat_config(
                    stochastic=self.training and self.config.expert_stochastic_rounding
                ),
            )
        return effective.transpose(-2, -1)

    def _effective_activation(self, values: Tensor) -> Tensor:
        if (
            not self.config.expert_fake_quantization
            or self.config.expert_activation_format == "bfloat16"
        ):
            return values
        return fake_quantize_expert_activation(
            values,
            self._qat_config(
                stochastic=self.training and self.config.expert_stochastic_rounding
            ),
        )

    def _qat_config(self, *, stochastic: bool) -> ExpertQATConfig:
        return self._qat_stochastic_config if stochastic else self._qat_deterministic_config

    @property
    def qat_backend_status(self) -> ExpertQATBackendStatus:
        """Resolved execution capability for logs and checkpoint diagnostics."""

        return expert_qat_backend_status(self._qat_config(stochastic=False))

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
        if not self.config.expert_fake_quantization:
            return errors
        squared_error: Tensor | None = None
        squared_weight: Tensor | None = None
        for name, parameter in (
            ("gate_proj", self.gate_proj),
            ("up_proj", self.up_proj),
            ("down_proj", self.down_proj),
        ):
            if self.expert_offload_enabled:
                if resident_ids is None:
                    raise RuntimeError("offloaded execution requires explicit resident expert ids")
                shadow = self._cached_projection_rows(name, resident_ids)
            else:
                shadow = parameter if resident_ids is None else parameter[resident_ids]
            if self.config.expert_weight_format == "legacy_int":
                low = fake_quantize_low_bit(
                    shadow,
                    bits=self.config.expert_quant_bits,
                    stochastic=False,
                    straight_through=False,
                )
            else:
                transposed = shadow.transpose(-2, -1).contiguous()
                low = fake_quantize_expert_weight(
                    transposed,
                    self._qat_config(stochastic=False),
                ).transpose(-2, -1)
            error = (shadow.float() - low.float()).square().sum(dim=(1, 2))
            weight = shadow.float().square().sum(dim=(1, 2))
            squared_error = error if squared_error is None else squared_error + error
            squared_weight = weight if squared_weight is None else squared_weight + weight
        assert squared_error is not None and squared_weight is not None
        measured = squared_error / squared_weight.clamp_min(1e-12)
        if resident_ids is None:
            errors = measured
        else:
            errors[resident_ids] = measured
        selected = _pair_counts(expert_ids.reshape(-1), self.num_experts) > 0
        return errors * selected.to(errors.dtype)

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

        if self.config.expert_execution == "loop":
            return self._execute_loop(
                hidden_states, token_index, expert_index, routing_weights, promoted_mask
            )
        return self._execute_grouped(
            hidden_states, token_index, expert_index, routing_weights, promoted_mask, resident_ids
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
        for expert in torch.unique(expert_index):
            slot_mask = expert_index == expert
            rows = token_index[slot_mask]
            outputs = self.expert_forward(hidden_states[rows], int(expert), promoted_mask)
            weighted = routing_weights[slot_mask].unsqueeze(-1) * outputs
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
        all_gate, all_up, all_down = self._stacked_effective_weights(
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
            expert_inputs = self._effective_activation(inputs)
            hidden = F.silu(torch.bmm(expert_inputs, gate_weight)) * torch.bmm(
                expert_inputs, up_weight
            )
            outputs = torch.bmm(self._effective_activation(hidden), down_weight)

            # Remove padding before retaining results for the final reduction.
            # Every real routing pair appears exactly once across the buckets.
            real_rows = pair_rows[valid]
            routed_rows.append(real_rows)
            routed_updates.append(
                outputs[valid] * routing_weights.index_select(0, real_rows).unsqueeze(-1)
            )

        pair_rows = torch.cat(routed_rows)
        weighted_updates = torch.cat(routed_updates)
        return update.index_add(0, token_index.index_select(0, pair_rows), weighted_updates)

    def _stacked_effective_weights(
        self,
        group_ids: Tensor | None,
        promoted_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Effective (quantized / promoted) weights of the given experts, stacked."""

        if self.expert_offload_enabled:
            if group_ids is None:
                raise RuntimeError("offloaded execution requires explicit resident expert ids")
            promoted = None if promoted_mask is None else promoted_mask[group_ids]
            stacks = (
                self._cached_projection_rows("gate_proj", group_ids),
                self._cached_projection_rows("up_proj", group_ids),
                self._cached_projection_rows("down_proj", group_ids),
            )
        elif group_ids is None:
            promoted = promoted_mask
            stacks = (self.gate_proj, self.up_proj, self.down_proj)
        else:
            promoted = None if promoted_mask is None else promoted_mask[group_ids]
            stacks = (
                self.gate_proj[group_ids],
                self.up_proj[group_ids],
                self.down_proj[group_ids],
            )
        gate, up, down = (self._effective_weight(stack, promoted) for stack in stacks)
        return gate, up, down


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
