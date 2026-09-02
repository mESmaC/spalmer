"""Expert residency: one request-level identity set shared by every layer (C13).

The ledger's candidate controller sequence is::

    load shared base + vocab + router + two-expert minimum
            -> evaluate observed NLL/CE and target-free predictive entropy
            -> rank eligible experts by predicted surprise
            -> add a bounded increment
            -> recompute
            -> retain, expand, or roll back within the current 10% soft cap

Two objects implement it:

- :class:`ExpertResidency` is the resident identity set. It is one module
  shared by all layer-local banks (like the router and the potentiation
  controller), so expert ``e`` is resident in every layer or in none. Per-token
  routing is restricted to residents. During a controller-managed request the
  resident count is also the requested per-token top-``k`` (bounded by
  ``max_active_experts``): accepted expansion therefore adds coherent expert
  ids *and* executes the added capacity. Rollback restores both the exact id
  set and the earlier top-``k``. Outside a request the full pool is resident,
  which is what training uses.
- :func:`choose_inference_residency` is the request-level controller applied
  at prefill. It starts from the minimum resident set, compares the prompt's
  effective surprise with the model's average surprise (C08: ``if effective
  NLL >= average surprise: include additional experts; recompute``), ranks the
  non-resident experts by their predicted surprise on the prompt, adds a
  bounded increment of ids, recomputes, and retains or rolls back.

This is logical residency only: which identities may execute. Physical
CPU/GPU migration, prefetch, and eviction are deferred.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from spalmer.experts.config import MicroExpertsConfig
from spalmer.modeling import CausalLMOutput, SPALMERCausalLM


@dataclass(frozen=True, slots=True)
class ResidencyState:
    """Exact request-local residency and execution-capacity state."""

    resident_ids: tuple[int, ...]
    active_experts_override: int | None


class ExpertResidency(nn.Module):
    """The resident expert identity set shared across layers.

    ``resident_mask`` (Boolean, ``[num_experts]``) and ``resident_ids`` (sorted
    long ids) are non-persistent buffers: residency is request-local state, not
    a learned weight, so it moves with the model between devices but is not
    written into checkpoints. A freshly built model has every expert resident.
    The Python-side id tuple is the source of truth, so reading ``ids``,
    ``size``, or ``is_full`` never synchronizes with the device.
    """

    def __init__(self, config: MicroExpertsConfig) -> None:
        super().__init__()
        self.config = config
        self._ids: tuple[int, ...] = tuple(range(config.num_experts))
        self._active_experts_override: int | None = None
        self._request_previous: ResidencyState | None = None
        self.register_buffer(
            "resident_mask",
            torch.ones(config.num_experts, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "resident_ids",
            torch.arange(config.num_experts, dtype=torch.long),
            persistent=False,
        )

    @property
    def num_experts(self) -> int:
        return self.config.num_experts

    @property
    def ids(self) -> tuple[int, ...]:
        """Exact resident expert ids, ascending."""

        return self._ids

    @property
    def size(self) -> int:
        return len(self._ids)

    @property
    def is_full(self) -> bool:
        return len(self._ids) == self.num_experts

    @property
    def request_open(self) -> bool:
        """Whether a controller-managed request currently owns the resident set."""

        return self._request_previous is not None

    @property
    def active_experts(self) -> int:
        """Experts actually executed per token at the current capacity."""

        if self._active_experts_override is not None:
            return self._active_experts_override
        return self.config.active_experts

    @property
    def active_experts_override(self) -> int | None:
        """Explicit per-token execution capacity, or the configured default."""

        return self._active_experts_override

    @property
    def max_active_experts(self) -> int:
        """Largest per-token execution capacity supported by this expert pool."""

        return min(self.config.max_active_experts, self.num_experts)

    def reset(self) -> None:
        """Make every expert resident (the training / no-request state)."""

        self._commit(tuple(range(self.num_experts)))

    def begin_request(
        self,
        ids: Sequence[int],
        *,
        active_experts: int | None = None,
    ) -> tuple[int, ...]:
        """Start a request from ``ids`` and remember the complete prior state.

        Nested calls keep the outermost prior set so that :meth:`end_request`
        always returns to the state before the first request. The controller
        supplies ``active_experts`` when resident capacity must also become
        executed capacity.
        """

        if active_experts is not None:
            self._validate_active_experts(active_experts)
        required_active = self.active_experts if active_experts is None else active_experts
        committed = self._validate_resident_set(ids, required_active=required_active)
        if self._request_previous is None:
            self._request_previous = self.snapshot_state()
        self._commit(committed)
        if active_experts is not None:
            self._active_experts_override = active_experts
        return committed

    def end_request(self) -> None:
        """Restore ids and top-``k`` from before :meth:`begin_request` (or no-op)."""

        previous_state = self._request_previous
        if previous_state is None:
            return
        self._request_previous = None
        self.restore_state(previous_state)

    def set_active_experts(self, count: int | None) -> None:
        """Set per-token execution capacity without changing resident identities."""

        if count is not None:
            self._validate_active_experts(count)
        effective = self.config.active_experts if count is None else count
        if effective > self.size:
            raise ValueError(
                f"per-token top-{effective} cannot be served by only "
                f"{self.size} resident experts"
            )
        self._active_experts_override = count

    def set(self, ids: Sequence[int]) -> tuple[int, ...]:
        """Replace the resident set with exactly ``ids`` (validated, deduplicated)."""

        unique = self._validate_resident_set(ids)
        self._commit(unique)
        return unique

    def expand(self, new_ids: Sequence[int]) -> tuple[int, ...]:
        """Add explicit ids to the resident set; existing residents are untouched.

        Returns the ids that were actually added. Ids that are already resident
        are rejected so that every expansion is an explicit, auditable change.
        """

        unique = self._validate_ids(new_ids)
        current = set(self._ids)
        already = [expert for expert in unique if expert in current]
        if already:
            raise ValueError(f"experts {already} are already resident")
        self._commit(tuple(sorted(current | set(unique))))
        return unique

    def snapshot(self) -> tuple[int, ...]:
        """Snapshot resident identities only (the legacy residency API)."""

        return self._ids

    def snapshot_state(self) -> ResidencyState:
        """Snapshot identities and the exact top-``k`` override for rollback."""

        return ResidencyState(self._ids, self._active_experts_override)

    def restore(self, ids: Sequence[int]) -> None:
        """Return to an exact earlier identity set without changing top-``k``."""

        if len(ids) >= self.num_experts:
            self.reset()
        else:
            self.set(ids)

    def restore_state(self, state: ResidencyState) -> None:
        """Restore exact identities and execution capacity from ``state``."""

        effective = (
            self.config.active_experts
            if state.active_experts_override is None
            else state.active_experts_override
        )
        if state.active_experts_override is not None:
            self._validate_active_experts(state.active_experts_override)
        ids = self._validate_resident_set(state.resident_ids, required_active=effective)
        # Commit the validated pair as one state. Either ordering through the
        # public setters can temporarily violate the opposite half when a
        # candidate expansion is rolled back.
        self._active_experts_override = state.active_experts_override
        self._commit(ids)

    @contextmanager
    def session(self, ids: Sequence[int] | None = None) -> Iterator[ExpertResidency]:
        """Scope a request: optionally start from ``ids``; restore the prior set on exit."""

        previous = self.snapshot_state()
        previous_request = self._request_previous
        try:
            if ids is not None:
                self.set(ids)
            yield self
        finally:
            self._request_previous = previous_request
            self.restore_state(previous)

    def _commit(self, ids: tuple[int, ...]) -> None:
        self._ids = ids
        device = self.resident_mask.device
        # Buffers are rebuilt as ordinary tensors even when the caller runs
        # under torch.inference_mode(), so a later grad-enabled forward (for
        # example training after a generation request) can still save them.
        with torch.inference_mode(False):
            mask = torch.zeros(self.num_experts, dtype=torch.bool, device=device)
            id_tensor = torch.tensor(ids, dtype=torch.long, device=device)
            mask[id_tensor] = True
        self.resident_mask = mask
        self.resident_ids = id_tensor

    def _validate_ids(self, ids: Sequence[int]) -> tuple[int, ...]:
        unique = tuple(sorted({int(expert) for expert in ids}))
        if not unique:
            raise ValueError("at least one expert id is required")
        if unique[0] < 0 or unique[-1] >= self.num_experts:
            raise ValueError(f"expert ids must lie in [0, {self.num_experts}); got {unique}")
        return unique

    def _validate_resident_set(
        self,
        ids: Sequence[int],
        *,
        required_active: int | None = None,
    ) -> tuple[int, ...]:
        unique = self._validate_ids(ids)
        required = self.active_experts if required_active is None else required_active
        if len(unique) < required:
            raise ValueError(
                f"a resident set needs at least active_experts={required} "
                f"ids; got {len(unique)}"
            )
        return unique

    def _validate_active_experts(self, count: int) -> None:
        if not self.config.min_active_experts <= count <= self.max_active_experts:
            raise ValueError(
                f"active expert count must be in [{self.config.min_active_experts}, "
                f"{self.max_active_experts}]; got {count}"
            )

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, resident={self.size}, "
            f"active_per_token={self.active_experts}"
        )


def rank_nonresident_experts(
    layer_metrics: Sequence[Mapping[str, Any]],
    resident_mask: Tensor,
) -> tuple[int, ...]:
    """Non-resident expert ids ordered by predicted surprise on the last pass.

    The router emits a predicted surprise for every expert at every token and
    layer, resident or not, so one prefill already ranks the candidates. Scores
    are averaged over tokens and layers because expert identity is coherent
    across layers.
    """

    accumulated: Tensor | None = None
    layers = 0
    for metrics in layer_metrics:
        scores = metrics.get("router_scores")
        if not isinstance(scores, Tensor):
            continue
        mean_scores = scores.detach().float().reshape(-1, scores.shape[-1]).mean(dim=0)
        accumulated = mean_scores if accumulated is None else accumulated + mean_scores
        layers += 1
    if accumulated is None or layers == 0:
        raise ValueError("layer metrics carry no router scores to rank experts with")
    candidates = (~resident_mask.to(accumulated.device)).nonzero(as_tuple=False).flatten()
    if candidates.numel() == 0:
        return ()
    order = (accumulated[candidates] / layers).argsort()
    return tuple(candidates[order].tolist())


@dataclass(slots=True)
class ResidencyDecision:
    """One request-level residency decision and the evidence behind it."""

    resident_ids: tuple[int, ...]
    active_experts: int
    effective_nll: float | None
    predictive_entropy: float
    average_surprise: float
    trace: tuple[tuple[int, float], ...]
    expansions: tuple[tuple[int, ...], ...]
    output: CausalLMOutput

    @property
    def effective_signal(self) -> float:
        """Observed prompt NLL when targets exist, else predictive entropy."""

        return self.predictive_entropy if self.effective_nll is None else self.effective_nll

    @property
    def resident_count(self) -> int:
        return len(self.resident_ids)


def choose_inference_residency(
    model: SPALMERCausalLM,
    prompt_ids: Tensor,
    *,
    initial_ids: Sequence[int] | None = None,
    increment: int | None = None,
    min_gain: float | None = None,
) -> ResidencyDecision:
    """Pick which experts this request keeps resident and leave the model set to it.

    Args:
        model: A routed SPALMER model with a shared :class:`ExpertResidency`.
            The chosen resident set stays applied on return so decoding
            continues with the same residency. The request is opened with
            ``residency.begin_request``; end it with ``residency.end_request()``
            (or ``model.end_residency_request()``), or wrap the whole request
            in ``model.residency_session()``. On any error the prior resident
            ids and top-``k`` are restored before the exception propagates.
        prompt_ids: ``[1, tokens]`` prompt. With two or more tokens the observed
            next-token NLL of the prompt is the effective signal; a one-token
            prompt falls back to the target-free predictive entropy.
        initial_ids: Explicit starting residents. Defaults to the most-utilized
            experts recorded by the potentiation telemetry (or the lowest ids
            when nothing has been observed), ``max(min_resident_experts, k)``
            of them where ``k`` is the per-token top-``k`` in effect. The
            starting set becomes the starting executed capacity, capped by
            ``max_active_experts``.
        increment: Expert ids added per expansion step (defaults to
            ``residency_increment``).
        min_gain: Smallest signal improvement, in nats per token, that keeps an
            expansion (defaults to ``residency_min_gain``). A negative value
            retains every expansion up to the cap.
    """

    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] == 0:
        raise ValueError("prompt_ids must describe one non-empty sequence")
    residency = model.residency
    if residency is None:
        raise TypeError("dynamic residency requires a model with a shared ExpertResidency")
    config = residency.config
    step = config.residency_increment if increment is None else increment
    gain = config.residency_min_gain if min_gain is None else min_gain
    if step <= 0:
        raise ValueError("increment must be positive")
    if not math.isfinite(gain):
        raise ValueError("min_gain must be finite")
    previous_per_token = model.active_experts or config.active_experts
    cap = min(
        max(config.resident_cap, previous_per_token),
        residency.max_active_experts,
    )
    average = model.average_surprise

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            start = (
                default_resident_ids(
                    model,
                    max(config.min_resident_experts, previous_per_token),
                )
                if initial_ids is None
                else tuple(initial_ids)
            )
            start_count = len(set(start))
            if start_count < previous_per_token:
                raise ValueError(
                    f"{start_count} starting residents cannot serve a per-token "
                    f"top-{previous_per_token}"
                )
            if start_count > cap:
                raise ValueError(
                    f"{start_count} starting residents exceed the dynamic controller "
                    f"cap of {cap}"
                )
            start_active = start_count
            residency.begin_request(start, active_experts=start_active)
            output, nll, entropy, signal = _evaluate(model, prompt_ids)
            trace = [(residency.size, signal)]
            expansions: list[tuple[int, ...]] = []
            while residency.size < cap and signal >= average:
                room = cap - residency.size
                candidates = rank_nonresident_experts(output.layer_metrics, residency.resident_mask)
                added = candidates[: min(step, room)]
                if not added:
                    break
                before = residency.snapshot_state()
                residency.expand(added)
                residency.set_active_experts(
                    min(residency.size, residency.max_active_experts)
                )
                candidate_output, candidate_nll, candidate_entropy, candidate_signal = _evaluate(
                    model, prompt_ids
                )
                trace.append((residency.size, candidate_signal))
                if signal - candidate_signal < gain:
                    residency.restore_state(before)
                    break
                expansions.append(added)
                output, nll, entropy, signal = (
                    candidate_output,
                    candidate_nll,
                    candidate_entropy,
                    candidate_signal,
                )
    except BaseException:
        residency.end_request()
        raise
    finally:
        model.train(was_training)

    return ResidencyDecision(
        resident_ids=residency.ids,
        active_experts=residency.active_experts,
        effective_nll=nll,
        predictive_entropy=entropy,
        average_surprise=average,
        trace=tuple(trace),
        expansions=tuple(expansions),
        output=output,
    )


def default_resident_ids(model: SPALMERCausalLM, count: int) -> tuple[int, ...]:
    """Starting residents: the most-utilized experts on record, else the lowest ids."""

    controller = model.potentiation_controller
    utilization = getattr(controller, "utilization_ema", None)
    if isinstance(utilization, Tensor) and bool((utilization > 0).any()):
        ranked = utilization.argsort(descending=True)[:count]
        return tuple(sorted(ranked.tolist()))
    return tuple(range(count))


def _evaluate(
    model: SPALMERCausalLM, prompt_ids: Tensor
) -> tuple[CausalLMOutput, float | None, float, float]:
    if prompt_ids.shape[1] >= 2:
        output = model(prompt_ids, labels=prompt_ids)
    else:
        output = model(prompt_ids)
    nll = None if output.token_nll is None else float(output.token_nll.float().mean())
    if output.predictive_entropy is None:
        raise RuntimeError("model did not return predictive entropy telemetry")
    entropy = float(output.predictive_entropy.float().mean())
    return output, nll, entropy, entropy if nll is None else nll


__all__ = [
    "ExpertResidency",
    "ResidencyDecision",
    "ResidencyState",
    "choose_inference_residency",
    "default_resident_ids",
    "rank_nonresident_experts",
]
