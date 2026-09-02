"""Mergeable repetition and router telemetry for held-out experiments."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

REPETITION_FORMAT = "spalmer.experiment.repetition"
REPETITION_VERSION = 1
ROUTER_FORMAT = "spalmer.experiment.router-telemetry"
ROUTER_VERSION = 1


@dataclass(frozen=True, slots=True)
class NgramRepetitionMetrics:
    order: int
    occurrences: int
    repeated_occurrences: int
    repetition_rate: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "order": self.order,
            "occurrences": self.occurrences,
            "repeated_occurrences": self.repeated_occurrences,
            "repetition_rate": self.repetition_rate,
        }


@dataclass(frozen=True, slots=True)
class RepetitionReport:
    sequences: int
    tokens: int
    transitions: int
    adjacent_repeats: int
    adjacent_repeat_rate: float
    ngrams: dict[int, NgramRepetitionMetrics]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequences": self.sequences,
            "tokens": self.tokens,
            "transitions": self.transitions,
            "adjacent_repeats": self.adjacent_repeats,
            "adjacent_repeat_rate": self.adjacent_repeat_rate,
            "ngrams": {str(order): item.to_dict() for order, item in self.ngrams.items()},
        }


class RepetitionAccumulator:
    """Measure within-sequence token and n-gram repetition while streaming.

    A repeated occurrence is every occurrence after the first matching n-gram
    in the same sequence. Matches do not cross or pool across sequences.
    """

    def __init__(self, orders: Sequence[int] = (1, 2, 3, 4)) -> None:
        normalized = tuple(sorted(set(int(order) for order in orders)))
        if not normalized or normalized[0] <= 0:
            raise ValueError("repetition orders must be positive and non-empty")
        self.orders = normalized
        self.sequences = 0
        self.tokens = 0
        self.transitions = 0
        self.adjacent_repeats = 0
        self._occurrences = {order: 0 for order in normalized}
        self._repeated = {order: 0 for order in normalized}

    def update(self, token_ids: Iterable[int]) -> None:
        sequence = tuple(int(token) for token in token_ids)
        self.sequences += 1
        self.tokens += len(sequence)
        self.transitions += max(0, len(sequence) - 1)
        self.adjacent_repeats += sum(
            left == right for left, right in zip(sequence, sequence[1:])
        )
        for order in self.orders:
            count = max(0, len(sequence) - order + 1)
            self._occurrences[order] += count
            if not count:
                continue
            frequencies = Counter(
                sequence[start : start + order] for start in range(count)
            )
            self._repeated[order] += sum(
                max(0, frequency - 1) for frequency in frequencies.values()
            )

    def merge(self, other: RepetitionAccumulator) -> None:
        if self.orders != other.orders:
            raise ValueError("cannot merge repetition accumulators with different orders")
        self.sequences += other.sequences
        self.tokens += other.tokens
        self.transitions += other.transitions
        self.adjacent_repeats += other.adjacent_repeats
        for order in self.orders:
            self._occurrences[order] += other._occurrences[order]
            self._repeated[order] += other._repeated[order]

    def finalize(self) -> RepetitionReport:
        metrics = {
            order: NgramRepetitionMetrics(
                order=order,
                occurrences=self._occurrences[order],
                repeated_occurrences=self._repeated[order],
                repetition_rate=(
                    self._repeated[order] / self._occurrences[order]
                    if self._occurrences[order]
                    else 0.0
                ),
            )
            for order in self.orders
        }
        return RepetitionReport(
            sequences=self.sequences,
            tokens=self.tokens,
            transitions=self.transitions,
            adjacent_repeats=self.adjacent_repeats,
            adjacent_repeat_rate=(
                self.adjacent_repeats / self.transitions if self.transitions else 0.0
            ),
            ngrams=metrics,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": REPETITION_FORMAT,
            "format_version": REPETITION_VERSION,
            "orders": list(self.orders),
            "sequences": self.sequences,
            "tokens": self.tokens,
            "transitions": self.transitions,
            "adjacent_repeats": self.adjacent_repeats,
            "occurrences": dict(self._occurrences),
            "repeated": dict(self._repeated),
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> RepetitionAccumulator:
        if payload.get("format") != REPETITION_FORMAT:
            raise ValueError("not a SPALMER repetition accumulator")
        if payload.get("format_version") != REPETITION_VERSION:
            raise ValueError(
                f"unsupported repetition version: {payload.get('format_version')!r}"
            )
        accumulator = cls(payload["orders"])
        accumulator.sequences = int(payload["sequences"])
        accumulator.tokens = int(payload["tokens"])
        accumulator.transitions = int(payload["transitions"])
        accumulator.adjacent_repeats = int(payload["adjacent_repeats"])
        accumulator._occurrences = _int_keyed_counts(payload["occurrences"], accumulator.orders)
        accumulator._repeated = _int_keyed_counts(payload["repeated"], accumulator.orders)
        accumulator._validate_state()
        return accumulator

    def _validate_state(self) -> None:
        if min(self.sequences, self.tokens, self.transitions, self.adjacent_repeats) < 0:
            raise ValueError("repetition state counts cannot be negative")
        if self.transitions > self.tokens or self.adjacent_repeats > self.transitions:
            raise ValueError("adjacent repeat count exceeds token transitions")
        for order in self.orders:
            if not 0 <= self._repeated[order] <= self._occurrences[order]:
                raise ValueError("invalid repeated n-gram count")


@dataclass(frozen=True, slots=True)
class RouterBatchStats:
    """Detached sufficient statistics for one layer's router output."""

    layer: int
    token_count: int
    selection_counts: tuple[int, ...]
    responsibility_mass: tuple[float, ...]
    entropy_sum: float
    active_experts_sum: int
    predicted_surprise_sum: float = 0.0
    predicted_surprise_count: int = 0

    def __post_init__(self) -> None:
        if self.layer < 0 or self.token_count < 0 or self.active_experts_sum < 0:
            raise ValueError("router layer and counts must be non-negative")
        if len(self.selection_counts) != len(self.responsibility_mass):
            raise ValueError("router expert vectors must have equal length")
        if not self.selection_counts:
            raise ValueError("router statistics require at least one expert")
        if any(count < 0 for count in self.selection_counts):
            raise ValueError("router selection counts cannot be negative")
        floats = (*self.responsibility_mass, self.entropy_sum, self.predicted_surprise_sum)
        if any(not math.isfinite(value) or value < 0 for value in floats):
            raise ValueError("router floating-point statistics must be finite and non-negative")
        if not 0 <= self.predicted_surprise_count <= self.token_count:
            raise ValueError("invalid predicted-surprise count")
        if self.predicted_surprise_count == 0 and self.predicted_surprise_sum:
            raise ValueError("predicted surprise sum requires observations")
        if not math.isclose(
            sum(self.responsibility_mass),
            float(self.token_count),
            rel_tol=1e-5,
            abs_tol=1e-7,
        ):
            raise ValueError("router responsibility mass must equal token count")
        if self.token_count == 0 and (
            sum(self.selection_counts)
            or sum(self.responsibility_mass)
            or self.entropy_sum
            or self.active_experts_sum
        ):
            raise ValueError("router batch without tokens cannot carry observations")

    @property
    def num_experts(self) -> int:
        return len(self.selection_counts)

    @classmethod
    def from_tensors(
        cls,
        layer: int,
        expert_ids: Tensor,
        routing_weights: Tensor,
        *,
        num_experts: int,
        valid_mask: Tensor | None = None,
        selected_scores: Tensor | None = None,
    ) -> RouterBatchStats:
        """Reduce router tensors without retaining their computation graph."""

        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if expert_ids.shape != routing_weights.shape or expert_ids.ndim < 1:
            raise ValueError("expert_ids and routing_weights must have one equal non-scalar shape")
        if routing_weights.shape[-1] == 0:
            raise ValueError("router tensors require at least one active expert")
        ids = expert_ids.detach().long()
        weights = routing_weights.detach().double()
        prefix_shape = ids.shape[:-1]
        if valid_mask is None:
            valid = torch.ones(prefix_shape, dtype=torch.bool, device=ids.device)
        else:
            if valid_mask.shape != prefix_shape:
                raise ValueError("valid_mask must match router tensors without the expert axis")
            valid = valid_mask.detach().bool().to(device=ids.device)
        selected_ids = ids[valid]
        selected_weights = weights[valid]
        token_count = selected_ids.shape[0]
        if selected_ids.numel() and (
            int(selected_ids.min().item()) < 0 or int(selected_ids.max().item()) >= num_experts
        ):
            raise ValueError("expert id is outside the configured expert range")
        if selected_weights.numel() and (
            not bool(torch.isfinite(selected_weights).all().item())
            or bool((selected_weights < 0).any().item())
        ):
            raise ValueError("routing weights must be finite and non-negative")
        if token_count and not torch.allclose(
            selected_weights.sum(dim=-1),
            torch.ones(token_count, dtype=torch.float64, device=selected_weights.device),
            rtol=1e-5,
            atol=1e-7,
        ):
            raise ValueError("routing weights must sum to one for each valid token")

        flat_ids = selected_ids.reshape(-1)
        flat_weights = selected_weights.reshape(-1)
        selection_counts = torch.bincount(flat_ids, minlength=num_experts)
        responsibility = torch.zeros(
            num_experts,
            dtype=torch.float64,
            device=selected_weights.device,
        )
        if flat_ids.numel():
            responsibility.scatter_add_(0, flat_ids, flat_weights)
        entropy_terms = torch.where(
            selected_weights > 0,
            -selected_weights * selected_weights.clamp_min(torch.finfo(torch.float64).tiny).log(),
            torch.zeros_like(selected_weights),
        )

        surprise_sum = 0.0
        surprise_count = 0
        if selected_scores is not None:
            if selected_scores.shape != expert_ids.shape:
                raise ValueError("selected_scores must have the router tensor shape")
            scores = selected_scores.detach().double().to(device=selected_weights.device)[valid]
            if scores.numel() and (
                not bool(torch.isfinite(scores).all().item())
                or bool((scores < 0).any().item())
            ):
                raise ValueError("selected surprise scores must be finite and non-negative")
            surprise_sum = float((scores * selected_weights).sum().item())
            surprise_count = token_count

        return cls(
            layer=layer,
            token_count=token_count,
            selection_counts=tuple(int(value) for value in selection_counts.cpu().tolist()),
            responsibility_mass=tuple(float(value) for value in responsibility.cpu().tolist()),
            entropy_sum=float(entropy_terms.sum().item()),
            active_experts_sum=int((selected_weights > 0).sum().item()),
            predicted_surprise_sum=surprise_sum,
            predicted_surprise_count=surprise_count,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "token_count": self.token_count,
            "selection_counts": list(self.selection_counts),
            "responsibility_mass": list(self.responsibility_mass),
            "entropy_sum": self.entropy_sum,
            "active_experts_sum": self.active_experts_sum,
            "predicted_surprise_sum": self.predicted_surprise_sum,
            "predicted_surprise_count": self.predicted_surprise_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RouterBatchStats:
        return cls(
            layer=int(payload["layer"]),
            token_count=int(payload["token_count"]),
            selection_counts=tuple(int(value) for value in payload["selection_counts"]),
            responsibility_mass=tuple(float(value) for value in payload["responsibility_mass"]),
            entropy_sum=float(payload["entropy_sum"]),
            active_experts_sum=int(payload["active_experts_sum"]),
            predicted_surprise_sum=float(payload.get("predicted_surprise_sum", 0.0)),
            predicted_surprise_count=int(payload.get("predicted_surprise_count", 0)),
        )


@dataclass(frozen=True, slots=True)
class RouterMetrics:
    token_count: int
    selection_counts: tuple[int, ...]
    responsibility_mass: tuple[float, ...]
    responsibility_fraction: tuple[float, ...]
    mean_entropy: float | None
    mean_active_experts: float | None
    mean_predicted_surprise: float | None
    utilized_experts: int
    dead_experts: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_count": self.token_count,
            "selection_counts": list(self.selection_counts),
            "responsibility_mass": list(self.responsibility_mass),
            "responsibility_fraction": list(self.responsibility_fraction),
            "mean_entropy": self.mean_entropy,
            "mean_active_experts": self.mean_active_experts,
            "mean_predicted_surprise": self.mean_predicted_surprise,
            "utilized_experts": self.utilized_experts,
            "dead_experts": list(self.dead_experts),
        }


@dataclass(frozen=True, slots=True)
class RouterTelemetryReport:
    num_experts: int
    overall: RouterMetrics
    by_layer: dict[int, RouterMetrics]

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_experts": self.num_experts,
            "overall": self.overall.to_dict(),
            "by_layer": {
                str(layer): metrics.to_dict() for layer, metrics in sorted(self.by_layer.items())
            },
        }


@dataclass(slots=True)
class _RouterTotals:
    token_count: int
    selection_counts: list[int]
    responsibility_mass: list[float]
    entropy_sum: float
    active_experts_sum: int
    predicted_surprise_sum: float
    predicted_surprise_count: int

    @classmethod
    def empty(cls, num_experts: int) -> _RouterTotals:
        return cls(0, [0] * num_experts, [0.0] * num_experts, 0.0, 0, 0.0, 0)

    def add(self, stats: RouterBatchStats) -> None:
        self.token_count += stats.token_count
        self.selection_counts = [
            left + right for left, right in zip(self.selection_counts, stats.selection_counts)
        ]
        self.responsibility_mass = [
            left + right
            for left, right in zip(self.responsibility_mass, stats.responsibility_mass)
        ]
        self.entropy_sum += stats.entropy_sum
        self.active_experts_sum += stats.active_experts_sum
        self.predicted_surprise_sum += stats.predicted_surprise_sum
        self.predicted_surprise_count += stats.predicted_surprise_count

    def metrics(self) -> RouterMetrics:
        mass = sum(self.responsibility_mass)
        fractions = tuple(value / mass if mass else 0.0 for value in self.responsibility_mass)
        dead = tuple(index for index, count in enumerate(self.selection_counts) if count == 0)
        return RouterMetrics(
            token_count=self.token_count,
            selection_counts=tuple(self.selection_counts),
            responsibility_mass=tuple(self.responsibility_mass),
            responsibility_fraction=fractions,
            mean_entropy=self.entropy_sum / self.token_count if self.token_count else None,
            mean_active_experts=(
                self.active_experts_sum / self.token_count if self.token_count else None
            ),
            mean_predicted_surprise=(
                self.predicted_surprise_sum / self.predicted_surprise_count
                if self.predicted_surprise_count
                else None
            ),
            utilized_experts=len(self.selection_counts) - len(dead),
            dead_experts=dead,
        )


class RouterTelemetryAccumulator:
    """Merge router selection, responsibility, entropy, and surprise summaries."""

    def __init__(self, num_experts: int) -> None:
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        self.num_experts = num_experts
        self._by_layer: dict[int, _RouterTotals] = {}

    def update(self, stats: RouterBatchStats) -> None:
        if stats.num_experts != self.num_experts:
            raise ValueError("router batch expert count does not match accumulator")
        self._by_layer.setdefault(stats.layer, _RouterTotals.empty(self.num_experts)).add(stats)

    def update_tensors(
        self,
        layer: int,
        expert_ids: Tensor,
        routing_weights: Tensor,
        *,
        valid_mask: Tensor | None = None,
        selected_scores: Tensor | None = None,
    ) -> None:
        self.update(
            RouterBatchStats.from_tensors(
                layer,
                expert_ids,
                routing_weights,
                num_experts=self.num_experts,
                valid_mask=valid_mask,
                selected_scores=selected_scores,
            )
        )

    def merge(self, other: RouterTelemetryAccumulator) -> None:
        if self.num_experts != other.num_experts:
            raise ValueError("cannot merge router accumulators with different expert counts")
        for layer, totals in other._by_layer.items():
            target = self._by_layer.setdefault(layer, _RouterTotals.empty(self.num_experts))
            target.add(
                RouterBatchStats(
                    layer=layer,
                    token_count=totals.token_count,
                    selection_counts=tuple(totals.selection_counts),
                    responsibility_mass=tuple(totals.responsibility_mass),
                    entropy_sum=totals.entropy_sum,
                    active_experts_sum=totals.active_experts_sum,
                    predicted_surprise_sum=totals.predicted_surprise_sum,
                    predicted_surprise_count=totals.predicted_surprise_count,
                )
            )

    def finalize(self) -> RouterTelemetryReport:
        overall = _RouterTotals.empty(self.num_experts)
        for layer, totals in self._by_layer.items():
            overall.add(
                RouterBatchStats(
                    layer=layer,
                    token_count=totals.token_count,
                    selection_counts=tuple(totals.selection_counts),
                    responsibility_mass=tuple(totals.responsibility_mass),
                    entropy_sum=totals.entropy_sum,
                    active_experts_sum=totals.active_experts_sum,
                    predicted_surprise_sum=totals.predicted_surprise_sum,
                    predicted_surprise_count=totals.predicted_surprise_count,
                )
            )
        return RouterTelemetryReport(
            num_experts=self.num_experts,
            overall=overall.metrics(),
            by_layer={layer: totals.metrics() for layer, totals in self._by_layer.items()},
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": ROUTER_FORMAT,
            "format_version": ROUTER_VERSION,
            "num_experts": self.num_experts,
            "by_layer": [
                RouterBatchStats(
                    layer=layer,
                    token_count=totals.token_count,
                    selection_counts=tuple(totals.selection_counts),
                    responsibility_mass=tuple(totals.responsibility_mass),
                    entropy_sum=totals.entropy_sum,
                    active_experts_sum=totals.active_experts_sum,
                    predicted_surprise_sum=totals.predicted_surprise_sum,
                    predicted_surprise_count=totals.predicted_surprise_count,
                ).to_dict()
                for layer, totals in sorted(self._by_layer.items())
            ],
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> RouterTelemetryAccumulator:
        if payload.get("format") != ROUTER_FORMAT:
            raise ValueError("not a SPALMER router-telemetry accumulator")
        if payload.get("format_version") != ROUTER_VERSION:
            raise ValueError(
                f"unsupported router-telemetry version: {payload.get('format_version')!r}"
            )
        accumulator = cls(int(payload["num_experts"]))
        for raw_batch in payload["by_layer"]:
            if not isinstance(raw_batch, Mapping):
                raise TypeError("router batch state must be a mapping")
            accumulator.update(RouterBatchStats.from_dict(raw_batch))
        return accumulator


def _int_keyed_counts(value: Any, orders: tuple[int, ...]) -> dict[int, int]:
    if not isinstance(value, Mapping):
        raise TypeError("n-gram count state must be a mapping")
    counts = {int(order): int(count) for order, count in value.items()}
    if set(counts) != set(orders):
        raise ValueError("n-gram count orders do not match accumulator")
    return counts


__all__ = [
    "NgramRepetitionMetrics",
    "REPETITION_FORMAT",
    "REPETITION_VERSION",
    "ROUTER_FORMAT",
    "ROUTER_VERSION",
    "RepetitionAccumulator",
    "RepetitionReport",
    "RouterBatchStats",
    "RouterMetrics",
    "RouterTelemetryAccumulator",
    "RouterTelemetryReport",
]
