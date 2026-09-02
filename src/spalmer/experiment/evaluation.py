"""Streaming, model-agnostic held-out causal language-model evaluation."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from torch import Tensor

ACCUMULATOR_FORMAT = "spalmer.experiment.causal-metrics"
ACCUMULATOR_VERSION = 1

BatchT = TypeVar("BatchT")


@dataclass(frozen=True, order=True, slots=True)
class EvaluationStratum:
    """An English-prose or programming-language evaluation stratum."""

    kind: Literal["english", "code"]
    language: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "english" and self.language is not None:
            raise ValueError("English stratum cannot carry a code language")
        if self.kind == "code" and (self.language is None or not self.language.strip()):
            raise ValueError("code stratum requires a language")

    @property
    def key(self) -> str:
        return "english" if self.kind == "english" else f"code:{self.language}"

    @classmethod
    def from_key(cls, key: str) -> EvaluationStratum:
        if key == "english":
            return cls("english")
        prefix = "code:"
        if key.startswith(prefix) and key[len(prefix) :]:
            return cls("code", key[len(prefix) :])
        raise ValueError(f"invalid evaluation stratum key: {key!r}")


ENGLISH = EvaluationStratum("english")


@dataclass(frozen=True, slots=True)
class CausalBatchStats:
    """Detached sufficient statistics for one homogeneous held-out batch.

    NLL is measured in nats. ``target_bytes`` is the number of original UTF-8
    payload bytes represented by predicted targets, excluding an unpredicted
    prefix and zero-payload special tokens.  That denominator makes bits per
    byte comparable across tokenizers and programming languages.
    """

    stratum: EvaluationStratum
    nll_sum: float
    target_tokens: int
    target_bytes: int
    documents: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.nll_sum) or self.nll_sum < 0:
            raise ValueError("nll_sum must be finite and non-negative")
        if min(self.target_tokens, self.target_bytes, self.documents) < 0:
            raise ValueError("target_tokens, target_bytes, and documents must be non-negative")
        if self.target_tokens == 0 and self.nll_sum != 0:
            raise ValueError("a batch without target tokens cannot carry NLL")

    @classmethod
    def from_token_nll(
        cls,
        stratum: EvaluationStratum,
        token_nll: Tensor,
        *,
        target_bytes: int,
        valid_mask: Tensor | None = None,
        documents: int = 0,
    ) -> CausalBatchStats:
        """Reduce a detached NLL tensor without retaining an autograd graph."""

        if not isinstance(token_nll, Tensor) or not token_nll.is_floating_point():
            raise TypeError("token_nll must be a floating-point tensor")
        values = token_nll.detach()
        if valid_mask is None:
            selected = values.reshape(-1)
        else:
            if not isinstance(valid_mask, Tensor) or valid_mask.shape != values.shape:
                raise ValueError("valid_mask must be a tensor with the token_nll shape")
            selected = values[valid_mask.detach().bool()]
        return cls(
            stratum=stratum,
            nll_sum=float(selected.double().sum().item()),
            target_tokens=selected.numel(),
            target_bytes=target_bytes,
            documents=documents,
        )


@dataclass(frozen=True, slots=True)
class CausalMetrics:
    nll_sum: float
    target_tokens: int
    target_bytes: int
    documents: int
    mean_nll: float | None
    perplexity: float | None
    bits_per_byte: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "nll_sum": self.nll_sum,
            "target_tokens": self.target_tokens,
            "target_bytes": self.target_bytes,
            "documents": self.documents,
            "mean_nll": self.mean_nll,
            "perplexity": self.perplexity,
            "bits_per_byte": self.bits_per_byte,
        }


@dataclass(frozen=True, slots=True)
class CrossStratumMetrics:
    strata: int
    mean_nll: float
    perplexity: float
    bits_per_byte: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "strata": self.strata,
            "mean_nll": self.mean_nll,
            "perplexity": self.perplexity,
            "bits_per_byte": self.bits_per_byte,
        }


@dataclass(frozen=True, slots=True)
class CausalEvaluationReport:
    overall: CausalMetrics
    by_stratum: dict[EvaluationStratum, CausalMetrics]
    macro: CrossStratumMetrics | None
    weighted: CrossStratumMetrics | None
    mixture_weights: dict[EvaluationStratum, float] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall.to_dict(),
            "by_stratum": {
                stratum.key: metrics.to_dict()
                for stratum, metrics in sorted(self.by_stratum.items())
            },
            "macro": None if self.macro is None else self.macro.to_dict(),
            "weighted": None if self.weighted is None else self.weighted.to_dict(),
            "mixture_weights": (
                None
                if self.mixture_weights is None
                else {
                    stratum.key: weight
                    for stratum, weight in sorted(self.mixture_weights.items())
                }
            ),
        }


@dataclass(slots=True)
class _Totals:
    nll_sum: float = 0.0
    target_tokens: int = 0
    target_bytes: int = 0
    documents: int = 0

    def add(self, stats: CausalBatchStats) -> None:
        self.nll_sum += stats.nll_sum
        self.target_tokens += stats.target_tokens
        self.target_bytes += stats.target_bytes
        self.documents += stats.documents

    def merge(self, other: _Totals) -> None:
        self.nll_sum += other.nll_sum
        self.target_tokens += other.target_tokens
        self.target_bytes += other.target_bytes
        self.documents += other.documents

    def metrics(self) -> CausalMetrics:
        mean_nll = self.nll_sum / self.target_tokens if self.target_tokens else None
        perplexity = _safe_exp(mean_nll) if mean_nll is not None else None
        bits_per_byte = (
            self.nll_sum / (math.log(2.0) * self.target_bytes)
            if self.target_bytes
            else None
        )
        return CausalMetrics(
            nll_sum=self.nll_sum,
            target_tokens=self.target_tokens,
            target_bytes=self.target_bytes,
            documents=self.documents,
            mean_nll=mean_nll,
            perplexity=perplexity,
            bits_per_byte=bits_per_byte,
        )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "nll_sum": self.nll_sum,
            "target_tokens": self.target_tokens,
            "target_bytes": self.target_bytes,
            "documents": self.documents,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> _Totals:
        totals = cls(
            nll_sum=float(payload["nll_sum"]),
            target_tokens=int(payload["target_tokens"]),
            target_bytes=int(payload["target_bytes"]),
            documents=int(payload["documents"]),
        )
        CausalBatchStats(ENGLISH, **totals.to_dict())
        return totals


class CausalMetricsAccumulator:
    """Mergeable sufficient-statistics accumulator for streaming evaluation."""

    def __init__(self) -> None:
        self._by_stratum: dict[EvaluationStratum, _Totals] = {}

    def update(self, stats: CausalBatchStats) -> None:
        self._by_stratum.setdefault(stats.stratum, _Totals()).add(stats)

    def merge(self, other: CausalMetricsAccumulator) -> None:
        for stratum, totals in other._by_stratum.items():
            self._by_stratum.setdefault(stratum, _Totals()).merge(totals)

    def finalize(
        self,
        mixture_weights: Mapping[EvaluationStratum | str, float] | None = None,
    ) -> CausalEvaluationReport:
        by_stratum = {
            stratum: totals.metrics()
            for stratum, totals in self._by_stratum.items()
            if totals.target_tokens
        }
        overall_totals = _Totals()
        for totals in self._by_stratum.values():
            overall_totals.merge(totals)
        macro = _cross_stratum(by_stratum, None)
        normalized_weights = _normalize_weights(mixture_weights, by_stratum)
        weighted = (
            None
            if normalized_weights is None
            else _cross_stratum(by_stratum, normalized_weights)
        )
        return CausalEvaluationReport(
            overall=overall_totals.metrics(),
            by_stratum=by_stratum,
            macro=macro,
            weighted=weighted,
            mixture_weights=normalized_weights,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": ACCUMULATOR_FORMAT,
            "format_version": ACCUMULATOR_VERSION,
            "by_stratum": {
                stratum.key: totals.to_dict()
                for stratum, totals in sorted(self._by_stratum.items())
            },
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> CausalMetricsAccumulator:
        if payload.get("format") != ACCUMULATOR_FORMAT:
            raise ValueError("not a SPALMER causal-metrics accumulator")
        if payload.get("format_version") != ACCUMULATOR_VERSION:
            raise ValueError(
                f"unsupported causal-metrics version: {payload.get('format_version')!r}"
            )
        raw_strata = payload.get("by_stratum")
        if not isinstance(raw_strata, Mapping):
            raise TypeError("by_stratum must be a mapping")
        accumulator = cls()
        for key, raw_totals in raw_strata.items():
            if not isinstance(raw_totals, Mapping):
                raise TypeError("stratum totals must be a mapping")
            accumulator._by_stratum[EvaluationStratum.from_key(str(key))] = _Totals.from_dict(
                raw_totals
            )
        return accumulator


def evaluate_stream(
    batches: Iterable[BatchT],
    evaluate_batch: Callable[
        [BatchT], CausalBatchStats | Iterable[CausalBatchStats]
    ],
    *,
    mixture_weights: Mapping[EvaluationStratum | str, float] | None = None,
    on_batch: Callable[[CausalBatchStats], None] | None = None,
) -> CausalEvaluationReport:
    """Accumulate callback-produced statistics without owning model execution."""

    accumulator = CausalMetricsAccumulator()
    for batch in batches:
        result = evaluate_batch(batch)
        stats_items = (result,) if isinstance(result, CausalBatchStats) else result
        for stats in stats_items:
            accumulator.update(stats)
            if on_batch is not None:
                on_batch(stats)
    return accumulator.finalize(mixture_weights)


def _cross_stratum(
    metrics: Mapping[EvaluationStratum, CausalMetrics],
    weights: Mapping[EvaluationStratum, float] | None,
) -> CrossStratumMetrics | None:
    if not metrics:
        return None
    effective = (
        {stratum: 1.0 / len(metrics) for stratum in metrics}
        if weights is None
        else weights
    )
    mean_nll = sum(
        effective[stratum] * _required(item.mean_nll)
        for stratum, item in metrics.items()
    )
    bpb_items = {
        stratum: item.bits_per_byte
        for stratum, item in metrics.items()
        if item.bits_per_byte is not None
    }
    if not bpb_items:
        bits_per_byte = None
    else:
        denominator = sum(effective[stratum] for stratum in bpb_items)
        bits_per_byte = sum(
            effective[stratum] * _required(value) for stratum, value in bpb_items.items()
        ) / denominator
    return CrossStratumMetrics(
        strata=len(metrics),
        mean_nll=mean_nll,
        perplexity=_safe_exp(mean_nll),
        bits_per_byte=bits_per_byte,
    )


def _normalize_weights(
    weights: Mapping[EvaluationStratum | str, float] | None,
    metrics: Mapping[EvaluationStratum, CausalMetrics],
) -> dict[EvaluationStratum, float] | None:
    if weights is None:
        return None
    converted: dict[EvaluationStratum, float] = {}
    for raw_stratum, raw_weight in weights.items():
        stratum = (
            EvaluationStratum.from_key(raw_stratum)
            if isinstance(raw_stratum, str)
            else raw_stratum
        )
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("mixture weights must be finite and non-negative")
        converted[stratum] = converted.get(stratum, 0.0) + weight
    missing_observations = {
        stratum for stratum, weight in converted.items() if weight > 0 and stratum not in metrics
    }
    if missing_observations:
        names = ", ".join(sorted(stratum.key for stratum in missing_observations))
        raise ValueError(f"weighted evaluation has no observations for: {names}")
    unspecified = set(metrics) - set(converted)
    if unspecified:
        names = ", ".join(sorted(stratum.key for stratum in unspecified))
        raise ValueError(f"mixture weights are missing evaluated strata: {names}")
    total = sum(converted[stratum] for stratum in metrics)
    if total <= 0:
        raise ValueError("at least one mixture weight must be positive")
    return {stratum: converted[stratum] / total for stratum in metrics}


def _safe_exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def _required(value: float | None) -> float:
    if value is None:
        raise ValueError("metric is unavailable")
    return value


__all__ = [
    "ACCUMULATOR_FORMAT",
    "ACCUMULATOR_VERSION",
    "CausalBatchStats",
    "CausalEvaluationReport",
    "CausalMetrics",
    "CausalMetricsAccumulator",
    "CrossStratumMetrics",
    "ENGLISH",
    "EvaluationStratum",
    "evaluate_stream",
]
