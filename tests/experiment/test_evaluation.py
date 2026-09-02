from __future__ import annotations

import math

import pytest
import torch

from spalmer.experiment.evaluation import (
    ENGLISH,
    CausalBatchStats,
    CausalMetricsAccumulator,
    EvaluationStratum,
    evaluate_stream,
)

PYTHON = EvaluationStratum("code", "python")


def test_streaming_causal_metrics_report_overall_macro_and_weighted_values():
    accumulator = CausalMetricsAccumulator()
    accumulator.update(CausalBatchStats(ENGLISH, 4.0, 2, 4, documents=1))
    accumulator.update(CausalBatchStats(PYTHON, 3.0, 3, 6, documents=2))

    report = accumulator.finalize({ENGLISH: 0.25, PYTHON: 0.75})
    assert report.overall.nll_sum == 7.0
    assert report.overall.target_tokens == 5
    assert report.overall.mean_nll == pytest.approx(1.4)
    assert report.overall.perplexity == pytest.approx(math.exp(1.4))
    assert report.overall.bits_per_byte == pytest.approx(7.0 / (10 * math.log(2)))
    assert report.by_stratum[ENGLISH].mean_nll == 2.0
    assert report.by_stratum[PYTHON].mean_nll == 1.0
    assert report.macro.mean_nll == 1.5
    assert report.weighted.mean_nll == 1.25
    assert report.weighted.perplexity == pytest.approx(math.exp(1.25))
    assert report.mixture_weights == {ENGLISH: 0.25, PYTHON: 0.75}


def test_tensor_reduction_honors_valid_mask_and_detaches_values():
    token_nll = torch.tensor([[0.5, 1.0], [2.0, 4.0]], requires_grad=True)
    valid = torch.tensor([[True, False], [True, False]])
    stats = CausalBatchStats.from_token_nll(
        PYTHON,
        token_nll,
        valid_mask=valid,
        target_bytes=7,
        documents=1,
    )
    assert stats.nll_sum == 2.5
    assert stats.target_tokens == 2
    assert stats.target_bytes == 7
    assert token_nll.grad is None


def test_accumulator_state_round_trip_is_mergeable():
    left = CausalMetricsAccumulator()
    left.update(CausalBatchStats(ENGLISH, 1.0, 1, 2))
    restored = CausalMetricsAccumulator.from_state_dict(left.state_dict())
    right = CausalMetricsAccumulator()
    right.update(CausalBatchStats(ENGLISH, 2.0, 1, 2))
    restored.merge(right)
    assert restored.finalize().overall.mean_nll == 1.5


def test_evaluate_stream_accepts_callback_and_emits_batch_hook():
    seen: list[str] = []

    def evaluate(value: int) -> CausalBatchStats:
        stratum = ENGLISH if value % 2 else PYTHON
        return CausalBatchStats(stratum, float(value), 1, value)

    report = evaluate_stream(
        [1, 2, 3],
        evaluate,
        mixture_weights={"english": 0.6, "code:python": 0.4},
        on_batch=lambda stats: seen.append(stats.stratum.key),
    )
    assert report.overall.target_tokens == 3
    assert seen == ["english", "code:python", "english"]


def test_weighted_report_rejects_missing_language_observations():
    accumulator = CausalMetricsAccumulator()
    accumulator.update(CausalBatchStats(ENGLISH, 1.0, 1, 1))
    with pytest.raises(ValueError, match="no observations"):
        accumulator.finalize({ENGLISH: 0.5, PYTHON: 0.5})


def test_explicit_zero_weight_can_exclude_an_observed_stratum():
    accumulator = CausalMetricsAccumulator()
    accumulator.update(CausalBatchStats(ENGLISH, 2.0, 1, 1))
    accumulator.update(CausalBatchStats(PYTHON, 10.0, 1, 1))
    report = accumulator.finalize({ENGLISH: 1.0, PYTHON: 0.0})
    assert report.weighted.mean_nll == 2.0
