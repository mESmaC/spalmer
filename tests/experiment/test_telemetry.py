from __future__ import annotations

import math

import pytest
import torch

from spalmer.experiment.telemetry import (
    RepetitionAccumulator,
    RouterTelemetryAccumulator,
)


def test_repetition_accumulator_measures_within_sequence_repeats_and_round_trips():
    accumulator = RepetitionAccumulator((1, 2, 3))
    accumulator.update([1, 2, 1, 2, 1])
    accumulator.update([])
    accumulator.update([3, 3, 3])

    report = accumulator.finalize()
    assert report.sequences == 3
    assert report.tokens == 8
    assert report.transitions == 6
    assert report.adjacent_repeats == 2
    assert report.adjacent_repeat_rate == pytest.approx(1 / 3)
    assert report.ngrams[1].occurrences == 8
    assert report.ngrams[1].repeated_occurrences == 5
    assert report.ngrams[2].occurrences == 6
    assert report.ngrams[2].repeated_occurrences == 3

    restored = RepetitionAccumulator.from_state_dict(accumulator.state_dict())
    assert restored.finalize() == report


def test_router_tensor_accumulator_tracks_responsibility_entropy_and_surprise():
    accumulator = RouterTelemetryAccumulator(num_experts=3)
    expert_ids = torch.tensor([[[0, 1], [1, 2]]])
    routing_weights = torch.tensor([[[0.75, 0.25], [0.5, 0.5]]])
    selected_scores = torch.tensor([[[1.0, 3.0], [2.0, 4.0]]])
    accumulator.update_tensors(
        0,
        expert_ids,
        routing_weights,
        valid_mask=torch.tensor([[True, False]]),
        selected_scores=selected_scores,
    )

    report = accumulator.finalize()
    layer = report.by_layer[0]
    assert layer.token_count == 1
    assert layer.selection_counts == (1, 1, 0)
    assert layer.responsibility_mass == pytest.approx((0.75, 0.25, 0.0))
    assert layer.responsibility_fraction == pytest.approx((0.75, 0.25, 0.0))
    assert layer.mean_entropy == pytest.approx(-(0.75 * math.log(0.75) + 0.25 * math.log(0.25)))
    assert layer.mean_active_experts == 2.0
    assert layer.mean_predicted_surprise == 1.5
    assert layer.utilized_experts == 2
    assert layer.dead_experts == (2,)

    restored = RouterTelemetryAccumulator.from_state_dict(accumulator.state_dict())
    assert restored.finalize() == report


def test_router_accumulator_rejects_non_normalized_weights():
    accumulator = RouterTelemetryAccumulator(num_experts=2)
    with pytest.raises(ValueError, match="sum to one"):
        accumulator.update_tensors(
            0,
            torch.tensor([[0, 1]]),
            torch.tensor([[0.8, 0.8]]),
        )
