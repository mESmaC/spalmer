from __future__ import annotations

import pytest

from spalmer.training import TrainingConfig


def test_token_budget_includes_accumulation() -> None:
    config = TrainingConfig(
        max_steps=10,
        micro_batch_size=3,
        sequence_length=128,
        gradient_accumulation_steps=4,
        warmup_steps=2,
    )

    assert config.tokens_per_optimizer_step == 1_536
    assert config.token_budget == 15_360


def test_schedule_warms_then_decays_to_floor() -> None:
    config = TrainingConfig(
        max_steps=10,
        micro_batch_size=1,
        sequence_length=8,
        warmup_steps=2,
        min_learning_rate_ratio=0.2,
    )

    assert config.learning_rate_multiplier(0) == pytest.approx(0.5)
    assert config.learning_rate_multiplier(1) == pytest.approx(1.0)
    assert config.learning_rate_multiplier(2) == pytest.approx(1.0)
    assert config.learning_rate_multiplier(9) == pytest.approx(0.2)


def test_invalid_warmup_is_rejected() -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        TrainingConfig(max_steps=5, micro_batch_size=1, sequence_length=8, warmup_steps=5)
