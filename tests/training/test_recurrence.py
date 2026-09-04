"""Contracts for the stateless per-step recurrence-depth sampler."""

from __future__ import annotations

import pytest
import torch

from spalmer.training import RecurrenceSampler, TrainingConfig


def _config(**overrides: object) -> TrainingConfig:
    values: dict[str, object] = {
        "max_steps": 10,
        "micro_batch_size": 1,
        "sequence_length": 4,
        "device": "cpu",
        "require_cuda": False,
        "compute_dtype": "float32",
        "parameter_dtype": "float32",
    }
    values.update(overrides)
    return TrainingConfig(**values)  # type: ignore[arg-type]


def test_sampler_is_stateless_and_seed_step_deterministic() -> None:
    sampler = RecurrenceSampler(mean_recurrence=8.0, mean_backprop_depth=4, seed=3)

    torch.manual_seed(0)
    before = torch.rand(1)
    draws = [sampler.sample(step) for step in range(64)]
    torch.manual_seed(0)
    after = torch.rand(1)

    # The sampler must never consume the global torch stream that PLE
    # stochastic rounding and expert QAT share.
    torch.testing.assert_close(before, after)
    assert [sampler.sample(step) for step in range(64)] == draws
    assert RecurrenceSampler(
        mean_recurrence=8.0, mean_backprop_depth=4, seed=3
    ).sample(7) == sampler.sample(7)
    assert RecurrenceSampler(
        mean_recurrence=8.0, mean_backprop_depth=4, seed=4
    ).sample(7) != sampler.sample(7)
    assert len({steps for steps, _ in draws}) > 1


def test_sampler_mean_and_bounds() -> None:
    sampler = RecurrenceSampler(mean_recurrence=8.0, mean_backprop_depth=4, sigma=0.5)
    draws = [sampler.sample(step) for step in range(20_000)]
    steps = [value for value, _ in draws]

    assert abs(sum(steps) / len(steps) - 8.0) < 0.15
    assert all(value >= 1 for value in steps)
    assert all(depth == min(4, value) for value, depth in draws)
    assert sampler.resolved_max_recurrence == 32
    assert max(steps) <= 32

    capped = RecurrenceSampler(mean_recurrence=8.0, mean_backprop_depth=4, max_recurrence=9)
    assert max(capped.sample(step)[0] for step in range(2_000)) <= 9

    flat = RecurrenceSampler(mean_recurrence=1.0, mean_backprop_depth=4)
    assert {flat.sample(step) for step in range(64)} == {(1, 1)}

    fixed = RecurrenceSampler(mean_recurrence=6.0, mean_backprop_depth=4, scheme="fixed")
    assert {fixed.sample(step) for step in range(32)} == {(6, 4)}

    uniform = RecurrenceSampler(mean_recurrence=4.0, mean_backprop_depth=2, scheme="uniform")
    uniform_steps = [uniform.sample(step)[0] for step in range(20_000)]
    assert set(uniform_steps) == {1, 2, 3, 4, 5, 6, 7}
    assert abs(sum(uniform_steps) / len(uniform_steps) - 4.0) < 0.15


def test_sampler_from_training_config_round_trips_every_field() -> None:
    assert RecurrenceSampler.from_training_config(_config()) is None

    sampler = RecurrenceSampler.from_training_config(
        _config(
            mean_recurrence=5.0,
            mean_backprop_depth=3,
            recurrence_sigma=0.25,
            recurrence_sampling="uniform",
            max_recurrence=11,
            seed=17,
        )
    )

    assert sampler == RecurrenceSampler(
        mean_recurrence=5.0,
        mean_backprop_depth=3,
        sigma=0.25,
        scheme="uniform",
        max_recurrence=11,
        seed=17,
    )


def test_training_config_recurrence_validation() -> None:
    assert _config().mean_recurrence is None
    assert _config().recurrence_sampling == "poisson_lognormal"
    assert _config().mean_backprop_depth == 4
    assert _config().max_recurrence is None

    with pytest.raises(ValueError, match="mean_recurrence must be at least 1"):
        _config(mean_recurrence=0.5)
    with pytest.raises(ValueError, match="mean_backprop_depth"):
        _config(mean_backprop_depth=0)
    with pytest.raises(ValueError, match="recurrence_sigma"):
        _config(recurrence_sigma=-0.1)
    with pytest.raises(ValueError, match="recurrence_sampling"):
        _config(recurrence_sampling="poisson")
    with pytest.raises(ValueError, match="max_recurrence must be positive"):
        _config(max_recurrence=0)
    with pytest.raises(ValueError, match="integer mean_recurrence"):
        _config(recurrence_sampling="fixed", mean_recurrence=4.5)
    with pytest.raises(ValueError, match="max_recurrence cannot be smaller"):
        _config(mean_recurrence=8.0, max_recurrence=4)

    recurrent = _config(mean_recurrence=8.0, max_recurrence=16)
    assert recurrent.to_dict()["mean_recurrence"] == 8.0
    assert recurrent.to_dict()["max_recurrence"] == 16


def test_sampler_rejects_inconsistent_construction() -> None:
    with pytest.raises(ValueError, match="mean_recurrence"):
        RecurrenceSampler(mean_recurrence=0.0)
    with pytest.raises(ValueError, match="unsupported recurrence scheme"):
        RecurrenceSampler(mean_recurrence=4.0, scheme="poisson")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integer mean_recurrence"):
        RecurrenceSampler(mean_recurrence=4.5, scheme="fixed")
    with pytest.raises(ValueError, match="step cannot be negative"):
        RecurrenceSampler(mean_recurrence=4.0).sample(-1)
