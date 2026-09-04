"""Per-optimizer-step depth sampling for the recurrent latent core.

The sampler is deliberately stateless: ``r`` is a pure function of the run
seed and the completed step index, so resuming a run needs no extra run-state
payload and the draws never touch the global Python/torch RNG streams that
PLE stochastic rounding and expert QAT depend on.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

from spalmer.training.config import TrainingConfig

RecurrenceScheme = Literal["poisson_lognormal", "uniform", "fixed"]

_KNUTH_RATE_LIMIT = 500.0


@dataclass(frozen=True, slots=True)
class RecurrenceSampler:
    """Draw ``(recurrence_steps, backprop_steps)`` for one optimizer step."""

    mean_recurrence: float
    mean_backprop_depth: int = 4
    sigma: float = 0.5
    scheme: RecurrenceScheme = "poisson_lognormal"
    max_recurrence: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mean_recurrence < 1:
            raise ValueError("mean_recurrence must be at least 1")
        if self.mean_backprop_depth < 1:
            raise ValueError("mean_backprop_depth must be positive")
        if self.sigma < 0:
            raise ValueError("sigma cannot be negative")
        if self.scheme not in {"poisson_lognormal", "uniform", "fixed"}:
            raise ValueError(f"unsupported recurrence scheme: {self.scheme!r}")
        if self.max_recurrence is not None and self.max_recurrence < 1:
            raise ValueError("max_recurrence must be positive or None")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if self.scheme == "fixed" and self.mean_recurrence != int(self.mean_recurrence):
            raise ValueError("fixed recurrence sampling requires an integer mean_recurrence")

    @classmethod
    def from_training_config(cls, config: TrainingConfig) -> RecurrenceSampler | None:
        """Return ``None`` for a non-recurrent :class:`TrainingConfig`."""

        if config.mean_recurrence is None:
            return None
        return cls(
            mean_recurrence=float(config.mean_recurrence),
            mean_backprop_depth=int(config.mean_backprop_depth),
            sigma=float(config.recurrence_sigma),
            scheme=config.recurrence_sampling,
            max_recurrence=config.max_recurrence,
            seed=int(config.seed),
        )

    @property
    def resolved_max_recurrence(self) -> int:
        """The hard cap on a drawn ``r`` (``4 * ceil(mean)`` when unset)."""

        if self.max_recurrence is not None:
            return int(self.max_recurrence)
        return max(1, 4 * math.ceil(self.mean_recurrence))

    def sample(self, step: int) -> tuple[int, int]:
        """Return ``(r, k_eff)`` for ``step`` (0-based completed steps)."""

        if step < 0:
            raise ValueError("step cannot be negative")
        rng = random.Random(f"spalmer-recurrence:{self.seed}:{step}")
        if self.scheme == "fixed":
            steps = int(self.mean_recurrence)
        elif self.mean_recurrence <= 1.0:
            steps = 1
        elif self.scheme == "uniform":
            steps = rng.randint(1, max(1, 2 * round(self.mean_recurrence) - 1))
        else:
            # rate ~ LogNormal(mu, sigma) with E[rate] = mean - 1, so that
            # E[r] = E[Poisson(rate)] + 1 == mean_recurrence exactly.
            mu = math.log(self.mean_recurrence - 1.0) - self.sigma**2 / 2
            rate = math.exp(rng.gauss(mu, self.sigma))
            steps = _poisson(rng, rate) + 1
        steps = max(1, min(steps, self.resolved_max_recurrence))
        return steps, min(self.mean_backprop_depth, steps)


def _poisson(rng: random.Random, rate: float) -> int:
    """Knuth's product method, with a normal approximation for large rates."""

    if rate <= 0:
        return 0
    if rate > _KNUTH_RATE_LIMIT:
        return max(0, round(rng.gauss(rate, math.sqrt(rate))))
    limit = math.exp(-rate)
    draws = 0
    product = 1.0
    while True:
        product *= rng.random()
        if product <= limit:
            return draws
        draws += 1


__all__ = ["RecurrenceSampler", "RecurrenceScheme"]
