"""Inference residency controller (SPALMER ledger C13, using the C08 rule).

The ledger's candidate controller sequence is::

    load shared base + vocab + router + two-expert minimum
            -> evaluate observed NLL/CE and target-free predictive entropy
            -> rank eligible experts by predicted surprise
            -> add a bounded increment
            -> recompute
            -> retain, expand, or roll back within the current 10% soft cap

This is the request-level form of that loop, applied at prefill: the active
expert count starts at the configured minimum and grows by bounded increments
while the effective surprise of the prompt is at or above the model's average
surprise (C08: ``if effective NLL >= average surprise: include additional
experts; recompute``). Each expansion is retained only if recomputing the
signal shows it bought at least ``residency_min_gain``; otherwise the
controller rolls back. Ranking the eligible experts by predicted surprise is
the router's job, so every evaluation already executes the least-surprised
experts for the current count.

Token-level changes during decoding, prefetch/eviction of nonresident
experts, and precision changes tied to residency remain open interfaces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from spalmer.modeling import CausalLMOutput, SPALMERCausalLM


@dataclass(slots=True)
class ResidencyDecision:
    """One request-level residency decision and the evidence behind it."""

    active_experts: int
    effective_nll: float | None
    predictive_entropy: float
    average_surprise: float
    trace: tuple[tuple[int, float], ...]
    output: CausalLMOutput

    @property
    def effective_signal(self) -> float:
        """Observed prompt NLL when targets exist, else predictive entropy."""

        return self.predictive_entropy if self.effective_nll is None else self.effective_nll


def choose_inference_residency(
    model: SPALMERCausalLM,
    prompt_ids: Tensor,
    *,
    increment: int | None = None,
    min_gain: float | None = None,
) -> ResidencyDecision:
    """Pick how many experts this request activates and leave the model set to it.

    Args:
        model: A routed SPALMER model. The chosen count stays applied on return
            so decoding continues with the same residency; callers restore it
            with ``model.set_active_experts(None)`` when they are done.
        prompt_ids: ``[1, tokens]`` prompt. With two or more tokens the observed
            next-token NLL of the prompt is the effective signal; a one-token
            prompt falls back to the target-free predictive entropy.
        increment: Experts added per expansion step (defaults to the expert
            configuration's ``residency_increment``).
        min_gain: Smallest signal improvement, in nats per token, that keeps an
            expansion (defaults to ``residency_min_gain``). A negative value
            retains every expansion up to the cap, which forces maximum
            residency for comparisons.
    """

    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] == 0:
        raise ValueError("prompt_ids must describe one non-empty sequence")
    mixer = _first_routed_mixer(model)
    config = mixer.config
    step = config.residency_increment if increment is None else increment
    gain = config.residency_min_gain if min_gain is None else min_gain
    if step <= 0:
        raise ValueError("increment must be positive")
    if not math.isfinite(gain):
        raise ValueError("min_gain must be finite")
    minimum = config.min_active_experts
    cap = mixer.max_active_experts
    average = model.average_surprise

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            count = minimum
            nll, entropy, signal = _measure(model, prompt_ids, count)
            trace = [(count, signal)]
            while count < cap and signal >= average:
                candidate = min(count + step, cap)
                candidate_nll, candidate_entropy, candidate_signal = _measure(
                    model, prompt_ids, candidate
                )
                trace.append((candidate, candidate_signal))
                if signal - candidate_signal < gain:
                    model.set_active_experts(count)
                    break
                count, nll, entropy, signal = (
                    candidate,
                    candidate_nll,
                    candidate_entropy,
                    candidate_signal,
                )
            # Trial outputs are released before this one authoritative prefill,
            # avoiding two prompt-sized logits/cache bundles in memory at once.
            model.set_active_experts(count)
            output = model(prompt_ids)
    finally:
        model.train(was_training)

    return ResidencyDecision(
        active_experts=count,
        effective_nll=nll,
        predictive_entropy=entropy,
        average_surprise=average,
        trace=tuple(trace),
        output=output,
    )


def _measure(
    model: SPALMERCausalLM, prompt_ids: Tensor, count: int
) -> tuple[float | None, float, float]:
    model.set_active_experts(count)
    if prompt_ids.shape[1] >= 2:
        output = model(prompt_ids, labels=prompt_ids)
    else:
        output = model(prompt_ids)
    nll = None if output.token_nll is None else float(output.token_nll.float().mean())
    if output.predictive_entropy is None:
        raise RuntimeError("model did not return predictive entropy telemetry")
    entropy = float(output.predictive_entropy.float().mean())
    return nll, entropy, entropy if nll is None else nll


def _first_routed_mixer(model: SPALMERCausalLM):
    for block in model.backbone.blocks:
        mixer = block.channel_mixer
        if callable(getattr(mixer, "set_active_experts", None)) and hasattr(mixer, "config"):
            return mixer
    raise TypeError("dynamic residency requires routed micro-expert channel mixers")


__all__ = ["ResidencyDecision", "choose_inference_residency"]
