"""Held-out model evaluation over finite host-backed batches."""

from __future__ import annotations

import contextlib
import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from spalmer.experiment import (
    CausalBatchStats,
    CausalEvaluationReport,
    CausalMetricsAccumulator,
    EvaluationStratum,
    RouterTelemetryAccumulator,
    RouterTelemetryReport,
)
from spalmer.modeling import CausalLMOutput, SPALMERCausalLM
from spalmer.tokenizer.backends import TokenizerBackend
from spalmer.training.engine import CausalBatch


@dataclass(frozen=True, slots=True)
class ModelEvaluationReport:
    causal: CausalEvaluationReport
    router: RouterTelemetryReport | None
    batches: int

    def to_dict(self) -> dict[str, object]:
        return {
            "causal": self.causal.to_dict(),
            "router": None if self.router is None else self.router.to_dict(),
            "batches": self.batches,
        }


@torch.inference_mode()
def evaluate_model_batches(
    model: SPALMERCausalLM,
    batches: Iterable[CausalBatch],
    tokenizer: TokenizerBackend,
    *,
    device: str | torch.device,
    compute_dtype: torch.dtype = torch.bfloat16,
    mixture_weights: Mapping[EvaluationStratum | str, float] | None = None,
    max_batches: int | None = None,
) -> ModelEvaluationReport:
    """Evaluate finite held-out batches without updating model/controller state."""

    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive or None")
    del tokenizer  # Retained for API compatibility; exact byte counts come from batches.
    target_device = torch.device(device)
    model_device = next(model.parameters()).device
    if model_device != target_device:
        raise ValueError(
            f"model is on {model_device}, but evaluation batches target {target_device}"
        )
    was_training = model.training
    model.eval()
    causal = CausalMetricsAccumulator()
    router: RouterTelemetryAccumulator | None = None
    observed_batches = 0
    try:
        selected_batches = (
            batches if max_batches is None else itertools.islice(batches, max_batches)
        )
        for host_batch in selected_batches:
            if not host_batch.strata:
                raise ValueError("held-out batches must identify every row's stratum")
            batch = host_batch.to(target_device)
            with _evaluation_autocast(target_device, compute_dtype):
                output = model(
                    batch.input_ids,
                    labels=batch.resolved_labels,
                    attention_mask=batch.attention_mask,
                    state_reset_mask=batch.state_reset_mask,
                )
            _accumulate_causal(causal, output, batch)
            router = _accumulate_router(router, output, batch)
            observed_batches += 1
    finally:
        model.train(was_training)
    if observed_batches == 0:
        raise ValueError("evaluation received no batches")
    return ModelEvaluationReport(
        causal=causal.finalize(mixture_weights),
        router=None if router is None else router.finalize(),
        batches=observed_batches,
    )


def _accumulate_causal(
    accumulator: CausalMetricsAccumulator,
    output: CausalLMOutput,
    batch: CausalBatch,
) -> None:
    if output.token_nll is None:
        raise RuntimeError("model evaluation did not return per-token NLL")
    labels = batch.resolved_labels[:, 1:]
    valid = labels != -100
    for key in sorted(set(batch.strata)):
        stratum = EvaluationStratum.from_key(key)
        rows = torch.tensor(
            [value == key for value in batch.strata],
            dtype=torch.bool,
            device=labels.device,
        )
        selected_valid = valid & rows.unsqueeze(-1)
        count = int(selected_valid.sum())
        if count == 0:
            continue
        nll_sum = float(output.token_nll[selected_valid].double().sum())
        byte_count = _target_bytes(batch, selected_valid)
        accumulator.update(
            CausalBatchStats(
                stratum=stratum,
                nll_sum=nll_sum,
                target_tokens=count,
                target_bytes=byte_count,
                documents=int(selected_valid.any(dim=-1).sum()),
            )
        )


def _target_bytes(
    batch: CausalBatch,
    selected_valid: Tensor,
) -> int:
    # Decoding arbitrary token suffixes is not an exact original-byte measure
    # for general tokenizers. A zero denominator deliberately suppresses BPB
    # unless the batch source supplies exact target-byte metadata.
    if not batch.token_utf8_bytes:
        return 0
    target_byte_lengths = torch.tensor(
        tuple(row[1:] for row in batch.token_utf8_bytes),
        dtype=torch.long,
        device=selected_valid.device,
    )
    if target_byte_lengths.shape != selected_valid.shape:
        raise ValueError("exact token-byte metadata does not match target mask shape")
    return int(target_byte_lengths[selected_valid].sum())


def _accumulate_router(
    accumulator: RouterTelemetryAccumulator | None,
    output: CausalLMOutput,
    batch: CausalBatch,
) -> RouterTelemetryAccumulator | None:
    valid = batch.resolved_labels[:, 1:] != -100
    for layer, metrics in enumerate(output.layer_metrics):
        ids = metrics.get("expert_ids")
        weights = metrics.get("routing_weights")
        scores = metrics.get("router_scores")
        if not all(isinstance(value, Tensor) for value in (ids, weights, scores)):
            continue
        ids = ids[:, :-1]
        weights = weights[:, :-1]
        scores = scores[:, :-1]
        selected_scores = scores.gather(-1, ids)
        if bool((selected_scores < 0).any().item()):
            # Identity-score routers expose a signed ordering utility rather
            # than a calibrated non-negative surprise estimate.
            selected_scores = None
        if accumulator is None:
            accumulator = RouterTelemetryAccumulator(scores.shape[-1])
        elif accumulator.num_experts != scores.shape[-1]:
            raise ValueError("router expert count changes between evaluated layers")
        accumulator.update_tensors(
            layer,
            ids,
            weights,
            valid_mask=valid,
            selected_scores=selected_scores,
        )
    return accumulator


def _evaluation_autocast(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda" or dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


__all__ = ["ModelEvaluationReport", "evaluate_model_batches"]
