"""Adapters from immutable data samplers to the training-engine contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from spalmer.data import WeightedWindowSampler
from spalmer.training.engine import CausalBatch


class MMapBatchSource:
    """Materialize only the currently requested mmap-backed token windows."""

    def __init__(
        self,
        sampler: WeightedWindowSampler,
        *,
        pin_memory: bool = True,
    ) -> None:
        self.sampler = sampler
        self.pin_memory = pin_memory

    def next_batch(self, *, batch_size: int, sequence_length: int) -> CausalBatch:
        if sequence_length != self.sampler.sequence_length:
            raise ValueError(
                "training sequence length does not match the configured window sampler: "
                f"{sequence_length} != {self.sampler.sequence_length}"
            )
        sampled = self.sampler.sample_batch(batch_size)
        token_ids = torch.tensor(sampled.token_ids, dtype=torch.long)
        if self.pin_memory and torch.cuda.is_available():
            token_ids = token_ids.pin_memory()
        exact_byte_rows = tuple(sample.token_byte_lengths for sample in sampled.samples)
        token_utf8_bytes = (
            tuple(lengths for lengths in exact_byte_rows if lengths is not None)
            if all(lengths is not None for lengths in exact_byte_rows)
            else ()
        )
        return CausalBatch(
            input_ids=token_ids,
            strata=tuple(
                _evaluation_stratum(sample.kind, sample.language) for sample in sampled.samples
            ),
            token_utf8_bytes=token_utf8_bytes,
        )

    def state_dict(self) -> Mapping[str, Any]:
        return self.sampler.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.sampler.load_state_dict(state)


def _evaluation_stratum(kind: str, language: str) -> str:
    if kind == "prose" and language == "en":
        return "english"
    if kind == "code" and language:
        return f"code:{language}"
    raise ValueError(f"unsupported evaluation stratum: kind={kind!r}, language={language!r}")


__all__ = ["MMapBatchSource"]
