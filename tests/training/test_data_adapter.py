from __future__ import annotations

from dataclasses import dataclass

from spalmer.data import TokenBatch, WindowSample
from spalmer.training import MMapBatchSource


@dataclass
class _Sampler:
    sequence_length: int = 3
    cursor: int = 0

    def sample_batch(self, batch_size: int) -> TokenBatch:
        samples = []
        for row in range(batch_size):
            start = self.cursor + row
            samples.append(
                WindowSample(
                    document_id=f"doc-{row}",
                    source="fixture",
                    kind="prose" if row == 0 else "code",
                    language="en" if row == 0 else "python",
                    stratum="fixture-prose" if row == 0 else "fixture-python",
                    shard_fingerprint="a" * 64,
                    document_offset=start,
                    tokens=tuple(range(start, start + self.sequence_length + 1)),
                    token_byte_lengths=(1,) * (self.sequence_length + 1),
                )
            )
        self.cursor += batch_size
        return TokenBatch(tuple(samples))

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state):
        self.cursor = int(state["cursor"])


def test_sampler_adapter_materializes_only_one_batch() -> None:
    sampler = _Sampler()
    source = MMapBatchSource(sampler, pin_memory=False)  # type: ignore[arg-type]

    batch = source.next_batch(batch_size=2, sequence_length=3)

    assert batch.input_ids.tolist() == [[0, 1, 2, 3], [1, 2, 3, 4]]
    assert batch.strata == ("english", "code:python")
    assert batch.token_utf8_bytes == ((1, 1, 1, 1), (1, 1, 1, 1))
    assert sampler.cursor == 2


def test_sampler_adapter_state_round_trips() -> None:
    sampler = _Sampler(cursor=7)
    source = MMapBatchSource(sampler, pin_memory=False)  # type: ignore[arg-type]
    state = source.state_dict()
    sampler.cursor = 99

    source.load_state_dict(state)

    assert sampler.cursor == 7
