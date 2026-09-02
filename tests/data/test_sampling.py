from __future__ import annotations

from dataclasses import replace

import pytest

from spalmer.data import (
    DocumentRecord,
    MMapTokenShard,
    SamplerState,
    SplitPolicy,
    WeightedWindowSampler,
    build_corpus_manifest,
    write_token_shard,
)

TOKENIZER_FINGERPRINT = "a" * 64
VOCAB_SIZE = 128


def _shards(tmp_path):
    records = [
        DocumentRecord.create(document_id="a", source="alpha", text="one two three"),
        DocumentRecord.create(document_id="b", source="beta", text="four five six"),
        DocumentRecord.create(document_id="c", source="beta", text="seven eight nine"),
    ]
    manifest = build_corpus_manifest(
        "sample",
        records,
        split_policy=SplitPolicy(weights=(("train", 1),)),
    )
    values = {
        "a": [1, 2, 3, 4, 5, 6],
        "b": [10, 11, 12, 13, 14],
        "c": [20, 21, 22, 23, 24, 25, 26],
    }
    first = write_token_shard(
        tmp_path,
        "first",
        manifest,
        ["a"],
        values.__getitem__,
        eod_token_id=99,
        tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
        vocab_size=VOCAB_SIZE,
    )
    second = write_token_shard(
        tmp_path,
        "second",
        manifest,
        ["b", "c"],
        values.__getitem__,
        eod_token_id=99,
        tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
        vocab_size=VOCAB_SIZE,
    )
    return MMapTokenShard(first), MMapTokenShard(second)


def _signature(sample):
    return sample.document_id, sample.document_offset, sample.tokens


def test_weighted_sampler_is_order_independent_and_state_restores_exactly(tmp_path) -> None:
    first, second = _shards(tmp_path)
    try:
        left = WeightedWindowSampler(
            [first, second], sequence_length=3, seed=123, document_weighting="windows"
        )
        right = WeightedWindowSampler(
            [second, first], sequence_length=3, seed=123, document_weighting="windows"
        )
        assert [_signature(left.sample()) for _ in range(20)] == [
            _signature(right.sample()) for _ in range(20)
        ]

        state = left.state_dict()
        expected = [_signature(left.sample()) for _ in range(10)]
        restored = WeightedWindowSampler(
            [first, second], sequence_length=3, seed=999, document_weighting="windows"
        )
        restored.load_state_dict(state)
        assert [_signature(restored.sample()) for _ in range(10)] == expected
        assert SamplerState.from_dict(state).draws > 0
    finally:
        first.close()
        second.close()


def test_stratum_weighting_and_host_batch_contract(tmp_path) -> None:
    first, second = _shards(tmp_path)
    try:
        probe = WeightedWindowSampler([first, second], sequence_length=2)
        alpha = next(stratum for stratum in probe.available_strata if '"alpha"' in stratum)
        sampler = WeightedWindowSampler(
            [first, second],
            sequence_length=2,
            stratum_weights={alpha: 1},
            seed=5,
        )
        batch = sampler.sample_batch(8)
        assert len(batch.samples) == 8
        assert all(sample.document_id == "a" for sample in batch.samples)
        assert all(isinstance(row, tuple) and len(row) == 3 for row in batch.token_ids)
        assert all(not hasattr(sample, "expert_domain") for sample in batch.samples)
    finally:
        first.close()
        second.close()


def test_sampler_rejects_incompatible_state_and_short_documents(tmp_path) -> None:
    first, second = _shards(tmp_path)
    try:
        sampler = WeightedWindowSampler([first, second], sequence_length=3, seed=1)
        assert sampler.eligibility.to_dict() == {
            "split": "train",
            "sequence_length": 3,
            "documents_in_split": 3,
            "eligible_documents": 3,
            "too_short_documents": 0,
            "eligible_windows": 12,
        }
        partial = WeightedWindowSampler([first, second], sequence_length=6, seed=1)
        assert partial.eligibility.eligible_documents == 2
        assert partial.eligibility.too_short_documents == 1
        assert partial.eligibility.eligible_windows == 3
        state = sampler.state_dict()
        incompatible = WeightedWindowSampler([first, second], sequence_length=2, seed=1)
        with pytest.raises(ValueError, match="different shard/configuration"):
            incompatible.load_state_dict(state)
        with pytest.raises(ValueError, match="long enough"):
            WeightedWindowSampler([first, second], sequence_length=100)
        with pytest.raises(ValueError, match="integer"):
            WeightedWindowSampler([first, second], sequence_length=2.5)  # type: ignore[arg-type]
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize(
    "field,value,label",
    [
        ("tokenizer_fingerprint", "b" * 64, "tokenizer fingerprint"),
        ("manifest_fingerprint", "b" * 64, "manifest fingerprint"),
        ("eod_token_id", 98, "EOD token id"),
        ("vocab_size", VOCAB_SIZE + 1, "vocabulary size"),
    ],
)
def test_sampler_rejects_heterogeneous_shard_contracts(
    tmp_path, field: str, value: object, label: str
) -> None:
    first, second = _shards(tmp_path)
    altered_index = tmp_path / f"altered-{field}.index.json"
    altered_descriptor = replace(second.descriptor, **{field: value})
    altered_index.write_text(altered_descriptor.to_json() + "\n", encoding="utf-8")
    altered = MMapTokenShard(altered_index)
    try:
        with pytest.raises(ValueError, match=label):
            WeightedWindowSampler([first, altered], sequence_length=3)
    finally:
        first.close()
        second.close()
        altered.close()
