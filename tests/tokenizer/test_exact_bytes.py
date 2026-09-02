from __future__ import annotations

import pytest

from spalmer.tokenizer import (
    ExactByteMappingUnavailable,
    HFTokenizerAdapter,
    RPDTokenizerAdapter,
    SpecialTokenIds,
    Vocab,
    append_byte_backstop,
)


class _OffsetTokenizer:
    vocab_size = 2
    all_special_ids: list[int] = []

    def __len__(self) -> int:
        return self.vocab_size

    def get_vocab(self) -> dict[str, int]:
        return {"a": 0, "é": 1}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del text, add_special_tokens
        return [0, 1]

    def decode(self, ids, *, skip_special_tokens: bool = False) -> str:
        del ids, skip_special_tokens
        return " aé"

    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        assert text == " aé"
        assert not add_special_tokens
        assert return_offsets_mapping
        return {"input_ids": [0, 1], "offset_mapping": [(1, 2), (2, 3)]}


class _OverlappingOffsetTokenizer(_OffsetTokenizer):
    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        del text, add_special_tokens, return_offsets_mapping
        return {"input_ids": [0, 1], "offset_mapping": [(0, 2), (1, 3)]}


def test_rpd_exact_byte_lengths_cover_multibyte_text() -> None:
    vocab = Vocab("bytes")
    append_byte_backstop(vocab)
    adapter = RPDTokenizerAdapter(
        vocab,
        special_tokens=SpecialTokenIds(eod_id=len(vocab)),
    )

    token_ids, byte_lengths = adapter.encode_with_byte_lengths("aé")

    assert sum(byte_lengths) == len("aé".encode())
    assert len(token_ids) == len(byte_lengths)
    assert adapter.encode_with_byte_lengths("")[1] == []


def test_fast_offsets_partition_normalization_gaps_and_unicode_exactly() -> None:
    adapter = HFTokenizerAdapter.from_tokenizer(_OffsetTokenizer())

    token_ids, byte_lengths = adapter.encode_with_byte_lengths(" aé")

    assert token_ids == [0, 1]
    assert byte_lengths == [2, 2]
    assert sum(byte_lengths) == len(" aé".encode())


def test_overlapping_fast_offsets_fail_closed() -> None:
    adapter = HFTokenizerAdapter.from_tokenizer(_OverlappingOffsetTokenizer())

    with pytest.raises(ExactByteMappingUnavailable, match="overlapping"):
        adapter.encode_with_byte_lengths(" aé")
