from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers import build_demo_vocab

from spalmer.tokenizer.backends import (
    CompatibleTokenizerAdapter,
    HFTokenizerAdapter,
    LocalSerializedTokenizerAdapter,
    RPDTokenizerAdapter,
    SpecialTokenIds,
    TokenizerBackend,
    fingerprint_local_artifact,
)


class FakeHFTokenizer:
    def __init__(self, vocab: dict[str, int]) -> None:
        self._vocab = vocab
        self.vocab_size = max(vocab.values()) + 1
        self.bos_token_id = vocab.get("<bos>")
        self.eos_token_id = vocab.get("<eos>")
        self.eod_token_id = vocab.get("<eod>")
        self.pad_token_id = vocab.get("<pad>")
        self.init_kwargs = {"normalization": "none"}
        self.special_tokens_map = {}

    def __len__(self) -> int:
        return self.vocab_size

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids = [self._vocab[part] for part in text.split()]
        if add_special_tokens:
            if self.bos_token_id is not None:
                ids.insert(0, self.bos_token_id)
            if self.eos_token_id is not None:
                ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        reverse = {token_id: token for token, token_id in self._vocab.items()}
        special_ids = {
            token_id
            for token_id in (
                self.bos_token_id,
                self.eos_token_id,
                self.eod_token_id,
                self.pad_token_id,
            )
            if token_id is not None
        }
        if skip_special_tokens:
            ids = [token_id for token_id in ids if token_id not in special_ids]
        return " ".join(reverse[token_id] for token_id in ids)


def test_rpd_adapter_is_deterministic_and_exposes_all_special_roles(tmp_path: Path) -> None:
    vocab = build_demo_vocab()
    first_special = len(vocab)
    specials = SpecialTokenIds(
        bos_id=first_special,
        eos_id=first_special + 1,
        eod_id=first_special + 2,
        pad_id=first_special + 3,
    )
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "copied.json"
    vocab.save(first_path)
    vocab.save(second_path)

    first = RPDTokenizerAdapter.from_file(first_path, special_tokens=specials)
    copied = RPDTokenizerAdapter.from_file(second_path, special_tokens=specials)
    plain = first.encode("the cat", kind="prose")
    wrapped = first.encode("the cat", kind="prose", add_special_tokens=True)

    assert isinstance(first, TokenizerBackend)
    assert first.special_tokens == specials
    assert wrapped == [specials.bos_id, *plain, specials.eos_id]
    assert first.decode(wrapped, skip_special_tokens=True) == "the cat"
    assert first.decode([specials.eod_id]) == "<|eod|>"
    assert first.content_vocab_size == len(vocab)
    assert first.vocab_size == len(vocab) + 4
    assert first.identity.artifact_fingerprint == vocab.fingerprint
    assert first.identity.fingerprint == copied.identity.fingerprint
    assert first.identity.source != copied.identity.source


def test_special_assignment_is_part_of_adapter_identity() -> None:
    vocab = build_demo_vocab()
    first_special = len(vocab)
    first = RPDTokenizerAdapter(
        vocab,
        special_tokens=SpecialTokenIds(bos_id=first_special, eos_id=first_special + 1),
    )
    second = RPDTokenizerAdapter(
        vocab,
        special_tokens=SpecialTokenIds(bos_id=first_special + 1, eos_id=first_special),
    )

    assert first.identity.artifact_fingerprint == second.identity.artifact_fingerprint
    assert first.identity.fingerprint != second.identity.fingerprint


def test_rpd_adapter_rejects_content_token_as_special() -> None:
    vocab = build_demo_vocab()

    with pytest.raises(ValueError, match="reserved suffix"):
        RPDTokenizerAdapter(vocab, special_tokens=SpecialTokenIds(eod_id=0))

    with pytest.raises(ValueError, match="reserved suffix"):
        RPDTokenizerAdapter(vocab, special_tokens=SpecialTokenIds(eod_id=len(vocab) + 1))


def test_compatible_and_hf_adapters_have_stable_semantic_identity() -> None:
    ordered_a = {"<bos>": 0, "<eos>": 1, "hello": 2, "world": 3}
    ordered_b = {"world": 3, "hello": 2, "<eos>": 1, "<bos>": 0}
    first = HFTokenizerAdapter.from_tokenizer(FakeHFTokenizer(ordered_a))
    second = HFTokenizerAdapter.from_tokenizer(FakeHFTokenizer(ordered_b))

    assert first.identity.fingerprint == second.identity.fingerprint
    assert first.special_tokens == SpecialTokenIds(bos_id=0, eos_id=1)
    assert first.encode("hello world", add_special_tokens=True, kind="code") == [0, 2, 3, 1]
    assert first.decode([0, 2, 3, 1], skip_special_tokens=True) == "hello world"


def test_local_serialized_adapter_hashes_artifact_content(tmp_path: Path) -> None:
    first_path = tmp_path / "one.json"
    copied_path = tmp_path / "two.json"
    payload = {"<bos>": 0, "<eos>": 1, "alpha": 2}
    first_path.write_text(json.dumps(payload), encoding="utf-8")
    copied_path.write_text(json.dumps(payload), encoding="utf-8")

    def loader(path: Path) -> FakeHFTokenizer:
        return FakeHFTokenizer(json.loads(path.read_text(encoding="utf-8")))

    first = LocalSerializedTokenizerAdapter.from_path(first_path, loader)
    copied = LocalSerializedTokenizerAdapter.from_path(copied_path, loader)
    assert first.identity.artifact_fingerprint == copied.identity.artifact_fingerprint
    assert first.identity.fingerprint == copied.identity.fingerprint

    copied_path.write_text(json.dumps({**payload, "beta": 3}), encoding="utf-8")
    changed = LocalSerializedTokenizerAdapter.from_path(copied_path, loader)
    assert first.identity.artifact_fingerprint != changed.identity.artifact_fingerprint
    assert first.identity.fingerprint != changed.identity.fingerprint


def test_directory_artifact_fingerprint_is_location_independent(tmp_path: Path) -> None:
    first = tmp_path / "one"
    copied = tmp_path / "two"
    first.mkdir()
    copied.mkdir()
    (first / "tokenizer.json").write_text('{"version":1}', encoding="utf-8")
    (copied / "tokenizer.json").write_text('{"version":1}', encoding="utf-8")

    assert fingerprint_local_artifact(first) == fingerprint_local_artifact(copied)
    (copied / "config.json").write_text("{}", encoding="utf-8")
    assert fingerprint_local_artifact(first) != fingerprint_local_artifact(copied)


def test_invalid_special_id_and_unfingerprintable_object_fail_closed() -> None:
    tokenizer = FakeHFTokenizer({"hello": 0})
    with pytest.raises(ValueError, match="outside vocabulary"):
        CompatibleTokenizerAdapter(tokenizer, special_tokens=SpecialTokenIds(eod_id=4))

    class OpaqueTokenizer:
        vocab_size = 1

        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            return [0]

        def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
            return "x"

    with pytest.raises(ValueError, match="artifact_fingerprint"):
        CompatibleTokenizerAdapter(OpaqueTokenizer())


def test_adapter_rejects_out_of_range_token_ids() -> None:
    adapter = HFTokenizerAdapter.from_tokenizer(FakeHFTokenizer({"hello": 0}))

    with pytest.raises(ValueError, match="token id -1"):
        adapter.decode([-1])

    adapter.tokenizer._vocab["bad"] = 2
    with pytest.raises(ValueError, match="token id 2"):
        adapter.encode("bad")
