from __future__ import annotations

import json
from pathlib import Path

import pytest

from spalmer.data import (
    CorpusManifest,
    JsonlAdapterConfig,
    MMapTokenShard,
    prepare_approved_jsonl,
)
from spalmer.tokenizer.backends import SpecialTokenIds, TokenizerIdentity


class _ByteTokenizer:
    special_tokens = SpecialTokenIds(eos_id=256, eod_id=256)
    identity = TokenizerIdentity(
        backend="fixture-byte",
        source="test",
        artifact_fingerprint="b" * 64,
        vocab_size=257,
        special_tokens=special_tokens,
    )
    vocab_size = 257

    def encode(self, text, *, add_special_tokens=False, kind="mixed"):
        del add_special_tokens, kind
        return list(text.encode("utf-8"))

    def decode(self, token_ids, *, skip_special_tokens=False):
        ids = list(token_ids)
        if skip_special_tokens:
            ids = [token_id for token_id in ids if token_id != 256]
        return bytes(ids).decode("utf-8")

    def encode_with_byte_lengths(self, text, *, add_special_tokens=False, kind="mixed"):
        token_ids = self.encode(text, add_special_tokens=add_special_tokens, kind=kind)
        return token_ids, [1] * len(token_ids)


class _FailingTokenizer(_ByteTokenizer):
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, text, *, add_special_tokens=False, kind="mixed"):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("fixture tokenization failure")
        return super().encode(text, add_special_tokens=add_special_tokens, kind=kind)


def _write_jsonl(path: Path) -> None:
    rows = [
        {
            "document_id": "english",
            "source": "alpha",
            "text": "Clear English communication.",
            "domain": "language",
            "lang": "en",
        },
        {
            "document_id": "python",
            "source": "beta/script.py",
            "text": "def answer():\n    return 42",
            "domain": "code",
            "lang": "",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_prepare_approved_jsonl_writes_reloadable_split_shards(tmp_path) -> None:
    source = tmp_path / "approved.jsonl"
    output = tmp_path / "prepared"
    _write_jsonl(source)

    prepared = prepare_approved_jsonl(
        [source],
        output,
        _ByteTokenizer(),  # type: ignore[arg-type]
        name="fixture",
        adapter_config=JsonlAdapterConfig(kind_field="domain", language_field="lang"),
        documents_per_shard=1,
    )

    manifest = CorpusManifest.load(prepared.manifest_path)
    assert prepared.documents_loaded == 2
    assert prepared.documents_selected == 2
    assert {item.language for item in manifest.selected_documents} == {"en", "python"}
    assert prepared.shard_indexes
    assert not (output / ".document-spool").exists()
    for index in prepared.shard_indexes:
        with MMapTokenShard(index) as shard:
            assert shard.descriptor.eod_token_id == 256
            assert shard.descriptor.tokenizer_fingerprint == _ByteTokenizer.identity.fingerprint
            assert shard.descriptor.vocab_size == _ByteTokenizer.vocab_size
            assert shard.descriptor.token_byte_file is not None
            assert shard.descriptor.documents


def test_prepare_rejects_ordinary_eod_and_empty_selection_without_output(tmp_path) -> None:
    source = tmp_path / "approved.jsonl"
    _write_jsonl(source)

    with pytest.raises(ValueError, match="assigned as a tokenizer special"):
        prepare_approved_jsonl(
            [source],
            tmp_path / "ordinary-eod",
            _ByteTokenizer(),  # type: ignore[arg-type]
            name="fixture",
            adapter_config=JsonlAdapterConfig(kind_field="domain", language_field="lang"),
            eod_token_id=1,
        )
    assert not (tmp_path / "ordinary-eod").exists()

    empty = tmp_path / "empty.jsonl"
    empty.write_text(
        json.dumps(
            {
                "document_id": "french",
                "source": "fixture",
                "text": "Texte en francais.",
                "domain": "language",
                "lang": "fr",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="selected no documents"):
        prepare_approved_jsonl(
            [empty],
            tmp_path / "empty-output",
            _ByteTokenizer(),  # type: ignore[arg-type]
            name="empty",
            adapter_config=JsonlAdapterConfig(kind_field="domain", language_field="lang"),
        )
    assert not (tmp_path / "empty-output").exists()


def test_prepare_publishes_atomically_and_never_overwrites(tmp_path) -> None:
    source = tmp_path / "approved.jsonl"
    _write_jsonl(source)
    output = tmp_path / "prepared"
    tokenizer = _FailingTokenizer()

    with pytest.raises(RuntimeError, match="fixture tokenization failure"):
        prepare_approved_jsonl(
            [source],
            output,
            tokenizer,  # type: ignore[arg-type]
            name="fixture",
            adapter_config=JsonlAdapterConfig(kind_field="domain", language_field="lang"),
            documents_per_shard=1,
        )
    assert not output.exists()
    assert not tuple(tmp_path.glob(".prepared.*.partial"))

    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("owned", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_approved_jsonl(
            [source],
            output,
            _ByteTokenizer(),  # type: ignore[arg-type]
            name="fixture",
            adapter_config=JsonlAdapterConfig(kind_field="domain", language_field="lang"),
        )
    assert marker.read_text(encoding="utf-8") == "owned"
