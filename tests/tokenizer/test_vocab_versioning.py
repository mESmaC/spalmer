from __future__ import annotations

import pytest
from helpers import build_demo_vocab

from spalmer.tokenizer import Tier, Vocab


def test_demo_vocab_represents_all_tiers():
    vocab = build_demo_vocab()
    for tier in Tier:
        assert any(entry.tier is tier for entry in vocab.entries), tier


def test_initial_version_is_frozen():
    vocab = build_demo_vocab()
    assert vocab.version == 1
    (record,) = vocab.history
    assert record.frozen
    assert record.entry_count == len(vocab)

    with pytest.raises(RuntimeError, match="sealed vocabulary"):
        vocab.append(Tier.SALVAGE, "new")


def test_append_only_extend_keeps_existing_ids():
    vocab = build_demo_vocab()
    before = list(vocab.entries)
    record = vocab.extend_append_only(
        "2026-09-02", "append one salvage merge", [(Tier.SALVAGE, "qx")]
    )
    assert record.version == 2
    assert record.entry_count == len(before) + 1
    assert vocab.entries[: len(before)] == before
    assert vocab.entries[-1].surface == "qx"
    assert vocab.entries[-1].token_id == len(before)


def test_duplicate_surface_in_same_tier_rejected():
    vocab = build_demo_vocab()
    with pytest.raises(ValueError):
        vocab.append(Tier.BYTE, "<byte:0x00>")


def test_json_roundtrip_preserves_entries_and_history():
    vocab = build_demo_vocab()
    clone = Vocab.from_json(vocab.to_json())
    assert clone.name == vocab.name
    assert clone.entries == vocab.entries
    assert clone.history == vocab.history
    assert clone.fingerprint == vocab.fingerprint


def test_tampered_history_is_rejected():
    data = build_demo_vocab().to_dict()
    data["history"][0]["entry_count"] = 999999
    with pytest.raises(ValueError):
        Vocab.from_dict(data)


def test_noncontiguous_ids_are_rejected():
    data = build_demo_vocab().to_dict()
    data["entries"][0]["id"] = 5
    with pytest.raises(ValueError):
        Vocab.from_dict(data)


def test_save_and_load(tmp_path):
    vocab = build_demo_vocab()
    path = tmp_path / "vocab.json"
    vocab.save(path)
    clone = Vocab.load(path)
    assert clone.entries == vocab.entries
    assert clone.history == vocab.history
