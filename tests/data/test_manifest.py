from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from spalmer.data import (
    CorpusManifest,
    DocumentRecord,
    JsonlAdapterConfig,
    SplitPolicy,
    build_corpus_manifest,
    discover_jsonl,
    iter_approved_jsonl,
    load_approved_jsonl,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_document_contract_is_versioned_immutable_and_roundtrips() -> None:
    record = DocumentRecord.create(
        document_id="doc-1",
        source="approved",
        text="Useful text",
        attributes={"score": 0.75, "tags": ["clean", "english"]},
    )
    assert DocumentRecord.from_dict(record.to_dict()) == record
    with pytest.raises(FrozenInstanceError):
        record.text = "changed"  # type: ignore[misc]
    damaged = record.to_dict()
    damaged["version"] = 99
    with pytest.raises(ValueError, match="unsupported document"):
        DocumentRecord.from_dict(damaged)


def test_jsonl_discovery_and_approved_adapter_are_deterministic() -> None:
    direct = discover_jsonl([FIXTURES / "approved.jsonl"])
    recursive = discover_jsonl([FIXTURES])
    assert direct[0].logical_name == "approved.jsonl"
    assert recursive[0].logical_name == "fixtures/approved.jsonl"

    records = load_approved_jsonl(
        recursive,
        JsonlAdapterConfig(approved_field="approved"),
    )
    assert [record.document_id for record in records] == [
        "en-a",
        "en-b",
        "en-c",
        "fr-a",
        "py-a",
        "py-b",
        "rs-a",
        "js-a",
    ]
    assert records[0].attributes == (
        ("input", '"fixtures/approved.jsonl"'),
        ("line", "1"),
        ("quality", "0.9"),
    )


def test_approved_jsonl_iterator_is_lazy(tmp_path) -> None:
    source = tmp_path / "lazy.jsonl"
    source.write_text('{"text":"first"}\nnot-json\n', encoding="utf-8")
    records = iter_approved_jsonl(discover_jsonl([source]))

    assert next(records).text == "first"
    with pytest.raises(ValueError, match="invalid JSON"):
        next(records)


def test_yylab_zstd_jsonl_and_source_extension_language_inference(tmp_path) -> None:
    zstd = pytest.importorskip("zstandard")
    source = tmp_path / "shard-00000.jsonl.zst"
    payload = (
        b'{"text":"int main(void) { return 0; }","domain":"code",'
        b'"source":"repo/main.c","lang":""}\n'
    )
    source.write_bytes(zstd.ZstdCompressor().compress(payload))

    records = load_approved_jsonl(
        discover_jsonl([source]),
        JsonlAdapterConfig(kind_field="domain", language_field="lang"),
    )

    assert len(records) == 1
    assert records[0].kind == "code"
    assert records[0].language == "c"


def test_yylab_news_and_conversation_domains_are_treated_as_prose(tmp_path) -> None:
    source = tmp_path / "approved.jsonl"
    source.write_text(
        '{"text":"Current event report","domain":"news","lang":"en"}\n'
        '{"text":"user: hello\\nassistant: hi","domain":"conversation","lang":"en"}\n',
        encoding="utf-8",
    )

    records = load_approved_jsonl(
        discover_jsonl([source]),
        JsonlAdapterConfig(kind_field="domain", language_field="lang"),
    )

    assert [record.kind for record in records] == ["prose", "prose"]


def test_manifest_dedup_language_inventory_and_splits_are_order_stable(tmp_path) -> None:
    records = load_approved_jsonl(
        discover_jsonl([FIXTURES]),
        JsonlAdapterConfig(approved_field="approved"),
    )
    policy = SplitPolicy(weights=(("train", 8), ("validation", 1), ("test", 1)), seed="x")
    first = build_corpus_manifest(
        "fixture",
        records,
        top_code_languages=2,
        split_policy=policy,
        inputs=("fixtures/approved.jsonl",),
    )
    second = build_corpus_manifest(
        "fixture",
        reversed(records),
        top_code_languages=2,
        split_policy=policy,
        inputs=("fixtures/approved.jsonl",),
    )
    assert first.to_json() == second.to_json()
    assert first.fingerprint == second.fingerprint

    documents = {document.document_id: document for document in first.documents}
    assert documents["en-a"].selected
    assert documents["en-b"].duplicate_of == "en-a"
    assert documents["en-b"].exclusion_reason == "normalized_duplicate"
    assert documents["en-c"].duplicate_of == "en-a"
    assert documents["en-c"].exclusion_reason == "exact_duplicate"
    assert documents["en-a"].exact_group == documents["en-c"].exact_group
    assert documents["en-a"].normalized_group == documents["en-b"].normalized_group
    assert documents["en-a"].split == documents["en-b"].split
    assert documents["fr-a"].exclusion_reason == "non_english_prose"
    assert documents["js-a"].exclusion_reason == "code_language_not_selected"
    assert [(item.language, item.selected) for item in first.code_languages] == [
        ("python", True),
        ("rust", True),
        ("javascript", False),
    ]
    assert all(not hasattr(document, "expert_domain") for document in first.documents)

    path = first.save(tmp_path / "manifest.json")
    assert CorpusManifest.load(path) == first
    with pytest.raises(FileExistsError):
        first.save(path)


def test_duplicate_ids_and_unknown_code_language_fail_closed() -> None:
    record = DocumentRecord.create(document_id="same", source="a", text="one")
    with pytest.raises(ValueError, match="unique"):
        build_corpus_manifest("bad", [record, record])

    unknown = DocumentRecord.create(
        document_id="unknown",
        source="code",
        text="x = 1",
        kind="code",
        language="unknown",
    )
    manifest = build_corpus_manifest("unknown", [unknown])
    assert manifest.documents[0].exclusion_reason == "unknown_code_language"
