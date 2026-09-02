from __future__ import annotations

import pytest

from spalmer.data import (
    DocumentRecord,
    MMapTokenShard,
    SplitPolicy,
    TokenizedDocument,
    TokenShardDescriptor,
    build_corpus_manifest,
    write_token_shard,
)

TOKENIZER_FINGERPRINT = "a" * 64
VOCAB_SIZE = 128


def _manifest():
    records = [
        DocumentRecord.create(document_id="a", source="one", text="alpha beta"),
        DocumentRecord.create(document_id="b", source="two", text="gamma delta"),
        DocumentRecord.create(document_id="c", source="one", text="epsilon zeta"),
    ]
    return build_corpus_manifest(
        "tokens",
        records,
        split_policy=SplitPolicy(weights=(("train", 1),)),
    )


def test_uint32_shard_is_checksummed_immutable_and_mmap_readable(tmp_path) -> None:
    manifest = _manifest()
    tokens = {"a": [1, 2, 3, 4], "b": [10, 11, 12], "c": [20, 21, 22, 23, 24]}
    calls: list[str] = []

    def source(document_id: str):
        calls.append(document_id)
        values = tokens[document_id]
        byte_count = next(
            document.utf8_bytes
            for document in manifest.selected_documents
            if document.document_id == document_id
        )
        byte_lengths = [1] * (len(values) - 1) + [byte_count - len(values) + 1]
        return TokenizedDocument(iter(values), byte_lengths)

    index_path = write_token_shard(
        tmp_path,
        "train-00000",
        manifest,
        ["c", "a", "b"],
        source,
        eod_token_id=99,
        tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
        vocab_size=VOCAB_SIZE,
    )
    assert calls == ["a", "b", "c"]
    descriptor = TokenShardDescriptor.load(index_path)
    assert descriptor.token_count == sum(map(len, tokens.values())) + 3
    assert descriptor.byte_count == descriptor.token_count * 4
    assert descriptor.manifest_fingerprint == manifest.fingerprint
    assert descriptor.tokenizer_fingerprint == TOKENIZER_FINGERPRINT
    assert descriptor.vocab_size == VOCAB_SIZE
    assert descriptor.token_byte_file is not None
    assert descriptor.token_byte_sha256 is not None

    with MMapTokenShard(index_path) as shard:
        assert shard.read_document("a") == (1, 2, 3, 4)
        assert shard.read_document("a", include_eod=True) == (1, 2, 3, 4, 99)
        assert shard.read_window("c", 2, 4) == (22, 23, 24, 99)
        assert shard.read_window_bytes("c", 2, 4) == (1, 1, 8, 0)
        with pytest.raises(IndexError):
            shard.read_window("b", 2, 3)
    with pytest.raises(ValueError, match="closed"):
        shard.read_tokens(0, 1)

    with pytest.raises(FileExistsError, match="immutable"):
        write_token_shard(
            tmp_path,
            "train-00000",
            manifest,
            ["a"],
            source,
            eod_token_id=99,
            tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
            vocab_size=VOCAB_SIZE,
        )


def test_shard_rejects_corruption_and_invalid_tokens(tmp_path) -> None:
    manifest = _manifest()
    index = write_token_shard(
        tmp_path,
        "corrupt",
        manifest,
        ["a"],
        lambda _document_id: [1, 2, 3],
        eod_token_id=9,
        tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
        vocab_size=VOCAB_SIZE,
    )
    descriptor = TokenShardDescriptor.load(index)
    token_path = index.parent / descriptor.token_file
    with token_path.open("r+b") as handle:
        handle.seek(0)
        handle.write(b"\xff")
    with pytest.raises(ValueError, match="checksum"):
        MMapTokenShard(index)

    with pytest.raises(ValueError, match="token id"):
        write_token_shard(
            tmp_path,
            "invalid",
            manifest,
            ["a"],
            lambda _document_id: [-1],
            eod_token_id=9,
            tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
            vocab_size=VOCAB_SIZE,
        )
    assert not (tmp_path / "invalid.tokens.u32").exists()

    with pytest.raises(ValueError, match="outside vocabulary"):
        write_token_shard(
            tmp_path,
            "out-of-vocab",
            manifest,
            ["a"],
            lambda _document_id: [VOCAB_SIZE],
            eod_token_id=9,
            tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
            vocab_size=VOCAB_SIZE,
        )

    with pytest.raises(ValueError, match="disagree with token count"):
        write_token_shard(
            tmp_path,
            "invalid-bytes",
            manifest,
            ["a"],
            lambda _document_id: TokenizedDocument([1, 2], [1]),
            eod_token_id=9,
            tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
            vocab_size=VOCAB_SIZE,
        )
    assert not (tmp_path / "invalid-bytes.tokens.u32").exists()
    assert not (tmp_path / "invalid-bytes.bytes.u32").exists()

    with pytest.raises(ValueError, match="original UTF-8 byte count"):
        write_token_shard(
            tmp_path,
            "invalid-byte-total",
            manifest,
            ["a"],
            lambda _document_id: TokenizedDocument([1, 2], [1, 1]),
            eod_token_id=9,
            tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
            vocab_size=VOCAB_SIZE,
        )
    assert not (tmp_path / "invalid-byte-total.tokens.u32").exists()
    assert not (tmp_path / "invalid-byte-total.bytes.u32").exists()
    assert not (tmp_path / "out-of-vocab.tokens.u32").exists()

    with pytest.raises(ValueError, match="outside vocabulary"):
        write_token_shard(
            tmp_path,
            "invalid-eod",
            manifest,
            ["a"],
            lambda _document_id: [1],
            eod_token_id=VOCAB_SIZE,
            tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
            vocab_size=VOCAB_SIZE,
        )


def test_excluded_documents_cannot_enter_a_shard(tmp_path) -> None:
    record = DocumentRecord.create(
        document_id="fr",
        source="foreign",
        text="bonjour",
        language="fr",
    )
    manifest = build_corpus_manifest("filtered", [record])
    with pytest.raises(ValueError, match="excluded"):
        write_token_shard(
            tmp_path,
            "filtered",
            manifest,
            ["fr"],
            lambda _document_id: [1],
            eod_token_id=2,
            tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
            vocab_size=VOCAB_SIZE,
        )
