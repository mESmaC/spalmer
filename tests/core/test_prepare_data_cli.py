from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import spalmer.__main__ as cli
from spalmer.tokenizer import Vocab, append_byte_backstop


def _write_content_vocab(path: Path) -> Vocab:
    vocab = Vocab("fixture")
    append_byte_backstop(vocab)
    vocab.seal_version("2026-09-02T00:00:00Z", "fixture")
    vocab.save(path)
    return vocab


def test_prepare_data_cli_reserves_rpd_eod_after_content_vocabulary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vocab_path = tmp_path / "vocab.json"
    vocab = _write_content_vocab(vocab_path)
    observed = {}

    def fake_prepare(inputs, output_directory, tokenizer, **kwargs):
        observed["tokenizer"] = tokenizer
        observed["eod_token_id"] = kwargs["eod_token_id"]
        return SimpleNamespace(to_dict=lambda: {"ok": True})

    monkeypatch.setattr(cli, "prepare_approved_jsonl", fake_prepare)
    cli.main(
        [
            "prepare-data",
            "approved.jsonl",
            "--output-directory",
            str(tmp_path / "prepared"),
            "--name",
            "fixture",
            "--tokenizer-rpd",
            str(vocab_path),
        ]
    )

    tokenizer = observed["tokenizer"]
    assert tokenizer.content_vocab_size == len(vocab)
    assert tokenizer.vocab_size == len(vocab) + 1
    assert tokenizer.special_tokens.eod_id == len(vocab)
    assert observed["eod_token_id"] is None


def test_prepare_data_cli_rejects_rpd_content_id_as_eod(tmp_path: Path) -> None:
    vocab_path = tmp_path / "vocab.json"
    _write_content_vocab(vocab_path)

    with pytest.raises(ValueError, match="reserved suffix"):
        cli.main(
            [
                "prepare-data",
                "approved.jsonl",
                "--output-directory",
                str(tmp_path / "prepared"),
                "--name",
                "fixture",
                "--tokenizer-rpd",
                str(vocab_path),
                "--eod-token-id",
                "0",
            ]
        )
