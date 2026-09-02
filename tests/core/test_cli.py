from __future__ import annotations

from pathlib import Path

import pytest
import torch

import spalmer.__main__ as cli


def test_legacy_training_invocations_keep_their_existing_dispatch(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "_run", calls.append)

    cli.main(["corpus.txt", "--layers", "4"])
    cli.main(["--smoke", "--layers", "4"])

    assert calls[0].text_file == Path("corpus.txt")
    assert calls[0].smoke is False
    assert calls[1].text_file is None
    assert calls[1].smoke is True


def test_generate_command_loads_saved_vocab_and_uses_cached_runtime(monkeypatch, capsys) -> None:
    vocab = _FakeVocab()
    model = _FakeModel()

    def fake_load_checkpoint(path, *, map_location):
        assert path == Path("trained.pt")
        assert map_location == "cpu"
        return model, vocab, {"tokens_seen": 10, "kind": "code"}

    class FakeEncoder:
        def __init__(self, loaded_vocab) -> None:
            assert loaded_vocab is vocab

        def encode(self, text: str, *, kind: str) -> list[int]:
            assert text == "prompt"
            assert kind == "code"
            return [1, 2]

    def fake_generate_tokens(loaded_model, prompt_ids, **kwargs):
        assert loaded_model is model
        torch.testing.assert_close(prompt_ids, torch.tensor([1, 2]))
        assert kwargs == {
            "max_new_tokens": 3,
            "temperature": 0.7,
            "top_k": 5,
            "dynamic_residency": True,
        }
        return torch.tensor([[1, 2, 3]])

    monkeypatch.setattr(cli, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(cli, "Encoder", FakeEncoder)
    monkeypatch.setattr(cli, "Vocab", _FakeVocab)
    monkeypatch.setattr(cli, "generate_tokens", fake_generate_tokens)

    cli.main(
        [
            "generate",
            "trained.pt",
            "--prompt",
            "prompt",
            "--device",
            "cpu",
            "--new-tokens",
            "3",
            "--temperature",
            "0.7",
            "--top-k",
            "5",
            "--dynamic-residency",
        ]
    )

    assert model.placement_calls == [
        {"dtype": torch.float32},
        {"device": torch.device("cpu")},
    ]
    assert model.offload_call is None
    assert model.eval_called
    assert capsys.readouterr().out == "ABC\n"


def test_cuda_generate_defaults_to_bounded_expert_offload(monkeypatch, capsys) -> None:
    vocab = _FakeVocab()
    model = _FakeModel()

    monkeypatch.setattr(
        cli,
        "load_checkpoint",
        lambda path, *, map_location: (model, vocab, {"kind": "prose"}),
    )

    class FakeEncoder:
        def __init__(self, loaded_vocab) -> None:
            assert loaded_vocab is vocab

        def encode(self, text: str, *, kind: str) -> list[int]:
            assert (text, kind) == ("prompt", "prose")
            return [1]

    monkeypatch.setattr(cli, "Encoder", FakeEncoder)
    monkeypatch.setattr(cli, "Vocab", _FakeVocab)
    monkeypatch.setattr(
        cli,
        "generate_tokens",
        lambda loaded_model, prompt_ids, **kwargs: torch.tensor([[1, 2]]),
    )

    cli.main(
        [
            "generate",
            "trained.pt",
            "--prompt",
            "prompt",
            "--device",
            "cuda",
            "--expert-cache-size",
            "4",
        ]
    )

    assert model.placement_calls == [{"dtype": torch.bfloat16}]
    assert model.offload_call == (
        torch.device("cuda"),
        {"cache_size": 4, "paging": True},
    )
    assert capsys.readouterr().out == "AB\n"


def test_cuda_dynamic_residency_explicitly_selects_legacy_offload(monkeypatch, capsys) -> None:
    vocab = _FakeVocab()
    model = _FakeModel()
    monkeypatch.setattr(
        cli,
        "load_checkpoint",
        lambda path, *, map_location: (model, vocab, {"kind": "prose"}),
    )

    class FakeEncoder:
        def __init__(self, loaded_vocab) -> None:
            assert loaded_vocab is vocab

        def encode(self, text: str, *, kind: str) -> list[int]:
            return [1]

    monkeypatch.setattr(cli, "Encoder", FakeEncoder)
    monkeypatch.setattr(cli, "Vocab", _FakeVocab)
    monkeypatch.setattr(
        cli,
        "generate_tokens",
        lambda loaded_model, prompt_ids, **kwargs: torch.tensor([[1, 2]]),
    )

    cli.main(
        [
            "generate",
            "trained.pt",
            "--prompt",
            "prompt",
            "--device",
            "cuda",
            "--dynamic-residency",
        ]
    )

    assert model.offload_call == (
        torch.device("cuda"),
        {"cache_size": None, "paging": False},
    )
    assert capsys.readouterr().out == "AB\n"


def test_training_rejects_incomplete_kda_mla_cycle() -> None:
    with pytest.raises(ValueError, match="positive multiple of 4"):
        cli.main(["--smoke", "--layers", "5"])


def test_terminal_output_escapes_unencodable_generation_bytes() -> None:
    assert cli._terminal_safe_text("A\ufffdB", encoding="cp1252") == r"A\ufffdB"


class _FakeModel:
    def __init__(self) -> None:
        self.placement_calls = []
        self.offload_call = None
        self.eval_called = False

    def to(self, **kwargs):
        self.placement_calls.append(kwargs)
        return self

    def enable_expert_offload(self, device, **kwargs):
        self.offload_call = (device, kwargs)

    def eval(self):
        self.eval_called = True
        return self


class _FakeToken:
    def __init__(self, value: bytes) -> None:
        self.value = value

    def payload(self) -> bytes:
        return self.value


class _FakeVocab:
    _TOKENS = {1: _FakeToken(b"A"), 2: _FakeToken(b"B"), 3: _FakeToken(b"C")}

    def get(self, token_id: int) -> _FakeToken:
        return self._TOKENS[token_id]
