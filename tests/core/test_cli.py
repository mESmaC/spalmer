from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import spalmer.__main__ as cli
from spalmer.config import RecurrenceConfig
from spalmer.modeling import AdaptiveExit


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


def test_training_recurrence_flags_build_config_and_metadata(tmp_path) -> None:
    destination = tmp_path / "recurrent.pt"

    cli.main(
        [
            "--smoke",
            "--steps",
            "2",
            "--layers",
            "4",
            "--d-model",
            "16",
            "--heads",
            "2",
            "--experts",
            "4",
            "--sequence-length",
            "16",
            "--batch-size",
            "2",
            "--new-tokens",
            "2",
            "--device",
            "cpu",
            "--recurrent-layers",
            "1",
            "2",
            "1",
            "--mean-recurrence",
            "3",
            "--backprop-depth",
            "2",
            "--default-steps",
            "5",
            "--output",
            str(destination),
        ]
    )

    model, _, metadata = cli.load_checkpoint(destination, map_location="cpu")
    assert model.config.recurrence == RecurrenceConfig(
        prelude_layers=1, core_layers=2, coda_layers=1, default_steps=5
    )
    block = metadata["recurrence"]
    assert block["prelude_layers"] == 1
    assert block["core_layers"] == 2
    assert block["coda_layers"] == 1
    assert block["default_steps"] == 5
    assert block["mean_recurrence"] == 3.0
    assert block["mean_backprop_depth"] == 2
    assert block["recurrence_sigma"] == 0.5
    assert block["mean_recurrence_steps"] >= 1


def test_training_recurrence_flags_fail_closed() -> None:
    parser = cli._training_parser()

    with pytest.raises(ValueError, match="must sum to --layers"):
        cli._resolve_training_recurrence(
            parser.parse_args(
                ["--smoke", "--layers", "4", "--recurrent-layers", "1", "1", "1"]
            )
        )
    with pytest.raises(ValueError, match="--mean-recurrence is required"):
        cli._resolve_training_recurrence(
            parser.parse_args(
                ["--smoke", "--layers", "4", "--recurrent-layers", "1", "2", "1"]
            )
        )
    with pytest.raises(ValueError, match="requires --recurrent-layers"):
        cli._resolve_training_recurrence(
            parser.parse_args(["--smoke", "--layers", "4", "--mean-recurrence", "4"])
        )

    config, sampler = cli._resolve_training_recurrence(
        parser.parse_args(["--smoke", "--layers", "4"])
    )
    assert (config, sampler) == (None, None)

    config, sampler = cli._resolve_training_recurrence(
        parser.parse_args(
            [
                "--smoke",
                "--layers",
                "4",
                "--recurrent-layers",
                "1",
                "2",
                "1",
                "--mean-recurrence",
                "6",
                "--max-recurrence",
                "9",
            ]
        )
    )
    assert config.default_steps == 6
    assert sampler.mean_recurrence == 6.0
    assert sampler.max_recurrence == 9


def test_generate_recurrence_flags_and_trace_line(monkeypatch, capsys) -> None:
    vocab = _FakeVocab()
    model = _FakeModel()
    model.config = SimpleNamespace(
        recurrence=SimpleNamespace(prelude_layers=1, core_layers=2, coda_layers=1)
    )
    captured: dict[str, object] = {}

    def fake_generate_tokens(loaded_model, prompt_ids, **kwargs):
        captured.update(kwargs)
        trace = kwargs["trace"]
        trace.max_steps = trace.prefill_steps = kwargs["recurrence_steps"]
        trace.decode_steps.extend([6, 4])
        trace.early_exits = 1
        return torch.tensor([[1, 2]])

    monkeypatch.setattr(
        cli,
        "load_checkpoint",
        lambda path, *, map_location: (model, vocab, {"kind": "prose"}),
    )
    monkeypatch.setattr(cli, "Encoder", lambda loaded: SimpleNamespace(encode=lambda *a, **k: [1]))
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
            "--recurrence-steps",
            "6",
            "--adaptive-exit",
            "latent_diff",
            "--exit-threshold",
            "0.05",
            "--min-steps",
            "3",
            "--warm-start",
            "--exit-state-policy",
            "skip",
        ]
    )

    assert captured["recurrence_steps"] == 6
    assert captured["warm_start"] is True
    policy = captured["adaptive_exit"]
    assert isinstance(policy, AdaptiveExit)
    assert (policy.criterion, policy.threshold, policy.min_steps) == ("latent_diff", 0.05, 3)
    assert policy.state_policy == "skip"
    assert captured["trace"].max_steps == 6
    output = capsys.readouterr().out.splitlines()
    assert output[-1] == (
        "recurrence: steps=6 prefill=6 decode mean=5.00 min=4 max=6 early_exits=1"
    )


def test_generate_recurrence_flags_rejected_on_a_flat_checkpoint(monkeypatch) -> None:
    vocab = _FakeVocab()
    model = _FakeModel()
    monkeypatch.setattr(
        cli,
        "load_checkpoint",
        lambda path, *, map_location: (model, vocab, {"kind": "prose"}),
    )
    monkeypatch.setattr(cli, "Encoder", lambda loaded: SimpleNamespace(encode=lambda *a, **k: [1]))
    monkeypatch.setattr(cli, "Vocab", _FakeVocab)
    monkeypatch.setattr(
        cli,
        "generate_tokens",
        lambda loaded_model, prompt_ids, **kwargs: torch.tensor([[1, 2]]),
    )

    with pytest.raises(SystemExit, match="no recurrent core"):
        cli.main(
            [
                "generate",
                "trained.pt",
                "--prompt",
                "prompt",
                "--device",
                "cpu",
                "--adaptive-exit",
                "kl",
            ]
        )


def test_plan_recurrence_columns(capsys) -> None:
    cli.main(["plan", "10m", "--vocab-size", "4096"])
    header, _, row = capsys.readouterr().out.splitlines()
    assert "core" in header.split()
    assert "eff" in header.split()
    assert row.split()[6] == "-"

    cli.main(
        [
            "plan",
            "10m",
            "--vocab-size",
            "4096",
            "--recurrence",
            "1",
            "1",
            "--recurrence-steps",
            "4",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    block = payload[0]["recurrence"]
    assert block["prelude_layers"] == 1
    assert block["coda_layers"] == 1
    assert block["default_steps"] == 4
    assert block["core_layers"] == payload[0]["config"]["n_layers"] - 2
    assert block["effective_depth"] == 1 + 4 * block["core_layers"] + 1
