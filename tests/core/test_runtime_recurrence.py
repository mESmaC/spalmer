"""Depth policy threading through the prototype train and generate loops."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from spalmer.modeling import AdaptiveExit, CausalLMOutput
from spalmer.runtime import RecurrenceTrace, generate_tokens, train_token_stream
from spalmer.training import RecurrenceSampler

_VOCAB = 8
_WIDTH = 4


def _recurrence_config() -> SimpleNamespace:
    return SimpleNamespace(
        prelude_layers=1,
        core_layers=2,
        coda_layers=1,
        default_steps=4,
        latent_init_std=1.0,
    )


class _FakeRecurrentModel(nn.Module):
    """Records forward kwargs; only ``config.recurrence`` marks it recurrent."""

    def __init__(self, *, recurrent: bool = True, decode_steps: int | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(_WIDTH, _VOCAB))
        self.execution_device = torch.device("cpu")
        self.average_surprise = 0.0
        self.calls: list[dict[str, object]] = []
        self._decode_steps = decode_steps
        self.config = SimpleNamespace(
            recurrence=_recurrence_config() if recurrent else None,
            n_layers=4,
        )

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> CausalLMOutput:
        self.calls.append({"mode": kwargs.get("execution_mode", "prefill"), **kwargs})
        anchor = self.weight.sum() * 0.0
        requested = kwargs.get("recurrence_steps")
        steps = None if requested is None else int(requested)
        if kwargs.get("execution_mode") == "decode" and self._decode_steps is not None:
            steps = self._decode_steps
        logits = anchor + torch.zeros((*input_ids.shape, _VOCAB))
        return CausalLMOutput(
            logits=logits,
            token_mixer_states=(None,),
            channel_mixer_states=(None,),
            loss=anchor + 1.0,
            predictive_entropy=anchor + 2.0,
            token_nll=anchor + torch.zeros((input_ids.shape[0], input_ids.shape[1] - 1)),
            layer_metrics=({},),
            recurrence_steps=steps,
            latent_states=(
                None if steps is None else torch.zeros((*input_ids.shape, _WIDTH))
            ),
        )

    def update_potentiation(self, layer_metrics):
        del layer_metrics
        return ()

    def observe_surprise(self, token_nll, mask) -> None:
        del token_nll, mask


def test_generate_tokens_threads_recurrence_and_trace() -> None:
    model = _FakeRecurrentModel()
    trace = RecurrenceTrace()
    exit_policy = AdaptiveExit(criterion="latent_diff", threshold=1e9, min_steps=2)

    generate_tokens(
        model,
        torch.tensor([1, 2, 3]),
        max_new_tokens=3,
        recurrence_steps=6,
        adaptive_exit=exit_policy,
        warm_start=True,
        trace=trace,
    )

    prefill, *decodes = model.calls
    assert prefill["recurrence_steps"] == 6
    assert isinstance(prefill["latent_generator"], torch.Generator)
    assert "adaptive_exit" not in prefill
    assert len(decodes) == 2
    for call in decodes:
        assert call["recurrence_steps"] == 6
        assert call["adaptive_exit"] is exit_policy
        assert tuple(call["latent_init"].shape) == (1, 1, _WIDTH)
    assert trace.max_steps == 6
    assert trace.prefill_steps == 6
    assert trace.decode_steps == [6, 6]
    assert trace.early_exits == 0
    assert trace.mean_decode_steps == pytest.approx(6.0)
    assert (trace.min_decode_steps, trace.max_decode_steps) == (6, 6)


def test_generate_tokens_counts_early_exits_and_defaults_to_checkpoint_depth() -> None:
    model = _FakeRecurrentModel(decode_steps=2)
    trace = RecurrenceTrace()

    generate_tokens(model, torch.tensor([1, 2]), max_new_tokens=3, trace=trace)

    assert model.calls[0]["recurrence_steps"] == 4
    assert trace.max_steps == 4
    assert trace.decode_steps == [2, 2]
    assert trace.early_exits == 2
    assert trace.mean_decode_steps == pytest.approx(2.0)


def test_generate_tokens_rejects_recurrence_args_on_flat_model() -> None:
    model = _FakeRecurrentModel(recurrent=False)

    for kwargs in (
        {"recurrence_steps": 3},
        {"adaptive_exit": AdaptiveExit(criterion="latent_diff")},
        {"warm_start": True},
    ):
        with pytest.raises(ValueError, match="model has no recurrent core"):
            generate_tokens(model, torch.tensor([1, 2]), max_new_tokens=1, **kwargs)

    trace = RecurrenceTrace()
    generate_tokens(model, torch.tensor([1, 2]), max_new_tokens=2, trace=trace)
    assert trace.decode_steps == []
    assert all("recurrence_steps" not in call for call in model.calls)


def test_generate_tokens_passes_recurrence_steps_to_dynamic_residency(monkeypatch) -> None:
    model = _FakeRecurrentModel()
    captured: dict[str, object] = {}

    def fake_choose(chosen_model, prompt_ids, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output=chosen_model(prompt_ids, recurrence_steps=5))

    monkeypatch.setattr(
        "spalmer.experts.residency.choose_inference_residency",
        fake_choose,
    )
    model.end_residency_request = lambda: None
    generate_tokens(
        model,
        torch.tensor([1, 2]),
        max_new_tokens=1,
        recurrence_steps=5,
        dynamic_residency=True,
    )

    assert captured == {"recurrence_steps": 5}


class _StreamModel(nn.Module):
    def __init__(self, *, recurrent: bool = True) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(_VOCAB, _WIDTH))
        self.average_surprise = 0.0
        self.calls: list[dict[str, object]] = []
        self.config = SimpleNamespace(
            recurrence=_recurrence_config() if recurrent else None,
            n_layers=4,
        )

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> CausalLMOutput:
        self.calls.append(dict(kwargs))
        anchor = self.embedding.sum()
        return CausalLMOutput(
            logits=anchor + torch.zeros((*input_ids.shape, _VOCAB)),
            token_mixer_states=(),
            channel_mixer_states=(),
            loss=anchor + 1.0,
            predictive_entropy=torch.tensor(2.0),
            token_nll=torch.zeros((input_ids.shape[0], input_ids.shape[1] - 1)),
            layer_metrics=(),
        )

    def update_potentiation(self, layer_metrics):
        del layer_metrics
        return ()

    def observe_surprise(self, token_nll, mask) -> None:
        del token_nll, mask


def test_train_token_stream_samples_r_per_step() -> None:
    model = _StreamModel()
    sampler = RecurrenceSampler(mean_recurrence=3.0, mean_backprop_depth=2, scheme="fixed")

    result = train_token_stream(
        model,
        torch.arange(64) % _VOCAB,
        steps=3,
        batch_size=2,
        sequence_length=4,
        recurrence=sampler,
    )

    assert [call["recurrence_steps"] for call in model.calls] == [3, 3, 3]
    assert [call["backprop_steps"] for call in model.calls] == [2, 2, 2]
    assert result.mean_recurrence_steps == pytest.approx(3.0)


def test_train_token_stream_fails_closed_on_recurrence_mismatch() -> None:
    stream = torch.arange(64) % _VOCAB
    sampler = RecurrenceSampler(mean_recurrence=3.0, scheme="fixed")

    with pytest.raises(ValueError, match="requires a RecurrenceSampler"):
        train_token_stream(
            _StreamModel(), stream, steps=1, batch_size=1, sequence_length=4
        )
    with pytest.raises(ValueError, match="model has no recurrent core"):
        train_token_stream(
            _StreamModel(recurrent=False),
            stream,
            steps=1,
            batch_size=1,
            sequence_length=4,
            recurrence=sampler,
        )

    flat = _StreamModel(recurrent=False)
    result = train_token_stream(flat, stream, steps=2, batch_size=1, sequence_length=4)
    assert result.mean_recurrence_steps is None
    assert all(
        "recurrence_steps" not in call and "backprop_steps" not in call
        for call in flat.calls
    )
