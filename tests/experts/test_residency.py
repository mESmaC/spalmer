"""Checks for the C13 inference residency controller and its C08 signal."""

from __future__ import annotations

import pytest
import torch

from spalmer.attention import KDAConfig, MLAConfig
from spalmer.checkpoint import load_checkpoint, save_checkpoint
from spalmer.config import SPALMERConfig
from spalmer.experts import MicroExpertsConfig, choose_inference_residency
from spalmer.factory import build_spalmer_model
from spalmer.runtime import generate_tokens, train_token_stream
from spalmer.tokenizer import Encoder, Sample, train


def _build(num_experts: int = 8, **overrides: object):
    vocab = train([Sample("alpha beta gamma delta epsilon zeta " * 8)])
    config = SPALMERConfig(
        vocab_size=len(vocab),
        d_model=8,
        n_layers=4,
        tokenizer_version=vocab.version,
        tokenizer_fingerprint=vocab.fingerprint,
        ple_expansion_factor=1,
    )
    kda = KDAConfig(hidden_size=8, num_heads=2, head_k_dim=4, head_v_dim=4, backend="reference")
    mla = MLAConfig(
        hidden_size=8, num_heads=2, head_k_dim=4, head_v_dim=4, q_latent_dim=4, kv_latent_dim=4
    )
    values: dict[str, object] = {
        "d_model": 8,
        "num_experts": num_experts,
        "expert_inter_dim": 3,
        "active_experts": 2,
        "max_active_experts": 6,
        "potentiation_budget": 0,
    }
    values.update(overrides)
    experts = MicroExpertsConfig(**values)
    torch.manual_seed(0)
    model = build_spalmer_model(config, kda, mla, experts).eval()
    return model, vocab, (kda, mla, experts)


def test_residency_override_is_bounded_and_reversible() -> None:
    model, _, _ = _build()
    assert model.active_experts == 2
    model.set_active_experts(4)
    assert model.active_experts == 4
    for block in model.backbone.blocks:
        assert block.channel_mixer.active_experts == 4
    with pytest.raises(ValueError, match="active expert count"):
        model.set_active_experts(7)
    model.set_active_experts(None)
    assert model.active_experts == 2


def test_controller_starts_at_minimum_expands_by_increment_and_stays_in_cap() -> None:
    model, vocab, _ = _build()
    prompt = torch.tensor([Encoder(vocab).encode("alpha beta gamma")])

    # With no observed average surprise every prompt looks hard, so the
    # controller keeps adding bounded increments while each one still pays.
    decision = choose_inference_residency(model, prompt, min_gain=-1.0)
    assert decision.trace[0][0] == 2
    assert [count for count, _ in decision.trace] == [2, 4, 6]
    assert decision.active_experts == 6
    assert model.active_experts == 6
    assert decision.effective_nll is not None and decision.effective_nll > 0
    assert decision.output.logits.shape[1] == prompt.shape[1]
    model.set_active_experts(None)


def test_residency_gain_must_be_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        _build(residency_min_gain=float("nan"))

    model, vocab, _ = _build()
    prompt = torch.tensor([Encoder(vocab).encode("alpha beta gamma")])
    with pytest.raises(ValueError, match="finite"):
        choose_inference_residency(model, prompt, min_gain=float("inf"))


def test_controller_rolls_back_an_expansion_that_does_not_pay() -> None:
    model, vocab, _ = _build()
    prompt = torch.tensor([Encoder(vocab).encode("alpha beta gamma")])

    decision = choose_inference_residency(model, prompt, min_gain=10.0)
    assert [count for count, _ in decision.trace] == [2, 4]
    assert decision.active_experts == 2
    assert model.active_experts == 2


def test_controller_uses_deterministic_inference_mode_and_restores_training() -> None:
    model, vocab, _ = _build()
    model.train()
    prompt = torch.tensor([Encoder(vocab).encode("alpha beta gamma")])

    first = choose_inference_residency(model, prompt)
    model.set_active_experts(None)
    second = choose_inference_residency(model, prompt)

    assert first.trace == second.trace
    assert first.active_experts == second.active_experts
    assert model.training


def test_controller_retains_the_minimum_when_surprise_is_below_average() -> None:
    model, vocab, _ = _build()
    prompt = torch.tensor([Encoder(vocab).encode("alpha beta gamma")])
    model.surprise_ema.fill_(1000.0)
    model.surprise_observations.fill_(1)

    decision = choose_inference_residency(model, prompt)
    assert decision.trace == ((2, decision.effective_signal),)
    assert decision.active_experts == 2
    assert decision.average_surprise == 1000.0


def test_single_token_prompt_uses_predictive_entropy() -> None:
    model, vocab, _ = _build()
    prompt = torch.tensor([Encoder(vocab).encode("alpha")[:1]])
    decision = choose_inference_residency(model, prompt, min_gain=10.0)
    assert decision.effective_nll is None
    assert decision.effective_signal == decision.predictive_entropy


def test_training_tracks_average_surprise_and_checkpoints_carry_it(tmp_path) -> None:
    model, vocab, (kda, mla, experts) = _build()
    tokens = torch.tensor(Encoder(vocab).encode("alpha beta gamma delta epsilon zeta " * 8))
    result = train_token_stream(model, tokens, steps=3, batch_size=2, sequence_length=8)
    assert result.average_surprise > 0
    assert int(model.surprise_observations) == 3
    assert model.surprise_ema.dtype == torch.float32

    path = tmp_path / "surprise.pt"
    save_checkpoint(path, model, vocab, kda_config=kda, mla_config=mla, experts_config=experts)
    restored, _, _ = load_checkpoint(path)
    assert restored.average_surprise == pytest.approx(model.average_surprise)

    # A version-2 bundle predates the buffers and the residency fields.
    payload = torch.load(path, weights_only=False)
    payload["format_version"] = 2
    payload["model_config"].pop("surprise_ema_decay")
    payload["model_state"].pop("surprise_ema")
    payload["model_state"].pop("surprise_observations")
    payload["experts_config"].pop("residency_increment")
    payload["experts_config"].pop("residency_min_gain")
    payload["metadata"]["average_surprise"] = 1.5
    legacy = tmp_path / "v2.pt"
    torch.save(payload, legacy)
    legacy_model, _, _ = load_checkpoint(legacy)
    assert legacy_model.average_surprise == pytest.approx(1.5)

    hybrid_payload = torch.load(path, weights_only=False)
    hybrid_payload["format_version"] = 2
    hybrid_payload["model_config"].pop("surprise_ema_decay")
    hybrid_payload["model_state"].pop("surprise_ema")
    hybrid_payload["experts_config"].pop("residency_increment")
    hybrid_payload["experts_config"].pop("residency_min_gain")
    hybrid = tmp_path / "hybrid-v2.pt"
    torch.save(hybrid_payload, hybrid)
    with pytest.raises(RuntimeError, match="surprise_ema"):
        load_checkpoint(hybrid)

    missing_config_payload = torch.load(path, weights_only=False)
    missing_config_payload["model_config"].pop("surprise_ema_decay")
    missing_config = tmp_path / "missing-model-config.pt"
    torch.save(missing_config_payload, missing_config)
    with pytest.raises(ValueError, match="surprise_ema_decay"):
        load_checkpoint(missing_config)

    missing_ple_payload = torch.load(path, weights_only=False)
    missing_ple_payload["model_config"].pop("ple_quant_bits")
    missing_ple = tmp_path / "missing-ple-config.pt"
    torch.save(missing_ple_payload, missing_ple)
    with pytest.raises(ValueError, match="ple_quant_bits"):
        load_checkpoint(missing_ple)

    missing_kda_payload = torch.load(path, weights_only=False)
    missing_kda_payload["kda_config"].pop("backend")
    missing_kda = tmp_path / "missing-kda-config.pt"
    torch.save(missing_kda_payload, missing_kda)
    with pytest.raises(ValueError, match="backend"):
        load_checkpoint(missing_kda)


def test_generation_restores_the_previous_residency_after_dynamic_prefill() -> None:
    model, vocab, _ = _build()
    prompt = torch.tensor(Encoder(vocab).encode("alpha beta"))
    generated = generate_tokens(model, prompt, max_new_tokens=3, dynamic_residency=True)
    assert generated.shape == (1, prompt.numel() + 3)
    assert model.active_experts_override is None
    assert model.active_experts == 2
