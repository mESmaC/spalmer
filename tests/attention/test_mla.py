"""Smoke tests for the global NoPE MLA token-mixer slice (ledger C04)."""

from __future__ import annotations

import pytest
import torch

from spalmer.attention.mla import MLAConfig, MLATokenMixer


@pytest.fixture()
def mla_config() -> MLAConfig:
    return MLAConfig(
        hidden_size=32,
        num_heads=4,
        head_k_dim=8,
        head_v_dim=8,
        q_latent_dim=16,
        kv_latent_dim=8,
    )


def test_output_shapes_and_cache_growth(mla_config: MLAConfig):
    torch.manual_seed(0)
    mixer = MLATokenMixer(mla_config)
    x = torch.randn(2, 9, mla_config.hidden_size)
    out, state = mixer(x)
    assert out.shape == x.shape
    assert state.latent_kv.shape == (2, 9, mla_config.kv_latent_dim)

    out_t, state_t = mixer.step(x[:, :1])
    assert out_t.shape == (2, 1, mla_config.hidden_size)
    assert state_t.latent_kv.shape == (2, 1, mla_config.kv_latent_dim)


def test_gradients_exist(mla_config: MLAConfig):
    torch.manual_seed(0)
    mixer = MLATokenMixer(mla_config)
    x = torch.randn(2, 6, mla_config.hidden_size)
    out, _ = mixer(x)
    out.sum().backward()
    for name, param in mixer.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_full_sequence_matches_recurrent_decode(mla_config: MLAConfig):
    torch.manual_seed(0)
    mixer = MLATokenMixer(mla_config)
    torch.manual_seed(1)
    x = torch.randn(2, 15, mla_config.hidden_size)

    out_full, state_full = mixer(x)

    state = mixer.create_state(2)
    outs = []
    for t in range(15):
        out_t, state = mixer.step(x[:, t : t + 1], state)
        outs.append(out_t)
    torch.testing.assert_close(torch.cat(outs, dim=1), out_full, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(state.latent_kv, state_full.latent_kv, rtol=1e-5, atol=1e-6)

    _, warm = mixer(x[:, :7])
    out_chunk, state_chunk = mixer(x[:, 7:], warm)
    state = warm
    outs = []
    for t in range(7, 15):
        out_t, state = mixer.step(x[:, t : t + 1], state)
        outs.append(out_t)
    torch.testing.assert_close(torch.cat(outs, dim=1), out_chunk, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(state.latent_kv, state_chunk.latent_kv, rtol=1e-5, atol=1e-6)


def test_unsupported_padding_and_reset_are_rejected(mla_config: MLAConfig):
    torch.manual_seed(0)
    mixer = MLATokenMixer(mla_config)
    x = torch.randn(2, 4, mla_config.hidden_size)

    padded = torch.ones(2, 4, dtype=torch.bool)
    padded[1, 3] = False
    with pytest.raises(NotImplementedError):
        mixer(x, attention_mask=padded)

    reset = torch.zeros(2, 4, dtype=torch.bool)
    reset[0, 2] = True
    with pytest.raises(NotImplementedError):
        mixer(x, state_reset_mask=reset)

    with pytest.raises(ValueError):
        mixer(x, attention_mask=torch.ones(2, 3, dtype=torch.bool))
