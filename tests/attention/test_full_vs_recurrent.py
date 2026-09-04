"""Full-sequence (chunk/prefill) versus token-by-token (recurrent/decode).

Both paths must produce identical outputs and identical final states, and
the gate algebra must satisfy the KDA contract.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from spalmer import RecurrenceConfig, RecurrentMixerStates, SPALMERConfig
from spalmer.attention import KDAConfig, KDATokenMixer, MLAConfig
from spalmer.attention.recurrence import (
    compute_forget_gate,
    init_dt_bias,
    kda_recurrent_reference,
)
from spalmer.experts import MicroExpertsConfig
from spalmer.factory import build_spalmer_model


def test_full_prefill_equals_recurrent_decode(small_config: KDAConfig):
    torch.manual_seed(0)
    mixer = KDATokenMixer(small_config)
    torch.manual_seed(1)
    T = 21
    x = torch.randn(2, T, small_config.hidden_size)

    out_full, state_full = mixer(x)

    state = mixer.create_state(2)
    outs = []
    for t in range(T):
        out_t, state = mixer.step(x[:, t : t + 1], state)
        outs.append(out_t)
    out_loop = torch.cat(outs, dim=1)

    torch.testing.assert_close(out_loop, out_full, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        state.recurrent_state, state_full.recurrent_state, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(state.conv_q, state_full.conv_q)
    torch.testing.assert_close(state.conv_k, state_full.conv_k)
    torch.testing.assert_close(state.conv_v, state_full.conv_v)


def test_full_prefill_equals_recurrent_decode_float64(small_config: KDAConfig):
    torch.manual_seed(0)
    mixer = KDATokenMixer(small_config).double()
    x = torch.randn(1, 13, small_config.hidden_size, dtype=torch.float64)
    out_full, state_full = mixer(x)
    state = mixer.create_state(1, dtype=torch.float64)
    outs = []
    for t in range(13):
        out_t, state = mixer.step(x[:, t : t + 1], state)
        outs.append(out_t)
    torch.testing.assert_close(torch.cat(outs, 1), out_full, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(state.recurrent_state, state_full.recurrent_state)


def test_mid_sequence_initial_state(small_config: KDAConfig):
    """Passing a warm state to the full prefill matches warm stepping."""

    torch.manual_seed(2)
    mixer = KDATokenMixer(small_config)
    x = torch.randn(2, 16, small_config.hidden_size)
    _, warm = mixer(x[:, :7])

    out_chunk, state_chunk = mixer(x[:, 7:], warm)
    state = warm
    outs = []
    for t in range(7, 16):
        out_t, state = mixer.step(x[:, t : t + 1], state)
        outs.append(out_t)
    torch.testing.assert_close(torch.cat(outs, 1), out_chunk, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        state.recurrent_state, state_chunk.recurrent_state, rtol=1e-5, atol=1e-6
    )


def test_alpha_is_channelwise_decay_in_open_unit_interval():
    torch.manual_seed(0)
    B, T, H, K = 2, 5, 3, 7
    f = torch.randn(B, T, H, K) * 2
    A_log = torch.log(torch.empty(H).uniform_(1, 16))
    dt_bias = init_dt_bias(K, H)
    g = compute_forget_gate(f, A_log, dt_bias, None)
    assert g.shape == (B, T, H, K)
    assert (g < 0).all()
    alpha = g.exp()
    assert (alpha > 0).all() and (alpha < 1).all()


def test_alpha_lower_bound_clamp():
    torch.manual_seed(0)
    f = torch.randn(1, 4, 2, 3) * 5
    A_log = torch.zeros(2)
    dt_bias = torch.zeros(2, 3)
    lb = -5.0
    g = compute_forget_gate(f, A_log, dt_bias, lb)
    assert (g >= lb).all() and (g < 0).all()


def test_beta_scalar_per_head_in_unit_interval(small_config: KDAConfig):
    torch.manual_seed(0)
    mixer = KDATokenMixer(small_config)
    x = torch.randn(2, 6, small_config.hidden_size)
    beta_raw = mixer.b_proj(x)
    assert beta_raw.shape == (2, 6, small_config.num_heads)
    assert (torch.sigmoid(beta_raw) > 0).all() and (torch.sigmoid(beta_raw) < 1).all()


def test_zero_write_strength_forgets_everything():
    """beta -> 0 means the state stays at its initial value and o equals the
    retrieval from the (zero) state: exactly the ledger's no-write limit."""

    torch.manual_seed(0)
    B, T, H, K, V = 2, 6, 3, 5, 4
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    g = -torch.rand(B, T, H, K)  # arbitrary valid log decay
    beta = torch.full((B, T, H), -1e9)  # sigmoid -> 0
    o, state = kda_recurrent_reference(q, k, v, g, beta)
    assert o.abs().max().item() == pytest.approx(0.0, abs=1e-20)
    assert state.abs().max().item() == pytest.approx(0.0, abs=1e-20)


def test_delta_rule_matches_ledger_matrix_form():
    """The delta-rule loop must equal the literal ledger update
    S_t = (I - beta k k^T) Diag(alpha) S_{t-1} + beta k v^T; o_t = S_t^T q_t."""

    torch.manual_seed(0)
    B, T, H, K, V = 1, 5, 1, 3, 2
    q = torch.randn(B, T, H, K, dtype=torch.float64)
    k = torch.randn(B, T, H, K, dtype=torch.float64)
    v = torch.randn(B, T, H, V, dtype=torch.float64)
    g = -torch.rand(B, T, H, K, dtype=torch.float64)
    beta_raw = torch.randn(B, T, H, dtype=torch.float64)

    o_ref, s_ref = kda_recurrent_reference(q, k, v, g, beta_raw)

    qn = F.normalize(q, dim=-1)
    kn = F.normalize(k, dim=-1)
    alpha = g.exp()
    beta = torch.sigmoid(beta_raw)
    s = torch.zeros(K, V, dtype=torch.float64)
    eye = torch.eye(K, dtype=torch.float64)
    outs = []
    for t in range(T):
        a = torch.diag(alpha[0, t, 0])
        kk = kn[0, t, 0]
        vv = v[0, t, 0]
        b = beta[0, t, 0]
        s = (eye - b * torch.outer(kk, kk)) @ (a @ s) + b * torch.outer(kk, vv)
        outs.append(s.T @ qn[0, t, 0])
    o_literal = torch.stack(outs).view(1, T, 1, V)
    torch.testing.assert_close(o_literal, o_ref, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(s, s_ref[0, 0], rtol=1e-10, atol=1e-12)


def test_allow_neg_eigval_doubles_beta():
    # With a zero initial state and a single step, S = beta * outer(k, v),
    # so doubling beta (sigmoid -> 2*sigmoid) exactly doubles the state.
    torch.manual_seed(0)
    B, T, H, K, V = 2, 1, 3, 5, 4
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    g = -torch.rand(B, T, H, K)
    beta = torch.zeros(B, T, H)  # sigmoid -> 0.5; neg eigval -> 1.0
    _, s_std = kda_recurrent_reference(q, k, v, g, beta)
    _, s_neg = kda_recurrent_reference(q, k, v, g, beta, allow_neg_eigval=True)
    assert s_neg.abs().max() > 0
    torch.testing.assert_close(s_neg, 2 * s_std, rtol=1e-5, atol=1e-7)


def test_output_is_alive(small_config: KDAConfig):
    torch.manual_seed(0)
    mixer = KDATokenMixer(small_config)
    x = torch.randn(4, 32, small_config.hidden_size)
    out, _ = mixer(x)
    assert out.std().item() > 1e-3
    assert torch.isfinite(out).all()


def _recurrent_model():
    """A tiny real-mixer model whose core iterates one KDA and one MLA block."""

    model_config = SPALMERConfig(
        vocab_size=32,
        d_model=16,
        n_layers=4,
        tokenizer_version=1,
        tokenizer_fingerprint="recurrent-core-v1",
        ple_expansion_factor=1,
        # Physical indices 1 (core) and 3 (coda) are MLA, so the iterated core
        # carries both a KDA recurrent state and a growing MLA latent cache.
        token_mixer_pattern=("kda", "mla", "kda", "mla"),
        # Deterministic latent start: prefill and decode must agree exactly.
        recurrence=RecurrenceConfig(1, 2, 1, default_steps=3, latent_init_std=0.0),
    )
    kda = KDAConfig(hidden_size=16, num_heads=2, head_k_dim=8, backend="reference")
    mla = MLAConfig(hidden_size=16, num_heads=2, head_k_dim=8, q_latent_dim=8, kv_latent_dim=8)
    experts = MicroExpertsConfig(d_model=16, num_experts=4, expert_inter_dim=8, active_experts=2)
    torch.manual_seed(0)
    return build_spalmer_model(model_config, kda, mla, experts).eval()


def test_recurrent_core_prefill_matches_decode():
    model = _recurrent_model()
    prompt = torch.randint(0, model.config.vocab_size, (1, 9))
    steps = 3

    with torch.no_grad():
        full = model(prompt, recurrence_steps=steps)
        warm = model(prompt[:, :-1], recurrence_steps=steps)
        stepped = model(
            prompt[:, -1:],
            execution_mode="decode",
            recurrence_steps=steps,
            token_mixer_states=warm.token_mixer_states,
            channel_mixer_states=warm.channel_mixer_states,
        )

    assert full.recurrence_steps == stepped.recurrence_steps == steps
    torch.testing.assert_close(
        stepped.logits[:, -1], full.logits[:, -1], rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        stepped.latent_states[:, -1], full.latent_states[:, -1], rtol=1e-4, atol=1e-4
    )
    # Each core block keeps one state per iteration through both paths.
    for slot in list(warm.token_mixer_states[1:3]) + list(stepped.token_mixer_states[1:3]):
        assert isinstance(slot, RecurrentMixerStates)
        assert len(slot) == steps
    # Distinct per-iteration KDA memories: one shared cache would collapse them.
    core_kda = stepped.token_mixer_states[2]
    assert not torch.allclose(
        core_kda[0].recurrent_state, core_kda[steps - 1].recurrent_state
    )


def test_recurrent_core_token_by_token_decode_matches_a_full_prefill():
    model = _recurrent_model()
    prompt = torch.randint(0, model.config.vocab_size, (1, 6))
    steps = 2

    with torch.no_grad():
        full = model(prompt, recurrence_steps=steps)
        output = model(prompt[:, :1], recurrence_steps=steps)
        for position in range(1, prompt.shape[1]):
            output = model(
                prompt[:, position : position + 1],
                execution_mode="decode",
                recurrence_steps=steps,
                token_mixer_states=output.token_mixer_states,
                channel_mixer_states=output.channel_mixer_states,
            )
            torch.testing.assert_close(
                output.logits[:, -1], full.logits[:, position], rtol=1e-4, atol=1e-4
            )
