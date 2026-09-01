"""Causality and cache-state threading tests.

The recurrent state, conv caches, and outputs must behave exactly like the
infinite causal history they summarize:

- prefix invariance: appending future tokens cannot change past outputs;
- prefill -> decode: stepping with the state from a prefix equals computing
  the longer sequence directly;
- chunk continuation: prefill(a) + prefill(b, state) == prefill(a + b);
- T=1 forward is the same call as step.
"""

from __future__ import annotations

import pytest
import torch

from spalmer.attention import KDAConfig, KDATokenMixer


def _make(config: KDAConfig, seed: int = 0) -> tuple[KDATokenMixer, torch.Tensor]:
    torch.manual_seed(seed)
    mixer = KDATokenMixer(config)
    torch.manual_seed(seed + 1)
    x = torch.randn(2, 24, config.hidden_size)
    return mixer, x


def test_prefix_invariance(small_config: KDAConfig):
    mixer, x = _make(small_config)
    out_short, _ = mixer(x[:, :10])
    out_long, _ = mixer(x[:, :24])
    torch.testing.assert_close(out_short, out_long[:, :10])


def test_step_with_prefill_state_matches_longer_prefill(small_config: KDAConfig):
    mixer, x = _make(small_config)
    out_prefix, state = mixer(x[:, :10])
    out_next, _ = mixer.step(x[:, 10:11], state)
    out_full, _ = mixer(x[:, :11])
    torch.testing.assert_close(out_prefix, out_full[:, :10])
    torch.testing.assert_close(out_next, out_full[:, 10:11])


def test_prefill_chunk_continuation(small_config: KDAConfig):
    mixer, x = _make(small_config)
    out_a, state_a = mixer(x[:, :9])
    out_b, state_b = mixer(x[:, 9:], state_a)
    out_full, state_full = mixer(x)
    torch.testing.assert_close(out_a, out_full[:, :9])
    torch.testing.assert_close(out_b, out_full[:, 9:])
    torch.testing.assert_close(state_b.recurrent_state, state_full.recurrent_state)
    torch.testing.assert_close(state_b.conv_q, state_full.conv_q)


def test_single_token_forward_equals_step(small_config: KDAConfig):
    mixer, x = _make(small_config)
    out_fwd, st_fwd = mixer(x[:, 5:6])
    out_step, st_step = mixer.step(x[:, 5:6])
    torch.testing.assert_close(out_fwd, out_step)
    torch.testing.assert_close(st_fwd.recurrent_state, st_step.recurrent_state)


def test_short_prefill_below_conv_width(small_config: KDAConfig):
    mixer, x = _make(small_config)
    for t in (1, 2, 3):  # conv_width - 1 = 3
        out, state = mixer(x[:, :t])
        assert out.shape == (2, t, small_config.hidden_size)
        assert torch.isfinite(out).all()
    # compare two-step build-up against direct T=2 prefill
    out_a, st = mixer(x[:, :1])
    out_b, _ = mixer(x[:, 1:2], st)
    out_direct, _ = mixer(x[:, :2])
    torch.testing.assert_close(torch.cat([out_a, out_b], dim=1), out_direct)


def test_state_batches_are_independent(small_config: KDAConfig):
    torch.manual_seed(3)
    mixer = KDATokenMixer(small_config)
    x1 = torch.randn(1, 12, small_config.hidden_size)
    x2 = torch.randn(1, 12, small_config.hidden_size)
    out_pair, state_pair = mixer(torch.cat([x1, x2], dim=0))
    out_single1, state_single1 = mixer(x1)
    out_single2, state_single2 = mixer(x2)
    torch.testing.assert_close(out_pair[0], out_single1[0])
    torch.testing.assert_close(out_pair[1], out_single2[0])
    torch.testing.assert_close(state_pair.recurrent_state[0], state_single1.recurrent_state[0])
    torch.testing.assert_close(state_pair.conv_v[1], state_single2.conv_v[0])


def test_causal_state_cannot_see_future_through_state(small_config: KDAConfig):
    """Changing the suffix of a sequence changes only the state after it."""

    mixer, x = _make(small_config)
    _, state_a = mixer(x[:, :12])
    x_mut = x.clone()
    x_mut[:, 10:] = torch.randn_like(x_mut[:, 10:])
    _, state_b = mixer(x_mut[:, :12])
    # states through position 9 are built from the unchanged prefix; the
    # recurrent state, however, summarizes all 12 tokens, so instead verify
    # the prefix outputs are unchanged and the states do differ.
    out_a, _ = mixer(x[:, :12])
    out_b, _ = mixer(x_mut[:, :12])
    torch.testing.assert_close(out_a[:, :10], out_b[:, :10])
    assert not torch.allclose(state_a.recurrent_state, state_b.recurrent_state)


def test_fresh_state_each_call_is_deterministic(small_config: KDAConfig):
    mixer, x = _make(small_config)
    out1, s1 = mixer(x)
    out2, s2 = mixer(x)
    torch.testing.assert_close(out1, out2)
    torch.testing.assert_close(s1.recurrent_state, s2.recurrent_state)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_state_threading_dtype(small_config: KDAConfig, dtype: torch.dtype):
    mixer = KDATokenMixer(small_config).to(dtype)
    x = torch.randn(2, 8, small_config.hidden_size, dtype=dtype)
    out_a, st = mixer(x[:, :4])
    out_b, _ = mixer(x[:, 4:], st)
    out_full, _ = mixer(x)
    torch.testing.assert_close(torch.cat([out_a, out_b], 1), out_full)
