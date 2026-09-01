"""Gradient smoke tests: full prefill, decode loop, and float64 gradcheck."""

from __future__ import annotations

import torch

from spalmer.attention import KDAConfig, KDATokenMixer
from spalmer.attention.recurrence import kda_recurrent_reference

ALL_PARAM_NAMES = {
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "q_conv.weight",
    "k_conv.weight",
    "v_conv.weight",
    "f_proj.0.weight",
    "f_proj.1.weight",
    "b_proj.weight",
    "A_log",
    "dt_bias",
    "g_proj.0.weight",
    "g_proj.1.weight",
    "g_proj.1.bias",
    "o_proj.weight",
}


def test_prefill_backward_all_parameters_recover_grad(small_config: KDAConfig):
    torch.manual_seed(0)
    mixer = KDATokenMixer(small_config)
    x = torch.randn(2, 12, small_config.hidden_size, requires_grad=True)
    out, _ = mixer(x)
    out.pow(2).mean().backward()
    for name, param in mixer.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert {name for name, _ in mixer.named_parameters()} == ALL_PARAM_NAMES


def test_decode_loop_backward_all_parameters_recover_grad(small_config: KDAConfig):
    torch.manual_seed(0)
    mixer = KDATokenMixer(small_config)
    x = torch.randn(2, 8, small_config.hidden_size, requires_grad=True)
    state = mixer.create_state(2)
    losses = []
    for t in range(8):
        out_t, state = mixer.step(x[:, t : t + 1], state)
        losses.append(out_t.pow(2).mean())
    torch.stack(losses).sum().backward()
    for name, param in mixer.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_gradient_flows_through_input_only_path(small_config: KDAConfig):
    """Input gradients must be nonzero (the mixer actually depends on x)."""

    torch.manual_seed(0)
    mixer = KDATokenMixer(small_config)
    x = torch.randn(1, 10, small_config.hidden_size, requires_grad=True)
    out, _ = mixer(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_recurrent_reference_gradcheck_float64():
    """Numerical vs analytical gradients of the reference recurrence."""

    torch.manual_seed(0)
    B, T, H, K, V = 1, 4, 2, 3, 2
    q = torch.randn(B, T, H, K, dtype=torch.float64, requires_grad=True)
    k = torch.randn(B, T, H, K, dtype=torch.float64, requires_grad=True)
    v = torch.randn(B, T, H, V, dtype=torch.float64, requires_grad=True)
    g = -torch.rand(B, T, H, K, dtype=torch.float64, requires_grad=True) - 0.1
    beta = torch.randn(B, T, H, dtype=torch.float64, requires_grad=True)
    state0 = torch.randn(B, H, K, V, dtype=torch.float64, requires_grad=True)

    def fn(q_, k_, v_, g_, b_, s_):
        o, s = kda_recurrent_reference(q_, k_, v_, g_, b_, s_)
        return o.sum() + s.sum()

    assert torch.autograd.gradcheck(fn, (q, k, v, g, beta, state0))


def test_mixer_gradcheck_float64():
    """Whole-module gradcheck on a tiny configuration (reference backend)."""

    from torch.func import functional_call

    torch.manual_seed(0)
    cfg = KDAConfig(hidden_size=6, num_heads=2, head_k_dim=3, head_v_dim=2, backend="reference")
    mixer = KDATokenMixer(cfg).double()
    x = torch.randn(1, 4, cfg.hidden_size, dtype=torch.float64)
    names = [n for n, _ in mixer.named_parameters()]
    params = tuple(p for _, p in mixer.named_parameters())

    def fn(*ps):
        return functional_call(mixer, dict(zip(names, ps)), (x,))[0]

    assert torch.autograd.gradcheck(fn, params, eps=1e-6, atol=1e-8)


def test_prefill_and_step_gradients_both_finite(small_config: KDAConfig):
    """Both paths produce finite gradients through every parameter.

    Note: gradients are NOT expected to match between the fully unrolled
    prefill and the decode loop; returned states are detached, so decode is
    truncated-BPTT by contract.
    """

    torch.manual_seed(0)
    x = torch.randn(1, 6, small_config.hidden_size)

    mixer_a = KDATokenMixer(small_config)
    out_a, _ = mixer_a(x)
    out_a.sum().backward()

    mixer_b = KDATokenMixer(small_config)
    mixer_b.load_state_dict(mixer_a.state_dict())
    state = mixer_b.create_state(1)
    outs = []
    for t in range(6):
        out_t, state = mixer_b.step(x[:, t : t + 1], state)
        outs.append(out_t)
    torch.cat(outs, 1).sum().backward()

    for (name, pa), (_, pb) in zip(mixer_a.named_parameters(), mixer_b.named_parameters()):
        assert pa.grad is not None, name
        assert pb.grad is not None, name
        assert torch.isfinite(pa.grad).all(), name
        assert torch.isfinite(pb.grad).all(), name
        assert pa.grad.abs().sum() > 0, name
        assert pb.grad.abs().sum() > 0, name
