"""Shape, dtype, state-contract, and configuration smoke tests."""

from __future__ import annotations

import pytest
import torch

from spalmer.attention import KDAConfig, KDAState, KDATokenMixer


def test_prefill_output_and_state_shapes(small_config: KDAConfig):
    torch.manual_seed(0)
    mixer = KDATokenMixer(small_config)
    x = torch.randn(2, 17, small_config.hidden_size)
    out, state = mixer(x)
    assert out.shape == (2, 17, small_config.hidden_size)
    assert state.recurrent_state.shape == (
        2,
        small_config.num_heads,
        small_config.head_k_dim,
        small_config.head_v_dim,
    )
    assert state.conv_q.shape == (2, small_config.key_dim, small_config.conv_width - 1)
    assert state.conv_k.shape == (2, small_config.key_dim, small_config.conv_width - 1)
    assert state.conv_v.shape == (2, small_config.value_dim, small_config.conv_width - 1)


def test_step_output_shape(small_config: KDAConfig):
    mixer = KDATokenMixer(small_config)
    x_t = torch.randn(3, 1, small_config.hidden_size)
    out_t, state = mixer.step(x_t)
    assert out_t.shape == (3, 1, small_config.hidden_size)
    assert state.recurrent_state.shape == (3, 4, 8, 8)


@pytest.mark.parametrize("T", [1, 2, 3, 4, 5, 16])
def test_prefill_various_lengths(small_config: KDAConfig, T: int):
    mixer = KDATokenMixer(small_config)
    x = torch.randn(2, T, small_config.hidden_size)
    out, state = mixer(x)
    assert out.shape == (2, T, small_config.hidden_size)
    assert state.conv_q.shape == (2, small_config.key_dim, small_config.conv_width - 1)
    assert torch.isfinite(out).all()


def test_dtype_preserved_and_state_matches_input(small_config: KDAConfig):
    for dtype in (torch.float32, torch.float64):
        mixer = KDATokenMixer(small_config).to(dtype)
        x = torch.randn(2, 9, small_config.hidden_size, dtype=dtype)
        out, state = mixer(x)
        assert out.dtype == dtype
        assert state.recurrent_state.dtype == dtype
        assert state.conv_q.dtype == dtype


def test_value_dim_differs_from_key_dim():
    cfg = KDAConfig(hidden_size=32, num_heads=4, head_k_dim=8, head_v_dim=6)
    mixer = KDATokenMixer(cfg)
    x = torch.randn(2, 7, 32)
    out, state = mixer(x)
    assert out.shape == (2, 7, 32)
    assert state.recurrent_state.shape == (2, 4, 8, 6)


def test_empty_state_factory(small_config: KDAConfig):
    state = KDAState.empty(5, small_config)
    assert state.recurrent_state.shape == (5, 4, 8, 8)
    assert state.conv_q.shape == (5, 32, 3)
    assert (state.recurrent_state == 0).all()
    assert (state.conv_q == 0).all()
    assert (state.conv_k == 0).all()
    assert (state.conv_v == 0).all()
    moved = state.to(dtype=torch.float64)
    assert moved.recurrent_state.dtype == torch.float64


def test_invalid_inputs_raise(small_config: KDAConfig):
    mixer = KDATokenMixer(small_config)
    with pytest.raises(ValueError):
        mixer(torch.randn(2, 3))  # not 3-D
    with pytest.raises(ValueError):
        mixer(torch.randn(2, 3, 31))  # wrong hidden size
    with pytest.raises(ValueError):
        mixer.step(torch.randn(2, 3, 32))  # step must be T=1


@pytest.mark.parametrize(
    ("kwargs",),
    [
        ({"hidden_size": 0},),
        ({"num_heads": 0},),
        ({"head_k_dim": -1},),
        ({"head_v_dim": -2},),
        ({"conv_width": 1},),
        ({"backend": "triton"},),
        ({"gate_lower_bound": 1.0},),
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        KDAConfig(**kwargs)


def test_auto_backend_is_reference_on_cpu():
    cfg = KDAConfig(hidden_size=16, num_heads=2, head_k_dim=4, backend="auto")
    mixer = KDATokenMixer(cfg)
    assert mixer.backend_name == "reference"


def test_fla_backend_raises_cleanly_when_missing(small_config: KDAConfig):
    from spalmer.attention.backends import fla_available

    if fla_available():
        pytest.skip("fla-core installed; guarded-error path not exercised")
    cfg = KDAConfig(**{**small_config.__dict__, "backend": "fla"})
    mixer = KDATokenMixer(cfg)
    with pytest.raises(ImportError, match="fla"):
        mixer(torch.randn(1, 4, cfg.hidden_size))


def test_parameters_registered(small_config: KDAConfig):
    mixer = KDATokenMixer(small_config)
    names = {name for name, _ in mixer.named_parameters()}
    expected = {
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
    assert expected <= names


def test_biasless_projections(small_config: KDAConfig):
    mixer = KDATokenMixer(small_config)
    assert mixer.q_proj.bias is None
    assert mixer.k_proj.bias is None
    assert mixer.v_proj.bias is None
    assert mixer.o_proj.bias is None
    assert mixer.f_proj[0].bias is None
    assert mixer.f_proj[1].bias is None
    assert mixer.b_proj.bias is None
    assert mixer.g_proj[1].bias is not None
