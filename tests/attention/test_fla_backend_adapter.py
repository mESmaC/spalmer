"""Focused tests for optimized-attention adapter contracts."""

from __future__ import annotations

import pytest
import torch

from spalmer.attention.backends import FlaKdaBackend


@pytest.mark.parametrize(("method", "sequence_length"), [("sequence", 4), ("step", 1)])
def test_fla_backend_flattens_dt_bias_without_changing_parameter_gradient_shape(
    method: str,
    sequence_length: int,
) -> None:
    backend = object.__new__(FlaKdaBackend)
    observed_shapes: list[torch.Size] = []

    def fake_kernel(**kwargs):
        observed_shapes.append(kwargs["dt_bias"].shape)
        output = kwargs["q"] + kwargs["dt_bias"].sum().to(kwargs["q"].dtype)
        return output, kwargs["initial_state"]

    backend._chunk_kda = fake_kernel
    backend._fused_recurrent_kda = fake_kernel

    batch, heads, key_dim = 2, 2, 3
    q = torch.randn(batch, sequence_length, heads, key_dim)
    state = torch.zeros(batch, heads, key_dim, key_dim)
    dt_bias = torch.randn(heads, key_dim, requires_grad=True)
    call = getattr(backend, method)
    output, _ = call(
        q=q,
        k=q,
        v=q,
        f=q,
        beta_raw=torch.zeros(batch, sequence_length, heads),
        A_log=torch.zeros(heads),
        dt_bias=dt_bias,
        state=state,
        lower_bound=None,
        allow_neg_eigval=False,
    )
    output.sum().backward()

    assert observed_shapes == [torch.Size([heads * key_dim])]
    assert dt_bias.grad is not None
    assert dt_bias.grad.shape == dt_bias.shape
    assert torch.isfinite(dt_bias.grad).all()


@pytest.mark.parametrize(
    ("key_width", "expected_kernel"),
    [(12, "chunk"), (16, "fused")],
)
def test_fla_decode_selects_safe_single_token_kernel(
    key_width: int,
    expected_kernel: str,
) -> None:
    backend = object.__new__(FlaKdaBackend)
    calls: list[tuple[str, int, torch.dtype]] = []

    def fake_chunk_kernel(**kwargs):
        calls.append(("chunk", kwargs["q"].shape[1], kwargs["initial_state"].dtype))
        return kwargs["q"], kwargs["initial_state"]

    def fake_fused_kernel(**kwargs):
        calls.append(("fused", kwargs["q"].shape[1], kwargs["initial_state"].dtype))
        return kwargs["q"], kwargs["initial_state"]

    backend._chunk_kda = fake_chunk_kernel
    backend._fused_recurrent_kda = fake_fused_kernel

    q = torch.randn(1, 1, 2, key_width)
    state = torch.zeros(1, 2, key_width, key_width)
    output, final_state = backend.step(
        q=q,
        k=q,
        v=q,
        f=q,
        beta_raw=torch.zeros(1, 1, 2),
        A_log=torch.zeros(2),
        dt_bias=torch.zeros(2, key_width),
        state=state,
        lower_bound=None,
        allow_neg_eigval=False,
    )

    assert calls == [(expected_kernel, 1, torch.float32)]
    assert output is q
    assert final_state is state
