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
