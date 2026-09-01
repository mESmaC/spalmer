"""Plain-PyTorch reference implementation of the KDA recurrence (ledger C04).

Implements exactly the ledger update

    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T q_t

with the Kimi Delta Attention parameterization (state rows indexed by K):

- ``alpha_t = exp(g_t)``, channel-wise forget gate over the key dimension,
  ``g_t = -exp(A_log) * softplus(f_t + dt_bias)`` in log space, per head;
- ``beta_t = sigmoid(b_t)`` (scalar per head), optionally ``2 * sigmoid``
  with ``allow_neg_eigval``;
- q and k are L2-normalized per head before entering the recurrence.

The delta-rule form used below is algebraically identical:

    S_dec = alpha_t[:, None] * S
    S_t   = S_dec + beta_t * outer(k_t, v_t - k_t @ S_dec)
    o_t   = q_t @ S_t

This module is the correctness reference; it does not recreate fla-core's
optimized chunkwise kernels.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def compute_forget_gate(
    f: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
) -> torch.Tensor:
    """Log-space forget gate ``g = -exp(A_log) * softplus(f + dt_bias)``.

    Args:
        f: ``[B, T, H, K]`` raw ``f_proj`` output.
        A_log: ``[H]`` float32 log decay scale.
        dt_bias: ``[H, K]`` float32 softplus-inverse step bias.
        lower_bound: optional clamp of g to ``[lower_bound, 0)``.

    Returns:
        g: ``[B, T, H, K]`` negative log decay.
    """

    compute_dtype = torch.float64 if f.dtype == torch.float64 else torch.float32
    a_scale = A_log.to(compute_dtype).reshape(1, 1, -1, 1)
    dt = dt_bias.to(compute_dtype).reshape(1, 1, *dt_bias.shape)
    g = -torch.exp(a_scale) * F.softplus(f.to(compute_dtype) + dt)
    if lower_bound is not None:
        g = g.clamp(min=lower_bound)
    return g


def init_dt_bias(gate_dim_per_head: int, num_heads: int) -> torch.Tensor:
    """Mamba-style dt init so ``softplus(dt_bias) ~ U(1e-3, 1e-1)`` at start."""

    dt = torch.exp(
        torch.rand(num_heads, gate_dim_per_head, dtype=torch.float32)
        * (math.log(0.1) - math.log(0.001))
        + math.log(0.001)
    ).clamp(min=1e-4)
    return dt + torch.log(-torch.expm1(-dt))


def kda_recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None = None,
    *,
    allow_neg_eigval: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the KDA recurrence over a sequence (reference path).

    Args:
        q, k: ``[B, T, H, K]`` post-conv/SiLU, NOT yet L2-normalized.
        v: ``[B, T, H, V]`` post-conv/SiLU.
        g: ``[B, T, H, K]`` log forget gate (already includes A_log/dt_bias).
        beta: ``[B, T, H]`` raw (pre-sigmoid) write strength.
        state: optional initial recurrent state ``[B, H, K, V]``.
        allow_neg_eigval: use ``beta = 2 * sigmoid``.

    Returns:
        o: ``[B, T, H, V]`` retrieved values (pre output-norm).
        final_state: ``[B, H, K, V]`` recurrent state after the sequence.
    """

    batch, seqlen, num_heads, head_k_dim = q.shape
    if k.shape != q.shape:
        raise ValueError(f"k shape {tuple(k.shape)} != q shape {tuple(q.shape)}")
    if v.shape[:3] != (batch, seqlen, num_heads):
        raise ValueError(f"v heads/time mismatch: {tuple(v.shape)}")
    if g.shape != q.shape:
        raise ValueError(f"g shape {tuple(g.shape)} != q shape {tuple(q.shape)}")
    if beta.shape != (batch, seqlen, num_heads):
        raise ValueError(f"beta shape {tuple(beta.shape)} unexpected")

    out_dtype = q.dtype
    compute_dtype = torch.float64 if out_dtype == torch.float64 else torch.float32

    q = F.normalize(q.to(compute_dtype), dim=-1)
    k = F.normalize(k.to(compute_dtype), dim=-1)
    v = v.to(compute_dtype)
    alpha = torch.exp(g.to(compute_dtype))
    b = torch.sigmoid(beta.to(compute_dtype))
    if allow_neg_eigval:
        b = b * 2.0

    if state is None:
        s = q.new_zeros(batch, num_heads, head_k_dim, v.shape[-1], dtype=compute_dtype)
    else:
        s = state.to(compute_dtype)

    outputs = []
    for t in range(seqlen):
        s = s * alpha[:, t].unsqueeze(-1)  # Diag(alpha_t) S
        pred = torch.einsum("bhk,bhkv->bhv", k[:, t], s)  # k^T Diag(alpha) S
        s = s + b[:, t].view(batch, num_heads, 1, 1) * torch.einsum(
            "bhk,bhv->bhkv", k[:, t], v[:, t] - pred
        )
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], s))
    o = torch.stack(outputs, dim=1)
    return o.to(out_dtype), s.to(out_dtype)
