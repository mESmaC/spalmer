"""KDATokenMixer: the SPALMER C04 Kimi Delta Attention token-mixer slice.

Structure (faithful to Kimi Linear's KDA layer, self-contained in plain
PyTorch; the fla-core kernels are used only through the guarded backend
adapter when available):

    x  -> biasless q/k/v projections
       -> causal depthwise conv width 4 + SiLU           (per q, k, v)
       -> per-head L2 normalization of q and k
       -> KDA recurrence
            alpha = exp(-exp(A_log) * softplus(f_proj(x) + dt_bias))  [per K channel]
            beta  = sigmoid(b_proj(x))                               [per head]
            S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
            o_t = S_t^T q_t
       -> per-head gated RMSNorm: rmsnorm(o) * sigmoid(g_proj(x))
       -> biasless output projection to the residual width

MLA, ATXY, MoE routing, and residual-add topology are intentionally out of
scope for this slice (ledger C04; the caller owns the residual connection).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spalmer.attention.backends import resolve_backend
from spalmer.attention.config import KDAConfig
from spalmer.attention.conv import CausalDepthwiseConvSiLU
from spalmer.attention.recurrence import init_dt_bias
from spalmer.attention.state import KDAState


class KDATokenMixer(nn.Module):
    """Kimi Delta Attention token mixer with explicit chunk and step APIs.

    Args:
        config: :class:`~spalmer.attention.config.KDAConfig`.

    Shape contract (``B`` batch, ``T`` time, ``H`` heads, ``K``/``V`` head
    dims, ``D`` hidden size):

    - :meth:`forward` (chunk/prefill): ``x [B, T, D]``, optional
      ``KDAState`` -> ``(out [B, T, D], KDAState)``.
    - :meth:`step` (recurrent/decode): ``x_t [B, 1, D]``, ``KDAState`` ->
      ``(out_t [B, 1, D], KDAState)``.

    Returned states are detached snapshots; gradients flow only through the
    outputs of the call that produced them.
    """

    def __init__(self, config: KDAConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        self.hidden_size = c.hidden_size
        self.num_heads = c.num_heads
        self.head_k_dim = c.head_k_dim
        self.head_v_dim = c.head_v_dim

        self.q_proj = nn.Linear(c.hidden_size, c.key_dim, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, c.key_dim, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, c.value_dim, bias=False)

        self.q_conv = CausalDepthwiseConvSiLU(c.key_dim, c.conv_width, c.conv_bias)
        self.k_conv = CausalDepthwiseConvSiLU(c.key_dim, c.conv_width, c.conv_bias)
        self.v_conv = CausalDepthwiseConvSiLU(c.value_dim, c.conv_width, c.conv_bias)

        # alpha (forget gate) path: low-rank bottleneck on hidden -> [H, K].
        self.f_proj = nn.Sequential(
            nn.Linear(c.hidden_size, c.head_v_dim, bias=False),
            nn.Linear(c.head_v_dim, c.key_dim, bias=False),
        )
        # beta (write strength) path: scalar per head.
        self.b_proj = nn.Linear(c.hidden_size, c.num_heads, bias=False)

        self.A_log = nn.Parameter(
            torch.log(torch.empty(c.num_heads, dtype=torch.float32).uniform_(1, 16))
        )
        self.dt_bias = nn.Parameter(init_dt_bias(c.head_k_dim, c.num_heads))
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True

        # Input-conditioned output gate (input-conditioned gate in C04 terms).
        self.g_proj = nn.Sequential(
            nn.Linear(c.hidden_size, c.head_v_dim, bias=False),
            nn.Linear(c.head_v_dim, c.value_dim, bias=True),
        )
        self.o_proj = nn.Linear(c.value_dim, c.hidden_size, bias=False)

        self._backend = None  # resolved lazily from the input device

    @property
    def backend_name(self) -> str:
        if self._backend is None:
            return resolve_backend(self.config.backend, torch.device("cpu")).name
        return self._backend.name

    def create_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> KDAState:
        """Fresh zero state (BOS) for a batch."""

        return KDAState.empty(batch_size, self.config, device=device, dtype=dtype)

    def _check_input(self, x: torch.Tensor, expected_t: int | None) -> None:
        if x.dim() != 3:
            raise ValueError(f"expected x [B, T, D], got {tuple(x.shape)}")
        if x.shape[-1] != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}, got {x.shape[-1]}")
        if expected_t is not None and x.shape[1] != expected_t:
            raise ValueError(f"expected T={expected_t}, got T={x.shape[1]}")

    def _core(
        self, x: torch.Tensor, state: KDAState, *, is_step: bool
    ) -> tuple[torch.Tensor, KDAState]:
        c = self.config
        if self._backend is None:
            self._backend = resolve_backend(c.backend, x.device)
        q, q_conv_state = self.q_conv(self.q_proj(x), state.conv_q)
        k, k_conv_state = self.k_conv(self.k_proj(x), state.conv_k)
        v, v_conv_state = self.v_conv(self.v_proj(x), state.conv_v)

        q = q.reshape(*q.shape[:2], self.num_heads, self.head_k_dim)
        k = k.reshape(*k.shape[:2], self.num_heads, self.head_k_dim)
        v = v.reshape(*v.shape[:2], self.num_heads, self.head_v_dim)
        f = self.f_proj(x).reshape(*x.shape[:2], self.num_heads, self.head_k_dim)
        beta_raw = self.b_proj(x)

        backend = self._backend
        call = backend.step if is_step else backend.sequence
        o, recurrent_state = call(
            q=q,
            k=k,
            v=v,
            f=f,
            beta_raw=beta_raw,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            state=state.recurrent_state,
            lower_bound=c.gate_lower_bound,
            allow_neg_eigval=c.allow_neg_eigval,
        )

        gate = self.g_proj(x).reshape(*x.shape[:2], self.num_heads, self.head_v_dim)
        o = self._output_norm(o, gate)
        out = self.o_proj(o.reshape(*x.shape[:2], c.value_dim))

        new_state = KDAState(
            recurrent_state=recurrent_state.detach(),
            conv_q=q_conv_state.detach(),
            conv_k=k_conv_state.detach(),
            conv_v=v_conv_state.detach(),
        )
        return out, new_state

    def _output_norm(self, o: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Per-head RMSNorm over V modulated by sigmoid(input gate)."""

        rms = o * torch.rsqrt(o.pow(2).mean(dim=-1, keepdim=True) + self.config.norm_eps)
        return rms * torch.sigmoid(gate)

    def forward(
        self,
        x: torch.Tensor,
        state: KDAState | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        state_reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KDAState]:
        """Chunk/prefill path: consume ``x [B, T, D]`` in one call."""

        self._check_input(x, expected_t=None)
        self._validate_initial_masking(x, attention_mask, state_reset_mask)
        if state is None:
            state = self.create_state(x.shape[0], device=x.device, dtype=x.dtype)
        return self._core(x, state, is_step=False)

    def step(
        self,
        x_t: torch.Tensor,
        state: KDAState | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        state_reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KDAState]:
        """Recurrent/decode path: consume exactly one token ``[B, 1, D]``."""

        self._check_input(x_t, expected_t=1)
        self._validate_initial_masking(x_t, attention_mask, state_reset_mask)
        if state is None:
            state = self.create_state(x_t.shape[0], device=x_t.device, dtype=x_t.dtype)
        return self._core(x_t, state, is_step=True)

    @staticmethod
    def _validate_initial_masking(
        x: torch.Tensor,
        attention_mask: torch.Tensor | None,
        state_reset_mask: torch.Tensor | None,
    ) -> None:
        """Keep the v0 path correct by accepting only unpadded fresh sequences."""

        expected_shape = x.shape[:2]
        if attention_mask is not None:
            if attention_mask.shape != expected_shape:
                raise ValueError(
                    f"attention_mask must have shape {tuple(expected_shape)}, "
                    f"got {tuple(attention_mask.shape)}"
                )
            if not bool(attention_mask.bool().all().item()):
                raise NotImplementedError(
                    "KDA v0 accepts only unpadded batches; pack fixed-length sequences"
                )
        if state_reset_mask is not None:
            if state_reset_mask.shape != expected_shape:
                raise ValueError(
                    f"state_reset_mask must have shape {tuple(expected_shape)}, "
                    f"got {tuple(state_reset_mask.shape)}"
                )
            if bool(state_reset_mask.bool().any().item()):
                raise NotImplementedError(
                    "in-chunk KDA state resets are not implemented; start a fresh state instead"
                )

    def extra_repr(self) -> str:
        c = self.config
        return (
            f"hidden_size={c.hidden_size}, heads={c.num_heads}, "
            f"K={c.head_k_dim}, V={c.head_v_dim}, conv_width={c.conv_width}, "
            f"backend={self.backend_name}"
        )


__all__ = ["KDATokenMixer", "KDAConfig", "KDAState"]
