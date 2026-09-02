"""MLATokenMixer: the SPALMER C04 global NoPE MLA token-mixer slice.

The periodic global MLA layer of the ledger's 3 KDA : 1 MLA pattern, written
with ordinary PyTorch attention primitives in the DeepSeek-V2 multi-head
latent-attention style used by Kimi Linear's global layers:

    x -> q path:   q_a_proj -> RMSNorm -> q_b_proj -> per-head queries
    x -> kv path:  kv_a_proj -> RMSNorm -> compressed latent (cached)
                                         -> kv_b_proj -> per-head keys/values
    -> causal scaled dot-product attention over cached + chunk keys/values
    -> o_proj back to the residual width

NoPE: no positional encoding of any kind is applied in this layer (ledger
C03/C04 baseline). The cache state stores the normalized compressed latent
``[B, L, kv_latent_dim]`` rather than per-head K/V, so the cache footprint is
independent of head count and head dims; ``kv_b_proj`` is re-applied to the
full cached latent on every call.

ATXY side paths, RoPE/YaRN experiments, quantized KV, and training
infrastructure are intentionally out of scope for this slice (ledger C04; the
caller owns the residual connection).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from spalmer.nn import RMSNorm


@dataclass
class MLAConfig:
    """Configuration of one global MLA token-mixer layer.

    Shapes per head: key ``K = head_k_dim``, value ``V = head_v_dim``.

    Args:
        hidden_size: Residual-stream width ``d_model``.
        num_heads: Number of attention heads ``H``.
        head_k_dim: Per-head key/query dimension ``K``.
        head_v_dim: Per-head value dimension ``V``. ``None`` means ``K``.
        q_latent_dim: Low-rank query bottleneck width.
        kv_latent_dim: Low-rank compressed latent KV width (the cached
            representation).
        norm_eps: Epsilon for the q and kv latent RMSNorms.
    """

    hidden_size: int = 512
    num_heads: int = 8
    head_k_dim: int = 64
    head_v_dim: int | None = None
    q_latent_dim: int = 256
    kv_latent_dim: int = 64
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.head_k_dim <= 0:
            raise ValueError(f"head_k_dim must be positive, got {self.head_k_dim}")
        if self.head_v_dim is None:
            self.head_v_dim = self.head_k_dim
        if self.head_v_dim <= 0:
            raise ValueError(f"head_v_dim must be positive, got {self.head_v_dim}")
        if self.q_latent_dim <= 0:
            raise ValueError(f"q_latent_dim must be positive, got {self.q_latent_dim}")
        if self.kv_latent_dim <= 0:
            raise ValueError(f"kv_latent_dim must be positive, got {self.kv_latent_dim}")
        if self.norm_eps <= 0:
            raise ValueError(f"norm_eps must be positive, got {self.norm_eps}")

    @property
    def key_dim(self) -> int:
        return self.num_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_heads * self.head_v_dim


@dataclass(frozen=True)
class MLAState:
    """Compressed latent KV cache of one global MLA layer for one batch.

    ``latent_kv`` is ``[B, L, kv_latent_dim]``: the RMS-normalized low-rank
    latent for the ``L`` tokens seen so far, oldest first. Tensors are
    detached snapshots; callers own gradient threading.
    """

    latent_kv: torch.Tensor

    @classmethod
    def empty(
        cls,
        batch_size: int,
        config: MLAConfig,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> MLAState:
        """Empty cache for a fresh sequence (BOS)."""

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        latent_kv = torch.zeros(
            batch_size,
            0,
            config.kv_latent_dim,
            device=device,
            dtype=dtype or torch.float32,
        )
        return cls(latent_kv=latent_kv)

    def to(
        self, *, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> MLAState:
        """Return a copy of the state moved to ``device``/``dtype``."""

        return MLAState(latent_kv=self.latent_kv.to(device=device, dtype=dtype))


class MLATokenMixer(nn.Module):
    """Global NoPE multi-head latent attention mixer with prefill and step APIs.

    Args:
        config: :class:`MLAConfig`.

    Shape contract (``B`` batch, ``T`` time, ``H`` heads, ``K``/``V`` head
    dims, ``D`` hidden size, ``C`` latent KV width):

    - :meth:`forward` (chunk/prefill): ``x [B, T, D]``, optional
      ``MLAState`` -> ``(out [B, T, D], MLAState)``.
    - :meth:`step` (recurrent/decode): ``x_t [B, 1, D]``, ``MLAState`` ->
      ``(out_t [B, 1, D], MLAState)``.

    Returned states are detached snapshots; gradients flow only through the
    outputs of the call that produced them.
    """

    def __init__(self, config: MLAConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        self.hidden_size = c.hidden_size
        self.num_heads = c.num_heads
        self.head_k_dim = c.head_k_dim
        self.head_v_dim = c.head_v_dim

        self.q_a_proj = nn.Linear(c.hidden_size, c.q_latent_dim, bias=False)
        self.q_norm = RMSNorm(c.q_latent_dim, eps=c.norm_eps)
        self.q_b_proj = nn.Linear(c.q_latent_dim, c.key_dim, bias=False)

        self.kv_a_proj = nn.Linear(c.hidden_size, c.kv_latent_dim, bias=False)
        self.kv_norm = RMSNorm(c.kv_latent_dim, eps=c.norm_eps)
        self.kv_b_proj = nn.Linear(
            c.kv_latent_dim, c.num_heads * (c.head_k_dim + c.head_v_dim), bias=False
        )

        self.o_proj = nn.Linear(c.value_dim, c.hidden_size, bias=False)

    def create_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> MLAState:
        """Empty cache for a fresh sequence (BOS) for a batch."""

        return MLAState.empty(batch_size, self.config, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        state: MLAState | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        state_reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MLAState]:
        """Chunk/prefill path: consume ``x [B, T, D]`` in one call."""

        self._check_input(x, expected_t=None)
        self._validate_initial_masking(x, attention_mask, state_reset_mask)
        if state is None:
            state = self.create_state(x.shape[0], device=x.device, dtype=x.dtype)
        return self._core(x, state)

    def step(
        self,
        x_t: torch.Tensor,
        state: MLAState | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        state_reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MLAState]:
        """Recurrent/decode path: consume exactly one token ``[B, 1, D]``."""

        self._check_input(x_t, expected_t=1)
        self._validate_initial_masking(x_t, attention_mask, state_reset_mask)
        if state is None:
            state = self.create_state(x_t.shape[0], device=x_t.device, dtype=x_t.dtype)
        return self._core(x_t, state)

    def _core(self, x: torch.Tensor, state: MLAState) -> tuple[torch.Tensor, MLAState]:
        c = self.config
        batch, new_len, _ = x.shape
        self._check_state(state, batch)
        cache_len = state.latent_kv.shape[1]

        latent = self.kv_norm(self.kv_a_proj(x))
        q = self.q_b_proj(self.q_norm(self.q_a_proj(x)))

        full_latent = torch.cat([state.latent_kv, latent], dim=1)
        kv = self.kv_b_proj(full_latent).view(
            batch, cache_len + new_len, self.num_heads, self.head_k_dim + self.head_v_dim
        )
        k, v = kv.split([self.head_k_dim, self.head_v_dim], dim=-1)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = q.view(batch, new_len, self.num_heads, self.head_k_dim).transpose(1, 2)

        o = self._attend(q, k, v, cache_len=cache_len)
        o = o.transpose(1, 2).reshape(batch, new_len, c.value_dim)
        out = self.o_proj(o)

        return out, MLAState(latent_kv=full_latent.detach())

    def _attend(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, cache_len: int
    ) -> torch.Tensor:
        """Causal NoPE attention; the new token(s) see the whole cache."""

        new_len = q.shape[-2]
        if new_len == 1:
            return F.scaled_dot_product_attention(q, k, v)
        if cache_len == 0:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        mask = torch.ones(new_len, cache_len + new_len, dtype=torch.bool, device=q.device)
        mask[:, cache_len:] = torch.tril(
            torch.ones(new_len, new_len, dtype=torch.bool, device=q.device)
        )
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    def _check_input(self, x: torch.Tensor, expected_t: int | None) -> None:
        if x.dim() != 3:
            raise ValueError(f"expected x [B, T, D], got {tuple(x.shape)}")
        if x.shape[-1] != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}, got {x.shape[-1]}")
        if expected_t is not None and x.shape[1] != expected_t:
            raise ValueError(f"expected T={expected_t}, got T={x.shape[1]}")

    def _check_state(self, state: MLAState, batch_size: int) -> None:
        latent = state.latent_kv
        if latent.dim() != 3:
            raise ValueError(
                f"expected latent_kv [B, L, kv_latent_dim], got {tuple(latent.shape)}"
            )
        if latent.shape[0] != batch_size:
            raise ValueError(
                f"state batch size {latent.shape[0]} does not match input batch {batch_size}"
            )
        if latent.shape[-1] != self.config.kv_latent_dim:
            raise ValueError(
                f"expected kv_latent_dim {self.config.kv_latent_dim}, got {latent.shape[-1]}"
            )

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
                    "MLA v0 accepts only unpadded batches; pack fixed-length sequences"
                )
        if state_reset_mask is not None:
            if state_reset_mask.shape != expected_shape:
                raise ValueError(
                    f"state_reset_mask must have shape {tuple(expected_shape)}, "
                    f"got {tuple(state_reset_mask.shape)}"
                )
            if bool(state_reset_mask.bool().any().item()):
                raise NotImplementedError(
                    "in-chunk MLA cache resets are not implemented; start a fresh state instead"
                )

    def extra_repr(self) -> str:
        c = self.config
        return (
            f"hidden_size={c.hidden_size}, heads={c.num_heads}, "
            f"K={c.head_k_dim}, V={c.head_v_dim}, q_latent={c.q_latent_dim}, "
            f"kv_latent={c.kv_latent_dim}"
        )


__all__ = ["MLAConfig", "MLAState", "MLATokenMixer"]
