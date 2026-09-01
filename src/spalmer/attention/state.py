"""Explicit cache-state contract for the SPALMER KDA token mixer.

The state is the only memory carried between calls:

- ``recurrent_state``: KDA matrix state ``S`` of shape ``[B, H, K, V]``.
  The FLA adapter uses its default K-first state layout to preserve this
  contract across the reference and optimized backends.
- ``conv_q`` / ``conv_k`` / ``conv_v``: causal depthwise-conv caches, each
  ``[B, channels, conv_width - 1]``, holding the previous ``conv_width - 1``
  raw (pre-activation) projected inputs, oldest first. Note this is the
  minimal ``W-1`` form; fla-core's ShortConvolution cache keeps the full
  ``W`` slots.

Tensors are detached snapshots; callers own gradient threading.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from spalmer.attention.config import KDAConfig


@dataclass(frozen=True)
class KDAState:
    """Recurrent + convolution cache of one KDA layer for one batch."""

    recurrent_state: torch.Tensor
    conv_q: torch.Tensor
    conv_k: torch.Tensor
    conv_v: torch.Tensor

    @classmethod
    def empty(
        cls,
        batch_size: int,
        config: KDAConfig,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> KDAState:
        """Zero state for a fresh sequence (BOS)."""

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        opts = {"device": device, "dtype": dtype or torch.float32}
        recurrent_state = torch.zeros(
            batch_size, config.num_heads, config.head_k_dim, config.head_v_dim, **opts
        )
        conv_q = torch.zeros(batch_size, config.key_dim, config.conv_width - 1, **opts)
        conv_k = conv_q.clone()
        conv_v = torch.zeros(batch_size, config.value_dim, config.conv_width - 1, **opts)
        return cls(recurrent_state=recurrent_state, conv_q=conv_q, conv_k=conv_k, conv_v=conv_v)

    def to(
        self, *, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> KDAState:
        """Return a copy of the state moved to ``device``/``dtype``."""

        def mv(t: torch.Tensor) -> torch.Tensor:
            return t.to(device=device, dtype=dtype)

        return KDAState(
            recurrent_state=mv(self.recurrent_state),
            conv_q=mv(self.conv_q),
            conv_k=mv(self.conv_k),
            conv_v=mv(self.conv_v),
        )
