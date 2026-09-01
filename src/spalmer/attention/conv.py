"""Causal depthwise conv + SiLU used on q/k/v (KDA short convolution).

Reference implementation in plain PyTorch. Semantics match fla-core's
``ShortConvolution`` with ``activation='silu'`` and ``bias=False``:

- depthwise (per-channel) FIR over time, causal left padding;
- weight layout ``[channels, width]`` with tap ``w`` multiplying the input
  from ``t - (width - 1) + w`` (tap ``width-1`` is the current token);
- SiLU applied after the convolution, never to the cached inputs.

Cache contract: ``[B, channels, width - 1]`` holding the previous
``width - 1`` raw inputs, oldest first.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalDepthwiseConvSiLU(nn.Module):
    """Causal depthwise conv of given width followed by SiLU."""

    def __init__(self, channels: int, width: int = 4, bias: bool = False) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if width < 2:
            raise ValueError(f"width must be >= 2, got {width}")
        self.channels = channels
        self.width = width
        self.weight = nn.Parameter(torch.empty(channels, width))
        if bias:
            self.bias_p = nn.Parameter(torch.zeros(channels))
        else:
            self.bias_p = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight, -1.0 / self.width, 1.0 / self.width)
        if self.bias_p is not None:
            nn.init.zeros_(self.bias_p)

    @property
    def state_width(self) -> int:
        return self.width - 1

    def forward(
        self, x: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convolve a chunk and return ``(y, new_state)``.

        Args:
            x: ``[B, T, channels]`` projected inputs.
            state: ``[B, channels, width - 1]`` previous raw inputs or None.

        Returns:
            y: ``[B, T, channels]`` SiLU-activated convolution output.
            new_state: ``[B, channels, width - 1]`` cache after consuming x.
        """

        batch, seqlen, channels = x.shape
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {channels}")
        x_t = x.transpose(1, 2)  # [B, C, T]
        if state is None:
            state = x_t.new_zeros(batch, self.channels, self.state_width)
        elif state.shape != (batch, self.channels, self.state_width):
            raise ValueError(
                f"expected state {tuple((batch, self.channels, self.state_width))}, "
                f"got {tuple(state.shape)}"
            )
        padded = torch.cat([state, x_t], dim=-1)  # [B, C, T + W - 1]
        window = padded.unfold(-1, self.width, 1)  # [B, C, T, W]
        y = torch.einsum("bctw,cw->bct", window, self.weight)
        if self.bias_p is not None:
            y = y + self.bias_p.view(1, -1, 1)
        y = F.silu(y)
        new_state = padded[..., -self.state_width :] if self.state_width else padded[..., :0]
        return y.transpose(1, 2), new_state
