"""Dense gated feed-forward primitive."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    """Conventional gated MLP: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, d_model: int, hidden_size: int, *, initializer_range: float | None = None):
        super().__init__()
        if d_model <= 0 or hidden_size <= 0:
            raise ValueError("d_model and hidden_size must be positive")
        self.gate_projection = nn.Linear(d_model, hidden_size, bias=False)
        self.up_projection = nn.Linear(d_model, hidden_size, bias=False)
        self.down_projection = nn.Linear(hidden_size, d_model, bias=False)
        if initializer_range is not None:
            for linear in (self.gate_projection, self.up_projection, self.down_projection):
                nn.init.normal_(linear.weight, mean=0.0, std=initializer_range)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_projection(
            F.silu(self.gate_projection(hidden_states)) * self.up_projection(hidden_states)
        )


__all__ = ["SwiGLU"]
