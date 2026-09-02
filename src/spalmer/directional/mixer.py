"""Feed Laterally NN with Lateral Active Silencing (SPALMER ledger C16).

Ported from ``agent/directional-v0`` (09a04ec) and reconciled with the
ledger's first compositional candidate::

    u_lateral = LateralNN(h, peer_summary)
    silence   = sigmoid(SilencingNN(h, peer_summary))
    h_next    = h + g_lateral * ((1 - silence) * u_lateral)

Scope of this slice, recorded explicitly:

- **Feed Forward** remains the existing shared/routed channel path and is not
  touched here.
- **Feed Backward** (the delayed block-group refinement pass) is deferred and
  intentionally not implemented.

The ``LateralSilencingMixer`` treats each token's features as
``num_feature_groups`` same-depth peers. A low-rank groups-by-groups mixing
matrix ``A @ B`` exchanges information across groups with its diagonal
zeroed, so a group can only receive from *other* groups — the lateral path
never becomes another ordinary per-group FFN. The silencing network sees the
group's own features **and** a summary of its peers (the mean of the other
groups), so inhibition can depend on what the peers are proposing, as the
ledger's ``SilencingNN(h, peer_summary)`` requires. One
``silence = sigmoid(logit)`` value is produced per token and group; only the
proposed lateral update is multiplied by ``(1 - silence)`` and by a learned
``residual_gate`` initialized to zero by default:

    directional_update = residual_gate * lateral_update * (1 - silence)

Every operation is per-token: the mixer never mixes across sequence
positions and is causal by construction. Only the proposed update is
silenced — the identity residual stream is never multiplied or erased.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from spalmer.directional.config import DirectionalConfig


class LateralSilencingMixer(nn.Module):
    """Low-rank cross-group lateral update with peer-aware active silencing.

    Args:
        config: Directional configuration. Must have ``enabled=True``;
            the feature gate is enforced by the factory and block wiring.
    """

    def __init__(self, config: DirectionalConfig) -> None:
        super().__init__()
        if not config.enabled:
            raise ValueError("LateralSilencingMixer requires an enabled DirectionalConfig")
        self.config = config
        std = config.initializer_range
        groups, rank, width = config.num_feature_groups, config.lateral_rank, config.group_width
        self.lateral_a = nn.Parameter(torch.randn(groups, rank) * std)
        self.lateral_b = nn.Parameter(torch.randn(rank, groups) * std)
        # SilencingNN(h, peer_summary): the group's own features concatenated
        # with the mean of its peers.
        self.silence_proj = nn.Linear(2 * width, 1)
        nn.init.zeros_(self.silence_proj.weight)
        nn.init.zeros_(self.silence_proj.bias)
        self.residual_gate = nn.Parameter(torch.tensor(float(config.residual_gate_init)))
        self.last_metrics: dict[str, Tensor] = {}

    @property
    def group_mixing_matrix(self) -> Tensor:
        """The diagonal-removed low-rank groups-by-groups mixing matrix."""

        mixing = self.lateral_a @ self.lateral_b
        return mixing - torch.diag(torch.diagonal(mixing))

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Map ``[batch, seq, d_model]`` to a same-shaped directional update."""

        if hidden_states.ndim != 3:
            raise ValueError(
                f"hidden_states must have shape [batch, seq, d_model]; "
                f"got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[-1] != self.config.d_model:
            raise ValueError(
                f"expected trailing dimension {self.config.d_model}; got {hidden_states.shape[-1]}"
            )
        batch, seq_len, d_model = hidden_states.shape
        groups = self.config.num_feature_groups
        width = self.config.group_width
        peer_view = hidden_states.reshape(batch, seq_len, groups, width)
        mixing = self.group_mixing_matrix
        lateral_update = torch.einsum("gp,bspw->bsgw", mixing, peer_view)
        # peer_summary: the mean of the *other* groups at the same token.
        total = peer_view.sum(dim=2, keepdim=True)
        peer_summary = (total - peer_view) / (groups - 1)
        silence = torch.sigmoid(
            self.silence_proj(torch.cat((peer_view, peer_summary), dim=-1))
        ).squeeze(-1)
        update = self.residual_gate * lateral_update * (1.0 - silence).unsqueeze(-1)
        self.last_metrics = {"mean_silence": silence.detach().float().mean()}
        return update.reshape(batch, seq_len, d_model)

    def extra_repr(self) -> str:
        config = self.config
        return (
            f"groups={config.num_feature_groups}, rank={config.lateral_rank}, "
            f"residual_gate_init={config.residual_gate_init}"
        )


__all__ = ["LateralSilencingMixer"]
