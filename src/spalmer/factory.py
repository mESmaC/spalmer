"""Small model builders used while the complete SPALMER block is assembled."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F

from spalmer.attention import KDAConfig, KDATokenMixer
from spalmer.config import SPALMERConfig
from spalmer.modeling import SPALMERBackbone, SPALMERBlock, SPALMERCausalLM


class DenseSwiGLU(nn.Module):
    """Conventional temporary channel mixer for bootstrap runs."""

    def __init__(self, d_model: int, hidden_size: int) -> None:
        super().__init__()
        self.gate_projection = nn.Linear(d_model, hidden_size, bias=False)
        self.up_projection = nn.Linear(d_model, hidden_size, bias=False)
        self.down_projection = nn.Linear(hidden_size, d_model, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_projection(
            F.silu(self.gate_projection(hidden_states)) * self.up_projection(hidden_states)
        )


def build_kda_bootstrap_model(
    config: SPALMERConfig,
    kda_config: KDAConfig,
    *,
    ffn_hidden_size: int | None = None,
) -> SPALMERCausalLM:
    """Build a trainable KDA-only model before global MLA/MoE integration.

    This helper is deliberately named ``bootstrap``: it validates the tokenizer,
    PLE, residual shell, KDA state path, vocabulary head, objective, optimizer,
    and generation loop without pretending to be the complete ledger architecture.
    """

    if kda_config.hidden_size != config.d_model:
        raise ValueError(
            f"KDA hidden_size={kda_config.hidden_size} does not match d_model={config.d_model}"
        )
    unsupported = {
        config.token_mixer_for_layer(index)
        for index in range(config.n_layers)
        if config.token_mixer_for_layer(index) != "kda"
    }
    if unsupported:
        raise ValueError(
            "the KDA bootstrap builder only accepts a KDA-only token_mixer_pattern"
        )

    hidden_size = ffn_hidden_size or 4 * config.d_model
    blocks = [
        SPALMERBlock(
            config.d_model,
            KDATokenMixer(kda_config),
            DenseSwiGLU(config.d_model, hidden_size),
            norm_eps=config.norm_eps,
        )
        for _ in range(config.n_layers)
    ]
    return SPALMERCausalLM(config, SPALMERBackbone(config, blocks))


__all__ = ["DenseSwiGLU", "build_kda_bootstrap_model"]

