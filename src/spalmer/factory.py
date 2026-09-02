"""Model builders assembling the SPALMER block from its research components."""

from __future__ import annotations

from spalmer.attention import KDAConfig, KDATokenMixer, MLAConfig, MLATokenMixer
from spalmer.config import SPALMERConfig
from spalmer.directional import DirectionalConfig, LateralSilencingMixer
from spalmer.experts import (
    ExpertPotentiationController,
    ExpertResidency,
    MicroExpertChannelMixer,
    MicroExpertsConfig,
    SurpriseRouter,
)
from spalmer.memory import ATXYConfig, ATXYInjection
from spalmer.modeling import SPALMERBackbone, SPALMERBlock, SPALMERCausalLM
from spalmer.nn import SwiGLU

# The bootstrap model's dense channel mixer; kept under its historical name.
DenseSwiGLU = SwiGLU


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
        raise ValueError("the KDA bootstrap builder only accepts a KDA-only token_mixer_pattern")

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


def build_spalmer_model(
    config: SPALMERConfig,
    kda_config: KDAConfig,
    mla_config: MLAConfig,
    experts_config: MicroExpertsConfig,
    *,
    directional_config: DirectionalConfig | None = None,
    atxy_config: ATXYConfig | None = None,
) -> SPALMERCausalLM:
    """Assemble the ledger-faithful KDA/MLA, shared-path, and micro-expert model.

    One router, one potentiation controller, and one :class:`ExpertResidency`
    are shared by every layer so that expert identity, promotion, and residency
    are coherent across depth.

    Feature gates:

    - ``directional_config`` (ledger C16) adds the lateral/silencing mixer as a
      third pre-norm residual branch in every block when ``enabled``; ``None``
      or a disabled config leaves the block untouched.
    - ``atxy_config`` (ledger C03/C04) attaches the ATXY address embedding and
      exact-value injection; ``None`` keeps the ATXY-free model. Even when
      attached, ATXY only acts on forward calls that pass an ``ATXYRequest``.
    """

    component_widths = {
        "KDA": kda_config.hidden_size,
        "MLA": mla_config.hidden_size,
        "micro-experts": experts_config.d_model,
    }
    use_directional = directional_config is not None and directional_config.enabled
    if use_directional:
        component_widths["directional"] = directional_config.d_model
    if atxy_config is not None:
        component_widths["ATXY"] = atxy_config.d_model
    mismatched = {
        name: width for name, width in component_widths.items() if width != config.d_model
    }
    if mismatched:
        detail = ", ".join(f"{name}={width}" for name, width in mismatched.items())
        raise ValueError(f"component widths must equal d_model={config.d_model}; got {detail}")

    shared_router = SurpriseRouter(experts_config)
    potentiation_controller = ExpertPotentiationController(experts_config)
    residency = ExpertResidency(experts_config)
    blocks: list[SPALMERBlock] = []
    for layer_index in range(config.n_layers):
        mixer_name = config.token_mixer_for_layer(layer_index)
        if mixer_name == "kda":
            token_mixer = KDATokenMixer(kda_config)
        elif mixer_name == "mla":
            token_mixer = MLATokenMixer(mla_config)
        else:  # SPALMERConfig validates this, but keep the assembly boundary closed.
            raise ValueError(f"unsupported token mixer: {mixer_name}")
        channel_mixer = MicroExpertChannelMixer(
            experts_config,
            router=shared_router,
            potentiation_controller=potentiation_controller,
            residency=residency,
        )
        directional_mixer = LateralSilencingMixer(directional_config) if use_directional else None
        blocks.append(
            SPALMERBlock(
                config.d_model,
                token_mixer,
                channel_mixer,
                directional_mixer=directional_mixer,
                norm_eps=config.norm_eps,
            )
        )
    atxy = None if atxy_config is None else ATXYInjection(atxy_config)
    return SPALMERCausalLM(
        config,
        SPALMERBackbone(config, blocks, atxy=atxy),
        potentiation_controller=potentiation_controller,
        residency=residency,
    )


__all__ = ["DenseSwiGLU", "build_kda_bootstrap_model", "build_spalmer_model"]
