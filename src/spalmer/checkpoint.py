"""Save and reload the first executable SPALMER model bundle."""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch

from spalmer.attention import KDAConfig, MLAConfig
from spalmer.config import SPALMERConfig
from spalmer.experts import MicroExpertsConfig
from spalmer.factory import build_spalmer_model
from spalmer.modeling import SPALMERCausalLM
from spalmer.tokenizer import Vocab

FORMAT_NAME = "spalmer.prototype.checkpoint"
FORMAT_VERSION = 3
_SUPPORTED_FORMAT_VERSIONS = {1, 2, FORMAT_VERSION}
# Version 3 added the shared average-surprise buffers on the language model and
# the C13 residency fields of the expert configuration.
_VERSION_3_EXPERT_FIELDS = {"residency_increment", "residency_min_gain"}
_VERSION_3_MODEL_BUFFERS = {"surprise_ema", "surprise_observations"}
_VERSION_3_MODEL_CONFIG_FIELDS = {"surprise_ema_decay"}
_VERSION_1_EXPERT_FIELDS = {
    "d_model",
    "num_experts",
    "expert_inter_dim",
    "active_experts",
    "min_active_experts",
    "max_active_experts",
    "router_bias",
    "initializer_range",
}


def save_checkpoint(
    path: str | Path,
    model: SPALMERCausalLM,
    vocab: Vocab,
    *,
    kda_config: KDAConfig,
    mla_config: MLAConfig,
    experts_config: MicroExpertsConfig,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist model weights, all construction configs, and exact vocabulary."""

    model.config.assert_tokenizer_compatible(
        version=vocab.version,
        fingerprint=vocab.fingerprint,
    )
    _validate_model_attention_configs(model, kda_config, mla_config)
    _validate_model_experts_config(model, experts_config)
    _validate_potentiation_state(model, experts_config)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "model_config": model.config.to_dict(),
            "kda_config": asdict(kda_config),
            "mla_config": asdict(mla_config),
            "experts_config": asdict(experts_config),
            "vocab": vocab.to_dict(),
            "model_state": model.state_dict(),
            "metadata": dict(metadata or {}),
        },
        destination,
    )
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[SPALMERCausalLM, Vocab, dict[str, Any]]:
    """Reconstruct a model and its tokenizer from a trusted local bundle."""

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != FORMAT_NAME:
        raise ValueError("not a SPALMER prototype checkpoint")
    checkpoint_version = payload.get("format_version")
    if checkpoint_version not in _SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(f"unsupported checkpoint version: {payload.get('format_version')}")

    raw_model_config = dict(payload["model_config"])
    required_model = {field.name for field in fields(SPALMERConfig)}
    if checkpoint_version < 3:
        required_model -= _VERSION_3_MODEL_CONFIG_FIELDS
    _require_config_schema("model", raw_model_config, required_model)
    config = SPALMERConfig(**raw_model_config)
    raw_kda_config = dict(payload["kda_config"])
    raw_mla_config = dict(payload["mla_config"])
    _require_config_schema("KDA", raw_kda_config, {field.name for field in fields(KDAConfig)})
    _require_config_schema("MLA", raw_mla_config, {field.name for field in fields(MLAConfig)})
    kda_config = KDAConfig(**raw_kda_config)
    mla_config = MLAConfig(**raw_mla_config)
    raw_experts_config = dict(payload["experts_config"])
    legacy_experts = checkpoint_version == 1
    if legacy_experts:
        _require_config_schema("expert", raw_experts_config, _VERSION_1_EXPERT_FIELDS)
        # Version-1 prototypes predate the low-bit expert substrate. Preserve
        # their dense execution semantics instead of silently quantizing them.
        raw_experts_config["expert_fake_quantization"] = False
        raw_experts_config["potentiation_budget"] = 0
        raw_experts_config["router_score_transform"] = "identity"
    else:
        required = {field.name for field in fields(MicroExpertsConfig)}
        if checkpoint_version == 2:
            required -= _VERSION_3_EXPERT_FIELDS
        _require_config_schema("expert", raw_experts_config, required)
    experts_config = MicroExpertsConfig(**raw_experts_config)
    vocab = Vocab.from_dict(payload["vocab"])
    config.assert_tokenizer_compatible(
        version=vocab.version,
        fingerprint=vocab.fingerprint,
    )
    if len(vocab) != config.vocab_size:
        raise ValueError(
            f"checkpoint vocabulary has {len(vocab)} entries, expected {config.vocab_size}"
        )

    model = build_spalmer_model(config, kda_config, mla_config, experts_config)
    if checkpoint_version == FORMAT_VERSION:
        model.load_state_dict(payload["model_state"], strict=True)
    else:
        # Older bundles predate some buffers; everything else must still match.
        allowed_missing = set(_VERSION_3_MODEL_BUFFERS)
        if legacy_experts:
            allowed_missing |= {
                f"potentiation_controller.{name}"
                for name in model.potentiation_controller.state_dict()
            }
        incompatible = model.load_state_dict(payload["model_state"], strict=False)
        missing = set(incompatible.missing_keys)
        invalid_missing = not missing <= allowed_missing
        # A real version-2 checkpoint predates both average-surprise buffers.
        # Requiring the pair prevents a damaged hybrid bundle from silently
        # initializing half of the signal from defaults.
        if checkpoint_version == 2:
            invalid_missing = missing != _VERSION_3_MODEL_BUFFERS
        elif checkpoint_version == 1:
            invalid_missing = missing != allowed_missing
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "legacy checkpoint state does not match its construction config; "
                f"missing={sorted(incompatible.missing_keys)}, "
                f"unexpected={sorted(incompatible.unexpected_keys)}"
            )
        metadata = payload.get("metadata") or {}
        if "average_surprise" in metadata:
            model.surprise_ema.fill_(float(metadata["average_surprise"]))
            model.surprise_observations.fill_(1)
    _validate_potentiation_state(model, experts_config)
    return model, vocab, dict(payload.get("metadata") or {})


def _validate_model_experts_config(
    model: SPALMERCausalLM,
    experts_config: MicroExpertsConfig,
) -> None:
    controller = model.potentiation_controller
    if controller is None or getattr(controller, "config", None) != experts_config:
        raise ValueError("experts_config does not match the model potentiation controller")
    for layer_index, block in enumerate(model.backbone.blocks):
        bank = getattr(block.channel_mixer, "experts", None)
        router = getattr(block.channel_mixer, "router", None)
        if getattr(bank, "config", None) != experts_config:
            raise ValueError(f"experts_config does not match expert bank at layer {layer_index}")
        if getattr(router, "config", None) != experts_config:
            raise ValueError(f"experts_config does not match router at layer {layer_index}")


def _validate_model_attention_configs(
    model: SPALMERCausalLM,
    kda_config: KDAConfig,
    mla_config: MLAConfig,
) -> None:
    for layer_index, block in enumerate(model.backbone.blocks):
        mixer_name = model.config.token_mixer_for_layer(layer_index)
        expected = kda_config if mixer_name == "kda" else mla_config
        if getattr(block.token_mixer, "config", None) != expected:
            raise ValueError(
                f"{mixer_name.upper()} config does not match token mixer at layer {layer_index}"
            )


def _require_config_schema(name: str, raw_config: dict[str, Any], required: set[str]) -> None:
    missing = required - raw_config.keys()
    unexpected = raw_config.keys() - required
    if missing or unexpected:
        raise ValueError(
            f"checkpoint has invalid {name} configuration fields: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def _validate_potentiation_state(
    model: SPALMERCausalLM,
    experts_config: MicroExpertsConfig,
) -> None:
    controller = model.potentiation_controller
    mask = getattr(controller, "promoted_mask", None)
    if not isinstance(mask, torch.Tensor) or mask.shape != (experts_config.num_experts,):
        raise ValueError("model has an invalid promoted expert mask")
    promoted = int(mask.count_nonzero())
    if promoted > experts_config.potentiation_budget:
        raise ValueError(
            f"promoted expert count {promoted} exceeds budget {experts_config.potentiation_budget}"
        )
    if not experts_config.expert_fake_quantization and promoted:
        raise ValueError("dense legacy expert execution cannot carry promoted experts")


__all__ = ["FORMAT_NAME", "FORMAT_VERSION", "load_checkpoint", "save_checkpoint"]
