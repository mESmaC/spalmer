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
FORMAT_VERSION = 2
_SUPPORTED_FORMAT_VERSIONS = {1, FORMAT_VERSION}


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

    config = SPALMERConfig(**payload["model_config"])
    kda_config = KDAConfig(**payload["kda_config"])
    mla_config = MLAConfig(**payload["mla_config"])
    raw_experts_config = dict(payload["experts_config"])
    legacy_experts = checkpoint_version == 1
    if legacy_experts:
        # Version-1 prototypes predate the low-bit expert substrate. Preserve
        # their dense execution semantics instead of silently quantizing them.
        raw_experts_config["expert_fake_quantization"] = False
        raw_experts_config["potentiation_budget"] = 0
        raw_experts_config["router_score_transform"] = "identity"
    else:
        required = {field.name for field in fields(MicroExpertsConfig)}
        missing_config = required - raw_experts_config.keys()
        if missing_config:
            raise ValueError(
                "current checkpoint is missing expert configuration fields: "
                f"{sorted(missing_config)}"
            )
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
    if legacy_experts:
        incompatible = model.load_state_dict(payload["model_state"], strict=False)
        expected_missing = {
            f"potentiation_controller.{name}"
            for name in model.potentiation_controller.state_dict()
        }
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "legacy checkpoint state does not match its construction config; "
                f"missing={sorted(incompatible.missing_keys)}, "
                f"unexpected={sorted(incompatible.unexpected_keys)}"
            )
    else:
        model.load_state_dict(payload["model_state"], strict=True)
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
            f"promoted expert count {promoted} exceeds budget "
            f"{experts_config.potentiation_budget}"
        )
    if not experts_config.expert_fake_quantization and promoted:
        raise ValueError("dense legacy expert execution cannot carry promoted experts")


__all__ = ["FORMAT_NAME", "FORMAT_VERSION", "load_checkpoint", "save_checkpoint"]
