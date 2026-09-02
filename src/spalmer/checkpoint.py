"""Save and reload the first executable SPALMER model bundle."""

from __future__ import annotations

from dataclasses import asdict
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
FORMAT_VERSION = 1


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
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint version: {payload.get('format_version')}")

    config = SPALMERConfig(**payload["model_config"])
    kda_config = KDAConfig(**payload["kda_config"])
    mla_config = MLAConfig(**payload["mla_config"])
    experts_config = MicroExpertsConfig(**payload["experts_config"])
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
    model.load_state_dict(payload["model_state"])
    return model, vocab, dict(payload.get("metadata") or {})


__all__ = ["FORMAT_NAME", "FORMAT_VERSION", "load_checkpoint", "save_checkpoint"]

