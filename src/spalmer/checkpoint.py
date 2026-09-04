"""Save and reload the first executable SPALMER model bundle."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch

from spalmer.attention import KDAConfig, MLAConfig
from spalmer.config import RecurrenceConfig, SPALMERConfig
from spalmer.directional import DirectionalConfig
from spalmer.experiment.state import (
    CheckpointBinding,
    RunStateEnvelope,
    tensor_state_sha256,
)
from spalmer.experts import MicroExpertsConfig
from spalmer.factory import build_spalmer_model
from spalmer.memory import ATXYConfig
from spalmer.modeling import SPALMERCausalLM
from spalmer.tokenizer import (
    HFTokenizerAdapter,
    RPDTokenizerAdapter,
    SpecialTokenIds,
    TokenizerBackend,
    TokenizerIdentity,
    Vocab,
)

FORMAT_NAME = "spalmer.prototype.checkpoint"
FORMAT_VERSION = 6
_SUPPORTED_FORMAT_VERSIONS = {1, 2, 3, 4, 5, FORMAT_VERSION}
CHECKPOINT_BINDING_METADATA_KEY = "_spalmer_checkpoint_binding"
_CHECKPOINT_BINDING_ATTRIBUTE = "_spalmer_checkpoint_binding"
_MODEL_PARAMETER_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
# Version 3 added the shared average-surprise buffers on the language model and
# the C13 residency fields of the expert configuration.
_VERSION_3_EXPERT_FIELDS = {"residency_increment", "residency_min_gain"}
_VERSION_3_MODEL_BUFFERS = {"surprise_ema", "surprise_observations"}
_VERSION_3_MODEL_CONFIG_FIELDS = {"surprise_ema_decay"}
# Version 6 added the optional depth-recurrent core to the model config.
_VERSION_6_MODEL_CONFIG_FIELDS = {"recurrence"}
_VERSION_5_EXPERT_FIELDS = {
    "expert_weight_format",
    "expert_activation_format",
    "expert_master_dtype",
    "expert_qat_backend",
    "expert_promotion_format",
}
_POST_V3_EXPERT_FIELDS = {
    "shared_inter_dim",
    "min_resident_experts",
    "max_resident_experts",
    "expert_execution",
}
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
    vocab: Vocab | TokenizerBackend,
    *,
    kda_config: KDAConfig,
    mla_config: MLAConfig,
    experts_config: MicroExpertsConfig,
    directional_config: DirectionalConfig | None = None,
    atxy_config: ATXYConfig | None = None,
    metadata: dict[str, Any] | None = None,
    run_state: RunStateEnvelope | None = None,
    checkpoint_binding: CheckpointBinding | None = None,
) -> Path:
    """Persist model weights, construction inputs, and an optional run-state binding."""

    _validate_model_tokenizer(model.config, vocab)
    _validate_model_attention_configs(model, kda_config, mla_config)
    _validate_model_experts_config(model, experts_config)
    _validate_optional_component_configs(model, directional_config, atxy_config)
    _validate_potentiation_state(model, experts_config)
    _validate_model_recurrence(model)
    model_parameter_dtype = _model_parameter_dtype(model.named_parameters())
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    caller_metadata = dict(metadata or {})
    if CHECKPOINT_BINDING_METADATA_KEY in caller_metadata:
        raise ValueError(f"checkpoint metadata key {CHECKPOINT_BINDING_METADATA_KEY!r} is reserved")
    model_state = model.state_dict()
    binding = _resolve_checkpoint_binding(
        model_state,
        run_state=run_state,
        checkpoint_binding=checkpoint_binding,
    )
    payload = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "model_parameter_dtype": model_parameter_dtype,
        "model_config": model.config.to_dict(),
        "kda_config": asdict(kda_config),
        "mla_config": asdict(mla_config),
        "experts_config": asdict(experts_config),
        "directional_config": (
            asdict(directional_config)
            if directional_config is not None and directional_config.enabled
            else None
        ),
        "atxy_config": None if atxy_config is None else asdict(atxy_config),
        "vocab": _embedded_vocab(vocab),
        "tokenizer_binding": _serialize_tokenizer_binding(vocab),
        "model_state": model_state,
        "checkpoint_binding": None if binding is None else binding.to_dict(),
        "metadata": caller_metadata,
    }
    _atomic_torch_save(payload, destination)
    object.__setattr__(model, _CHECKPOINT_BINDING_ATTRIBUTE, binding)
    return destination


def _model_parameter_dtype(
    named_parameters: Iterable[tuple[str, torch.Tensor]],
) -> str:
    """Return one supported dtype after checking every trainable parameter."""

    observed: dict[torch.dtype, list[str]] = {}
    non_floating: list[str] = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if not parameter.is_floating_point():
            non_floating.append(name)
            continue
        observed.setdefault(parameter.dtype, []).append(name)
    if non_floating:
        raise ValueError(
            "checkpoint trainable parameters must be floating point; "
            f"non-floating={sorted(non_floating)}"
        )
    if not observed:
        raise ValueError("checkpoint model has no floating trainable parameters")
    if len(observed) != 1:
        detail = ", ".join(
            f"{dtype}: {sorted(names)}" for dtype, names in sorted(observed.items(), key=str)
        )
        raise ValueError(f"checkpoint model has mixed trainable parameter dtypes: {detail}")
    dtype = next(iter(observed))
    for name, expected in _MODEL_PARAMETER_DTYPES.items():
        if dtype == expected:
            return name
    raise ValueError(
        f"checkpoint model trainable dtype {dtype} is unsupported; "
        f"expected one of {sorted(_MODEL_PARAMETER_DTYPES)}"
    )


def _load_model_parameter_dtype(
    payload: Mapping[str, Any],
    checkpoint_version: int,
    experts_config: MicroExpertsConfig,
) -> str:
    if checkpoint_version < 5:
        # Historical checkpoints predate the model-wide field. Their migrated
        # expert recipe preserves the parameter dtype used by those bundles.
        return experts_config.expert_master_dtype
    raw_dtype = payload.get("model_parameter_dtype")
    if not isinstance(raw_dtype, str) or raw_dtype not in _MODEL_PARAMETER_DTYPES:
        raise ValueError(
            "version-5 checkpoint has invalid model_parameter_dtype; "
            f"expected one of {sorted(_MODEL_PARAMETER_DTYPES)}"
        )
    if raw_dtype != experts_config.expert_master_dtype:
        raise ValueError(
            "checkpoint model_parameter_dtype disagrees with expert_master_dtype: "
            f"{raw_dtype!r} != {experts_config.expert_master_dtype!r}"
        )
    return raw_dtype


def _validate_checkpoint_parameter_state_dtype(
    named_parameters: Iterable[tuple[str, torch.Tensor]],
    model_state: Mapping[str, Any],
    expected_dtype: str,
) -> None:
    expected = _MODEL_PARAMETER_DTYPES[expected_dtype]
    mismatched: list[str] = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad or name not in model_state:
            continue
        saved = model_state[name]
        if (
            not isinstance(saved, torch.Tensor)
            or not saved.is_floating_point()
            or saved.dtype != expected
        ):
            observed = saved.dtype if isinstance(saved, torch.Tensor) else type(saved).__name__
            mismatched.append(f"{name}={observed}")
    if mismatched:
        raise ValueError(
            "checkpoint trainable tensor payload disagrees with model_parameter_dtype "
            f"{expected_dtype!r}: {sorted(mismatched)}"
        )


def _embedded_vocab(tokenizer: Vocab | TokenizerBackend) -> dict[str, Any] | None:
    if isinstance(tokenizer, Vocab):
        return tokenizer.to_dict()
    if isinstance(tokenizer, RPDTokenizerAdapter):
        return tokenizer.vocab.to_dict()
    return None


def _serialize_tokenizer_binding(tokenizer: Vocab | TokenizerBackend) -> dict[str, Any]:
    """Persist either a self-contained RPD tokenizer or a verified HF reference."""

    if isinstance(tokenizer, Vocab):
        return {
            "kind": "spalmer_vocab",
            "version": tokenizer.version,
            "fingerprint": tokenizer.fingerprint,
            "vocab_size": len(tokenizer),
        }
    if isinstance(tokenizer, RPDTokenizerAdapter):
        return {
            "kind": "spalmer_rpd",
            "identity": tokenizer.identity.to_dict(),
        }
    if isinstance(tokenizer, HFTokenizerAdapter):
        if tokenizer.identity.source == "huggingface:in-memory":
            raise ValueError(
                "an in-memory Hugging Face tokenizer cannot be reconstructed from a checkpoint; "
                "create the adapter with its local path or Hub identifier as source"
            )
        identity = tokenizer.identity.to_dict()
        local_source = Path(tokenizer.identity.source)
        if local_source.exists():
            identity["source"] = str(local_source.resolve())
        return {
            "kind": "huggingface",
            "identity": identity,
        }
    raise TypeError(
        "checkpoint tokenizers must be Vocab, RPDTokenizerAdapter, or HFTokenizerAdapter; "
        "caller-defined compatible tokenizers need a reconstructible checkpoint format"
    )


def _validate_model_tokenizer(
    config: SPALMERConfig,
    tokenizer: Vocab | TokenizerBackend,
) -> None:
    if isinstance(tokenizer, Vocab):
        version = tokenizer.version
        fingerprint = tokenizer.fingerprint
        vocab_size = len(tokenizer)
    else:
        identity = tokenizer.identity
        fingerprint = identity.fingerprint
        vocab_size = identity.vocab_size
        if isinstance(tokenizer, RPDTokenizerAdapter):
            try:
                version = int(identity.revision or "")
            except ValueError as exc:
                raise ValueError(
                    "RPD tokenizer identity has no numeric vocabulary version"
                ) from exc
        else:
            # External tokenizers have content identity rather than SPALMER's
            # integer vocabulary history. The fingerprint is authoritative;
            # tokenizer_version remains the experiment schema version.
            version = config.tokenizer_version
    config.assert_tokenizer_compatible(version=version, fingerprint=fingerprint)
    if vocab_size != config.vocab_size:
        raise ValueError(
            f"checkpoint tokenizer has {vocab_size} entries, expected {config.vocab_size}"
        )


def _load_checkpoint_tokenizer(
    payload: Mapping[str, Any],
    checkpoint_version: int,
    *,
    tokenizer_override: TokenizerBackend | None,
) -> Vocab | TokenizerBackend:
    if checkpoint_version < 5:
        if tokenizer_override is not None:
            raise ValueError(
                "tokenizer_override is unavailable for legacy embedded-vocab checkpoints"
            )
        return Vocab.from_dict(payload["vocab"])

    if "tokenizer_binding" not in payload:
        raise ValueError("version-5 checkpoint is missing its tokenizer_binding field")
    binding = dict(_require_mapping(payload["tokenizer_binding"], "tokenizer binding"))
    kind = binding.get("kind")
    if kind == "spalmer_vocab":
        _require_config_schema(
            "tokenizer binding",
            binding,
            {"kind", "version", "fingerprint", "vocab_size"},
        )
        if tokenizer_override is not None:
            raise ValueError("tokenizer_override cannot replace an embedded SPALMER vocabulary")
        vocab = Vocab.from_dict(payload["vocab"])
        if (
            int(binding["version"]) != vocab.version
            or str(binding["fingerprint"]) != vocab.fingerprint
            or int(binding["vocab_size"]) != len(vocab)
        ):
            raise ValueError("embedded SPALMER vocabulary disagrees with its tokenizer binding")
        return vocab
    if kind not in {"spalmer_rpd", "huggingface"}:
        raise ValueError(f"unsupported checkpoint tokenizer kind: {kind!r}")
    _require_config_schema("tokenizer binding", binding, {"kind", "identity"})
    identity = _tokenizer_identity_from_dict(binding["identity"])
    expected_backend = "spalmer-rpd-v2" if kind == "spalmer_rpd" else "huggingface"
    if identity.backend != expected_backend:
        raise ValueError(
            f"checkpoint tokenizer kind {kind!r} disagrees with backend {identity.backend!r}"
        )
    if tokenizer_override is not None:
        if kind != "huggingface":
            raise ValueError("tokenizer_override is only supported for Hugging Face references")
        _assert_tokenizer_identity(identity, tokenizer_override.identity)
        return tokenizer_override
    if kind == "spalmer_rpd":
        vocab = Vocab.from_dict(payload["vocab"])
        tokenizer = RPDTokenizerAdapter(
            vocab,
            special_tokens=identity.special_tokens,
            source=identity.source,
        )
        _assert_tokenizer_identity(identity, tokenizer.identity)
        return tokenizer

    local_source = Path(identity.source)
    tokenizer = HFTokenizerAdapter.from_pretrained(
        identity.source,
        revision=identity.revision,
        local_files_only=local_source.exists(),
        trust_remote_code=False,
        special_tokens=identity.special_tokens,
    )
    _assert_tokenizer_identity(identity, tokenizer.identity)
    return tokenizer


def _tokenizer_identity_from_dict(value: Any) -> TokenizerIdentity:
    raw = dict(_require_mapping(value, "tokenizer identity"))
    _require_config_schema(
        "tokenizer identity",
        raw,
        {
            "backend",
            "source",
            "artifact_fingerprint",
            "vocab_size",
            "special_tokens",
            "revision",
            "fingerprint",
        },
    )
    raw_specials = dict(_require_mapping(raw["special_tokens"], "tokenizer special tokens"))
    _require_config_schema(
        "tokenizer special tokens",
        raw_specials,
        {"bos_id", "eos_id", "eod_id", "pad_id"},
    )
    identity = TokenizerIdentity(
        backend=str(raw["backend"]),
        source=str(raw["source"]),
        artifact_fingerprint=str(raw["artifact_fingerprint"]),
        vocab_size=int(raw["vocab_size"]),
        special_tokens=SpecialTokenIds(**raw_specials),
        revision=None if raw["revision"] is None else str(raw["revision"]),
    )
    if raw["fingerprint"] != identity.fingerprint:
        raise ValueError("checkpoint tokenizer identity fingerprint is invalid")
    return identity


def _assert_tokenizer_identity(
    expected: TokenizerIdentity,
    observed: TokenizerIdentity,
) -> None:
    if (
        expected.backend != observed.backend
        or expected.artifact_fingerprint != observed.artifact_fingerprint
        or expected.vocab_size != observed.vocab_size
        or expected.special_tokens != observed.special_tokens
        or expected.fingerprint != observed.fingerprint
    ):
        raise ValueError(
            "checkpoint tokenizer reference resolved to a different tokenizer identity"
        )


def _resolve_checkpoint_binding(
    model_state: Mapping[str, Any],
    *,
    run_state: RunStateEnvelope | None,
    checkpoint_binding: CheckpointBinding | None,
) -> CheckpointBinding | None:
    if run_state is not None:
        if run_state.checkpoint_binding is None:
            raise ValueError("run state has no checkpoint binding")
        if checkpoint_binding is not None and checkpoint_binding != run_state.checkpoint_binding:
            raise ValueError("explicit checkpoint binding disagrees with run state")
        checkpoint_binding = run_state.checkpoint_binding
    if checkpoint_binding is None:
        return None
    observed_digest = tensor_state_sha256(model_state)
    if checkpoint_binding.model_state_sha256 != observed_digest:
        raise ValueError("checkpoint binding does not match the model state")
    if run_state is not None and checkpoint_binding.step != run_state.progress.step:
        raise ValueError("checkpoint binding step disagrees with run progress")
    return checkpoint_binding


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _flush_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _flush_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    tokenizer_override: TokenizerBackend | None = None,
) -> tuple[SPALMERCausalLM, Vocab | TokenizerBackend, dict[str, Any]]:
    """Reconstruct a model and its tokenizer from a trusted local bundle."""

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != FORMAT_NAME:
        raise ValueError("not a SPALMER prototype checkpoint")
    checkpoint_version = payload.get("format_version")
    if checkpoint_version not in _SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(f"unsupported checkpoint version: {payload.get('format_version')}")

    raw_binding = payload.get("checkpoint_binding")
    if checkpoint_version >= 4 and "checkpoint_binding" not in payload:
        raise ValueError("checkpoint version 4 or newer is missing its binding field")
    binding = (
        None
        if raw_binding is None
        else CheckpointBinding.from_dict(_require_mapping(raw_binding, "checkpoint binding"))
    )
    if binding is not None:
        observed_digest = tensor_state_sha256(
            _require_mapping(payload["model_state"], "model state")
        )
        if observed_digest != binding.model_state_sha256:
            raise ValueError("checkpoint model state does not match its binding digest")

    raw_model_config = dict(payload["model_config"])
    required_model = {field.name for field in fields(SPALMERConfig)}
    if checkpoint_version < 3:
        required_model -= _VERSION_3_MODEL_CONFIG_FIELDS
    if checkpoint_version < 6:
        # Pre-recurrence bundles construct a plain physical stack. A null
        # recurrence field carries no information, so it is dropped rather than
        # rejected; a populated one would contradict the declared version.
        required_model -= _VERSION_6_MODEL_CONFIG_FIELDS
        if raw_model_config.get("recurrence") is None:
            raw_model_config.pop("recurrence", None)
        else:
            raise ValueError(
                f"checkpoint version {checkpoint_version} predates the recurrent core "
                "but carries a recurrence configuration"
            )
    _require_config_schema("model", raw_model_config, required_model)
    raw_recurrence = raw_model_config.get("recurrence")
    if isinstance(raw_recurrence, Mapping):
        _require_config_schema(
            "recurrence",
            dict(raw_recurrence),
            {field.name for field in fields(RecurrenceConfig)},
        )
    config = SPALMERConfig(**raw_model_config)
    raw_kda_config = dict(payload["kda_config"])
    raw_mla_config = dict(payload["mla_config"])
    _require_config_schema("KDA", raw_kda_config, {field.name for field in fields(KDAConfig)})
    _require_config_schema("MLA", raw_mla_config, {field.name for field in fields(MLAConfig)})
    kda_config = KDAConfig(**raw_kda_config)
    mla_config = MLAConfig(**raw_mla_config)
    raw_experts_config = _migrate_experts_config(
        dict(payload["experts_config"]),
        checkpoint_version,
    )
    legacy_experts = checkpoint_version == 1
    experts_config = MicroExpertsConfig(**raw_experts_config)
    model_parameter_dtype = _load_model_parameter_dtype(
        payload,
        checkpoint_version,
        experts_config,
    )
    directional_config, atxy_config = _load_optional_component_configs(
        payload,
        checkpoint_version,
    )
    tokenizer = _load_checkpoint_tokenizer(
        payload,
        checkpoint_version,
        tokenizer_override=tokenizer_override,
    )
    _validate_model_tokenizer(config, tokenizer)

    model = build_spalmer_model(
        config,
        kda_config,
        mla_config,
        experts_config,
        directional_config=directional_config,
        atxy_config=atxy_config,
    )
    model_state = _require_mapping(payload["model_state"], "model state")
    _validate_checkpoint_parameter_state_dtype(
        model.named_parameters(remove_duplicate=False),
        model_state,
        model_parameter_dtype,
    )
    model.to(dtype=_MODEL_PARAMETER_DTYPES[model_parameter_dtype])
    if checkpoint_version >= 3:
        model.load_state_dict(model_state, strict=True)
    else:
        # Older bundles predate some buffers; everything else must still match.
        allowed_missing = set(_VERSION_3_MODEL_BUFFERS)
        if legacy_experts:
            allowed_missing |= {
                f"potentiation_controller.{name}"
                for name in model.potentiation_controller.state_dict()
            }
        incompatible = model.load_state_dict(model_state, strict=False)
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
    observed_parameter_dtype = _model_parameter_dtype(model.named_parameters())
    if observed_parameter_dtype != model_parameter_dtype:
        raise ValueError(
            "loaded model parameter dtype disagrees with checkpoint metadata: "
            f"{observed_parameter_dtype!r} != {model_parameter_dtype!r}"
        )
    _validate_model_experts_config(model, experts_config)
    _validate_potentiation_state(model, experts_config)
    _validate_optional_component_configs(model, directional_config, atxy_config)
    _validate_model_recurrence(model)
    object.__setattr__(model, _CHECKPOINT_BINDING_ATTRIBUTE, binding)
    metadata = dict(payload.get("metadata") or {})
    metadata.pop(CHECKPOINT_BINDING_METADATA_KEY, None)
    if binding is not None:
        metadata[CHECKPOINT_BINDING_METADATA_KEY] = binding.to_dict()
    return model, tokenizer, metadata


def checkpoint_binding_from_metadata(
    metadata: dict[str, Any],
) -> CheckpointBinding | None:
    """Extract the authoritative binding returned by :func:`load_checkpoint`."""

    raw = metadata.get(CHECKPOINT_BINDING_METADATA_KEY)
    if raw is None:
        return None
    return CheckpointBinding.from_dict(_require_mapping(raw, "checkpoint binding metadata"))


def _migrate_experts_config(
    raw_config: dict[str, Any],
    checkpoint_version: int,
) -> dict[str, Any]:
    """Upgrade historical expert configs without changing their numerics."""

    current_fields = {field.name for field in fields(MicroExpertsConfig)}
    # The v5 and v6 expert schemas are identical; an equality gate here would
    # reject every v5 bundle the moment FORMAT_VERSION moves past it.
    if checkpoint_version >= 5:
        _require_config_schema("expert", raw_config, current_fields)
        return raw_config
    if checkpoint_version == 1:
        _require_config_schema("expert", raw_config, _VERSION_1_EXPERT_FIELDS)
    else:
        minimum_fields = current_fields - _VERSION_5_EXPERT_FIELDS - _POST_V3_EXPERT_FIELDS
        if checkpoint_version == 2:
            minimum_fields -= _VERSION_3_EXPERT_FIELDS
        missing = minimum_fields - raw_config.keys()
        unexpected = raw_config.keys() - (current_fields - _VERSION_5_EXPERT_FIELDS)
        if missing or unexpected:
            raise ValueError(
                "checkpoint has invalid expert configuration fields: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )

    active = int(raw_config["active_experts"])
    num_experts = int(raw_config["num_experts"])
    min_active = int(raw_config.get("min_active_experts", min(2, active)))
    max_active = int(raw_config.get("max_active_experts", max(active, min(20, num_experts))))
    dense_v1 = checkpoint_version == 1
    compatibility = MicroExpertsConfig(
        d_model=int(raw_config["d_model"]),
        num_experts=num_experts,
        expert_inter_dim=raw_config.get("expert_inter_dim"),
        shared_inter_dim=0,
        active_experts=active,
        min_active_experts=min_active,
        max_active_experts=max_active,
        min_resident_experts=active,
        max_resident_experts=num_experts,
        expert_execution="loop",
        expert_weight_format="legacy_int",
        expert_activation_format="bfloat16",
        expert_master_dtype="float32",
        expert_qat_backend="reference",
        expert_promotion_format="bfloat16",
        expert_fake_quantization=(
            False if dense_v1 else bool(raw_config.get("expert_fake_quantization", True))
        ),
        potentiation_budget=(
            0 if dense_v1 else int(raw_config.get("potentiation_budget", 0))
        ),
        router_score_transform=(
            "identity" if dense_v1 else str(raw_config.get("router_score_transform", "identity"))
        ),
    )
    migrated = asdict(compatibility)
    migrated.update(raw_config)
    # These fields did not exist before v5. Preserve the old integer fake-QAT
    # path and FP32 parameter payload rather than silently changing outputs.
    migrated.update(
        expert_weight_format="legacy_int",
        expert_activation_format="bfloat16",
        expert_master_dtype="float32",
        expert_qat_backend="reference",
        expert_promotion_format="bfloat16",
    )
    # Pre-v5 residency and per-token execution capacity were independent, as
    # they are now. Preserve both historical counts exactly.
    if dense_v1:
        migrated.update(
            expert_fake_quantization=False,
            potentiation_budget=0,
            router_score_transform="identity",
        )
    _require_config_schema("expert", migrated, current_fields)
    return migrated


def _load_optional_component_configs(
    payload: Mapping[str, Any],
    checkpoint_version: int,
) -> tuple[DirectionalConfig | None, ATXYConfig | None]:
    if checkpoint_version < 5:
        return None, None
    missing = {"directional_config", "atxy_config"} - payload.keys()
    if missing:
        raise ValueError(f"version-5 checkpoint is missing fields: {sorted(missing)}")
    raw_directional = payload.get("directional_config")
    raw_atxy = payload.get("atxy_config")
    directional = None
    if raw_directional is not None:
        directional_fields = {field.name for field in fields(DirectionalConfig)}
        directional_mapping = dict(_require_mapping(raw_directional, "directional config"))
        _require_config_schema("directional", directional_mapping, directional_fields)
        directional = DirectionalConfig(**directional_mapping)
        if not directional.enabled:
            raise ValueError("checkpoint directional_config must be enabled or null")
    atxy = None
    if raw_atxy is not None:
        atxy_fields = {field.name for field in fields(ATXYConfig)}
        atxy_mapping = dict(_require_mapping(raw_atxy, "ATXY config"))
        _require_config_schema("ATXY", atxy_mapping, atxy_fields)
        atxy = ATXYConfig(**atxy_mapping)
    return directional, atxy


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


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
        expected_dtype = (
            torch.bfloat16
            if experts_config.expert_master_dtype == "bfloat16"
            else torch.float32
        )
        observed_dtypes = {
            parameter.dtype for parameter in bank.parameters(recurse=False)
        }
        if observed_dtypes != {expected_dtype}:
            raise ValueError(
                f"expert bank at layer {layer_index} has master dtypes "
                f"{sorted(map(str, observed_dtypes))}; expected {expected_dtype}"
            )


def _validate_model_recurrence(model: SPALMERCausalLM) -> None:
    """The latent recurrence module must exist exactly when the config says so."""

    configured = model.config.recurrence is not None
    attached = getattr(model.backbone, "recurrence", None) is not None
    if configured != attached:
        raise ValueError(
            "model recurrence configuration disagrees with the backbone: "
            f"config.recurrence={'set' if configured else 'None'}, "
            f"backbone.recurrence={'present' if attached else 'absent'}"
        )


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


def _validate_optional_component_configs(
    model: SPALMERCausalLM,
    directional_config: DirectionalConfig | None,
    atxy_config: ATXYConfig | None,
) -> None:
    expected_directional = (
        directional_config
        if directional_config is not None and directional_config.enabled
        else None
    )
    for layer_index, block in enumerate(model.backbone.blocks):
        mixer = getattr(block, "directional_mixer", None)
        observed = None if mixer is None else getattr(mixer, "config", None)
        if observed != expected_directional:
            raise ValueError(
                f"directional_config does not match directional mixer at layer {layer_index}"
            )
    atxy = getattr(model.backbone, "atxy", None)
    observed_atxy = None if atxy is None else getattr(atxy, "config", None)
    if observed_atxy != atxy_config:
        raise ValueError("atxy_config does not match the model ATXY module")


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


__all__ = [
    "CHECKPOINT_BINDING_METADATA_KEY",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "checkpoint_binding_from_metadata",
    "load_checkpoint",
    "save_checkpoint",
]
