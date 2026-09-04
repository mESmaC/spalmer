"""Bridge the training engine to versioned, fail-closed experiment state."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from spalmer.experiment import (
    ArtifactHashes,
    CheckpointBinding,
    DependencyProvenance,
    GitProvenance,
    ResumePayloads,
    RunProgress,
    RunProvenance,
    RunStateCompatibilityError,
    RunStateEnvelope,
    capture_dependency_provenance,
    capture_git_provenance,
    capture_rng_state,
    restore_rng_state,
    tensor_state_sha256,
    validate_rng_state_compatibility,
    validate_run_state_compatibility,
)
from spalmer.training.config import TrainingConfig
from spalmer.training.engine import ExperimentTrainer, TrainerProgress

_CHECKPOINT_BINDING_ATTRIBUTE = "_spalmer_checkpoint_binding"
_AUTHORITATIVE_METADATA_KEYS = {
    "dependency_packages",
    "repository",
    "runtime",
    "training_config",
}


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible identity with deterministic dataclass handling."""

    normalized = _json_identity(value)
    payload = json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_artifact_hashes(
    *,
    config: Any,
    tokenizer: Any,
    corpus_manifest: Any,
    shard_manifest: Any,
    mixture: Any,
) -> ArtifactHashes:
    """Build the five immutable identities required for safe resumption."""

    return ArtifactHashes(
        config_sha256=_artifact_digest(config),
        tokenizer_sha256=_artifact_digest(tokenizer),
        corpus_manifest_sha256=_artifact_digest(corpus_manifest),
        shard_manifest_sha256=_artifact_digest(shard_manifest),
        mixture_sha256=_artifact_digest(mixture),
    )


def capture_trainer_run_state(
    trainer: ExperimentTrainer,
    *,
    run_id: str,
    artifacts: ArtifactHashes,
    repository: str | Path,
    dependency_packages: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
    generation_id: str | None = None,
    checkpoint_binding: CheckpointBinding | None = None,
) -> RunStateEnvelope:
    """Capture non-weight state to store beside an atomic model checkpoint."""

    caller_metadata = dict(metadata or {})
    reserved = _AUTHORITATIVE_METADATA_KEYS & caller_metadata.keys()
    if reserved:
        raise ValueError("run-state metadata uses reserved keys: " + ", ".join(sorted(reserved)))
    if generation_id is not None and checkpoint_binding is not None:
        raise ValueError("generation_id and checkpoint_binding are mutually exclusive")

    progress = RunProgress(
        step=trainer.progress.completed_steps,
        tokens_seen=trainer.progress.tokens_seen,
    )
    model_state = trainer.model.state_dict()
    if checkpoint_binding is None:
        checkpoint_binding = CheckpointBinding.create(
            generation_id=generation_id,
            step=progress.step,
            model_state=model_state,
        )
    elif (
        checkpoint_binding.step != progress.step
        or checkpoint_binding.model_state_sha256 != tensor_state_sha256(model_state)
    ):
        raise ValueError("checkpoint binding does not match trainer progress/model state")

    controller = getattr(trainer.model, "potentiation_controller", None)
    controller_state = None if controller is None else _clone_payload(dict(controller.state_dict()))
    normalized_packages = tuple(sorted(set(str(name) for name in dependency_packages)))
    repository_path = str(Path(repository).resolve())
    runtime_metadata = {
        "training_config": trainer.config.to_dict(),
        "runtime": _runtime_identity(trainer),
        "repository": repository_path,
        "dependency_packages": list(normalized_packages),
    }
    runtime_metadata.update(caller_metadata)
    return RunStateEnvelope(
        run_id=run_id,
        created_at_utc=datetime.now(UTC).isoformat(),
        progress=progress,
        artifacts=artifacts,
        provenance=RunProvenance(
            git=capture_git_provenance(repository_path),
            dependencies=capture_dependency_provenance(normalized_packages),
        ),
        rng=capture_rng_state(include_cuda=trainer.runtime.device.type == "cuda"),
        checkpoint_binding=checkpoint_binding,
        payloads=ResumePayloads(
            optimizer=_clone_payload(trainer.optimizer.state_dict()),
            scheduler={
                "kind": "linear-warmup-cosine",
                "completed_steps": trainer.progress.completed_steps,
            },
            scaler=None,
            sampler=_clone_payload(dict(trainer.batches.state_dict())),
            controller=controller_state,
        ),
        metadata=runtime_metadata,
    )


def restore_trainer_run_state(
    trainer: ExperimentTrainer,
    state: RunStateEnvelope,
    *,
    expected_artifacts: ArtifactHashes,
    strict_cuda_rng: bool = True,
    checkpoint_binding: CheckpointBinding | Mapping[str, Any] | None = None,
    repository: str | Path | None = None,
    dependency_packages: Sequence[str] | None = None,
    expected_git: GitProvenance | None = None,
    expected_dependencies: DependencyProvenance | None = None,
    require_clean_git: bool = False,
    allow_unbound_legacy: bool = False,
) -> None:
    """Restore state after the caller has loaded the matching model weights."""

    payloads = state.payloads
    if payloads.optimizer is None or payloads.sampler is None:
        raise ValueError("run state is missing optimizer or sampler state")
    schedule = payloads.scheduler or {}
    if schedule.get("kind") != "linear-warmup-cosine":
        raise ValueError("run state uses an incompatible learning-rate schedule")
    if int(schedule.get("completed_steps", -1)) != state.progress.step:
        raise ValueError("scheduler progress disagrees with run progress")
    _validate_authoritative_metadata(trainer, state, allow_legacy=allow_unbound_legacy)
    resolved_binding = _resolve_loaded_checkpoint_binding(trainer, checkpoint_binding)
    if state.checkpoint_binding is None:
        if not allow_unbound_legacy:
            raise RunStateCompatibilityError({"checkpoint.binding": (None, "required")})
    else:
        if resolved_binding is None:
            raise RunStateCompatibilityError(
                {"checkpoint.binding": (None, state.checkpoint_binding)}
            )
        if resolved_binding != state.checkpoint_binding:
            raise RunStateCompatibilityError(
                {"checkpoint.binding": (resolved_binding, state.checkpoint_binding)}
            )
        observed_digest = tensor_state_sha256(trainer.model.state_dict())
        if observed_digest != state.checkpoint_binding.model_state_sha256:
            raise RunStateCompatibilityError(
                {
                    "checkpoint.model_state_sha256": (
                        observed_digest,
                        state.checkpoint_binding.model_state_sha256,
                    )
                }
            )

    if expected_git is None:
        repository_path = repository or state.metadata.get("repository")
        if repository_path is None:
            if not allow_unbound_legacy:
                raise ValueError("run state lacks authoritative repository provenance")
        else:
            expected_git = capture_git_provenance(repository_path)
    if expected_dependencies is None:
        package_names = dependency_packages
        if package_names is None:
            raw_packages = state.metadata.get("dependency_packages")
            if raw_packages is None:
                if not allow_unbound_legacy:
                    raise ValueError("run state lacks authoritative dependency package names")
                package_names = ()
            elif not isinstance(raw_packages, list | tuple):
                raise TypeError("run-state dependency_packages metadata must be a sequence")
            else:
                package_names = tuple(str(name) for name in raw_packages)
        expected_dependencies = capture_dependency_provenance(package_names)
    validate_run_state_compatibility(
        state,
        expected_artifacts,
        expected_git=expected_git,
        expected_dependencies=expected_dependencies,
        expected_checkpoint=resolved_binding,
        require_checkpoint_binding=not allow_unbound_legacy,
        require_clean_git=require_clean_git,
    )

    current_optimizer = _clone_payload(trainer.optimizer.state_dict())
    current_sampler = _clone_payload(dict(trainer.batches.state_dict()))
    _validate_optimizer_topology(current_optimizer, payloads.optimizer)
    _validate_sampler_topology(current_sampler, payloads.sampler)
    controller = getattr(trainer.model, "potentiation_controller", None)
    if payloads.controller is not None:
        if controller is None:
            raise ValueError("run state contains controller state but the model has none")
        current_controller = dict(controller.state_dict())
        _validate_state_topology("controller", current_controller, payloads.controller)
        if not _payloads_equal(current_controller, payloads.controller):
            raise RunStateCompatibilityError(
                {"checkpoint.controller_state": ("model checkpoint", "run state")}
            )
    elif controller is not None:
        raise ValueError("model has a controller but run state contains no controller state")
    if state.progress.step > trainer.config.max_steps:
        raise RunStateCompatibilityError(
            {"progress.step": (state.progress.step, f"<= {trainer.config.max_steps}")}
        )
    effective_strict_cuda = strict_cuda_rng and trainer.runtime.device.type == "cuda"
    validate_rng_state_compatibility(state.rng, strict_cuda=effective_strict_cuda)

    previous_progress = trainer.progress
    previous_rng = capture_rng_state(include_cuda=torch.cuda.is_available())
    try:
        trainer.optimizer.load_state_dict(_clone_payload(payloads.optimizer))
        trainer.batches.load_state_dict(_clone_payload(payloads.sampler))
        trainer.progress = TrainerProgress(
            completed_steps=state.progress.step,
            tokens_seen=state.progress.tokens_seen,
            started_at=time.time(),
        )
        restore_rng_state(state.rng, strict_cuda=effective_strict_cuda)
    except BaseException as restore_error:
        rollback_errors: list[BaseException] = []
        for restore_previous in (
            lambda: trainer.optimizer.load_state_dict(_clone_payload(current_optimizer)),
            lambda: trainer.batches.load_state_dict(_clone_payload(current_sampler)),
            lambda: restore_rng_state(previous_rng, strict_cuda=False),
        ):
            try:
                restore_previous()
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
        trainer.progress = previous_progress
        if rollback_errors:
            raise RuntimeError(
                "run-state restore failed and rollback was incomplete: "
                + "; ".join(str(error) for error in rollback_errors)
            ) from restore_error
        raise


def _runtime_identity(trainer: ExperimentTrainer) -> dict[str, Any]:
    return {
        "device": str(trainer.runtime.device),
        "compute_dtype": str(trainer.runtime.compute_dtype),
        "cuda_name": trainer.runtime.cuda_name,
        "compute_capability": trainer.runtime.compute_capability,
    }


def _validate_authoritative_metadata(
    trainer: ExperimentTrainer,
    state: RunStateEnvelope,
    *,
    allow_legacy: bool,
) -> None:
    mismatches: dict[str, tuple[Any, Any]] = {}
    expected = {
        "training_config": trainer.config.to_dict(),
        "runtime": _runtime_identity(trainer),
    }
    for name, current in expected.items():
        saved = state.metadata.get(name)
        if saved is None and allow_legacy:
            continue
        if name == "training_config":
            saved = _normalized_training_config(saved, current)
        if saved != current:
            mismatches[f"metadata.{name}"] = (saved, current)
    if mismatches:
        raise RunStateCompatibilityError(mismatches)


def _normalized_training_config(saved: Any, current: Mapping[str, Any]) -> Any:
    """Fill defaults into a run state written before a field was introduced.

    Only keys the saved mapping never carried are supplied; any value the run
    state does record still has to match, so a real configuration change still
    raises :class:`RunStateCompatibilityError`.
    """

    if not isinstance(saved, Mapping):
        return saved
    missing = set(current) - set(saved)
    if not missing:
        return saved
    try:
        return TrainingConfig(**dict(saved)).to_dict()
    except (TypeError, ValueError):
        return saved


def _resolve_loaded_checkpoint_binding(
    trainer: ExperimentTrainer,
    explicit: CheckpointBinding | Mapping[str, Any] | None,
) -> CheckpointBinding | None:
    if isinstance(explicit, Mapping):
        explicit = CheckpointBinding.from_dict(explicit)
    attached = getattr(trainer.model, _CHECKPOINT_BINDING_ATTRIBUTE, None)
    if isinstance(attached, Mapping):
        attached = CheckpointBinding.from_dict(attached)
    if attached is not None and not isinstance(attached, CheckpointBinding):
        raise TypeError("model checkpoint binding has an invalid type")
    if explicit is not None and attached is not None and explicit != attached:
        raise RunStateCompatibilityError({"checkpoint.binding": (attached, explicit)})
    return attached if explicit is None else explicit


def _validate_optimizer_topology(current: Any, saved: Any) -> None:
    current_topology = _optimizer_topology(current)
    saved_topology = _optimizer_topology(saved)
    if current_topology != saved_topology:
        raise RunStateCompatibilityError({"optimizer.topology": (saved_topology, current_topology)})


def _optimizer_topology(payload: Any) -> dict[str, tuple[int, ...] | None]:
    if not isinstance(payload, Mapping) or set(payload) != {"dense", "sparse"}:
        raise TypeError("optimizer state must contain exactly dense and sparse lanes")
    topology: dict[str, tuple[int, ...] | None] = {}
    for lane in ("dense", "sparse"):
        state = payload[lane]
        if state is None:
            topology[lane] = None
            continue
        if not isinstance(state, Mapping):
            raise TypeError(f"optimizer {lane} lane must be a mapping or None")
        groups = state.get("param_groups")
        if not isinstance(groups, list | tuple):
            raise TypeError(f"optimizer {lane} param_groups must be a sequence")
        counts: list[int] = []
        for group in groups:
            if not isinstance(group, Mapping):
                raise TypeError(f"optimizer {lane} parameter group must be a mapping")
            parameters = group.get("params")
            if not isinstance(parameters, list | tuple):
                raise TypeError(f"optimizer {lane} group params must be a sequence")
            counts.append(len(parameters))
        topology[lane] = tuple(counts)
    return topology


def _validate_sampler_topology(current: Any, saved: Any) -> None:
    if not isinstance(current, Mapping) or not isinstance(saved, Mapping):
        raise TypeError("sampler states must be mappings")
    if set(current) != set(saved):
        raise RunStateCompatibilityError({"sampler.state_keys": (sorted(saved), sorted(current))})
    identity_keys = ("format", "version", "sampler_fingerprint")
    mismatches = {
        f"sampler.{key}": (saved.get(key), current.get(key))
        for key in identity_keys
        if key in saved or key in current
        if saved.get(key) != current.get(key)
    }
    if mismatches:
        raise RunStateCompatibilityError(mismatches)


def _validate_state_topology(name: str, current: Any, saved: Any) -> None:
    current_signature = _state_signature(current)
    saved_signature = _state_signature(saved)
    if current_signature != saved_signature:
        raise RunStateCompatibilityError({f"{name}.topology": (saved_signature, current_signature)})


def _state_signature(value: Any) -> Any:
    if isinstance(value, Tensor):
        return ("tensor", str(value.dtype), tuple(value.shape), str(value.layout))
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                (str(key), _state_signature(item))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ),
        )
    if isinstance(value, list | tuple):
        return (type(value).__name__, tuple(_state_signature(item) for item in value))
    return type(value).__name__


def _payloads_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        return bool(torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _payloads_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list | tuple) and isinstance(right, list | tuple):
        return len(left) == len(right) and all(
            _payloads_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return bool(left == right)


def _clone_payload(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _clone_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_payload(item) for item in value)
    if isinstance(value, list):
        return [_clone_payload(item) for item in value]
    return copy.deepcopy(value)


def _artifact_digest(value: Any) -> str:
    fingerprint = getattr(value, "fingerprint", None)
    if callable(fingerprint):
        fingerprint = fingerprint()
    if isinstance(fingerprint, str) and _looks_like_sha256(fingerprint):
        return fingerprint
    if isinstance(value, str) and _looks_like_sha256(value):
        return value
    return canonical_sha256(value)


def _looks_like_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _json_identity(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_identity(asdict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_identity(to_dict())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("artifact identity mapping keys must be strings")
        return {key: _json_identity(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_identity(item) for item in value]
    raise TypeError(f"cannot derive a stable artifact identity from {type(value).__name__}")


__all__ = [
    "build_artifact_hashes",
    "canonical_sha256",
    "capture_trainer_run_state",
    "restore_trainer_run_state",
]
