"""Versioned, model-agnostic state for reproducible experiment resumption.

The central checkpoint owns model weights.  This module owns the orthogonal
state needed to resume the *experiment* that produced those weights: progress,
optimizer-family payloads, random-number generators, sampler/controller state,
and exact input-artifact identity.  Payload slots are intentionally opaque so
this package does not depend on a particular model, optimizer, or scheduler.

Run-state files are trusted local artifacts.  They use :func:`torch.save` so
tensor-valued optimizer and RNG payloads retain their dtype without conversion.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

FORMAT_NAME = "spalmer.experiment.run-state"
FORMAT_VERSION = 2
_SUPPORTED_FORMAT_VERSIONS = {1, FORMAT_VERSION}
_TENSOR_STATE_HASH_FORMAT = "spalmer.tensor-state.v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class RunStateCompatibilityError(ValueError):
    """Raised when a saved run cannot safely resume against current inputs."""

    def __init__(self, mismatches: Mapping[str, tuple[Any, Any]]) -> None:
        self.mismatches = dict(mismatches)
        details = ", ".join(
            f"{name}: saved={saved!r}, expected={expected!r}"
            for name, (saved, expected) in sorted(self.mismatches.items())
        )
        super().__init__(f"run-state compatibility check failed: {details}")


def _validate_sha256(name: str, value: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class ArtifactHashes:
    """Content identities that define one experiment's immutable inputs."""

    config_sha256: str
    tokenizer_sha256: str
    corpus_manifest_sha256: str
    shard_manifest_sha256: str
    mixture_sha256: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _validate_sha256(name, getattr(self, name))

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArtifactHashes:
        _require_exact_keys(payload, set(cls.__dataclass_fields__), "artifact hashes")
        return cls(**{name: str(payload[name]) for name in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class CheckpointBinding:
    """Identity shared by one model checkpoint and its orthogonal run state."""

    generation_id: str
    step: int
    model_state_sha256: str

    def __post_init__(self) -> None:
        try:
            normalized = str(uuid.UUID(self.generation_id))
        except (ValueError, AttributeError) as exc:
            raise ValueError("generation_id must be a canonical UUID") from exc
        if normalized != self.generation_id:
            raise ValueError("generation_id must be a canonical lowercase UUID")
        if self.step < 0:
            raise ValueError("checkpoint binding step must be non-negative")
        _validate_sha256("model_state_sha256", self.model_state_sha256)

    @classmethod
    def create(
        cls,
        *,
        step: int,
        model_state: Mapping[str, Tensor],
        generation_id: str | None = None,
    ) -> CheckpointBinding:
        return cls(
            generation_id=str(uuid.uuid4()) if generation_id is None else generation_id,
            step=step,
            model_state_sha256=tensor_state_sha256(model_state),
        )

    def to_dict(self) -> dict[str, str | int]:
        return {
            "generation_id": self.generation_id,
            "step": self.step,
            "model_state_sha256": self.model_state_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CheckpointBinding:
        _require_exact_keys(
            payload,
            {"generation_id", "step", "model_state_sha256"},
            "checkpoint binding",
        )
        return cls(
            generation_id=str(payload["generation_id"]),
            step=int(payload["step"]),
            model_state_sha256=str(payload["model_state_sha256"]),
        )


def tensor_state_sha256(state: Mapping[str, Tensor]) -> str:
    """Hash a tensor state mapping independent of device and mapping order."""

    if not isinstance(state, Mapping):
        raise TypeError("tensor state must be a mapping")
    digest = hashlib.sha256()
    _hash_field(digest, _TENSOR_STATE_HASH_FORMAT.encode("ascii"))
    for name in sorted(state):
        if not isinstance(name, str):
            raise TypeError("tensor state keys must be strings")
        value = state[name]
        if not isinstance(value, Tensor):
            raise TypeError(f"tensor state value {name!r} must be a tensor")
        if value.layout != torch.strided:
            raise TypeError(f"tensor state value {name!r} must use strided layout")
        materialized = value.detach().cpu().contiguous()
        byte_view = materialized.reshape(-1).view(torch.uint8)
        _hash_field(digest, name.encode("utf-8"))
        _hash_field(digest, str(materialized.dtype).encode("ascii"))
        _hash_field(
            digest,
            json.dumps(list(materialized.shape), separators=(",", ":")).encode("ascii"),
        )
        _hash_field(digest, byte_view.numpy().tobytes())
    return digest.hexdigest()


def _hash_field(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


@dataclass(frozen=True, slots=True)
class GitProvenance:
    """Source revision and a content-sensitive working-tree fingerprint."""

    revision: str
    dirty: bool
    branch: str | None = None
    status_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.revision.strip():
            raise ValueError("git revision cannot be empty")
        if self.status_sha256 is not None:
            _validate_sha256("status_sha256", self.status_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "dirty": self.dirty,
            "branch": self.branch,
            "status_sha256": self.status_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GitProvenance:
        _require_exact_keys(
            payload,
            {"revision", "dirty", "branch", "status_sha256"},
            "git provenance",
        )
        return cls(
            revision=str(payload["revision"]),
            dirty=bool(payload["dirty"]),
            branch=None if payload.get("branch") is None else str(payload["branch"]),
            status_sha256=(
                None if payload.get("status_sha256") is None else str(payload["status_sha256"])
            ),
        )


@dataclass(frozen=True, slots=True)
class DependencyProvenance:
    """Interpreter, Torch/CUDA, platform, and selected package versions."""

    python_version: str
    torch_version: str
    cuda_version: str | None
    platform: str
    packages: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.python_version or not self.torch_version or not self.platform:
            raise ValueError("Python, Torch, and platform provenance cannot be empty")
        normalized = tuple(sorted((str(name), str(version)) for name, version in self.packages))
        if normalized != self.packages:
            raise ValueError("dependency packages must be sorted by name and version")
        if any(not name or not version for name, version in self.packages):
            raise ValueError("dependency package names and versions cannot be empty")
        if len({name for name, _ in self.packages}) != len(self.packages):
            raise ValueError("dependency package names must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "platform": self.platform,
            "packages": [list(item) for item in self.packages],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DependencyProvenance:
        _require_exact_keys(
            payload,
            {"python_version", "torch_version", "cuda_version", "platform", "packages"},
            "dependency provenance",
        )
        return cls(
            python_version=str(payload["python_version"]),
            torch_version=str(payload["torch_version"]),
            cuda_version=(
                None if payload.get("cuda_version") is None else str(payload["cuda_version"])
            ),
            platform=str(payload["platform"]),
            packages=tuple((str(name), str(version)) for name, version in payload["packages"]),
        )


@dataclass(frozen=True, slots=True)
class RunProvenance:
    git: GitProvenance
    dependencies: DependencyProvenance

    def to_dict(self) -> dict[str, Any]:
        return {"git": self.git.to_dict(), "dependencies": self.dependencies.to_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunProvenance:
        _require_exact_keys(payload, {"git", "dependencies"}, "run provenance")
        return cls(
            git=GitProvenance.from_dict(_mapping(payload["git"], "git provenance")),
            dependencies=DependencyProvenance.from_dict(
                _mapping(payload["dependencies"], "dependency provenance")
            ),
        )


@dataclass(frozen=True, slots=True)
class RunProgress:
    step: int
    tokens_seen: int

    def __post_init__(self) -> None:
        if self.step < 0 or self.tokens_seen < 0:
            raise ValueError("step and tokens_seen must be non-negative")

    def to_dict(self) -> dict[str, int]:
        return {"step": self.step, "tokens_seen": self.tokens_seen}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunProgress:
        _require_exact_keys(payload, {"step", "tokens_seen"}, "run progress")
        return cls(step=int(payload["step"]), tokens_seen=int(payload["tokens_seen"]))


@dataclass(frozen=True, slots=True)
class RNGSnapshot:
    """Python plus CPU and per-device CUDA RNG state."""

    python_state: tuple[Any, ...]
    torch_cpu_state: Tensor
    torch_cuda_states: tuple[Tensor, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.python_state, tuple):
            raise TypeError("python RNG state must be a tuple")
        _validate_rng_tensor("torch_cpu_state", self.torch_cpu_state)
        for index, state in enumerate(self.torch_cuda_states):
            _validate_rng_tensor(f"torch_cuda_states[{index}]", state)

    @property
    def cuda_device_count(self) -> int:
        return len(self.torch_cuda_states)

    def to_dict(self) -> dict[str, Any]:
        return {
            "python": self.python_state,
            "torch_cpu": self.torch_cpu_state,
            "torch_cuda": list(self.torch_cuda_states),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RNGSnapshot:
        _require_exact_keys(payload, {"python", "torch_cpu", "torch_cuda"}, "RNG state")
        return cls(
            python_state=tuple(payload["python"]),
            torch_cpu_state=_tensor(payload["torch_cpu"], "torch CPU RNG state"),
            torch_cuda_states=tuple(
                _tensor(state, "torch CUDA RNG state") for state in payload["torch_cuda"]
            ),
        )


def _validate_rng_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
        raise TypeError(f"{name} must be a one-dimensional uint8 tensor")


@dataclass(slots=True)
class ResumePayloads:
    """Opaque framework state paired with, but separate from, model weights."""

    optimizer: dict[str, Any] | None = None
    scheduler: dict[str, Any] | None = None
    scaler: dict[str, Any] | None = None
    sampler: dict[str, Any] | None = None
    controller: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "scaler": self.scaler,
            "sampler": self.sampler,
            "controller": self.controller,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResumePayloads:
        _require_exact_keys(payload, set(cls.__dataclass_fields__), "resume payloads")
        values: dict[str, dict[str, Any] | None] = {}
        for name in cls.__dataclass_fields__:
            value = payload.get(name)
            if value is not None and not isinstance(value, dict):
                raise TypeError(f"{name} resume payload must be a dictionary or None")
            values[name] = value
        return cls(**values)


@dataclass(slots=True)
class RunStateEnvelope:
    """Complete model-agnostic state required beside a model checkpoint."""

    run_id: str
    created_at_utc: str
    progress: RunProgress
    artifacts: ArtifactHashes
    provenance: RunProvenance
    rng: RNGSnapshot
    checkpoint_binding: CheckpointBinding | None = None
    payloads: ResumePayloads = field(default_factory=ResumePayloads)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id.strip() or not self.created_at_utc.strip():
            raise ValueError("run_id and created_at_utc cannot be empty")
        if not isinstance(self.metadata, dict):
            raise TypeError("run-state metadata must be a dictionary")
        if (
            self.checkpoint_binding is not None
            and self.checkpoint_binding.step != self.progress.step
        ):
            raise ValueError("checkpoint binding step must equal run progress step")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "progress": self.progress.to_dict(),
            "artifacts": self.artifacts.to_dict(),
            "provenance": self.provenance.to_dict(),
            "rng": self.rng.to_dict(),
            "checkpoint_binding": (
                None if self.checkpoint_binding is None else self.checkpoint_binding.to_dict()
            ),
            "payloads": self.payloads.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, raw_payload: Mapping[str, Any]) -> RunStateEnvelope:
        payload = _mapping(raw_payload, "run state")
        version = payload.get("format_version")
        if version not in _SUPPORTED_FORMAT_VERSIONS:
            raise ValueError(f"unsupported run-state format version: {version!r}")
        required = {
            "format",
            "format_version",
            "run_id",
            "created_at_utc",
            "progress",
            "artifacts",
            "provenance",
            "rng",
            "payloads",
            "metadata",
        }
        if version >= 2:
            required.add("checkpoint_binding")
        _require_exact_keys(payload, required, "run state")
        if payload.get("format") != FORMAT_NAME:
            raise ValueError("not a SPALMER experiment run-state artifact")
        raw_binding = payload.get("checkpoint_binding")
        return cls(
            run_id=str(payload["run_id"]),
            created_at_utc=str(payload["created_at_utc"]),
            progress=RunProgress.from_dict(_mapping(payload["progress"], "progress")),
            artifacts=ArtifactHashes.from_dict(_mapping(payload["artifacts"], "artifacts")),
            provenance=RunProvenance.from_dict(_mapping(payload["provenance"], "provenance")),
            rng=RNGSnapshot.from_dict(_mapping(payload["rng"], "RNG state")),
            checkpoint_binding=(
                None
                if raw_binding is None
                else CheckpointBinding.from_dict(_mapping(raw_binding, "checkpoint binding"))
            ),
            payloads=ResumePayloads.from_dict(_mapping(payload["payloads"], "payloads")),
            metadata=dict(_mapping(payload.get("metadata", {}), "metadata")),
        )


def capture_rng_state(*, include_cuda: bool = True) -> RNGSnapshot:
    """Capture independent copies of every RNG state used by experiment code."""

    cuda_states: tuple[Tensor, ...] = ()
    if include_cuda and torch.cuda.is_available():
        cuda_states = tuple(state.cpu().clone() for state in torch.cuda.get_rng_state_all())
    return RNGSnapshot(
        python_state=random.getstate(),
        torch_cpu_state=torch.get_rng_state().cpu().clone(),
        torch_cuda_states=cuda_states,
    )


def restore_rng_state(snapshot: RNGSnapshot, *, strict_cuda: bool = True) -> None:
    """Restore a snapshot, optionally requiring identical CUDA device topology."""

    validate_rng_state_compatibility(snapshot, strict_cuda=strict_cuda)
    previous = capture_rng_state(include_cuda=torch.cuda.is_available())
    try:
        _apply_rng_state(snapshot)
    except BaseException:
        _apply_rng_state(previous)
        raise


def validate_rng_state_compatibility(
    snapshot: RNGSnapshot,
    *,
    strict_cuda: bool = True,
) -> None:
    """Validate RNG engine state and CUDA topology without mutating global streams."""

    available_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if strict_cuda and available_cuda != snapshot.cuda_device_count:
        raise RunStateCompatibilityError(
            {"cuda_device_count": (snapshot.cuda_device_count, available_cuda)}
        )
    python_probe = random.Random()
    python_probe.setstate(snapshot.python_state)
    torch.Generator(device="cpu").set_state(snapshot.torch_cpu_state.cpu())
    for index, state in enumerate(snapshot.torch_cuda_states[:available_cuda]):
        torch.Generator(device=f"cuda:{index}").set_state(state.cpu())


def _apply_rng_state(snapshot: RNGSnapshot) -> None:
    available_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    random.setstate(snapshot.python_state)
    torch.set_rng_state(snapshot.torch_cpu_state.cpu())
    if snapshot.torch_cuda_states and available_cuda:
        states = snapshot.torch_cuda_states[:available_cuda]
        torch.cuda.set_rng_state_all([state.cpu() for state in states])


def capture_dependency_provenance(
    package_names: Iterable[str] = (),
) -> DependencyProvenance:
    """Record runtime versions without importing optional packages."""

    packages: list[tuple[str, str]] = []
    for name in sorted(set(package_names)):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "not-installed"
        packages.append((name, version))
    return DependencyProvenance(
        python_version=sys.version,
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        platform=platform.platform(),
        packages=tuple(packages),
    )


def capture_git_provenance(repository: str | Path) -> GitProvenance:
    """Read a revision and content-sensitive working-tree fingerprint."""

    root = Path(repository).resolve()

    def git(*arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
        )
        return completed.stdout

    revision = git("rev-parse", "HEAD").decode("ascii").strip()
    branch = git("rev-parse", "--abbrev-ref", "HEAD").decode("utf-8").strip()
    status = git("status", "--porcelain=v1", "-z", "--untracked-files=normal")
    worktree = hashlib.sha256()
    _hash_field(worktree, status)
    _hash_field(worktree, git("diff", "--binary", "HEAD", "--"))
    untracked = git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    for encoded_path in sorted(path for path in untracked if path):
        relative = encoded_path.decode("utf-8", errors="surrogateescape")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"untracked Git path escapes repository: {relative!r}") from exc
        _hash_field(worktree, encoded_path)
        _hash_field(worktree, candidate.read_bytes())
    return GitProvenance(
        revision=revision,
        dirty=bool(status),
        branch=branch,
        status_sha256=worktree.hexdigest(),
    )


def validate_run_state_compatibility(
    state: RunStateEnvelope,
    expected_artifacts: ArtifactHashes,
    *,
    expected_git: GitProvenance | None = None,
    expected_dependencies: DependencyProvenance | None = None,
    expected_checkpoint: CheckpointBinding | None = None,
    require_checkpoint_binding: bool = False,
    require_clean_git: bool = False,
) -> None:
    """Fail closed on artifact drift and on requested provenance constraints."""

    mismatches: dict[str, tuple[Any, Any]] = {}
    for name in ArtifactHashes.__dataclass_fields__:
        saved = getattr(state.artifacts, name)
        expected = getattr(expected_artifacts, name)
        if saved != expected:
            mismatches[name] = (saved, expected)
    if expected_git is not None:
        if state.provenance.git.revision != expected_git.revision:
            mismatches["git.revision"] = (
                state.provenance.git.revision,
                expected_git.revision,
            )
        if state.provenance.git.dirty != expected_git.dirty:
            mismatches["git.dirty"] = (state.provenance.git.dirty, expected_git.dirty)
        if state.provenance.git.status_sha256 != expected_git.status_sha256:
            mismatches["git.status_sha256"] = (
                state.provenance.git.status_sha256,
                expected_git.status_sha256,
            )
    if require_clean_git and state.provenance.git.dirty:
        mismatches["git.clean"] = (False, True)
    if expected_dependencies is not None and state.provenance.dependencies != expected_dependencies:
        mismatches["dependencies"] = (
            state.provenance.dependencies,
            expected_dependencies,
        )
    if require_checkpoint_binding and state.checkpoint_binding is None:
        mismatches["checkpoint.binding"] = (None, "required")
    if expected_checkpoint is not None and state.checkpoint_binding != expected_checkpoint:
        mismatches["checkpoint.binding"] = (
            state.checkpoint_binding,
            expected_checkpoint,
        )
    if mismatches:
        raise RunStateCompatibilityError(mismatches)


def save_run_state_atomic(path: str | Path, state: RunStateEnvelope) -> Path:
    """Atomically replace ``path`` with a fully flushed run-state artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(state.to_payload(), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _flush_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_run_state(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> RunStateEnvelope:
    """Load and validate a trusted local run-state artifact."""

    payload = torch.load(path, map_location=map_location, weights_only=False)
    return RunStateEnvelope.from_payload(_mapping(payload, "run state"))


def _flush_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _require_exact_keys(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    keys = set(payload)
    missing = expected - keys
    unexpected = keys - expected
    if missing or unexpected:
        raise ValueError(
            f"invalid {name} schema: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def _tensor(value: Any, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    return value


__all__ = [
    "ArtifactHashes",
    "CheckpointBinding",
    "DependencyProvenance",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "GitProvenance",
    "RNGSnapshot",
    "ResumePayloads",
    "RunProgress",
    "RunProvenance",
    "RunStateCompatibilityError",
    "RunStateEnvelope",
    "capture_dependency_provenance",
    "capture_git_provenance",
    "capture_rng_state",
    "load_run_state",
    "restore_rng_state",
    "save_run_state_atomic",
    "tensor_state_sha256",
    "validate_rng_state_compatibility",
    "validate_run_state_compatibility",
]
