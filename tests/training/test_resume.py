from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from spalmer.experiment import RunStateCompatibilityError
from spalmer.training import (
    TrainingConfig,
    build_artifact_hashes,
    canonical_sha256,
    capture_trainer_run_state,
)
from spalmer.training.resume import _runtime_identity, _validate_authoritative_metadata


@dataclass(frozen=True)
class _Artifact:
    name: str
    revision: int


class _Fingerprint:
    fingerprint = "a" * 64


def test_canonical_hash_ignores_mapping_insertion_order() -> None:
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})


def test_artifact_hashes_use_content_identity_or_explicit_fingerprint() -> None:
    hashes = build_artifact_hashes(
        config=_Artifact("ten-million", 1),
        tokenizer=_Fingerprint(),
        corpus_manifest={"documents": ["one", "two"]},
        shard_manifest={"shards": ["train-0"]},
        mixture={"english": 0.5, "code:python": 0.5},
    )

    assert hashes.tokenizer_sha256 == "a" * 64
    assert hashes.config_sha256 == canonical_sha256(_Artifact("ten-million", 1))
    assert len(set(hashes.to_dict().values())) == 5


def test_canonical_hash_rejects_ambiguous_non_string_mapping_keys() -> None:
    with pytest.raises(TypeError, match="mapping keys must be strings"):
        canonical_sha256({1: "one", "1": "string-one"})


def test_capture_rejects_authoritative_metadata_override_before_touching_trainer() -> None:
    with pytest.raises(ValueError, match="reserved keys"):
        capture_trainer_run_state(
            object(),  # type: ignore[arg-type]
            run_id="not-reached",
            artifacts=build_artifact_hashes(
                config={},
                tokenizer={},
                corpus_manifest={},
                shard_manifest={},
                mixture={},
            ),
            repository=".",
            metadata={"training_config": {"seed": 999}},
        )


def _training_config(**overrides: object) -> TrainingConfig:
    values: dict[str, object] = {
        "max_steps": 4,
        "micro_batch_size": 1,
        "sequence_length": 2,
        "device": "cpu",
        "require_cuda": False,
        "compute_dtype": "float32",
        "parameter_dtype": "float32",
    }
    values.update(overrides)
    return TrainingConfig(**values)  # type: ignore[arg-type]


def _metadata_trainer(config: TrainingConfig) -> SimpleNamespace:
    return SimpleNamespace(
        config=config,
        runtime=SimpleNamespace(
            device="cpu",
            compute_dtype="torch.float32",
            cuda_name=None,
            compute_capability=None,
        ),
    )


def test_resume_accepts_legacy_training_config_without_recurrence_keys() -> None:
    config = _training_config()
    trainer = _metadata_trainer(config)
    legacy = {
        key: value
        for key, value in config.to_dict().items()
        if key
        not in {
            "mean_recurrence",
            "mean_backprop_depth",
            "recurrence_sigma",
            "recurrence_sampling",
            "max_recurrence",
        }
    }
    state = SimpleNamespace(
        metadata={"training_config": legacy, "runtime": _runtime_identity(trainer)}
    )

    _validate_authoritative_metadata(trainer, state, allow_legacy=False)


def test_resume_still_rejects_a_real_training_config_change() -> None:
    trainer = _metadata_trainer(_training_config())
    legacy = {
        key: value
        for key, value in _training_config(seed=11).to_dict().items()
        if not key.startswith("mean_") and not key.startswith("recurrence_")
        and key != "max_recurrence"
    }
    state = SimpleNamespace(
        metadata={"training_config": legacy, "runtime": _runtime_identity(trainer)}
    )

    with pytest.raises(RunStateCompatibilityError):
        _validate_authoritative_metadata(trainer, state, allow_legacy=False)


def test_resume_rejects_an_explicit_recurrence_mismatch() -> None:
    trainer = _metadata_trainer(_training_config(mean_recurrence=8.0))
    saved = _training_config(mean_recurrence=4.0).to_dict()
    state = SimpleNamespace(
        metadata={"training_config": saved, "runtime": _runtime_identity(trainer)}
    )

    with pytest.raises(RunStateCompatibilityError):
        _validate_authoritative_metadata(trainer, state, allow_legacy=False)
