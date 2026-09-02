from __future__ import annotations

import random

import pytest
import torch

from spalmer.experiment.state import (
    FORMAT_VERSION,
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
    capture_rng_state,
    load_run_state,
    restore_rng_state,
    save_run_state_atomic,
    tensor_state_sha256,
    validate_run_state_compatibility,
)


def _hash(character: str) -> str:
    return character * 64


def _artifacts(*, config: str = "a") -> ArtifactHashes:
    return ArtifactHashes(
        config_sha256=_hash(config),
        tokenizer_sha256=_hash("b"),
        corpus_manifest_sha256=_hash("c"),
        shard_manifest_sha256=_hash("d"),
        mixture_sha256=_hash("e"),
    )


def _state(*, step: int = 7) -> RunStateEnvelope:
    dependencies = DependencyProvenance(
        python_version="3.12.test",
        torch_version="2.6.test",
        cuda_version=None,
        platform="test-platform",
        packages=(("alpha", "1.0"),),
    )
    return RunStateEnvelope(
        run_id="synthetic-run",
        created_at_utc="2026-09-02T00:00:00Z",
        progress=RunProgress(step=step, tokens_seen=step * 128),
        artifacts=_artifacts(),
        provenance=RunProvenance(
            git=GitProvenance(
                revision="0123456789abcdef",
                dirty=False,
                branch="main",
                status_sha256=_hash("0"),
            ),
            dependencies=dependencies,
        ),
        rng=capture_rng_state(include_cuda=False),
        payloads=ResumePayloads(
            optimizer={"state": {0: {"moment": torch.tensor([1.0, 2.0])}}},
            scheduler={"last_epoch": step},
            scaler={"scale": 1024.0},
            sampler={"cursor": 19},
            controller={"promoted": [2, 5]},
        ),
        metadata={"note": "synthetic only"},
    )


def test_atomic_run_state_round_trip_and_replace(tmp_path):
    destination = tmp_path / "nested" / "run-state.pt"
    first = _state(step=7)
    assert save_run_state_atomic(destination, first) == destination

    restored = load_run_state(destination)
    assert restored.progress == RunProgress(step=7, tokens_seen=896)
    assert restored.artifacts == first.artifacts
    assert restored.provenance == first.provenance
    assert restored.payloads.scheduler == {"last_epoch": 7}
    moment = restored.payloads.optimizer["state"][0]["moment"]
    torch.testing.assert_close(moment, torch.tensor([1.0, 2.0]))

    save_run_state_atomic(destination, _state(step=8))
    assert load_run_state(destination).progress.step == 8
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))


def test_rng_snapshot_restores_python_and_torch_cpu_streams():
    original = capture_rng_state(include_cuda=False)
    try:
        random.seed(317)
        torch.manual_seed(911)
        snapshot = capture_rng_state(include_cuda=False)
        expected_python = [random.random() for _ in range(3)]
        expected_torch = torch.rand(4)

        random.seed(1)
        torch.manual_seed(2)
        restore_rng_state(snapshot, strict_cuda=False)
        assert [random.random() for _ in range(3)] == expected_python
        torch.testing.assert_close(torch.rand(4), expected_torch)
    finally:
        restore_rng_state(original, strict_cuda=False)


def test_compatibility_validation_reports_every_artifact_drift():
    state = _state()
    expected = _artifacts(config="f")
    with pytest.raises(RunStateCompatibilityError) as captured:
        validate_run_state_compatibility(state, expected)
    assert captured.value.mismatches == {
        "config_sha256": (_hash("a"), _hash("f")),
    }

    validate_run_state_compatibility(state, state.artifacts)
    with pytest.raises(RunStateCompatibilityError, match="git.clean"):
        dirty = _state()
        dirty.provenance = RunProvenance(
            git=GitProvenance("revision", True),
            dependencies=dirty.provenance.dependencies,
        )
        validate_run_state_compatibility(dirty, dirty.artifacts, require_clean_git=True)


def test_envelope_rejects_wrong_format_version_and_bad_hash():
    payload = _state().to_payload()
    payload["format_version"] = FORMAT_VERSION + 1
    with pytest.raises(ValueError, match="unsupported run-state"):
        RunStateEnvelope.from_payload(payload)
    payload = _state().to_payload()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="invalid run state schema"):
        RunStateEnvelope.from_payload(payload)
    with pytest.raises(ValueError, match="SHA-256"):
        _artifacts(config="not-a-digest")


def test_checkpoint_binding_hash_is_order_independent_and_versioned_state_round_trips():
    first = {"right": torch.tensor([3, 4]), "left": torch.tensor([1.0])}
    second = {"left": first["left"].clone(), "right": first["right"].clone()}
    assert tensor_state_sha256(first) == tensor_state_sha256(second)

    state = _state()
    state.checkpoint_binding = CheckpointBinding.create(
        step=state.progress.step,
        model_state=first,
        generation_id="12345678-1234-5678-9234-567812345678",
    )
    restored = RunStateEnvelope.from_payload(state.to_payload())
    assert restored.checkpoint_binding == state.checkpoint_binding


def test_version_one_run_state_loads_as_explicitly_unbound_legacy_state():
    payload = _state().to_payload()
    payload["format_version"] = 1
    payload.pop("checkpoint_binding")
    assert RunStateEnvelope.from_payload(payload).checkpoint_binding is None


def test_dependency_capture_is_sorted_and_marks_missing_packages():
    provenance = capture_dependency_provenance(
        ["package-that-does-not-exist-spalmer-test", "pytest"]
    )
    assert [name for name, _ in provenance.packages] == sorted(
        name for name, _ in provenance.packages
    )
    assert dict(provenance.packages)["package-that-does-not-exist-spalmer-test"] == (
        "not-installed"
    )
