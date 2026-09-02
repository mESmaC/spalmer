from __future__ import annotations

from dataclasses import dataclass

import pytest

from spalmer.training import (
    build_artifact_hashes,
    canonical_sha256,
    capture_trainer_run_state,
)


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
