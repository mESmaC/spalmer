"""Immutable, versioned contracts for SPALMER corpus preparation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

DOCUMENT_FORMAT = "spalmer.data.document"
DOCUMENT_VERSION = 1
MANIFEST_FORMAT = "spalmer.data.corpus-manifest"
MANIFEST_VERSION = 1

DocumentKind = Literal["prose", "code"]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    """Return the lowercase SHA-256 of UTF-8 ``text``."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def freeze_attributes(attributes: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
    """Freeze arbitrary JSON-compatible metadata into deterministic strings."""

    if not attributes:
        return ()
    frozen: list[tuple[str, str]] = []
    for key, value in attributes.items():
        if not isinstance(key, str) or not key:
            raise ValueError("document attribute keys must be non-empty strings")
        frozen.append((key, _canonical_json(value)))
    frozen.sort()
    return tuple(frozen)


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """One approved source document before corpus filtering and deduplication."""

    document_id: str
    source: str
    text: str
    kind: DocumentKind = "prose"
    language: str = "en"
    attributes: tuple[tuple[str, str], ...] = ()
    contract_version: int = DOCUMENT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != DOCUMENT_VERSION:
            raise ValueError(f"unsupported document contract version: {self.contract_version}")
        if not self.document_id.strip():
            raise ValueError("document_id cannot be empty")
        if not self.source.strip():
            raise ValueError("source cannot be empty")
        if self.kind not in {"prose", "code"}:
            raise ValueError(f"unknown document kind: {self.kind!r}")
        if not self.language.strip():
            raise ValueError("language cannot be empty")
        if tuple(sorted(self.attributes)) != self.attributes:
            raise ValueError("attributes must be sorted for deterministic serialization")
        if len({key for key, _ in self.attributes}) != len(self.attributes):
            raise ValueError("document attribute keys must be unique")

    @classmethod
    def create(
        cls,
        *,
        document_id: str,
        source: str,
        text: str,
        kind: DocumentKind = "prose",
        language: str = "en",
        attributes: Mapping[str, Any] | None = None,
    ) -> DocumentRecord:
        return cls(
            document_id=document_id,
            source=source,
            text=text,
            kind=kind,
            language=language,
            attributes=freeze_attributes(attributes),
        )

    @property
    def exact_sha256(self) -> str:
        return sha256_text(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": DOCUMENT_FORMAT,
            "version": self.contract_version,
            "document_id": self.document_id,
            "source": self.source,
            "text": self.text,
            "kind": self.kind,
            "language": self.language,
            "attributes": {key: json.loads(value) for key, value in self.attributes},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DocumentRecord:
        if data.get("format") != DOCUMENT_FORMAT:
            raise ValueError("not a SPALMER document record")
        version = int(data.get("version", 0))
        if version != DOCUMENT_VERSION:
            raise ValueError(f"unsupported document contract version: {version}")
        return cls.create(
            document_id=str(data["document_id"]),
            source=str(data["source"]),
            text=str(data["text"]),
            kind=str(data.get("kind", "prose")),  # type: ignore[arg-type]
            language=str(data.get("language", "en")),
            attributes=data.get("attributes") or {},
        )


@dataclass(frozen=True, slots=True)
class SplitPolicy:
    """Content-stable split assignment using integer weights and a named seed."""

    weights: tuple[tuple[str, int], ...] = (
        ("train", 9800),
        ("validation", 100),
        ("test", 100),
    )
    seed: str = "spalmer-split-v1"

    def __post_init__(self) -> None:
        if not self.seed:
            raise ValueError("split seed cannot be empty")
        if not self.weights:
            raise ValueError("at least one split is required")
        if len({name for name, _ in self.weights}) != len(self.weights):
            raise ValueError("split names must be unique")
        for name, weight in self.weights:
            if not name or weight <= 0:
                raise ValueError("split names must be non-empty and weights positive")

    def assign(self, group_id: str) -> str:
        digest = hashlib.sha256(f"{self.seed}\0{group_id}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "little") % sum(weight for _, weight in self.weights)
        cursor = 0
        for name, weight in self.weights:
            cursor += weight
            if bucket < cursor:
                return name
        raise AssertionError("split bucket was not assigned")

    def to_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "weights": [[name, weight] for name, weight in self.weights]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SplitPolicy:
        return cls(
            weights=tuple((str(name), int(weight)) for name, weight in data["weights"]),
            seed=str(data["seed"]),
        )


@dataclass(frozen=True, slots=True)
class ManifestDocument:
    """Stable corpus decision for one :class:`DocumentRecord`."""

    document_id: str
    source: str
    kind: DocumentKind
    language: str
    character_count: int
    utf8_bytes: int
    exact_sha256: str
    normalized_sha256: str
    exact_group: str
    normalized_group: str
    split: str
    stratum: str
    selected: bool
    exclusion_reason: str | None = None
    duplicate_of: str | None = None

    def __post_init__(self) -> None:
        if self.character_count < 0 or self.utf8_bytes < 0:
            raise ValueError("document sizes cannot be negative")
        if self.selected and (self.exclusion_reason is not None or self.duplicate_of is not None):
            raise ValueError("selected documents cannot carry an exclusion or duplicate target")
        if not self.selected and self.exclusion_reason is None:
            raise ValueError("excluded documents require an exclusion_reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source": self.source,
            "kind": self.kind,
            "language": self.language,
            "character_count": self.character_count,
            "utf8_bytes": self.utf8_bytes,
            "exact_sha256": self.exact_sha256,
            "normalized_sha256": self.normalized_sha256,
            "exact_group": self.exact_group,
            "normalized_group": self.normalized_group,
            "split": self.split,
            "stratum": self.stratum,
            "selected": self.selected,
            "exclusion_reason": self.exclusion_reason,
            "duplicate_of": self.duplicate_of,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ManifestDocument:
        return cls(**dict(data))


@dataclass(frozen=True, slots=True)
class CodeLanguageStat:
    language: str
    rank: int
    document_count: int
    character_count: int
    utf8_bytes: int
    selected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "rank": self.rank,
            "document_count": self.document_count,
            "character_count": self.character_count,
            "utf8_bytes": self.utf8_bytes,
            "selected": self.selected,
        }


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """Immutable record of filtering, deduplication, language ranking, and splits."""

    name: str
    documents: tuple[ManifestDocument, ...]
    code_languages: tuple[CodeLanguageStat, ...]
    split_policy: SplitPolicy
    inputs: tuple[str, ...] = ()
    contract_version: int = MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != MANIFEST_VERSION:
            raise ValueError(f"unsupported corpus manifest version: {self.contract_version}")
        if not self.name.strip():
            raise ValueError("manifest name cannot be empty")
        ids = [document.document_id for document in self.documents]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("manifest documents must have unique, sorted document ids")
        if tuple(sorted(set(self.inputs))) != self.inputs:
            raise ValueError("manifest inputs must be unique and sorted")

    @property
    def selected_documents(self) -> tuple[ManifestDocument, ...]:
        return tuple(document for document in self.documents if document.selected)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": MANIFEST_FORMAT,
            "version": self.contract_version,
            "name": self.name,
            "inputs": list(self.inputs),
            "split_policy": self.split_policy.to_dict(),
            "code_languages": [language.to_dict() for language in self.code_languages],
            "documents": [document.to_dict() for document in self.documents],
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(self.to_json() + "\n")
        return destination

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CorpusManifest:
        if data.get("format") != MANIFEST_FORMAT:
            raise ValueError("not a SPALMER corpus manifest")
        version = int(data.get("version", 0))
        if version != MANIFEST_VERSION:
            raise ValueError(f"unsupported corpus manifest version: {version}")
        return cls(
            name=str(data["name"]),
            inputs=tuple(str(item) for item in data.get("inputs", ())),
            split_policy=SplitPolicy.from_dict(data["split_policy"]),
            code_languages=tuple(CodeLanguageStat(**item) for item in data["code_languages"]),
            documents=tuple(ManifestDocument.from_dict(item) for item in data["documents"]),
            contract_version=version,
        )

    @classmethod
    def from_json(cls, text: str) -> CorpusManifest:
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path: str | Path) -> CorpusManifest:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "CodeLanguageStat",
    "CorpusManifest",
    "DOCUMENT_FORMAT",
    "DOCUMENT_VERSION",
    "DocumentKind",
    "DocumentRecord",
    "MANIFEST_FORMAT",
    "MANIFEST_VERSION",
    "ManifestDocument",
    "SplitPolicy",
    "freeze_attributes",
    "sha256_text",
]
