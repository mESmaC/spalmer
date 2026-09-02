"""Stable filtering, deduplication, code-language ranking, and splitting."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .contracts import (
    CodeLanguageStat,
    CorpusManifest,
    DocumentKind,
    DocumentRecord,
    ManifestDocument,
    SplitPolicy,
)

_ENGLISH = {"en", "eng", "english", "en-us", "en-gb", "en-ca", "en-au"}
_LANGUAGE_ALIASES = {
    "c++": "cpp",
    "c#": "csharp",
    "js": "javascript",
    "py": "python",
    "rb": "ruby",
    "rs": "rust",
    "ts": "typescript",
}


def canonical_language(language: str, *, kind: str) -> str:
    value = language.strip().lower().replace("_", "-")
    if kind == "prose" and value in _ENGLISH:
        return "en"
    return _LANGUAGE_ALIASES.get(value, value)


def normalize_for_dedup(text: str, *, kind: str) -> str:
    """Conservative, deterministic normalization for duplicate grouping."""

    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n")).strip()
    if kind == "prose":
        normalized = re.sub(r"\s+", " ", normalized).casefold()
    return normalized


def stratum_key(*, source: str, kind: str, language: str) -> str:
    """Return an unambiguous scheduling key, not a token- or expert-domain label."""

    return json.dumps([source, kind, language], ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class _Candidate:
    document_id: str
    source: str
    kind: DocumentKind
    language: str
    character_count: int
    utf8_bytes: int
    exact_sha256: str
    normalized_sha256: str
    basic_exclusion: str | None


def build_corpus_manifest(
    name: str,
    records: Iterable[DocumentRecord],
    *,
    top_code_languages: int = 6,
    split_policy: SplitPolicy | None = None,
    inputs: Iterable[str] = (),
) -> CorpusManifest:
    """Build a manifest whose decisions do not depend on discovery order."""

    if top_code_languages < 0:
        raise ValueError("top_code_languages cannot be negative")
    split_policy = split_policy or SplitPolicy()
    candidates = tuple(sorted((_candidate(record) for record in records), key=_candidate_id))
    ids = [candidate.document_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("document ids must be unique before manifest construction")
    eligible = [candidate for candidate in candidates if candidate.basic_exclusion is None]
    normalized_groups: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in eligible:
        normalized_groups[candidate.normalized_sha256].append(candidate)
    canonical_by_group = {
        digest: min(group, key=_candidate_id)
        for digest, group in normalized_groups.items()
    }
    exact_canonical_by_group = {
        (digest, exact): min(
            (candidate for candidate in group if candidate.exact_sha256 == exact),
            key=_candidate_id,
        )
        for digest, group in normalized_groups.items()
        for exact in {candidate.exact_sha256 for candidate in group}
    }

    code_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for canonical in canonical_by_group.values():
        if canonical.kind != "code":
            continue
        totals = code_totals[canonical.language]
        totals[0] += 1
        totals[1] += canonical.character_count
        totals[2] += canonical.utf8_bytes
    ranked = sorted(
        code_totals.items(),
        key=lambda item: (-item[1][2], -item[1][0], item[0]),
    )
    selected_code = {language for language, _ in ranked[:top_code_languages]}
    language_stats = tuple(
        CodeLanguageStat(
            language=language,
            rank=index + 1,
            document_count=totals[0],
            character_count=totals[1],
            utf8_bytes=totals[2],
            selected=language in selected_code,
        )
        for index, (language, totals) in enumerate(ranked)
    )

    manifest_documents: list[ManifestDocument] = []
    for candidate in candidates:
        canonical = canonical_by_group.get(candidate.normalized_sha256)
        duplicate_of = None
        exclusion = candidate.basic_exclusion
        exact_canonical = exact_canonical_by_group.get(
            (candidate.normalized_sha256, candidate.exact_sha256)
        )
        if exclusion is None and exact_canonical is not None:
            if exact_canonical.document_id != candidate.document_id:
                duplicate_of = exact_canonical.document_id
                exclusion = "exact_duplicate"
            elif canonical is not None and canonical.document_id != candidate.document_id:
                duplicate_of = canonical.document_id
                exclusion = "normalized_duplicate"
        if (
            exclusion is None
            and candidate.kind == "code"
            and candidate.language not in selected_code
        ):
            exclusion = "code_language_not_selected"

        normalized_group = f"normalized-sha256:{candidate.normalized_sha256}"
        manifest_documents.append(
            ManifestDocument(
                document_id=candidate.document_id,
                source=candidate.source,
                kind=candidate.kind,
                language=candidate.language,
                character_count=candidate.character_count,
                utf8_bytes=candidate.utf8_bytes,
                exact_sha256=candidate.exact_sha256,
                normalized_sha256=candidate.normalized_sha256,
                exact_group=f"exact-sha256:{candidate.exact_sha256}",
                normalized_group=normalized_group,
                split=split_policy.assign(normalized_group),
                stratum=stratum_key(
                    source=candidate.source,
                    kind=candidate.kind,
                    language=candidate.language,
                ),
                selected=exclusion is None,
                exclusion_reason=exclusion,
                duplicate_of=duplicate_of,
            )
        )
    return CorpusManifest(
        name=name,
        documents=tuple(manifest_documents),
        code_languages=language_stats,
        split_policy=split_policy,
        inputs=tuple(sorted(set(inputs))),
    )


def _candidate(record: DocumentRecord) -> _Candidate:
    language = canonical_language(record.language, kind=record.kind)
    exact = record.exact_sha256
    normalized = normalize_for_dedup(record.text, kind=record.kind)
    digest = hashlib.sha256(f"{record.kind}\0{normalized}".encode()).hexdigest()
    exclusion = None
    if not record.text.strip():
        exclusion = "empty"
    elif record.kind == "prose" and language != "en":
        exclusion = "non_english_prose"
    elif record.kind == "code" and language in {"", "unknown", "und"}:
        exclusion = "unknown_code_language"
    return _Candidate(
        document_id=record.document_id,
        source=record.source,
        kind=record.kind,
        language=language,
        character_count=len(record.text),
        utf8_bytes=len(record.text.encode("utf-8")),
        exact_sha256=exact,
        normalized_sha256=digest,
        basic_exclusion=exclusion,
    )


def _candidate_id(candidate: _Candidate) -> str:
    return candidate.document_id


__all__ = [
    "build_corpus_manifest",
    "canonical_language",
    "normalize_for_dedup",
    "stratum_key",
]
