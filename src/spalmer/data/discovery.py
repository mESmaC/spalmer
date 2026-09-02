"""Deterministic discovery and a thin adapter for approved JSONL exports."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import DocumentRecord


@dataclass(frozen=True, slots=True)
class DiscoveredInput:
    path: Path
    logical_name: str


@dataclass(frozen=True, slots=True)
class JsonlAdapterConfig:
    """Field mapping for yyLab-style approved document exports."""

    text_field: str = "text"
    id_field: str = "document_id"
    source_field: str = "source"
    kind_field: str = "kind"
    language_field: str = "language"
    approved_field: str | None = None
    default_source: str | None = None
    default_kind: str = "prose"
    default_prose_language: str = "en"

    def __post_init__(self) -> None:
        fields = (
            self.text_field,
            self.id_field,
            self.source_field,
            self.kind_field,
            self.language_field,
        )
        if any(not field for field in fields):
            raise ValueError("JSONL field names cannot be empty")
        if self.default_kind not in {"prose", "code"}:
            raise ValueError("default_kind must be 'prose' or 'code'")


def discover_jsonl(inputs: Sequence[str | Path]) -> tuple[DiscoveredInput, ...]:
    """Find JSONL files in stable logical-name order.

    Directory inputs are recursive. Their directory name prefixes each logical
    path, keeping manifests independent of absolute checkout paths.
    """

    found: list[DiscoveredInput] = []
    roots = sorted((Path(value).resolve() for value in inputs), key=lambda path: path.as_posix())
    for root in roots:
        if root.is_file():
            if not _is_jsonl(root):
                raise ValueError(f"discovery input is not JSONL: {root}")
            found.append(DiscoveredInput(root, root.name))
        elif root.is_dir():
            files = (path for path in root.rglob("*") if path.is_file() and _is_jsonl(path))
            for path in sorted(files, key=lambda item: item.as_posix()):
                relative = path.relative_to(root).as_posix()
                found.append(DiscoveredInput(path.resolve(), f"{root.name}/{relative}"))
        else:
            raise FileNotFoundError(root)
    logical_names = [item.logical_name for item in found]
    if len(logical_names) != len(set(logical_names)):
        raise ValueError("discovered JSONL logical names collide; use distinct input roots")
    return tuple(sorted(found, key=lambda item: item.logical_name))


def load_approved_jsonl(
    inputs: Iterable[DiscoveredInput],
    config: JsonlAdapterConfig | None = None,
) -> tuple[DocumentRecord, ...]:
    """Materialize :func:`iter_approved_jsonl` for callers that need a tuple."""

    return tuple(iter_approved_jsonl(inputs, config))


def iter_approved_jsonl(
    inputs: Iterable[DiscoveredInput],
    config: JsonlAdapterConfig | None = None,
) -> Iterator[DocumentRecord]:
    """Stream approved exports without importing yyLab's ingestion stack.

    When ``approved_field`` is configured, only truthy rows are consumed. If it
    is ``None``, every row is trusted as already approved by the exporting
    system. Declared document ids are preferred; deterministic file/line ids are
    generated only when they are absent.
    """

    config = config or JsonlAdapterConfig()
    ordered = sorted(inputs, key=lambda item: item.logical_name)
    for discovered in ordered:
        with _open_jsonl(discovered.path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in {discovered.logical_name}:{line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"expected a JSON object in {discovered.logical_name}:{line_number}"
                    )
                if config.approved_field is not None and not row.get(config.approved_field):
                    continue
                yield _adapt_row(row, discovered, line_number, config)


def _adapt_row(
    row: Mapping[str, Any],
    discovered: DiscoveredInput,
    line_number: int,
    config: JsonlAdapterConfig,
) -> DocumentRecord:
    if config.text_field not in row or not isinstance(row[config.text_field], str):
        raise ValueError(
            f"missing string field {config.text_field!r} in "
            f"{discovered.logical_name}:{line_number}"
        )
    text = row[config.text_field]
    source = str(row.get(config.source_field) or config.default_source or discovered.logical_name)
    kind = _canonical_kind(row.get(config.kind_field), config.default_kind)
    default_language = config.default_prose_language if kind == "prose" else "unknown"
    language = str(row.get(config.language_field) or "").strip() or default_language
    if kind == "code" and language in {"", "unknown", "und"}:
        language = _code_language_from_source(source)
    declared_id = row.get(config.id_field)
    if declared_id is None or not str(declared_id).strip():
        identity = f"{discovered.logical_name}\0{line_number}\0{source}".encode()
        document_id = f"jsonl-{hashlib.sha256(identity).hexdigest()}"
    else:
        document_id = str(declared_id)

    consumed = {
        config.text_field,
        config.id_field,
        config.source_field,
        config.kind_field,
        config.language_field,
    }
    if config.approved_field is not None:
        consumed.add(config.approved_field)
    attributes = {key: value for key, value in row.items() if key not in consumed}
    attributes["input"] = discovered.logical_name
    attributes["line"] = line_number
    return DocumentRecord.create(
        document_id=document_id,
        source=source,
        text=text,
        kind=kind,  # type: ignore[arg-type]
        language=language,
        attributes=attributes,
    )


def _canonical_kind(value: object, default: str) -> str:
    raw = str(value or default).strip().lower()
    if raw == "code":
        return "code"
    if raw in {
        "prose",
        "language",
        "math",
        "science",
        "web",
        "news",
        "conversation",
        "reasoning",
        "instruction",
        "other",
    }:
        return "prose"
    raise ValueError(f"unknown document kind/domain: {raw!r}")


_CODE_EXTENSIONS = {
    ".bash": "shell",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".cxx": "cpp",
    ".go": "go",
    ".h": "c",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".lua": "lua",
    ".mjs": "javascript",
    ".php": "php",
    ".py": "python",
    ".pyi": "python",
    ".rb": "ruby",
    ".rs": "rust",
    ".scala": "scala",
    ".sh": "shell",
    ".sql": "sql",
    ".swift": "swift",
    ".ts": "typescript",
    ".tsx": "typescript",
}


def _code_language_from_source(source: str) -> str:
    return _CODE_EXTENSIONS.get(Path(source).suffix.lower(), "unknown")


def _is_jsonl(path: Path) -> bool:
    lowered = path.name.lower()
    return lowered.endswith(".jsonl") or lowered.endswith(".jsonl.zst")


@contextmanager
def _open_jsonl(path: Path):
    if not path.name.lower().endswith(".zst"):
        with path.open("r", encoding="utf-8") as handle:
            yield handle
        return
    try:
        import zstandard as zstd
    except ImportError as exc:  # pragma: no cover - depends on optional data environment
        raise ImportError(
            "reading yyLab .jsonl.zst shards requires the optional 'zstandard' package"
        ) from exc
    with path.open("rb") as compressed:
        with zstd.ZstdDecompressor().stream_reader(compressed) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8") as handle:
                yield handle


__all__ = [
    "DiscoveredInput",
    "JsonlAdapterConfig",
    "discover_jsonl",
    "iter_approved_jsonl",
    "load_approved_jsonl",
]
