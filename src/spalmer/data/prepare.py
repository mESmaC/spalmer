"""Bounded-memory preparation of approved JSONL into experiment-ready token shards."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from spalmer.tokenizer.backends import ExactByteMappingUnavailable, TokenizerBackend

from .contracts import CorpusManifest
from .discovery import DiscoveredInput, JsonlAdapterConfig, discover_jsonl, iter_approved_jsonl
from .manifest import build_corpus_manifest, canonical_language
from .shards import TokenizedDocument, write_token_shard


@dataclass(frozen=True, slots=True)
class PreparedCorpus:
    output_directory: Path
    manifest_path: Path
    tokenizer_identity_path: Path
    shard_indexes: tuple[Path, ...]
    documents_loaded: int
    documents_selected: int

    def to_dict(self) -> dict[str, object]:
        return {
            "output_directory": str(self.output_directory),
            "manifest": self.manifest_path.name,
            "tokenizer_identity": self.tokenizer_identity_path.name,
            "shard_indexes": [path.name for path in self.shard_indexes],
            "documents_loaded": self.documents_loaded,
            "documents_selected": self.documents_selected,
        }


def prepare_approved_jsonl(
    inputs: list[str | Path],
    output_directory: str | Path,
    tokenizer: TokenizerBackend,
    *,
    name: str,
    adapter_config: JsonlAdapterConfig | None = None,
    top_code_languages: int = 6,
    documents_per_shard: int = 20_000,
    eod_token_id: int | None = None,
) -> PreparedCorpus:
    """Filter/deduplicate and encode through a verified two-pass disk spool."""

    if documents_per_shard <= 0:
        raise ValueError("documents_per_shard must be positive")
    if eod_token_id is not None and eod_token_id not in tokenizer.special_tokens.assigned:
        raise ValueError("explicit EOD token id must already be assigned as a tokenizer special")
    resolved_eod = tokenizer.special_tokens.eod_id if eod_token_id is None else eod_token_id
    if resolved_eod is None:
        resolved_eod = tokenizer.special_tokens.eos_id
    if resolved_eod is None:
        raise ValueError("tokenizer must define EOD/EOS or receive an explicit eod_token_id")
    if not 0 <= resolved_eod < tokenizer.vocab_size:
        raise ValueError("EOD token id is outside the tokenizer vocabulary")

    discovered = discover_jsonl(inputs)
    manifest = build_corpus_manifest(
        name,
        iter_approved_jsonl(discovered, adapter_config),
        top_code_languages=top_code_languages,
        inputs=(item.logical_name for item in discovered),
    )
    if not manifest.selected_documents:
        raise ValueError("corpus preparation selected no documents")

    destination = Path(output_directory).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"prepared corpus destination already exists: {destination}")
    if not destination.name:
        raise ValueError("prepared corpus destination must name a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".partial",
        )
    )

    published = False
    try:
        staged_manifest = manifest.save(staging / "corpus-manifest.json")
        staged_identity = staging / "tokenizer-identity.json"
        with staged_identity.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(tokenizer.identity.to_dict(), ensure_ascii=False, indent=2) + "\n"
            )

        spool_directory = staging / ".document-spool"
        source_paths = _spool_selected_documents(
            discovered,
            adapter_config,
            manifest,
            spool_directory,
        )

        manifest_document_by_id = {
            document.document_id: document for document in manifest.selected_documents
        }

        exact_byte_mode: bool | None = None

        def token_source(document_id: str):
            nonlocal exact_byte_mode
            document = manifest_document_by_id[document_id]
            text = source_paths[document_id].read_bytes().decode("utf-8")
            encode_exact = getattr(tokenizer, "encode_with_byte_lengths", None)
            if exact_byte_mode is not False and callable(encode_exact):
                try:
                    token_ids, byte_lengths = encode_exact(
                        text,
                        add_special_tokens=False,
                        kind=document.kind,
                    )
                except ExactByteMappingUnavailable:
                    if exact_byte_mode:
                        raise RuntimeError(
                            "tokenizer lost exact byte-offset support during preparation"
                        ) from None
                    exact_byte_mode = False
                else:
                    exact_byte_mode = True
                    return TokenizedDocument(token_ids, byte_lengths)
            exact_byte_mode = False
            return tokenizer.encode(text, add_special_tokens=False, kind=document.kind)

        staged_indexes: list[Path] = []
        split_names = sorted({document.split for document in manifest.selected_documents})
        for split in split_names:
            document_ids = [
                document.document_id
                for document in manifest.selected_documents
                if document.split == split
            ]
            for shard_number, start in enumerate(range(0, len(document_ids), documents_per_shard)):
                selected = document_ids[start : start + documents_per_shard]
                staged_indexes.append(
                    write_token_shard(
                        staging,
                        f"{split}-{shard_number:05d}",
                        manifest,
                        selected,
                        token_source,
                        eod_token_id=resolved_eod,
                        tokenizer_fingerprint=tokenizer.identity.fingerprint,
                        vocab_size=tokenizer.vocab_size,
                    )
                )

        if not staged_indexes:
            raise ValueError("corpus preparation produced no token shards")
        shutil.rmtree(spool_directory)
        prepared = PreparedCorpus(
            output_directory=destination,
            manifest_path=destination / staged_manifest.name,
            tokenizer_identity_path=destination / staged_identity.name,
            shard_indexes=tuple(destination / path.name for path in staged_indexes),
            documents_loaded=len(manifest.documents),
            documents_selected=len(manifest.selected_documents),
        )
        summary_path = staging / "prepared-corpus.json"
        with summary_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(prepared.to_dict(), ensure_ascii=False, indent=2) + "\n")

        os.rename(staging, destination)
        published = True
        return prepared
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def _spool_selected_documents(
    discovered: tuple[DiscoveredInput, ...],
    adapter_config: JsonlAdapterConfig | None,
    manifest: CorpusManifest,
    spool_directory: Path,
) -> dict[str, Path]:
    """Verify a second input pass and spill only selected text to bounded storage."""

    expected = {document.document_id: document for document in manifest.documents}
    selected_positions = {
        document.document_id: position
        for position, document in enumerate(manifest.selected_documents)
    }
    source_paths = {
        document_id: spool_directory / f"{position:012d}.utf8"
        for document_id, position in selected_positions.items()
    }
    spool_directory.mkdir()
    seen: set[str] = set()
    for record in iter_approved_jsonl(discovered, adapter_config):
        document = expected.get(record.document_id)
        if document is None or record.document_id in seen:
            raise RuntimeError("corpus inputs changed between manifest and tokenization passes")
        seen.add(record.document_id)
        language = canonical_language(record.language, kind=record.kind)
        if (
            record.exact_sha256 != document.exact_sha256
            or record.source != document.source
            or record.kind != document.kind
            or language != document.language
        ):
            raise RuntimeError("corpus inputs changed between manifest and tokenization passes")
        source_path = source_paths.get(record.document_id)
        if source_path is not None:
            with source_path.open("xb") as handle:
                handle.write(record.text.encode("utf-8"))
    if seen != set(expected):
        raise RuntimeError("corpus inputs changed between manifest and tokenization passes")
    return source_paths


__all__ = ["PreparedCorpus", "prepare_approved_jsonl"]
