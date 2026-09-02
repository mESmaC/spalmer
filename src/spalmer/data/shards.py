"""Checksummed immutable uint32 token shards with mmap-backed reads."""

from __future__ import annotations

import hashlib
import json
import mmap
import operator
import os
import struct
import tempfile
from array import array
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import CorpusManifest

SHARD_FORMAT = "spalmer.data.token-shard"
SHARD_VERSION = 2
UINT32_MAX = (1 << 32) - 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class TokenShardDocument:
    document_id: str
    source: str
    kind: str
    language: str
    split: str
    stratum: str
    token_start: int
    token_count: int
    eod_offset: int

    @property
    def stored_token_count(self) -> int:
        return self.token_count + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source": self.source,
            "kind": self.kind,
            "language": self.language,
            "split": self.split,
            "stratum": self.stratum,
            "token_start": self.token_start,
            "token_count": self.token_count,
            "eod_offset": self.eod_offset,
        }


@dataclass(frozen=True, slots=True)
class TokenizedDocument:
    """One document's token IDs and optional exact UTF-8 contribution per token."""

    token_ids: Iterable[int]
    token_byte_lengths: Iterable[int] | None = None


@dataclass(frozen=True, slots=True)
class TokenShardDescriptor:
    """Immutable token-shard metadata.

    When present, ``token_byte_file`` is a little-endian uint32 sidecar with
    one exact source-byte contribution per stored token. Writers must leave it
    absent unless those contributions come from exact tokenizer offsets.
    """

    name: str
    token_file: str
    token_sha256: str
    token_count: int
    byte_count: int
    eod_token_id: int
    tokenizer_fingerprint: str
    vocab_size: int
    manifest_fingerprint: str
    documents: tuple[TokenShardDocument, ...]
    token_byte_file: str | None = None
    token_byte_sha256: str | None = None
    contract_version: int = SHARD_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != SHARD_VERSION:
            raise ValueError(f"unsupported token shard version: {self.contract_version}")
        if not self.name or Path(self.token_file).name != self.token_file:
            raise ValueError("shard name and token_file must be local names")
        vocab_size = _as_vocab_size(self.vocab_size)
        _validate_token_id(self.eod_token_id, vocab_size, "eod_token_id")
        if self.token_count < 0 or self.byte_count != self.token_count * 4:
            raise ValueError("token shard byte_count must equal token_count * 4")
        _validate_sha256(self.token_sha256, "token_sha256")
        _validate_sha256(self.tokenizer_fingerprint, "tokenizer_fingerprint")
        _validate_sha256(self.manifest_fingerprint, "manifest_fingerprint")
        if (self.token_byte_file is None) != (self.token_byte_sha256 is None):
            raise ValueError("token byte sidecar name and checksum must be supplied together")
        if self.token_byte_file is not None:
            if Path(self.token_byte_file).name != self.token_byte_file:
                raise ValueError("token byte sidecar must be a local filename")
            _validate_sha256(self.token_byte_sha256, "token_byte_sha256")
        ids = [document.document_id for document in self.documents]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("shard document ids must be unique and sorted")
        cursor = 0
        for document in self.documents:
            if document.token_count <= 0:
                raise ValueError("tokenized documents cannot be empty")
            if document.token_start != cursor:
                raise ValueError("shard document spans must be contiguous")
            if document.eod_offset != document.token_start + document.token_count:
                raise ValueError("document EOD offset must follow its content tokens")
            cursor += document.stored_token_count
        if cursor != self.token_count:
            raise ValueError("document spans do not cover the token shard")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": SHARD_FORMAT,
            "version": self.contract_version,
            "name": self.name,
            "token_file": self.token_file,
            "token_sha256": self.token_sha256,
            "token_count": self.token_count,
            "byte_count": self.byte_count,
            "eod_token_id": self.eod_token_id,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "vocab_size": self.vocab_size,
            "manifest_fingerprint": self.manifest_fingerprint,
            "documents": [document.to_dict() for document in self.documents],
            "token_byte_file": self.token_byte_file,
            "token_byte_sha256": self.token_byte_sha256,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TokenShardDescriptor:
        if data.get("format") != SHARD_FORMAT:
            raise ValueError("not a SPALMER token shard descriptor")
        version = int(data.get("version", 0))
        if version != SHARD_VERSION:
            raise ValueError(f"unsupported token shard version: {version}")
        return cls(
            name=str(data["name"]),
            token_file=str(data["token_file"]),
            token_sha256=str(data["token_sha256"]),
            token_count=int(data["token_count"]),
            byte_count=int(data["byte_count"]),
            eod_token_id=int(data["eod_token_id"]),
            tokenizer_fingerprint=str(data["tokenizer_fingerprint"]),
            vocab_size=int(data["vocab_size"]),
            manifest_fingerprint=str(data["manifest_fingerprint"]),
            documents=tuple(TokenShardDocument(**item) for item in data["documents"]),
            token_byte_file=(
                str(data["token_byte_file"]) if data.get("token_byte_file") is not None else None
            ),
            token_byte_sha256=(
                str(data["token_byte_sha256"])
                if data.get("token_byte_sha256") is not None
                else None
            ),
            contract_version=version,
        )

    @classmethod
    def load(cls, path: str | Path) -> TokenShardDescriptor:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_token_shard(
    directory: str | Path,
    name: str,
    manifest: CorpusManifest,
    document_ids: Sequence[str],
    token_source: Callable[[str], Iterable[int] | TokenizedDocument],
    *,
    eod_token_id: int,
    tokenizer_fingerprint: str,
    vocab_size: int,
) -> Path:
    """Write one immutable shard while materializing at most one document.

    ``token_source`` is called once per selected manifest document in stable id
    order. It can tokenize from a document store lazily; this API never asks for
    a corpus-sized token mapping.
    """

    if not name or Path(name).name != name:
        raise ValueError("shard name must be one local filename component")
    vocab_size = _as_vocab_size(vocab_size)
    eod_token_id = _validate_token_id(eod_token_id, vocab_size, "eod_token_id")
    _validate_sha256(tokenizer_fingerprint, "tokenizer_fingerprint")
    requested = tuple(sorted(document_ids))
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("document_ids must be a non-empty unique collection")
    selected = {document.document_id: document for document in manifest.selected_documents}
    unknown = set(requested) - selected.keys()
    if unknown:
        raise ValueError(f"documents are absent or excluded by the manifest: {sorted(unknown)}")

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    token_path = destination / f"{name}.tokens.u32"
    byte_path = destination / f"{name}.bytes.u32"
    index_path = destination / f"{name}.index.json"
    if token_path.exists() or byte_path.exists() or index_path.exists():
        raise FileExistsError(f"immutable shard already exists: {name}")

    temporary_path: Path | None = None
    temporary_byte_path: Path | None = None
    temporary_index_path: Path | None = None
    complete = False
    try:
        with (
            tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination,
                prefix=f".{name}.tokens.",
                suffix=".partial",
                delete=False,
            ) as handle,
            tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination,
                prefix=f".{name}.bytes.",
                suffix=".partial",
                delete=False,
            ) as byte_handle,
        ):
            temporary_path = Path(handle.name)
            temporary_byte_path = Path(byte_handle.name)
            checksum = hashlib.sha256()
            byte_checksum = hashlib.sha256()
            documents: list[TokenShardDocument] = []
            cursor = 0
            exact_byte_mode: bool | None = None
            for document_id in requested:
                manifest_document = selected[document_id]
                source_value = token_source(document_id)
                if isinstance(source_value, TokenizedDocument):
                    token_values = list(source_value.token_ids)
                    byte_values = (
                        None
                        if source_value.token_byte_lengths is None
                        else list(source_value.token_byte_lengths)
                    )
                else:
                    token_values = source_value
                    byte_values = None
                document_exact = byte_values is not None
                if exact_byte_mode is None:
                    exact_byte_mode = document_exact
                elif exact_byte_mode != document_exact:
                    raise ValueError("token source changed exact byte-metadata mode within a shard")
                count = _write_uint32_values(
                    handle,
                    checksum,
                    token_values,
                    vocab_size=vocab_size,
                )
                if count == 0:
                    raise ValueError(f"token source returned an empty document: {document_id}")
                if byte_values is not None:
                    if len(byte_values) != count:
                        raise ValueError(
                            f"token byte lengths disagree with token count for {document_id}"
                        )
                    if sum(byte_values) != manifest_document.utf8_bytes:
                        raise ValueError(
                            "token byte lengths disagree with original UTF-8 byte count "
                            f"for {document_id}"
                        )
                    _write_uint32_values(
                        byte_handle,
                        byte_checksum,
                        (*byte_values, 0),
                        vocab_size=UINT32_MAX + 1,
                    )
                _write_uint32_values(handle, checksum, (eod_token_id,), vocab_size=vocab_size)
                documents.append(
                    TokenShardDocument(
                        document_id=document_id,
                        source=manifest_document.source,
                        kind=manifest_document.kind,
                        language=manifest_document.language,
                        split=manifest_document.split,
                        stratum=manifest_document.stratum,
                        token_start=cursor,
                        token_count=count,
                        eod_offset=cursor + count,
                    )
                )
                cursor += count + 1
            handle.flush()
            os.fsync(handle.fileno())
            byte_handle.flush()
            os.fsync(byte_handle.fileno())
        descriptor = TokenShardDescriptor(
            name=name,
            token_file=token_path.name,
            token_sha256=checksum.hexdigest(),
            token_count=cursor,
            byte_count=cursor * 4,
            eod_token_id=eod_token_id,
            tokenizer_fingerprint=tokenizer_fingerprint,
            vocab_size=vocab_size,
            manifest_fingerprint=manifest.fingerprint,
            documents=tuple(documents),
            token_byte_file=byte_path.name if exact_byte_mode else None,
            token_byte_sha256=byte_checksum.hexdigest() if exact_byte_mode else None,
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination,
            prefix=f".{name}.index.",
            suffix=".partial",
            delete=False,
        ) as index:
            temporary_index_path = Path(index.name)
            index.write(descriptor.to_json() + "\n")
            index.flush()
            os.fsync(index.fileno())
        os.rename(temporary_path, token_path)
        temporary_path = None
        if exact_byte_mode:
            os.rename(temporary_byte_path, byte_path)
            temporary_byte_path = None
        else:
            temporary_byte_path.unlink()
            temporary_byte_path = None
        os.rename(temporary_index_path, index_path)
        temporary_index_path = None
        complete = True
        return index_path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if temporary_byte_path is not None:
            temporary_byte_path.unlink(missing_ok=True)
        if temporary_index_path is not None:
            temporary_index_path.unlink(missing_ok=True)
        if not complete:
            index_path.unlink(missing_ok=True)
            token_path.unlink(missing_ok=True)
            byte_path.unlink(missing_ok=True)


def _write_uint32_values(
    handle,
    checksum: Any,
    values: Iterable[int],
    *,
    vocab_size: int,
) -> int:
    buffer = array("I")
    count = 0

    def flush() -> None:
        if not buffer:
            return
        if buffer.itemsize != 4:
            raise RuntimeError("this platform does not provide a four-byte unsigned int array")
        payload = array("I", buffer)
        if os.sys.byteorder != "little":
            payload.byteswap()
        raw = payload.tobytes()
        handle.write(raw)
        checksum.update(raw)
        del buffer[:]

    for value in values:
        integer = _validate_token_id(value, vocab_size, "token id")
        buffer.append(integer)
        count += 1
        if len(buffer) >= 65_536:
            flush()
    flush()
    return count


def _as_uint32(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}") from exc
    if not 0 <= integer <= UINT32_MAX:
        raise ValueError(f"{name} must be in [0, {UINT32_MAX}]; got {integer}")
    return integer


def _as_vocab_size(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("vocab_size must be a positive integer, not bool")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"vocab_size must be a positive integer; got {value!r}") from exc
    if not 0 < integer <= UINT32_MAX + 1:
        raise ValueError(f"vocab_size must be in [1, {UINT32_MAX + 1}]; got {integer}")
    return integer


def _validate_token_id(value: object, vocab_size: int, name: str) -> int:
    integer = _as_uint32(value, name)
    if integer >= vocab_size:
        raise ValueError(f"{name} {integer} is outside vocabulary size {vocab_size}")
    return integer


def _validate_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    if value != value.lower() or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def sha256_file(path: str | Path) -> str:
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


class MMapTokenShard:
    """Read-only token access that materializes only requested windows."""

    def __init__(self, index_path: str | Path, *, verify_checksum: bool = True) -> None:
        self.index_path = Path(index_path)
        self.descriptor = TokenShardDescriptor.load(self.index_path)
        self.token_path = self.index_path.parent / self.descriptor.token_file
        stat = self.token_path.stat()
        if stat.st_size != self.descriptor.byte_count:
            raise ValueError("token shard length does not match its descriptor")
        if verify_checksum and sha256_file(self.token_path) != self.descriptor.token_sha256:
            raise ValueError("token shard checksum mismatch")
        self._byte_handle = None
        self._byte_mmap = None
        if self.descriptor.token_byte_file is not None:
            byte_path = self.index_path.parent / self.descriptor.token_byte_file
            if byte_path.stat().st_size != self.descriptor.token_count * 4:
                raise ValueError("token byte sidecar length does not match its descriptor")
            if verify_checksum and sha256_file(byte_path) != self.descriptor.token_byte_sha256:
                raise ValueError("token byte sidecar checksum mismatch")
            self._byte_handle = byte_path.open("rb")
            self._byte_mmap = mmap.mmap(
                self._byte_handle.fileno(), length=0, access=mmap.ACCESS_READ
            )
        self._handle = self.token_path.open("rb")
        self._mmap = mmap.mmap(self._handle.fileno(), length=0, access=mmap.ACCESS_READ)
        self._documents = {document.document_id: document for document in self.descriptor.documents}

    def __enter__(self) -> MMapTokenShard:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if getattr(self, "_mmap", None) is not None:
            self._mmap.close()
            self._mmap = None
        if getattr(self, "_handle", None) is not None:
            self._handle.close()
            self._handle = None
        if getattr(self, "_byte_mmap", None) is not None:
            self._byte_mmap.close()
            self._byte_mmap = None
        if getattr(self, "_byte_handle", None) is not None:
            self._byte_handle.close()
            self._byte_handle = None

    @property
    def documents(self) -> tuple[TokenShardDocument, ...]:
        return self.descriptor.documents

    def document(self, document_id: str) -> TokenShardDocument:
        try:
            return self._documents[document_id]
        except KeyError as exc:
            raise KeyError(f"document is not present in shard: {document_id}") from exc

    def read_tokens(self, start: int, count: int) -> tuple[int, ...]:
        if self._mmap is None:
            raise ValueError("token shard is closed")
        if start < 0 or count < 0 or start + count > self.descriptor.token_count:
            raise IndexError("token range is outside the shard")
        if count == 0:
            return ()
        return struct.unpack_from(f"<{count}I", self._mmap, start * 4)

    def read_token_bytes(self, start: int, count: int) -> tuple[int, ...] | None:
        """Read exact per-token byte contributions, or ``None`` when unavailable."""

        if self._mmap is None:
            raise ValueError("token shard is closed")
        if start < 0 or count < 0 or start + count > self.descriptor.token_count:
            raise IndexError("token byte range is outside the shard")
        if self._byte_mmap is None:
            return None
        if count == 0:
            return ()
        return struct.unpack_from(f"<{count}I", self._byte_mmap, start * 4)

    def read_document(self, document_id: str, *, include_eod: bool = False) -> tuple[int, ...]:
        document = self.document(document_id)
        count = document.stored_token_count if include_eod else document.token_count
        return self.read_tokens(document.token_start, count)

    def read_window(self, document_id: str, start: int, count: int) -> tuple[int, ...]:
        document = self.document(document_id)
        if start < 0 or count <= 0 or start + count > document.stored_token_count:
            raise IndexError("window is outside the document including its EOD token")
        return self.read_tokens(document.token_start + start, count)

    def read_window_bytes(self, document_id: str, start: int, count: int) -> tuple[int, ...] | None:
        document = self.document(document_id)
        if start < 0 or count <= 0 or start + count > document.stored_token_count:
            raise IndexError("window is outside the document including its EOD token")
        return self.read_token_bytes(document.token_start + start, count)


__all__ = [
    "MMapTokenShard",
    "SHARD_FORMAT",
    "SHARD_VERSION",
    "TokenShardDescriptor",
    "TokenShardDocument",
    "TokenizedDocument",
    "UINT32_MAX",
    "sha256_file",
    "write_token_shard",
]
