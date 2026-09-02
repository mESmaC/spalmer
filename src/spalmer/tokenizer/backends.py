"""Replaceable tokenizer boundary for scale experiments.

The core protocol is deliberately smaller than either the RPD or Hugging Face
APIs.  ``transformers`` is imported only by ``HFTokenizerAdapter.from_pretrained``;
using RPD artifacts or caller-supplied compatible objects adds no dependency.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from spalmer.tokenizer.code_regions import InputKind
from spalmer.tokenizer.encoder import Encoder
from spalmer.tokenizer.vocab import Vocab


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SpecialTokenIds:
    """Explicit model-facing identities; ``None`` means the role is unavailable."""

    bos_id: int | None = None
    eos_id: int | None = None
    eod_id: int | None = None
    pad_id: int | None = None

    def __post_init__(self) -> None:
        for name, token_id in asdict(self).items():
            if token_id is not None and (not isinstance(token_id, int) or token_id < 0):
                raise ValueError(f"{name} token id must be a non-negative integer or None")

    def validate_for_vocab(self, vocab_size: int) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        for name, token_id in asdict(self).items():
            if token_id is not None and token_id >= vocab_size:
                raise ValueError(
                    f"{name} token id {token_id} is outside vocabulary size {vocab_size}"
                )

    @property
    def assigned(self) -> frozenset[int]:
        return frozenset(token_id for token_id in asdict(self).values() if token_id is not None)

    def to_dict(self) -> dict[str, int | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TokenizerIdentity:
    """Content identity plus the model-visible adapter contract.

    ``source`` is audit metadata and is intentionally excluded from
    ``fingerprint``: copying byte-identical artifacts to a new directory does
    not create a new tokenizer identity.
    """

    backend: str
    source: str
    artifact_fingerprint: str
    vocab_size: int
    special_tokens: SpecialTokenIds
    revision: str | None = None

    def __post_init__(self) -> None:
        if not self.backend.strip():
            raise ValueError("backend cannot be empty")
        if not self.source.strip():
            raise ValueError("source cannot be empty")
        if not self.artifact_fingerprint.strip():
            raise ValueError("artifact_fingerprint cannot be empty")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.special_tokens.validate_for_vocab(self.vocab_size)

    @property
    def fingerprint(self) -> str:
        return _canonical_digest(
            {
                "backend": self.backend,
                "artifact_fingerprint": self.artifact_fingerprint,
                "vocab_size": self.vocab_size,
                "special_tokens": self.special_tokens.to_dict(),
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "source": self.source,
            "artifact_fingerprint": self.artifact_fingerprint,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens.to_dict(),
            "revision": self.revision,
            "fingerprint": self.fingerprint,
        }


@runtime_checkable
class TokenizerBackend(Protocol):
    """Minimal tokenizer interface consumed by experiment infrastructure."""

    @property
    def identity(self) -> TokenizerIdentity: ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def special_tokens(self) -> SpecialTokenIds: ...

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        kind: InputKind = "mixed",
    ) -> list[int]: ...

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = False) -> str: ...


class ExactByteMappingUnavailable(RuntimeError):
    """Raised when a tokenizer cannot map tokens back to original UTF-8 bytes."""


@runtime_checkable
class ExactByteTokenizerBackend(TokenizerBackend, Protocol):
    """Optional capability used to persist exact BPB denominators in shards."""

    def encode_with_byte_lengths(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        kind: InputKind = "mixed",
    ) -> tuple[list[int], list[int]]: ...


class RPDTokenizerAdapter:
    """Adapter for the repository's versioned RPD ``Vocab`` artifact.

    RPD vocabulary entries are all reachable from ordinary input.  Model-only
    special tokens therefore live in a virtual, contiguous id suffix after the
    content vocabulary.  Treating an existing content id as EOD would make a
    naturally occurring token indistinguishable from a document boundary, so
    the adapter rejects that configuration.
    """

    def __init__(
        self,
        vocab: Vocab,
        *,
        special_tokens: SpecialTokenIds = SpecialTokenIds(),
        source: str | None = None,
    ) -> None:
        content_vocab_size = len(vocab)
        assigned = sorted(special_tokens.assigned)
        if assigned and assigned != list(
            range(content_vocab_size, content_vocab_size + len(assigned))
        ):
            raise ValueError(
                "RPD special token ids must form a contiguous reserved suffix "
                f"starting at content vocabulary size {content_vocab_size}"
            )
        model_vocab_size = content_vocab_size + len(assigned)
        special_tokens.validate_for_vocab(model_vocab_size)
        self.vocab = vocab
        self.encoder = Encoder(vocab)
        self._content_vocab_size = content_vocab_size
        self._identity = TokenizerIdentity(
            backend="spalmer-rpd-v2",
            source=source or f"rpd:{vocab.name}",
            artifact_fingerprint=vocab.fingerprint,
            vocab_size=model_vocab_size,
            special_tokens=special_tokens,
            revision=str(vocab.version),
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        special_tokens: SpecialTokenIds = SpecialTokenIds(),
    ) -> RPDTokenizerAdapter:
        source = str(Path(path).resolve())
        return cls(Vocab.load(path), special_tokens=special_tokens, source=source)

    @property
    def identity(self) -> TokenizerIdentity:
        return self._identity

    @property
    def vocab_size(self) -> int:
        return self.identity.vocab_size

    @property
    def special_tokens(self) -> SpecialTokenIds:
        return self.identity.special_tokens

    @property
    def content_vocab_size(self) -> int:
        return self._content_vocab_size

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        kind: InputKind = "mixed",
    ) -> list[int]:
        token_ids = self.encoder.encode(text, kind=kind)
        if add_special_tokens:
            if self.special_tokens.bos_id is not None:
                token_ids.insert(0, self.special_tokens.bos_id)
            if self.special_tokens.eos_id is not None:
                token_ids.append(self.special_tokens.eos_id)
        return token_ids

    def encode_with_byte_lengths(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        kind: InputKind = "mixed",
    ) -> tuple[list[int], list[int]]:
        token_ids = self.encode(text, add_special_tokens=add_special_tokens, kind=kind)
        byte_lengths = [
            (
                0
                if token_id in self.special_tokens.assigned
                else len(self.vocab.get(token_id).payload())
            )
            for token_id in token_ids
        ]
        if sum(byte_lengths) != len(text.encode("utf-8")):
            raise AssertionError("RPD byte accounting must exactly cover the original input")
        return token_ids, byte_lengths

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = False,
        errors: str = "strict",
    ) -> str:
        ids = _validated_token_ids(token_ids, self.vocab_size)
        assigned = self.special_tokens.assigned
        unknown_reserved = next(
            (
                token_id
                for token_id in ids
                if token_id >= self.content_vocab_size and token_id not in assigned
            ),
            None,
        )
        if unknown_reserved is not None:
            raise ValueError(f"token id {unknown_reserved} is not an assigned RPD special token")
        payload = bytearray()
        for token_id in ids:
            if token_id in assigned:
                if not skip_special_tokens:
                    payload.extend(self._special_surface(token_id).encode("utf-8"))
                continue
            payload.extend(self.vocab.get(token_id).payload())
        return payload.decode("utf-8", errors=errors)

    def _special_surface(self, token_id: int) -> str:
        roles = (
            ("bos", self.special_tokens.bos_id),
            ("eos", self.special_tokens.eos_id),
            ("eod", self.special_tokens.eod_id),
            ("pad", self.special_tokens.pad_id),
        )
        role = next(name for name, assigned_id in roles if assigned_id == token_id)
        return f"<|{role}|>"


class CompatibleTokenizerAdapter:
    """Adapter for an in-memory object with HF-style ``encode``/``decode`` methods."""

    def __init__(
        self,
        tokenizer: object,
        *,
        backend: str = "compatible-object",
        source: str | None = None,
        revision: str | None = None,
        artifact_fingerprint: str | None = None,
        special_tokens: SpecialTokenIds | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        resolved_specials = special_tokens or _special_tokens_from_object(tokenizer)
        vocab_size = _vocab_size_from_object(tokenizer)
        resolved_specials.validate_for_vocab(vocab_size)
        self._identity = TokenizerIdentity(
            backend=backend,
            source=source or f"python:{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            artifact_fingerprint=artifact_fingerprint or _object_artifact_fingerprint(tokenizer),
            vocab_size=vocab_size,
            special_tokens=resolved_specials,
            revision=revision,
        )

    @property
    def identity(self) -> TokenizerIdentity:
        return self._identity

    @property
    def vocab_size(self) -> int:
        return self.identity.vocab_size

    @property
    def special_tokens(self) -> SpecialTokenIds:
        return self.identity.special_tokens

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        kind: InputKind = "mixed",
    ) -> list[int]:
        del kind  # Generic/HF tokenizers do not consume RPD's prose/code routing hint.
        encode = getattr(self.tokenizer, "encode", None)
        if not callable(encode):
            raise TypeError(
                "compatible tokenizer must implement encode(text, add_special_tokens=...)"
            )
        result = encode(text, add_special_tokens=add_special_tokens)
        return _validated_token_ids(result, self.vocab_size)

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = False) -> str:
        decode = getattr(self.tokenizer, "decode", None)
        if not callable(decode):
            raise TypeError(
                "compatible tokenizer must implement decode(ids, skip_special_tokens=...)"
            )
        ids = _validated_token_ids(token_ids, self.vocab_size)
        return str(decode(ids, skip_special_tokens=skip_special_tokens))

    def encode_with_byte_lengths(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        kind: InputKind = "mixed",
    ) -> tuple[list[int], list[int]]:
        """Use fast-tokenizer offsets to partition every original UTF-8 byte.

        Offsets are interpreted against the original Python string. Bytes in a
        normalization gap are assigned to the following token, and trailing
        bytes to the final content token. This makes the partition exact while
        retaining deterministic per-token accounting for arbitrary windows.
        """

        del kind
        tokenize = self.tokenizer if callable(self.tokenizer) else None
        if tokenize is None:
            raise ExactByteMappingUnavailable(
                "tokenizer is not callable with return_offsets_mapping=True"
            )
        try:
            encoded = tokenize(
                text,
                add_special_tokens=add_special_tokens,
                return_offsets_mapping=True,
            )
        except (NotImplementedError, TypeError, ValueError) as exc:
            raise ExactByteMappingUnavailable(
                "tokenizer does not expose usable original-text offset mappings"
            ) from exc
        if not isinstance(encoded, Mapping):
            raise ExactByteMappingUnavailable("tokenizer offset result must be a mapping")
        token_ids = _flat_integer_sequence(encoded.get("input_ids"), "input_ids")
        offsets = _flat_offset_sequence(encoded.get("offset_mapping"))
        if len(token_ids) != len(offsets):
            raise ExactByteMappingUnavailable("token ids and offsets have different lengths")
        _validated_token_ids(token_ids, self.vocab_size)
        byte_lengths = _partition_utf8_bytes(
            text,
            token_ids,
            offsets,
            self.special_tokens.assigned,
        )
        return token_ids, byte_lengths


class LocalSerializedTokenizerAdapter(CompatibleTokenizerAdapter):
    """Load a caller-defined compatible serialization and hash all artifact bytes."""

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        loader: Callable[[Path], object],
        *,
        special_tokens: SpecialTokenIds | None = None,
        revision: str | None = None,
    ) -> LocalSerializedTokenizerAdapter:
        artifact = Path(path).resolve()
        tokenizer = loader(artifact)
        return cls(
            tokenizer,
            backend="local-compatible",
            source=str(artifact),
            revision=revision,
            artifact_fingerprint=fingerprint_local_artifact(artifact),
            special_tokens=special_tokens,
        )


class HFTokenizerAdapter(CompatibleTokenizerAdapter):
    """Optional Hugging Face ``AutoTokenizer`` adapter."""

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: object,
        *,
        source: str = "huggingface:in-memory",
        revision: str | None = None,
        artifact_fingerprint: str | None = None,
        special_tokens: SpecialTokenIds | None = None,
    ) -> HFTokenizerAdapter:
        return cls(
            tokenizer,
            backend="huggingface",
            source=source,
            revision=revision,
            artifact_fingerprint=artifact_fingerprint,
            special_tokens=special_tokens,
        )

    @classmethod
    def from_pretrained(
        cls,
        identifier: str | Path,
        *,
        revision: str | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        special_tokens: SpecialTokenIds | None = None,
        **kwargs: Any,
    ) -> HFTokenizerAdapter:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional environment
            raise ImportError(
                "HFTokenizerAdapter.from_pretrained requires the optional 'transformers' package"
            ) from exc

        source = str(identifier)
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        local_path = Path(identifier)
        artifact_fingerprint = (
            fingerprint_local_artifact(local_path.resolve())
            if local_path.exists()
            else _object_artifact_fingerprint(tokenizer)
        )
        return cls.from_tokenizer(
            tokenizer,
            source=source,
            revision=revision,
            artifact_fingerprint=artifact_fingerprint,
            special_tokens=special_tokens,
        )


def fingerprint_local_artifact(path: str | Path) -> str:
    """Content-hash one file or a directory tree, independent of root location."""

    artifact = Path(path)
    if not artifact.exists():
        raise FileNotFoundError(artifact)
    digest = hashlib.sha256()
    if artifact.is_file():
        digest.update(b"file\0")
        _update_digest_from_file(digest, artifact)
        return digest.hexdigest()
    if not artifact.is_dir():
        raise ValueError(f"tokenizer artifact is neither a file nor directory: {artifact}")
    digest.update(b"directory\0")
    files = sorted(
        (candidate for candidate in artifact.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(artifact).as_posix(),
    )
    for candidate in files:
        relative = candidate.relative_to(artifact).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        _update_digest_from_file(digest, candidate)
    return digest.hexdigest()


def _update_digest_from_file(digest: Any, path: Path) -> None:
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(len(chunk).to_bytes(8, "big"))
            digest.update(chunk)


def _special_tokens_from_object(tokenizer: object) -> SpecialTokenIds:
    def token_id(*names: str) -> int | None:
        for name in names:
            value = getattr(tokenizer, name, None)
            if value is not None:
                return int(value)
        return None

    return SpecialTokenIds(
        bos_id=token_id("bos_token_id"),
        eos_id=token_id("eos_token_id"),
        eod_id=token_id("eod_token_id", "eod_id"),
        pad_id=token_id("pad_token_id"),
    )


def _validated_token_ids(token_ids: Iterable[int], vocab_size: int) -> list[int]:
    result = [int(token_id) for token_id in token_ids]
    invalid = next((token_id for token_id in result if not 0 <= token_id < vocab_size), None)
    if invalid is not None:
        raise ValueError(f"token id {invalid} is outside vocabulary size {vocab_size}")
    return result


def _flat_integer_sequence(value: object, name: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ExactByteMappingUnavailable(f"tokenizer {name} must be a flat sequence")
    if value and isinstance(value[0], Sequence):
        raise ExactByteMappingUnavailable(f"tokenizer {name} must not contain a batch axis")
    return [int(item) for item in value]


def _flat_offset_sequence(value: object) -> list[tuple[int, int]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ExactByteMappingUnavailable("tokenizer offset_mapping must be a flat sequence")
    offsets: list[tuple[int, int]] = []
    for item in value:
        if (
            not isinstance(item, Sequence)
            or isinstance(item, str | bytes | bytearray)
            or len(item) != 2
        ):
            raise ExactByteMappingUnavailable("every tokenizer offset must be a start/end pair")
        offsets.append((int(item[0]), int(item[1])))
    return offsets


def _partition_utf8_bytes(
    text: str,
    token_ids: Sequence[int],
    offsets: Sequence[tuple[int, int]],
    special_ids: frozenset[int],
) -> list[int]:
    prefix = [0]
    for character in text:
        prefix.append(prefix[-1] + len(character.encode("utf-8")))
    byte_lengths = [0] * len(token_ids)
    covered_character = 0
    content_indices: list[int] = []
    for index, (token_id, (start, stop)) in enumerate(zip(token_ids, offsets, strict=True)):
        if token_id in special_ids:
            continue
        if not 0 <= start <= stop <= len(text):
            raise ExactByteMappingUnavailable("tokenizer offset is outside the original text")
        if start < covered_character:
            raise ExactByteMappingUnavailable(
                "overlapping tokenizer offsets cannot provide exact per-token byte ownership"
            )
        if stop == start and text:
            raise ExactByteMappingUnavailable(
                "zero-width content-token offsets cannot provide exact byte ownership"
            )
        content_indices.append(index)
        # Include any normalization/whitespace gap before this token so every
        # original byte has exactly one deterministic owner.
        byte_lengths[index] = prefix[stop] - prefix[covered_character]
        covered_character = stop
    if text and not content_indices:
        raise ExactByteMappingUnavailable("non-empty input produced no content-token offsets")
    if covered_character < len(text):
        byte_lengths[content_indices[-1]] += prefix[-1] - prefix[covered_character]
    if sum(byte_lengths) != prefix[-1]:
        raise ExactByteMappingUnavailable("token offsets do not exactly cover original UTF-8 bytes")
    return byte_lengths


def _vocab_mapping(tokenizer: object) -> dict[str, int]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        return {}
    raw = get_vocab()
    if not isinstance(raw, Mapping):
        raise TypeError("compatible tokenizer get_vocab() must return a mapping")
    return {str(token): int(token_id) for token, token_id in raw.items()}


def _vocab_size_from_object(tokenizer: object) -> int:
    candidates: list[int] = []
    try:
        candidates.append(int(len(tokenizer)))  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        pass
    declared = getattr(tokenizer, "vocab_size", None)
    if declared is not None:
        candidates.append(int(declared))
    vocab = _vocab_mapping(tokenizer)
    if vocab:
        candidates.append(max(vocab.values()) + 1)
    if not candidates or max(candidates) <= 0:
        raise ValueError("cannot infer a positive vocabulary size from compatible tokenizer")
    return max(candidates)


def _object_artifact_fingerprint(tokenizer: object) -> str:
    backend_json: str | None = None
    backend = getattr(tokenizer, "backend_tokenizer", None)
    to_str = getattr(backend, "to_str", None)
    if callable(to_str):
        backend_json = str(to_str())
    payload = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "vocab": sorted(_vocab_mapping(tokenizer).items()),
        "backend_json": backend_json,
        "init_kwargs": _json_safe(getattr(tokenizer, "init_kwargs", None)),
        "special_tokens_map": _json_safe(getattr(tokenizer, "special_tokens_map", None)),
    }
    if not payload["vocab"] and backend_json is None:
        raise ValueError(
            "compatible tokenizer must expose get_vocab() or backend_tokenizer.to_str() "
            "when artifact_fingerprint is not supplied"
        )
    return _canonical_digest(payload)


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_safe(item) for item in value]
    return str(value)


__all__ = [
    "CompatibleTokenizerAdapter",
    "ExactByteMappingUnavailable",
    "ExactByteTokenizerBackend",
    "HFTokenizerAdapter",
    "LocalSerializedTokenizerAdapter",
    "RPDTokenizerAdapter",
    "SpecialTokenIds",
    "TokenizerBackend",
    "TokenizerIdentity",
    "fingerprint_local_artifact",
]
