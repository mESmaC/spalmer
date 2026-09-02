"""Deterministic weighted stratum -> document -> window sampling."""

from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import reduce
from math import gcd
from typing import Any

from .shards import MMapTokenShard, TokenShardDocument

SAMPLER_STATE_FORMAT = "spalmer.data.window-sampler-state"
SAMPLER_STATE_VERSION = 1
_MASK64 = (1 << 64) - 1
_SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15


@dataclass(frozen=True, slots=True)
class WindowSample:
    document_id: str
    source: str
    kind: str
    language: str
    stratum: str
    shard_fingerprint: str
    document_offset: int
    tokens: tuple[int, ...]
    token_byte_lengths: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class TokenBatch:
    """Host-side batch; callers choose if and where to create a device tensor."""

    samples: tuple[WindowSample, ...]

    @property
    def token_ids(self) -> tuple[tuple[int, ...], ...]:
        return tuple(sample.tokens for sample in self.samples)


@dataclass(frozen=True, slots=True)
class WindowEligibility:
    """Deterministic accounting for document-local fixed-length windows."""

    split: str
    sequence_length: int
    documents_in_split: int
    eligible_documents: int
    too_short_documents: int
    eligible_windows: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "split": self.split,
            "sequence_length": self.sequence_length,
            "documents_in_split": self.documents_in_split,
            "eligible_documents": self.eligible_documents,
            "too_short_documents": self.too_short_documents,
            "eligible_windows": self.eligible_windows,
        }


@dataclass(frozen=True, slots=True)
class SamplerState:
    sampler_fingerprint: str
    rng_state: int
    draws: int
    contract_version: int = SAMPLER_STATE_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != SAMPLER_STATE_VERSION:
            raise ValueError(f"unsupported sampler state version: {self.contract_version}")
        if len(self.sampler_fingerprint) != 64:
            raise ValueError("sampler_fingerprint must be a SHA-256 digest")
        if not 0 <= self.rng_state <= _MASK64 or self.draws < 0:
            raise ValueError("invalid sampler RNG state")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": SAMPLER_STATE_FORMAT,
            "version": self.contract_version,
            "sampler_fingerprint": self.sampler_fingerprint,
            "rng_state": self.rng_state,
            "draws": self.draws,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SamplerState:
        if data.get("format") != SAMPLER_STATE_FORMAT:
            raise ValueError("not a SPALMER sampler state")
        version = int(data.get("version", 0))
        if version != SAMPLER_STATE_VERSION:
            raise ValueError(f"unsupported sampler state version: {version}")
        return cls(
            sampler_fingerprint=str(data["sampler_fingerprint"]),
            rng_state=int(data["rng_state"]),
            draws=int(data["draws"]),
            contract_version=version,
        )


@dataclass(frozen=True, slots=True)
class _Candidate:
    shard: MMapTokenShard
    document: TokenShardDocument
    window_count: int
    shard_fingerprint: str


class WeightedWindowSampler:
    """Serializable deterministic sampler over one or more immutable shards."""

    def __init__(
        self,
        shards: Sequence[MMapTokenShard],
        *,
        sequence_length: int,
        split: str = "train",
        stratum_weights: Mapping[str, float | int | str] | None = None,
        document_weighting: str = "windows",
        seed: int = 0,
    ) -> None:
        sequence_length = _as_integer(sequence_length, "sequence_length")
        seed = _as_integer(seed, "seed")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if document_weighting not in {"uniform", "windows"}:
            raise ValueError("document_weighting must be 'uniform' or 'windows'")
        if not shards:
            raise ValueError("at least one token shard is required")
        self.shards = tuple(shards)
        _validate_homogeneous_shards(self.shards)
        self.sequence_length = sequence_length
        self.window_tokens = sequence_length + 1
        self.split = split
        self.document_weighting = document_weighting
        self._rng_state = seed & _MASK64
        self._draws = 0

        candidates: dict[str, list[_Candidate]] = {}
        seen_documents: set[str] = set()
        documents_in_split = 0
        eligible_documents = 0
        too_short_documents = 0
        eligible_windows = 0
        for shard in sorted(shards, key=lambda value: value.descriptor.fingerprint):
            shard_fingerprint = shard.descriptor.fingerprint
            for document in shard.documents:
                if document.document_id in seen_documents:
                    raise ValueError(
                        f"document occurs in more than one shard: {document.document_id}"
                    )
                seen_documents.add(document.document_id)
                if document.split != split:
                    continue
                documents_in_split += 1
                if document.stored_token_count < self.window_tokens:
                    too_short_documents += 1
                    continue
                window_count = document.stored_token_count - self.window_tokens + 1
                eligible_documents += 1
                eligible_windows += window_count
                candidate = _Candidate(
                    shard=shard,
                    document=document,
                    window_count=window_count,
                    shard_fingerprint=shard_fingerprint,
                )
                candidates.setdefault(document.stratum, []).append(candidate)
        self.eligibility = WindowEligibility(
            split=split,
            sequence_length=sequence_length,
            documents_in_split=documents_in_split,
            eligible_documents=eligible_documents,
            too_short_documents=too_short_documents,
            eligible_windows=eligible_windows,
        )
        if not candidates:
            raise ValueError(
                "no documents in the requested split are long enough for a window "
                f"(documents={documents_in_split}, too_short={too_short_documents}, "
                f"sequence_length={sequence_length})"
            )
        self._candidates = {
            stratum: tuple(
                sorted(
                    group,
                    key=lambda item: (item.document.document_id, item.shard_fingerprint),
                )
            )
            for stratum, group in sorted(candidates.items())
        }
        self._stratum_weights = _effective_weights(tuple(self._candidates), stratum_weights)
        self._active_strata = tuple(
            stratum for stratum in self._candidates if self._stratum_weights[stratum] > 0
        )
        if not self._active_strata:
            raise ValueError("at least one available stratum must have positive weight")
        self._fingerprint = self._calculate_fingerprint()

    @property
    def available_strata(self) -> tuple[str, ...]:
        return tuple(self._candidates)

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def sample(self) -> WindowSample:
        stratum = self._weighted_choice(
            self._active_strata,
            tuple(self._stratum_weights[value] for value in self._active_strata),
        )
        candidates = self._candidates[stratum]
        if self.document_weighting == "windows":
            document_weights = tuple(candidate.window_count for candidate in candidates)
        else:
            document_weights = (1,) * len(candidates)
        candidate = self._weighted_choice(candidates, document_weights)
        offset = self._randbelow(candidate.window_count)
        tokens = candidate.shard.read_window(
            candidate.document.document_id,
            offset,
            self.window_tokens,
        )
        byte_lengths = candidate.shard.read_window_bytes(
            candidate.document.document_id,
            offset,
            self.window_tokens,
        )
        return WindowSample(
            document_id=candidate.document.document_id,
            source=candidate.document.source,
            kind=candidate.document.kind,
            language=candidate.document.language,
            stratum=stratum,
            shard_fingerprint=candidate.shard_fingerprint,
            document_offset=offset,
            tokens=tokens,
            token_byte_lengths=byte_lengths,
        )

    def sample_batch(self, batch_size: int) -> TokenBatch:
        batch_size = _as_integer(batch_size, "batch_size")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return TokenBatch(tuple(self.sample() for _ in range(batch_size)))

    def state_dict(self) -> dict[str, Any]:
        return SamplerState(
            sampler_fingerprint=self.fingerprint,
            rng_state=self._rng_state,
            draws=self._draws,
        ).to_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        restored = SamplerState.from_dict(state)
        if restored.sampler_fingerprint != self.fingerprint:
            raise ValueError("sampler state belongs to a different shard/configuration")
        self._rng_state = restored.rng_state
        self._draws = restored.draws

    def _calculate_fingerprint(self) -> str:
        identity = {
            "shards": sorted(shard.descriptor.fingerprint for shard in self.shards),
            "sequence_length": self.sequence_length,
            "split": self.split,
            "document_weighting": self.document_weighting,
            "stratum_weights": self._stratum_weights,
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _next_u64(self) -> int:
        self._rng_state = (self._rng_state + _SPLITMIX_INCREMENT) & _MASK64
        value = self._rng_state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
        self._draws += 1
        return (value ^ (value >> 31)) & _MASK64

    def _randbelow(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("random upper bound must be positive")
        if upper > 1 << 64:
            raise ValueError("random upper bound exceeds the sampler's 64-bit range")
        limit = (1 << 64) - ((1 << 64) % upper)
        while True:
            value = self._next_u64()
            if value < limit:
                return value % upper

    def _weighted_choice(self, values: Sequence[Any], weights: Sequence[int]):
        total = sum(weights)
        target = self._randbelow(total)
        cursor = 0
        for value, weight in zip(values, weights, strict=True):
            cursor += weight
            if target < cursor:
                return value
        raise AssertionError("weighted choice was not assigned")


def _effective_weights(
    strata: tuple[str, ...],
    requested: Mapping[str, float | int | str] | None,
) -> dict[str, int]:
    if requested is None:
        return {stratum: 1 for stratum in strata}
    unknown = set(requested) - set(strata)
    if unknown:
        raise ValueError(f"stratum weights name unavailable strata: {sorted(unknown)}")
    decimals: list[Decimal] = []
    for stratum in strata:
        raw_value = requested.get(stratum, 0)
        if isinstance(raw_value, bool):
            raise ValueError("stratum weights must be numeric, not bool")
        try:
            value = Decimal(str(raw_value))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"invalid stratum weight for {stratum!r}: {raw_value!r}") from exc
        if not value.is_finite() or value < 0:
            raise ValueError("stratum weights must be finite and non-negative")
        decimals.append(value)
    minimum_exponent = min(value.as_tuple().exponent for value in decimals)
    scale = Decimal(10) ** max(0, -minimum_exponent)
    integers = [int(value * scale) for value in decimals]
    divisor = reduce(gcd, (value for value in integers if value), 0) or 1
    effective = {stratum: value // divisor for stratum, value in zip(strata, integers, strict=True)}
    if sum(effective.values()) > 1 << 64:
        raise ValueError("normalized stratum weights exceed the sampler's 64-bit range")
    return effective


def _validate_homogeneous_shards(shards: Sequence[MMapTokenShard]) -> None:
    fields = (
        ("tokenizer fingerprint", "tokenizer_fingerprint"),
        ("manifest fingerprint", "manifest_fingerprint"),
        ("EOD token id", "eod_token_id"),
        ("vocabulary size", "vocab_size"),
    )
    for label, attribute in fields:
        values = {getattr(shard.descriptor, attribute) for shard in shards}
        if len(values) != 1:
            raise ValueError(f"token shards have heterogeneous {label} values")
    sidecar_presence = {shard.descriptor.token_byte_file is not None for shard in shards}
    if len(sidecar_presence) != 1:
        raise ValueError("token shards have heterogeneous exact token-byte metadata")


def _as_integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}") from exc


__all__ = [
    "SAMPLER_STATE_FORMAT",
    "SAMPLER_STATE_VERSION",
    "SamplerState",
    "TokenBatch",
    "WeightedWindowSampler",
    "WindowEligibility",
    "WindowSample",
]
