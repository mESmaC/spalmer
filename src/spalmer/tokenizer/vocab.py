"""Versioned, serializable static vocabulary with append-only version metadata.

Lifecycle contract (draft section 9): version 1 is frozen, and later vocabulary
merges are append-only. Existing token ids and sealed version records are never
rewritten; `Vocab.extend_append_only` adds entries and seals a new record.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .tiers import Tier

FORMAT_NAME = "spalmer.tokenizer.vocab"
FORMAT_VERSION = 1


def byte_surface(value: int) -> str:
    if not 0 <= value <= 255:
        raise ValueError(f"byte value out of range: {value}")
    return f"<byte:0x{value:02x}>"


def append_byte_backstop(vocab: Vocab) -> None:
    for value in range(256):
        if vocab.find(Tier.BYTE, byte_surface(value)) is None:
            vocab.append(Tier.BYTE, byte_surface(value), byte=value)


@dataclass(frozen=True)
class TokenEntry:
    token_id: int
    surface: str
    tier: Tier
    byte: int | None = None

    @property
    def is_byte(self) -> bool:
        return self.byte is not None

    def payload(self) -> bytes:
        if self.byte is not None:
            return bytes((self.byte,))
        return self.surface.encode("utf-8")

    def to_dict(self) -> dict:
        return {
            "id": self.token_id,
            "surface": self.surface,
            "tier": int(self.tier),
            "byte": self.byte,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TokenEntry:
        return cls(
            token_id=int(data["id"]),
            surface=str(data["surface"]),
            tier=Tier(int(data["tier"])),
            byte=data.get("byte"),
        )


@dataclass(frozen=True)
class VersionRecord:
    version: int
    created: str
    note: str
    frozen: bool
    entry_count: int

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "created": self.created,
            "note": self.note,
            "frozen": self.frozen,
            "entry_count": self.entry_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> VersionRecord:
        return cls(
            version=int(data["version"]),
            created=str(data["created"]),
            note=str(data["note"]),
            frozen=bool(data["frozen"]),
            entry_count=int(data["entry_count"]),
        )


def _validate_history(history: list[VersionRecord], total_entries: int) -> None:
    expected_version = 1
    last_count = 0
    for record in history:
        if record.version != expected_version:
            raise ValueError("version records must increase strictly from 1")
        if not last_count <= record.entry_count <= total_entries:
            raise ValueError("version entry_count must be non-decreasing and in range")
        last_count = record.entry_count
        expected_version += 1
    if history and history[-1].entry_count != total_entries:
        raise ValueError("the latest version record must seal every vocabulary entry")


class Vocab:
    def __init__(self, name: str) -> None:
        self.name = name
        self.entries: list[TokenEntry] = []
        self.history: list[VersionRecord] = []
        self._surface_index: dict[tuple[int, str], int] = {}

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def version(self) -> int:
        return self.history[-1].version if self.history else 0

    @property
    def fingerprint(self) -> str:
        """Stable identity for the token-id mapping consumed by a model."""

        identity = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "version": self.version,
            "entries": [entry.to_dict() for entry in self.entries],
        }
        payload = json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, token_id: int) -> TokenEntry:
        return self.entries[token_id]

    def find(self, tier: Tier, surface: str) -> int | None:
        return self._surface_index.get((int(tier), surface))

    def append(self, tier: Tier, surface: str, byte: int | None = None) -> TokenEntry:
        if (int(tier), surface) in self._surface_index:
            raise ValueError(f"duplicate surface in tier {tier!r}: {surface!r}")
        if self.history:
            raise RuntimeError("sealed vocabulary is immutable; use extend_append_only")
        return self._append_unchecked(tier, surface, byte)

    def _append_unchecked(
        self,
        tier: Tier,
        surface: str,
        byte: int | None = None,
    ) -> TokenEntry:
        if (int(tier), surface) in self._surface_index:
            raise ValueError(f"duplicate surface in tier {tier!r}: {surface!r}")
        entry = TokenEntry(token_id=len(self.entries), surface=surface, tier=tier, byte=byte)
        self.entries.append(entry)
        self._surface_index[(int(tier), surface)] = entry.token_id
        return entry

    def seal_version(self, created: str, note: str, frozen: bool = True) -> VersionRecord:
        record = VersionRecord(
            version=self.version + 1,
            created=created,
            note=note,
            frozen=frozen,
            entry_count=len(self.entries),
        )
        self.history.append(record)
        return record

    def extend_append_only(
        self, created: str, note: str, items: Iterable[tuple[Tier, str]]
    ) -> VersionRecord:
        for tier, surface in items:
            self._append_unchecked(tier, surface)
        return self.seal_version(created, note, frozen=True)

    def to_dict(self) -> dict:
        return {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "name": self.name,
            "entries": [entry.to_dict() for entry in self.entries],
            "history": [record.to_dict() for record in self.history],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Vocab:
        if data.get("format") != FORMAT_NAME:
            raise ValueError("not a spalmer tokenizer vocabulary")
        if int(data.get("format_version", 0)) != FORMAT_VERSION:
            raise ValueError(f"unsupported format version: {data.get('format_version')!r}")
        vocab = cls(str(data["name"]))
        for raw in data["entries"]:
            entry = TokenEntry.from_dict(raw)
            if entry.token_id != len(vocab.entries):
                raise ValueError("token ids must be contiguous starting at 0")
            vocab.append(entry.tier, entry.surface, entry.byte)
        history = [VersionRecord.from_dict(raw) for raw in data.get("history", [])]
        _validate_history(history, len(vocab.entries))
        vocab.history = history
        return vocab

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, text: str) -> Vocab:
        return cls.from_dict(json.loads(text))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Vocab:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
