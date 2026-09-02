"""Deterministic ranked-precedence, longest-match encoding and lossless decoding.

Greedy ranked matching is the v0 behavior: an early long match can occasionally
block a globally better later split. The draft designates Viterbi/DP search over
the same tiered candidates as the measured alternative; it is not implemented.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from .tiers import TIER_ORDER, Tier
from .vocab import Vocab


def build_tier_tables(
    vocab: Vocab, tiers: Sequence[Tier]
) -> tuple[dict[Tier, dict[str, int]], dict[Tier, int]]:
    wanted = set(tiers)
    tables: dict[Tier, dict[str, int]] = {tier: {} for tier in tiers}
    max_lens: dict[Tier, int] = {tier: 0 for tier in tiers}
    for entry in vocab.entries:
        if entry.is_byte or entry.tier not in wanted:
            continue
        tables[entry.tier][entry.surface] = entry.token_id
        if len(entry.surface) > max_lens[entry.tier]:
            max_lens[entry.tier] = len(entry.surface)
    return tables, max_lens


def longest_match_in(
    tables: Mapping[Tier, Mapping[str, int]],
    max_lens: Mapping[Tier, int],
    text: str,
    pos: int,
    tiers: Sequence[Tier],
) -> tuple[int, Tier, str] | None:
    for tier in tiers:
        table = tables.get(tier)
        if not table:
            continue
        end = min(len(text), pos + max_lens[tier])
        for stop in range(end, pos, -1):
            surface = text[pos:stop]
            token_id = table.get(surface)
            if token_id is not None and _is_eligible_match(tier, surface, text, pos, stop):
                return token_id, tier, surface
    return None


def _is_eligible_match(tier: Tier, surface: str, text: str, start: int, stop: int) -> bool:
    """Keep identifier-like lexer entries on identifier boundaries.

    Operators and delimiters retain ordinary maximal-munch behavior. Python
    keywords, however, must not claim a prefix inside a larger prose word or
    source identifier (for example, ``in`` in ``inside`` or ``for`` in
    ``format``).
    """

    if tier is not Tier.LEXER or not surface.isidentifier():
        return True
    if start > 0 and _is_identifier_continue(text[start - 1]):
        return False
    return stop >= len(text) or not _is_identifier_continue(text[stop])


def _is_identifier_continue(character: str) -> bool:
    """Return whether one character can continue a Python identifier."""

    return ("a" + character).isidentifier()


class Encoder:
    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab
        self._text_tiers = tuple(tier for tier in TIER_ORDER if tier is not Tier.BYTE)
        self._tables, self._max_lens = build_tier_tables(vocab, self._text_tiers)
        self._payloads = [entry.payload() for entry in vocab.entries]
        self._byte_ids: dict[int, int] = {}
        for entry in vocab.entries:
            if entry.is_byte:
                self._byte_ids[entry.byte] = entry.token_id
        if set(self._byte_ids) != set(range(256)):
            raise ValueError("vocabulary must contain the complete 256-byte backstop")

    def encode(self, text: str) -> list[int]:
        return [token_id for token_id, _ in self.encode_with_tiers(text)]

    def encode_with_tiers(self, text: str) -> list[tuple[int, Tier]]:
        out: list[tuple[int, Tier]] = []
        pos = 0
        length = len(text)
        while pos < length:
            match = longest_match_in(self._tables, self._max_lens, text, pos, self._text_tiers)
            if match is not None:
                token_id, tier, surface = match
                out.append((token_id, tier))
                pos += len(surface)
            else:
                for value in text[pos].encode("utf-8"):
                    out.append((self._byte_ids[value], Tier.BYTE))
                pos += 1
        return out

    def decode(self, token_ids: Iterable[int]) -> str:
        payload = b"".join(self._payloads[token_id] for token_id in token_ids)
        return payload.decode("utf-8")
