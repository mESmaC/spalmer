"""Runtime tier stack for the Ranked-Precedence Distributional Tokenizer (C01).

Encoding matches left-to-right: the highest eligible tier (lowest rank value)
claims first, and the longest match wins within a tier. This generalizes
maximal munch with ranked precedence.
"""

from __future__ import annotations

from enum import IntEnum


class Tier(IntEnum):
    LEXER = 1
    PHRASE = 2
    WORD = 3
    SALVAGE = 4
    LANGUAGE_FALLBACK = 5
    ATOM = 6
    BYTE = 7


TIER_ORDER: tuple[Tier, ...] = tuple(Tier)

TIER_LABELS: dict[Tier, str] = {
    Tier.LEXER: "lexer",
    Tier.PHRASE: "phrase",
    Tier.WORD: "word",
    Tier.SALVAGE: "salvage",
    Tier.LANGUAGE_FALLBACK: "language_fallback",
    Tier.ATOM: "atom",
    Tier.BYTE: "byte",
}
