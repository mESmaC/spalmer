"""Minimal trainer producing a small Ranked-Precedence vocabulary from samples.

Implemented stages of the C01 draft:
- Phase 1 lexer reservation: deterministic keywords, operators, delimiters.
- Phase 2 code-region routing (lite): comments and string-literal contents feed
  the prose statistics; identifier material is routed to the identifier path;
  string delimiters stay atomic.
- Phase 3 identifier preprocessing: camelCase and snake_case boundary splits.
- Phase 5 fallthrough logging: one left-to-right pass over tiers 1-3 records the
  residual stream used by salvage.
- Phase 6 frequency salvage: bottom-up pairwise merges over residual runs,
  capped at quadgram length.
- Phase 7 reserved tiers: English letters and bigrams, whitespace/punctuation/
  digit/letter atoms, and the 256-byte backstop.

Partially implemented stages, identified explicitly:
- Phase 0 corpus sufficiency (Heaps' law): not implemented.
- Phase 4 recursive distributional splitting: replaced by whole-word frequency
  selection plus count-based phrase selection; no proxy-model surprisal yet.
- Phase 6 merge scoring uses raw counts, not PMI or proxy-model surprisal.
- Phase 8 utilization audit: not implemented (the draft's letter duplication
  between the language-fallback and atom tiers is preserved as drafted).
- Greedy ranked matching only; the Viterbi/DP alternative is not implemented.
- No special tokens; that interface remains open in the ledger.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from datetime import date

from .encoder import build_tier_tables, longest_match_in
from .tiers import Tier
from .vocab import Vocab, append_byte_backstop

PYTHON_KEYWORDS = (
    "False",
    "None",
    "True",
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
)

OPERATORS = (
    "**=",
    "//=",
    ">>=",
    "<<=",
    "...",
    "->",
    ":=",
    "==",
    "!=",
    "<=",
    ">=",
    "**",
    "//",
    "<<",
    ">>",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "&=",
    "|=",
    "^=",
    "@=",
    "+",
    "-",
    "*",
    "/",
    "%",
    "&",
    "|",
    "^",
    "~",
    "<",
    ">",
    "=",
)

DELIMITERS = ("(", ")", "[", "]", "{", "}", ",", ":", ".", ";", "@")

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
STRING_RE = re.compile(
    r'"""(?:[^"\\]|\\.|\n)*?"""'
    r"|'''(?:[^'\\]|\\.|\n)*?'''"
    r'|"(?:[^"\\\n]|\\.)*"'
    r"|'(?:[^'\\\n]|\\.)*'"
)
COMMENT_RE = re.compile(r"#[^\n]*")
CAMEL_BOUNDARY_RE = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])"
    r"|(?<=[A-Z])(?=[A-Z][a-z])"
    r"|(?<=[A-Za-z])(?=[0-9])"
    r"|(?<=[0-9])(?=[A-Za-z])"
)


@dataclass(frozen=True)
class Sample:
    text: str
    kind: str = "prose"


@dataclass(frozen=True)
class TrainerConfig:
    min_word_count: int = 2
    max_word_tokens: int = 256
    min_phrase_count: int = 2
    max_phrase_words: int = 3
    max_phrase_tokens: int = 64
    max_salvage_merges: int = 32
    min_salvage_count: int = 2
    max_fallback_bigrams: int = 32
    min_fallback_bigram_count: int = 2


def split_identifier(identifier: str) -> list[str]:
    parts: list[str] = []
    for chunk in identifier.split("_"):
        if chunk:
            parts.extend(piece for piece in CAMEL_BOUNDARY_RE.split(chunk) if piece)
    return parts or [identifier]


def lexer_surfaces() -> list[str]:
    seen: dict[str, None] = {}
    ordered = (
        *sorted(PYTHON_KEYWORDS),
        *sorted(OPERATORS, key=lambda s: (-len(s), s)),
        *sorted(DELIMITERS),
    )
    for surface in ordered:
        seen.setdefault(surface, None)
    return list(seen)


def train(
    samples: list[Sample],
    config: TrainerConfig | None = None,
    name: str = "spalmer-rpdt",
    created: str | None = None,
) -> Vocab:
    if not samples:
        raise ValueError("at least one sample is required")
    config = config or TrainerConfig()
    created = created or date.today().isoformat()

    prose_texts: list[str] = []
    word_counts: Counter[str] = Counter()
    phrase_streams: list[list[str]] = []

    for sample in samples:
        if sample.kind == "prose":
            prose_texts.append(sample.text)
        elif sample.kind == "code":
            comments = [m.lstrip("#") for m in COMMENT_RE.findall(sample.text)]
            strings = [m[1:-1] for m in STRING_RE.findall(sample.text)]
            prose_texts.extend(t for t in comments + strings if t.strip())
            body = STRING_RE.sub(" ", COMMENT_RE.sub(" ", sample.text))
            for identifier in IDENTIFIER_RE.findall(body):
                word_counts.update(split_identifier(identifier))
        else:
            raise ValueError(f"unknown sample kind: {sample.kind!r}")

    for text in prose_texts:
        words = WORD_RE.findall(text)
        if words:
            phrase_streams.append(words)
            word_counts.update(words)

    vocab = Vocab(name)

    for surface in lexer_surfaces():
        vocab.append(Tier.LEXER, surface)

    for phrase, _count in _top_phrases(phrase_streams, config):
        vocab.append(Tier.PHRASE, phrase)

    for word, _count in _top_words(word_counts, config):
        vocab.append(Tier.WORD, word)

    _learn_salvage(vocab, samples, config)

    for letter in string.ascii_lowercase + string.ascii_uppercase:
        vocab.append(Tier.LANGUAGE_FALLBACK, letter)
    for bigram, _count in _top_fallback_bigrams(prose_texts, config):
        vocab.append(Tier.LANGUAGE_FALLBACK, bigram)

    for surface in _atom_surfaces():
        vocab.append(Tier.ATOM, surface)

    append_byte_backstop(vocab)

    vocab.seal_version(created, "version 1: initial trainer vocabulary (frozen)", frozen=True)
    return vocab


def _top_phrases(streams: list[list[str]], config: TrainerConfig) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for words in streams:
        for n in range(2, config.max_phrase_words + 1):
            for i in range(len(words) - n + 1):
                counts[" ".join(words[i : i + n])] += 1
    ranked = sorted(
        ((phrase, c) for phrase, c in counts.items() if c >= config.min_phrase_count),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[: config.max_phrase_tokens]


def _top_words(word_counts: Counter[str], config: TrainerConfig) -> list[tuple[str, int]]:
    ranked = sorted(
        (
            (word, c)
            for word, c in word_counts.items()
            if c >= config.min_word_count and len(word) > 1
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[: config.max_word_tokens]


def _learn_salvage(vocab: Vocab, samples: list[Sample], config: TrainerConfig) -> None:
    tiers = (Tier.LEXER, Tier.PHRASE, Tier.WORD)
    tables, max_lens = build_tier_tables(vocab, tiers)
    runs: list[list[str]] = []
    for sample in samples:
        runs.extend(_residual_runs(tables, max_lens, tiers, sample.text))
    for _ in range(config.max_salvage_merges):
        pair_counts: Counter[tuple[str, str]] = Counter()
        for pieces in runs:
            for left, right in zip(pieces, pieces[1:]):
                if len(left) + len(right) <= 4:
                    pair_counts[(left, right)] += 1
        eligible = sorted(
            ((pair, c) for pair, c in pair_counts.items() if c >= config.min_salvage_count),
            key=lambda item: (-item[1], item[0]),
        )
        if not eligible:
            break
        chosen = eligible[0][0]
        runs = [_merge_pair(pieces, chosen) for pieces in runs]
        surface = chosen[0] + chosen[1]
        if vocab.find(Tier.SALVAGE, surface) is None:
            vocab.append(Tier.SALVAGE, surface)


def _residual_runs(tables, max_lens, tiers, text: str) -> list[list[str]]:
    runs: list[list[str]] = []
    current: list[str] = []
    pos = 0
    while pos < len(text):
        match = longest_match_in(tables, max_lens, text, pos, tiers)
        if match is None:
            character = text[pos]
            if character.isalpha():
                current.append(character)
            elif current:
                runs.append(current)
                current = []
            pos += 1
        else:
            if current:
                runs.append(current)
                current = []
            pos += len(match[2])
    if current:
        runs.append(current)
    return runs


def _merge_pair(pieces: list[str], pair: tuple[str, str]) -> list[str]:
    merged: list[str] = []
    i = 0
    while i < len(pieces):
        if i < len(pieces) - 1 and pieces[i] == pair[0] and pieces[i + 1] == pair[1]:
            merged.append(pair[0] + pair[1])
            i += 2
        else:
            merged.append(pieces[i])
            i += 1
    return merged


def _top_fallback_bigrams(prose_texts: list[str], config: TrainerConfig) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for text in prose_texts:
        lowered = text.lower()
        for i in range(len(lowered) - 1):
            pair = lowered[i : i + 2]
            if pair.isascii() and pair.isalpha():
                counts[pair] += 1
    ranked = sorted(
        ((p, c) for p, c in counts.items() if c >= config.min_fallback_bigram_count),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[: config.max_fallback_bigrams]


def _atom_surfaces() -> list[str]:
    lexer_set = set(lexer_surfaces())
    surfaces = [" ", "\n", "\t", "\r", "\x0b", "\x0c"]
    surfaces.extend(c for c in string.punctuation if c not in lexer_set)
    surfaces.extend(string.digits)
    surfaces.extend(string.ascii_letters)
    return surfaces
