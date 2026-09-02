"""Minimal trainer producing a small Ranked-Precedence vocabulary from samples.

Implemented stages of the C01 draft:
- Phase 1 lexer reservation: deterministic keywords, operators, delimiters.
- Phase 2 code-region routing (lite): comments and docstrings feed the English
  distributional path; identifier material is routed to the identifier path;
  ordinary string-literal contents are left to the out-of-distribution fallback
  path (they contribute no word/phrase statistics) and string delimiters stay
  atomic.
- Phase 3 identifier preprocessing: camelCase and snake_case boundary splits.
- Phase 4 recursive distributional splitting: each run of space-separated words
  is split top-down at the boundary with the highest conditional surprisal
  under a character n-gram proxy model until a multi-word span is predictable
  enough to stay one phrase. Words are recursively split the same way into
  alphabetic Word-tier segments; the proxy signal decides, and frequency only
  sets the coverage floor and orders the size cap.
- Phase 5 fallthrough logging: one left-to-right pass over tiers 1-3 records the
  residual stream used by salvage.
- Phase 6 salvage: bottom-up pairwise merges over residual runs, scored by PMI
  rather than raw counts, extended while the merged span stays within the
  configured length.
- Phase 7 reserved tiers: English letters and bigrams, whitespace/punctuation/
  digit/letter atoms, and the 256-byte backstop.

Partially implemented stages, identified explicitly:
- Phase 0 corpus sufficiency (Heaps' law): not implemented.
- Phase 4's proxy model is an in-sample character n-gram, not a held-out or
  neural proxy; co-occurrence embeddings and TF-IDF candidates are not used.
- Phase 8 utilization audit lives in :mod:`spalmer.tokenizer.evaluation`; the
  draft's letter duplication between the language-fallback and atom tiers is
  preserved as drafted.
- Greedy ranked matching only; the Viterbi/DP alternative is not implemented.
- No special tokens; that interface remains open in the ledger.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from dataclasses import dataclass
from datetime import date
from itertools import chain

from .code_regions import analyze_python_code, route_regions, tiers_for_role
from .encoder import build_tier_tables, longest_match_in
from .proxy import ProxyCharModel
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
    """Knobs of the v0 trainer.

    Surprisal ratios compare a span's conditional surprisal (bits per
    character under the proxy model) with the corpus-typical value of the same
    measurement: a word's continuation against typical word cohesion, a phrase's
    continuation against the typical surprisal of the next word. A span at or
    below its ratio is "predictable enough" to stop splitting (draft Phase 4).
    Count floors are the coverage fallback; they never promote a span the proxy
    rejected.
    """

    min_word_count: int = 2
    max_word_tokens: int = 256
    word_surprisal_ratio: float = 1.5
    min_phrase_count: int = 2
    max_phrase_words: int = 3
    max_phrase_tokens: int = 64
    phrase_surprisal_ratio: float = 0.6
    max_salvage_merges: int = 32
    min_salvage_count: int = 2
    max_salvage_length: int = 4
    min_salvage_pmi_bits: float = 0.0
    max_fallback_bigrams: int = 32
    min_fallback_bigram_count: int = 2
    proxy_order: int = 4


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
    identifier_pieces: Counter[str] = Counter()

    for sample in samples:
        if sample.kind == "prose":
            prose_texts.append(sample.text)
        elif sample.kind == "code":
            routed_prose, identifiers = _route_python_code(sample.text)
            prose_texts.extend(routed_prose)
            identifier_pieces.update(identifiers)
        else:
            raise ValueError(f"unknown sample kind: {sample.kind!r}")

    proxy = ProxyCharModel(config.proxy_order).fit(
        chain(prose_texts, identifier_pieces.elements())
    )
    corpus = _collect_corpus(prose_texts, identifier_pieces, config.max_phrase_words)
    word_baseline, phrase_baseline = _baseline_surprisal(corpus, proxy)
    thresholds = _Thresholds(
        word_bits=config.word_surprisal_ratio * word_baseline,
        phrase_bits=config.phrase_surprisal_ratio * phrase_baseline,
    )
    phrases, words = _distributional_segments(corpus, proxy, config, thresholds)

    vocab = Vocab(name)

    for surface in lexer_surfaces():
        vocab.append(Tier.LEXER, surface)

    for phrase in phrases:
        vocab.append(Tier.PHRASE, phrase)

    for word in words:
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


def _route_code_prose(text: str) -> list[str]:
    """Phase 2: comments and docstrings take the English distributional path.

    Ordinary string literals are out-of-distribution material: their contents
    contribute nothing here and are covered at runtime by the salvage,
    language-fallback, atom, and byte tiers.
    """

    prose, _ = _route_python_code(text)
    return prose


def _route_python_code(text: str) -> tuple[list[str], Counter[str]]:
    """Separate Python comments/docstrings from identifiers without regex leakage.

    Tokenization prevents ``#`` inside an ordinary string from becoming a
    comment and keeps all string contents out of the identifier stream.  The
    AST supplies the semantic distinction between a true docstring and an
    arbitrary triple-quoted string expression or assignment.
    """

    analysis = analyze_python_code(text)
    identifiers: Counter[str] = Counter()
    for identifier in analysis.identifiers:
        identifiers.update(split_identifier(identifier))
    return list(analysis.prose_chunks), identifiers


def _word_runs(text: str) -> list[tuple[str, list[str]]]:
    """Runs of words separated by exactly one space, with their left context.

    A phrase token consumes the spaces inside it, so only words that are
    adjacent across a single space can form one. Punctuation, newlines, and
    multiple spaces end a run.
    """

    runs: list[tuple[str, list[str]]] = []
    context = ""
    words: list[str] = []
    previous_end = -1
    for match in WORD_RE.finditer(text):
        if words and text[previous_end : match.start()] == " ":
            words.append(match.group())
        else:
            if words:
                runs.append((context, words))
            context = text[max(0, match.start() - 8) : match.start()]
            words = [match.group()]
        previous_end = match.end()
    if words:
        runs.append((context, words))
    return runs


@dataclass(frozen=True)
class _Thresholds:
    word_bits: float
    phrase_bits: float


@dataclass(frozen=True)
class _Corpus:
    """Word runs with their left context, plus the counts the recursion needs."""

    runs: list[tuple[str, list[str]]]
    word_counts: Counter[str]
    window_counts: Counter[str]


def _collect_corpus(
    prose_texts: list[str], identifier_pieces: Counter[str], max_phrase_words: int
) -> _Corpus:
    runs: list[tuple[str, list[str]]] = []
    word_counts: Counter[str] = Counter(identifier_pieces)
    window_counts: Counter[str] = Counter()
    for text in prose_texts:
        for context, words in _word_runs(text):
            runs.append((context, words))
            word_counts.update(words)
            for size in range(2, max_phrase_words + 1):
                for start in range(len(words) - size + 1):
                    window_counts[" ".join(words[start : start + size])] += 1
    return _Corpus(runs=runs, word_counts=word_counts, window_counts=window_counts)


def _baseline_surprisal(corpus: _Corpus, proxy: ProxyCharModel) -> tuple[float, float]:
    """Typical word cohesion and typical next-word surprisal, in bits per character.

    Both are measured exactly the way the split criteria measure a candidate,
    so a ratio compares a span with its peers rather than with an absolute
    number that would drift with corpus size or repetition. Words are weighted
    by type so a few very frequent words do not define what "typical" means.
    """

    word_bits = 0.0
    word_characters = 0
    for word in corpus.word_counts:
        word_bits += proxy.surprisal(word[1:], word[0])
        word_characters += len(word) - 1

    pair_bits = 0.0
    pair_characters = 0
    for context, words in corpus.runs:
        if not words:
            continue
        history = _advance_context(context, words[0], proxy)
        for word in words[1:]:
            continuation = " " + word
            pair_bits += proxy.surprisal(continuation, history)
            pair_characters += len(continuation)
            history = _advance_context(history, continuation, proxy)
    return (
        word_bits / word_characters if word_characters else 0.0,
        pair_bits / pair_characters if pair_characters else 0.0,
    )


def _distributional_segments(
    corpus: _Corpus,
    proxy: ProxyCharModel,
    config: TrainerConfig,
    thresholds: _Thresholds,
) -> tuple[list[str], list[str]]:
    """Phase 4: proxy-scored recursive splitting into phrase and word tokens."""

    phrase_surprisal: dict[str, float] = {}
    for context, words in corpus.runs:
        for span_context, span in _split_run(context, words, corpus, proxy, config, thresholds):
            if len(span) > 1:
                surface = " ".join(span)
                bits = _continuation_surprisal(span_context, span, proxy)
                phrase_surprisal[surface] = min(bits, phrase_surprisal.get(surface, bits))

    # The proxy decided eligibility above (a phrase leaf survived splitting; a
    # word segment is cohesive enough to stay one token). Frequency only
    # enforces the coverage floor and orders the size cap. Lexer surfaces may
    # also exist here: region routing disables Lexer for prose and keyword
    # boundary checks keep it from pre-empting segments inside identifiers.
    phrases = sorted(
        (-corpus.window_counts[phrase], bits, phrase)
        for phrase, bits in phrase_surprisal.items()
        if all(word.isalpha() for word in phrase.split(" "))
    )
    words = _distributional_word_segments(
        corpus.word_counts,
        proxy,
        config,
        thresholds.word_bits,
    )
    return (
        [phrase for _, _, phrase in phrases[: config.max_phrase_tokens]],
        words,
    )


def _word_cohesion(word: str, proxy: ProxyCharModel) -> float:
    """Bits per character of a word's continuation given its first character.

    Cohesion asks whether the rest of the span follows from its start, which is
    the split question; the unavoidable cost of the first character is not part
    of it.
    """

    return proxy.mean_surprisal(word[1:], word[0])


@dataclass(frozen=True)
class _WordSplitTree:
    word: str
    spans: tuple[tuple[int, int], ...]
    children: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]


def _distributional_word_segments(
    word_counts: Counter[str],
    proxy: ProxyCharModel,
    config: TrainerConfig,
    threshold: float,
) -> list[str]:
    """Recursively retain predictable within-word spans for the Word tier."""

    trees: list[tuple[_WordSplitTree, int]] = []
    segment_counts: Counter[str] = Counter()
    for word, count in word_counts.items():
        if len(word) <= 1:
            continue
        tree = _build_word_split_tree(word, proxy)
        trees.append((tree, count))
        for start, end in tree.spans:
            if end - start > 1:
                segment_counts[word[start:end]] += count

    candidates: dict[str, float] = {}
    for tree, _ in trees:
        stack = [(0, len(tree.word))]
        while stack:
            span = stack.pop()
            start, end = span
            if end - start <= 1:
                continue
            surface = tree.word[start:end]
            cohesion = _word_span_cohesion(tree.word, start, end, proxy)
            if (
                surface.isalpha()
                and segment_counts[surface] >= config.min_word_count
                and cohesion <= threshold
            ):
                candidates[surface] = min(cohesion, candidates.get(surface, cohesion))
                continue
            left, right = tree.children[span]
            stack.append(right)
            stack.append(left)

    ranked = sorted(
        (-segment_counts[surface], cohesion, surface) for surface, cohesion in candidates.items()
    )
    return [surface for _, _, surface in ranked[: config.max_word_tokens]]


def _build_word_split_tree(word: str, proxy: ProxyCharModel) -> _WordSplitTree:
    boundary_surprisal: list[float] = []
    history = word[0]
    for character in word[1:]:
        boundary_surprisal.append(-math.log2(proxy.probability(character, history)))
        history = _advance_context(history, character, proxy)
    maxima = _RangeMaximum(boundary_surprisal)

    spans: list[tuple[int, int]] = []
    children: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] = {}
    stack = [(0, len(word))]
    while stack:
        span = stack.pop()
        spans.append(span)
        start, end = span
        if end - start <= 1:
            continue
        boundary = maxima.query(start, end - 1) + 1
        left = (start, boundary)
        right = (boundary, end)
        children[span] = (left, right)
        stack.append(right)
        stack.append(left)
    return _WordSplitTree(word, tuple(spans), children)


def _word_span_cohesion(word: str, start: int, end: int, proxy: ProxyCharModel) -> float:
    history_start = max(0, start + 1 - (proxy.order - 1))
    history = word[history_start : start + 1]
    return proxy.mean_surprisal(word[start + 1 : end], history)


class _RangeMaximum:
    """Static leftmost-maximum queries used by iterative recursive splitting."""

    def __init__(self, values: list[float]) -> None:
        self.values = values
        size = 1
        while size < len(values):
            size *= 2
        self.size = size
        self.tree = [-1] * (2 * size)
        for index in range(len(values)):
            self.tree[size + index] = index
        for node in range(size - 1, 0, -1):
            self.tree[node] = self._better(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, start: int, stop: int) -> int:
        if not 0 <= start < stop <= len(self.values):
            raise ValueError("maximum-query range must be non-empty and in bounds")
        start += self.size
        stop += self.size
        best = -1
        while start < stop:
            if start & 1:
                best = self._better(best, self.tree[start])
                start += 1
            if stop & 1:
                stop -= 1
                best = self._better(best, self.tree[stop])
            start //= 2
            stop //= 2
        return best

    def _better(self, left: int, right: int) -> int:
        if left < 0:
            return right
        if right < 0:
            return left
        if self.values[left] > self.values[right]:
            return left
        if self.values[right] > self.values[left]:
            return right
        return min(left, right)


def _split_run(
    context: str,
    words: list[str],
    corpus: _Corpus,
    proxy: ProxyCharModel,
    config: TrainerConfig,
    thresholds: _Thresholds,
) -> list[tuple[str, list[str]]]:
    """Split one run top-down until every span is predictable enough to keep.

    A span stops splitting when the proxy finds its continuation predictable
    and it recurs often enough to be worth a vocabulary entry (the coverage
    floor); a predictable span seen only once keeps splitting so that the
    phrases inside it can still be found.
    """

    if len(words) <= 1:
        return [(context, words)]

    contexts = [_context_tail(context, proxy)]
    history = _advance_context(contexts[0], words[0], proxy)
    boundary_surprisal: list[float] = []
    for word in words[1:]:
        continuation = " " + word
        boundary_surprisal.append(proxy.mean_surprisal(continuation, history))
        contexts.append(_advance_context(history, " ", proxy))
        history = _advance_context(history, continuation, proxy)
    maxima = _RangeMaximum(boundary_surprisal)

    leaves: list[tuple[int, str, list[str]]] = []
    stack = [(0, len(words))]
    while stack:
        start, end = stack.pop()
        span = words[start:end]
        span_context = contexts[start]
        if end - start == 1:
            leaves.append((start, span_context, span))
            continue
        surface = " ".join(span) if end - start <= config.max_phrase_words else ""
        if (
            surface
            and corpus.window_counts[surface] >= config.min_phrase_count
            and _continuation_surprisal(span_context, span, proxy) <= thresholds.phrase_bits
        ):
            leaves.append((start, span_context, span))
            continue
        boundary = maxima.query(start, end - 1) + 1
        stack.append((boundary, end))
        stack.append((start, boundary))
    leaves.sort(key=lambda item: item[0])
    return [(span_context, span) for _, span_context, span in leaves]


def _context_tail(context: str, proxy: ProxyCharModel) -> str:
    width = proxy.order - 1
    return context[-width:] if width else ""


def _advance_context(context: str, text: str, proxy: ProxyCharModel) -> str:
    return _context_tail(context + text, proxy)


def _continuation_surprisal(context: str, words: list[str], proxy: ProxyCharModel) -> float:
    """Bits per character of everything after the first word, given the prefix."""

    return proxy.mean_surprisal(" " + " ".join(words[1:]), context + words[0])


def _learn_salvage(vocab: Vocab, samples: list[Sample], config: TrainerConfig) -> None:
    """Phase 5/6: log residual runs once, then merge them bottom-up by PMI."""

    tiers = (Tier.LEXER, Tier.PHRASE, Tier.WORD)
    tables, max_lens = build_tier_tables(vocab, tiers)
    runs: list[list[str]] = []
    for sample in samples:
        runs.extend(_residual_runs(tables, max_lens, tiers, sample.text, kind=sample.kind))
    for _ in range(config.max_salvage_merges):
        chosen = _best_salvage_pair(runs, config)
        if chosen is None:
            break
        runs = [_merge_pair(pieces, chosen) for pieces in runs]
        surface = chosen[0] + chosen[1]
        if vocab.find(Tier.SALVAGE, surface) is None:
            vocab.append(Tier.SALVAGE, surface)


def _best_salvage_pair(runs: list[list[str]], config: TrainerConfig) -> tuple[str, str] | None:
    """Highest-PMI adjacent pair above the count floor, or ``None``.

    ``pmi(l, r) = log2(p(l, r) / (p(l) p(r)))`` over adjacent residual pieces.
    Raw counts only break ties and enforce the coverage floor, since counting
    alone over-merges boundary artifacts (draft Phase 6).
    """

    pair_counts: Counter[tuple[str, str]] = Counter()
    left_counts: Counter[str] = Counter()
    right_counts: Counter[str] = Counter()
    for pieces in runs:
        for left, right in zip(pieces, pieces[1:]):
            pair_counts[(left, right)] += 1
            left_counts[left] += 1
            right_counts[right] += 1
    total_pairs = sum(pair_counts.values())
    if not total_pairs:
        return None

    candidates: list[tuple[float, int, tuple[str, str]]] = []
    for pair, count in pair_counts.items():
        left, right = pair
        if (
            count < config.min_salvage_count
            or len(left) + len(right) > config.max_salvage_length
        ):
            continue
        joint = count / total_pairs
        marginal = (left_counts[left] / total_pairs) * (
            right_counts[right] / total_pairs
        )
        pmi = math.log2(joint / marginal)
        if pmi >= config.min_salvage_pmi_bits:
            candidates.append((-pmi, -count, pair))
    if not candidates:
        return None
    return min(candidates)[2]


def _residual_runs(tables, max_lens, tiers, text: str, *, kind: str) -> list[list[str]]:
    runs: list[list[str]] = []
    for region in route_regions(text, kind):
        region_text = text[region.start : region.end]
        region_tiers = tiers_for_role(tiers, region.role)
        current: list[str] = []
        pos = 0
        while pos < len(region_text):
            match = longest_match_in(tables, max_lens, region_text, pos, region_tiers)
            if match is None:
                character = region_text[pos]
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
