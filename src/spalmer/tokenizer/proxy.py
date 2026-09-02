"""Character n-gram proxy model supplying conditional surprisal (C01 Phase 4/6).

The tokenizer draft scores splits and salvage merges with "conditional
surprisal under a proxy model" and lists n-gram models as the proxy in its
statistical toolbox. This is the easiest faithful proxy: an order-``n``
character model with Witten-Bell interpolation down to a uniform floor, which
needs no tuned smoothing constant and behaves sensibly on both tiny and
repetitive corpora. It is a construction-time instrument only; nothing about
it is stored in the vocabulary or needed at runtime.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable


class ProxyCharModel:
    """Order-``n`` character model returning surprisal in bits.

    Args:
        order: Maximum context length plus one (``order=4`` conditions each
            character on up to three preceding characters).
    """

    def __init__(self, order: int = 4) -> None:
        if order < 1:
            raise ValueError("order must be at least 1")
        self.order = order
        self._continuations: dict[str, Counter[str]] = defaultdict(Counter)
        self._alphabet: set[str] = set()

    def fit(self, texts: Iterable[str]) -> ProxyCharModel:
        for text in texts:
            for position, character in enumerate(text):
                self._alphabet.add(character)
                start = max(0, position - (self.order - 1))
                for begin in range(start, position + 1):
                    self._continuations[text[begin:position]][character] += 1
        return self

    @property
    def alphabet_size(self) -> int:
        # One extra slot keeps unseen characters at a finite surprisal.
        return len(self._alphabet) + 1

    def probability(self, character: str, context: str) -> float:
        """Witten-Bell interpolated ``p(character | context)``."""

        history = context[max(0, len(context) - (self.order - 1)) :]
        probability = 1.0 / self.alphabet_size
        for length in range(len(history) + 1):
            counts = self._continuations.get(history[len(history) - length :] if length else "")
            if not counts:
                continue
            total = sum(counts.values())
            types = len(counts)
            probability = (counts.get(character, 0) + types * probability) / (total + types)
        return probability

    def surprisal(self, text: str, context: str = "") -> float:
        """Total conditional surprisal of ``text`` in bits given ``context``."""

        bits = 0.0
        history = context
        for character in text:
            bits -= math.log2(self.probability(character, history))
            history = (history + character)[-(self.order - 1) :] if self.order > 1 else ""
        return bits

    def mean_surprisal(self, text: str, context: str = "") -> float:
        """Bits per character of ``text`` given ``context`` (``0.0`` if empty)."""

        if not text:
            return 0.0
        return self.surprisal(text, context) / len(text)


__all__ = ["ProxyCharModel"]
