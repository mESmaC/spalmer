"""Vocabulary evaluation (C01 draft Phase 8 and section 7).

The draft evaluates a vocabulary by held-out utilization (flagging dead
tokens), compression in bits per character, token-stream entropy, and the
singleton count; downstream LM loss stays with the model (ledger C15).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from .code_regions import InputKind
from .encoder import Encoder
from .tiers import Tier
from .vocab import Vocab


@dataclass(frozen=True)
class VocabEvaluation:
    token_count: int
    character_count: int
    bits_per_character: float
    token_stream_entropy_bits: float
    singleton_count: int
    dead_token_ids: tuple[int, ...]
    firings: dict[int, int] = field(default_factory=dict)
    tier_firings: dict[Tier, int] = field(default_factory=dict)

    @property
    def utilization(self) -> float:
        """Fraction of vocabulary entries that fired at least once."""

        total = len(self.dead_token_ids) + len(self.firings)
        return len(self.firings) / total if total else 0.0


def evaluate_vocab(
    vocab: Vocab, held_out_text: str, *, kind: InputKind = "mixed"
) -> VocabEvaluation:
    """Encode ``held_out_text`` and report the draft's evaluation metrics.

    ``bits_per_character`` charges every token ``log2(len(vocab))`` bits, the
    language-independent compression baseline of section 7; the token-stream
    entropy is the empirical entropy of the emitted token ids.
    """

    encoder = Encoder(vocab)
    encoded = encoder.encode_with_tiers(held_out_text, kind=kind)
    firings: Counter[int] = Counter(token_id for token_id, _ in encoded)
    tier_firings: Counter[Tier] = Counter(tier for _, tier in encoded)
    token_count = len(encoded)
    character_count = len(held_out_text)

    entropy = 0.0
    for count in firings.values():
        probability = count / token_count
        entropy -= probability * math.log2(probability)
    bits = token_count * math.log2(len(vocab)) if len(vocab) > 1 else 0.0
    dead = tuple(entry.token_id for entry in vocab.entries if entry.token_id not in firings)
    return VocabEvaluation(
        token_count=token_count,
        character_count=character_count,
        bits_per_character=bits / character_count if character_count else 0.0,
        token_stream_entropy_bits=entropy,
        singleton_count=sum(1 for count in firings.values() if count == 1),
        dead_token_ids=dead,
        firings=dict(firings),
        tier_firings=dict(tier_firings),
    )


__all__ = ["VocabEvaluation", "evaluate_vocab"]
