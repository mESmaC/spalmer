from __future__ import annotations

from spalmer.tokenizer import Sample, Tier, TrainerConfig, Vocab, append_byte_backstop, train

PROSE_SAMPLES = (
    "the cat sat on the mat. the cat saw the dog. the dog saw the cat.",
    "of the cat and of the dog, the mat of the cat sat in the sun.",
    "the quick brown fox jumps over the lazy dog. the lazy dog sleeps in the sun.",
)

CODE_SAMPLE = """def fib(n):
    # compute the fibonacci number iteratively
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


class PointXY:
    \"\"\"a point with x and y coordinates\"\"\"

    def __init__(self, x_value, y_value):
        self.xValue = x_value
        self.y_value = y_value
"""


def demo_config() -> TrainerConfig:
    return TrainerConfig(
        min_word_count=2,
        min_phrase_count=2,
        min_salvage_count=2,
        min_fallback_bigram_count=2,
    )


def demo_samples() -> list[Sample]:
    samples = [Sample(text, "prose") for text in PROSE_SAMPLES]
    samples.append(Sample(CODE_SAMPLE, "code"))
    return samples


def build_demo_vocab() -> Vocab:
    return train(demo_samples(), demo_config(), created="2026-09-01")


def precedence_vocab() -> Vocab:
    vocab = Vocab("precedence")
    vocab.append(Tier.LEXER, "==")
    vocab.append(Tier.WORD, "==x")
    vocab.append(Tier.ATOM, "=")
    vocab.append(Tier.ATOM, "x")
    append_byte_backstop(vocab)
    return vocab


def within_tier_vocab() -> Vocab:
    vocab = Vocab("within-tier")
    vocab.append(Tier.WORD, "a")
    vocab.append(Tier.WORD, "ab")
    vocab.append(Tier.WORD, "abc")
    vocab.append(Tier.ATOM, "a")
    vocab.append(Tier.ATOM, "b")
    vocab.append(Tier.ATOM, "c")
    append_byte_backstop(vocab)
    return vocab
