from __future__ import annotations

import pytest
from helpers import build_demo_vocab, within_tier_vocab

from spalmer.tokenizer import Encoder

ROUND_TRIP_CASES = [
    "",
    "the cat sat on the mat.",
    "Hello, world! 123 456",
    "def fib(n):\n    return n\n",
    "a == b and c != d  # comment",
    "h\u00e9llo w\u00f6rld \u2014 na\u00efve caf\u00e9",
    "emoji \U0001f600 \U0001f4a5",
    "\u4e2d\u6587\u6d4b\u8bd5",
    "\t\r\n\x0b\x0c spacing",
    "\x00\x01\x02 control",
    "unregistered zzz qqq xxx junk",
]


@pytest.mark.parametrize("text", ROUND_TRIP_CASES)
def test_roundtrip_on_trained_vocab(text):
    encoder = Encoder(build_demo_vocab())
    assert encoder.decode(encoder.encode(text)) == text


def test_roundtrip_on_tiny_vocab():
    encoder = Encoder(within_tier_vocab())
    text = "ab cab 123 \u00e9"
    assert encoder.decode(encoder.encode(text)) == text


def test_empty_roundtrip():
    encoder = Encoder(within_tier_vocab())
    assert encoder.encode("") == []
    assert encoder.decode([]) == ""
