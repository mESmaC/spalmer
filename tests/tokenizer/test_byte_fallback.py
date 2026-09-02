from __future__ import annotations

from helpers import build_demo_vocab, precedence_vocab

from spalmer.tokenizer import Encoder, Tier


def test_byte_backstop_is_complete():
    vocab = build_demo_vocab()
    byte_entries = [entry for entry in vocab.entries if entry.tier is Tier.BYTE]
    assert len(byte_entries) == 256
    for entry in byte_entries:
        assert entry.is_byte
        assert entry.payload() == bytes([entry.byte])


def test_unclaimed_text_falls_to_byte_tier():
    encoder = Encoder(precedence_vocab())
    result = encoder.encode_with_tiers("\u00e9\u4e16")
    assert result
    assert all(tier is Tier.BYTE for _, tier in result)
    assert encoder.decode([token_id for token_id, _ in result]) == "\u00e9\u4e16"


def test_byte_fallback_mixes_with_higher_tiers():
    encoder = Encoder(precedence_vocab())
    result = encoder.encode_with_tiers("==\u00e9")
    tiers = [tier for _, tier in result]
    assert tiers[0] is Tier.LEXER
    assert tiers[1:] and all(tier is Tier.BYTE for tier in tiers[1:])


def test_every_byte_value_round_trips_through_payloads():
    vocab = build_demo_vocab()
    reconstructed = bytes(
        next(e for e in vocab.entries if e.tier is Tier.BYTE and e.byte == value).byte
        for value in range(256)
    )
    assert reconstructed == bytes(range(256))
