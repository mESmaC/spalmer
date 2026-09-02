from __future__ import annotations

import pytest
from helpers import build_demo_vocab, precedence_vocab, within_tier_vocab

from spalmer.tokenizer import Encoder, Tier, Vocab


def test_encode_is_deterministic(tmp_path):
    vocab = build_demo_vocab()
    text = "the cat sat on the mat; def fib(n): return n // 2"
    first = Encoder(vocab).encode(text)
    again = Encoder(vocab).encode(text)
    assert first == again
    path = tmp_path / "vocab.json"
    vocab.save(path)
    reloaded = Vocab.load(path)
    assert Encoder(reloaded).encode(text) == first


def test_higher_tier_beats_longer_lower_tier_match():
    vocab = precedence_vocab()
    encoder = Encoder(vocab)
    result = encoder.encode_with_tiers("==x")
    assert [vocab.get(t).surface for t, _ in result] == ["==", "x"]
    assert result[0][1] is Tier.LEXER
    assert result[1][1] is Tier.ATOM


def test_longest_match_wins_within_a_tier():
    vocab = within_tier_vocab()
    encoder = Encoder(vocab)
    assert [vocab.get(t).surface for t, _ in encoder.encode_with_tiers("abc")] == ["abc"]
    assert [vocab.get(t).surface for t, _ in encoder.encode_with_tiers("aba")] == ["ab", "a"]
    assert [vocab.get(t).surface for t, _ in encoder.encode_with_tiers("ac")] == ["a", "c"]


def test_phrase_token_consumes_internal_space():
    vocab = build_demo_vocab()
    encoder = Encoder(vocab)
    phrases = [entry.surface for entry in vocab.entries if entry.tier is Tier.PHRASE]
    assert phrases and all(" " in phrase for phrase in phrases)
    for phrase in phrases:
        token_id, tier = encoder.encode_with_tiers(phrase + " zzz", kind="prose")[0]
        assert tier is Tier.PHRASE
        assert vocab.get(token_id).surface == phrase


def test_multi_char_operator_outmatches_its_prefix():
    vocab = build_demo_vocab()
    encoder = Encoder(vocab)
    result = encoder.encode_with_tiers("a == b")
    surfaces = [(vocab.get(t).surface, tier) for t, tier in result]
    assert ("==", Tier.LEXER) in surfaces


@pytest.mark.parametrize(
    "text",
    [
        "inside format island ordinary assertion notable android",
        "x_in in_value for2 2for className returnValue",
    ],
)
def test_identifier_like_lexer_tokens_do_not_match_inside_words(text):
    vocab = build_demo_vocab()
    encoder = Encoder(vocab)
    result = encoder.encode_with_tiers(text)

    identifier_lexer_matches = [
        vocab.get(token_id).surface
        for token_id, tier in result
        if tier is Tier.LEXER and vocab.get(token_id).surface.isidentifier()
    ]
    assert identifier_lexer_matches == []
    assert encoder.decode(token_id for token_id, _ in result) == text


def test_identifier_like_lexer_tokens_still_match_at_boundaries():
    vocab = build_demo_vocab()
    result = Encoder(vocab).encode_with_tiers("if x in items: for item in items: return item")
    keyword_matches = [
        vocab.get(token_id).surface
        for token_id, tier in result
        if tier is Tier.LEXER and vocab.get(token_id).surface.isidentifier()
    ]
    assert keyword_matches == ["if", "in", "for", "in", "return"]


def test_code_routing_keeps_lexer_tokens_out_of_literals_and_english_regions():
    vocab = build_demo_vocab()
    encoder = Encoder(vocab)
    source = (
        '"""return and == are prose here"""\n'
        'value = "return == literal"\n'
        "# return == comment\n"
        "return value == other\n"
    )
    encoded = encoder.encode_with_tiers(source, kind="code")
    surfaces = [(vocab.get(token_id).surface, tier) for token_id, tier in encoded]

    lexer_surfaces = [surface for surface, tier in surfaces if tier is Tier.LEXER]
    assert lexer_surfaces == ["=", "return", "=="]
    quote_tiers = [tier for surface, tier in surfaces if surface == '"']
    assert quote_tiers and all(tier is Tier.ATOM for tier in quote_tiers)
    assert encoder.decode(token_id for token_id, _ in encoded) == source
