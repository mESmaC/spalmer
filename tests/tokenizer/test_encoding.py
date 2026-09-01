from __future__ import annotations

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
    result = encoder.encode_with_tiers("abc")
    assert [vocab.get(t).surface for t, _ in result] == ["ab", "c"]
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
    token_id, tier = encoder.encode_with_tiers("the cat saw")[0]
    assert tier is Tier.PHRASE
    assert vocab.get(token_id).surface == "the cat"


def test_multi_char_operator_outmatches_its_prefix():
    vocab = build_demo_vocab()
    encoder = Encoder(vocab)
    result = encoder.encode_with_tiers("a == b")
    surfaces = [(vocab.get(t).surface, tier) for t, tier in result]
    assert ("==", Tier.LEXER) in surfaces
