"""Checks for the proxy-scored construction stages of the C01 draft."""

from __future__ import annotations

from helpers import build_demo_vocab, demo_config

from spalmer.tokenizer import (
    Encoder,
    ProxyCharModel,
    Sample,
    Tier,
    TrainerConfig,
    evaluate_vocab,
    train,
)
from spalmer.tokenizer.code_regions import (
    _mark_legacy_fstring_fields,
    analyze_python_code,
)


def _surfaces(vocab, tier: Tier) -> list[str]:
    return [entry.surface for entry in vocab.entries if entry.tier is tier]


def test_legacy_fstring_format_text_stays_out_of_code_routing():
    content = "{value:for} {other!r:>10} {nested:{width}}"
    roles = ["fallback"] * len(content)

    _mark_legacy_fstring_fields(roles, content, 0)

    for literal in ("for", ">10"):
        start = content.index(literal)
        assert set(roles[start : start + len(literal)]) == {"fallback"}
    for expression in ("value", "other", "nested", "width"):
        start = content.index(expression)
        assert set(roles[start : start + len(expression)]) == {"code"}


def test_proxy_surprisal_is_lower_for_seen_continuations():
    proxy = ProxyCharModel(order=3).fit(["abcabcabc"])
    assert proxy.mean_surprisal("bc", "a") < proxy.mean_surprisal("xq", "a")
    assert proxy.probability("z", "ab") > 0


def test_phrase_selection_follows_the_proxy_not_the_count():
    # "alpha beta" and "gamma delta" both occur twice, but gamma is followed by
    # many different words, so only the predictable continuation is a phrase.
    text = (
        "alpha beta. alpha beta. gamma delta. gamma delta. "
        "gamma eta. gamma zeta. gamma iota. gamma kappa. gamma mu. gamma nu."
    )
    vocab = train([Sample(text)], demo_config(), created="2026-09-02")
    phrases = _surfaces(vocab, Tier.PHRASE)
    assert "alpha beta" in phrases
    assert "gamma delta" not in phrases


def test_word_tier_can_keep_proxy_scored_segments_instead_of_whole_words():
    text = "unhappy undoing untold remake redo replay " * 4
    vocab = train(
        [Sample(text)],
        TrainerConfig(min_phrase_count=999, max_salvage_merges=0),
        created="2026-09-02",
    )
    words = _surfaces(vocab, Tier.WORD)
    assert "redo" not in words
    assert "re" in words and "do" in words


def test_lexer_words_remain_available_to_prose_and_identifier_segments():
    text = "in the house and in the garden " * 8
    vocab = train([Sample(text)], demo_config(), created="2026-09-02")
    encoder = Encoder(vocab)

    prose = encoder.encode_with_tiers("in the house and", kind="prose")
    assert all(tier is not Tier.LEXER for _, tier in prose)
    assert any(
        vocab.get(token_id).surface.startswith("in")
        and tier in {Tier.PHRASE, Tier.WORD}
        for token_id, tier in prose
    )
    code = encoder.encode_with_tiers("in values", kind="code")
    assert vocab.get(code[0][0]).surface == "in" and code[0][1] is Tier.LEXER


def test_word_and_phrase_tiers_do_not_merge_atomic_digits_or_apostrophes():
    samples = [
        Sample("don't won't don't won't " * 4),
        Sample("var123 = var123\n" * 4, "code"),
    ]
    vocab = train(samples, demo_config(), created="2026-09-02")
    for tier in (Tier.PHRASE, Tier.WORD):
        assert all(
            all(character.isalpha() or character == " " for character in entry.surface)
            for entry in vocab.entries
            if entry.tier is tier
        )
    encoder = Encoder(vocab)
    contraction = encoder.encode_with_tiers("don't", kind="prose")
    digits = encoder.encode_with_tiers("var123", kind="code")
    assert any(
        vocab.get(token_id).surface == "'" and tier is Tier.ATOM
        for token_id, tier in contraction
    )
    assert all(
        tier is Tier.ATOM
        for token_id, tier in digits
        if vocab.get(token_id).surface.isdigit()
    )


def test_long_word_run_uses_iterative_splitting_without_recursion_failure():
    text = " ".join(["a"] * 1_050)
    vocab = train(
        [Sample(text)],
        TrainerConfig(max_salvage_merges=0),
        created="2026-09-02",
    )
    assert len(vocab) > 256


def test_salvage_merges_are_ranked_by_pmi_rather_than_raw_count():
    # Every word occurs once, so all letters reach the residual stream. The
    # pair (a, b) is the most frequent but "a" starts many other words; (q, z)
    # occurs only together, so PMI ranks it first.
    text = "ab abx aby ac ad ae af ag ah ai aj ak al am an qzy qzw"
    vocab = train(
        [Sample(text)],
        TrainerConfig(
            min_salvage_count=2,
            max_salvage_merges=2,
            word_surprisal_ratio=0.0,
        ),
        created="2026-09-02",
    )
    salvage = _surfaces(vocab, Tier.SALVAGE)
    assert salvage[0] == "qz"


def test_string_literal_contents_take_the_fallback_path_but_docstrings_do_not():
    code = (
        '"""module guidance words module guidance words"""\n'
        'payload = """ordinary payload words ordinary payload words"""\n'
        "def f():\n"
        '    """helpful docstring words helpful docstring words"""\n'
        '    marker = "# fake comment words fake comment words"\n'
        "    # genuine comment words genuine comment words\n"
        '    return "secret literal words secret literal words"\n'
    )
    vocab = train([Sample(code, "code")], demo_config(), created="2026-09-02")
    words = _surfaces(vocab, Tier.WORD)
    assert "helpful" in words and "docstring" in words
    assert "guidance" in words and "genuine" in words
    assert "secret" not in words and "literal" not in words
    assert "payload" not in words and "fake" not in words
    # Delimiters stay atomic and the literal still round-trips.
    encoder = Encoder(vocab)
    assert encoder.decode(encoder.encode(code)) == code


def test_implicitly_concatenated_docstrings_route_every_literal_as_prose():
    code = 'def f():\n    "alpha words " "beta words"\n'
    analysis = analyze_python_code(code)
    assert analysis.prose_chunks == ("alpha words beta words",)
    for word in ("alpha", "beta"):
        position = code.index(word)
        assert any(
            region.start <= position < region.end and region.role == "prose"
            for region in analysis.regions
        )


def test_evaluation_reports_utilization_compression_and_singletons():
    vocab = build_demo_vocab()
    report = evaluate_vocab(vocab, "the cat sat on the mat. the dog saw the cat.")
    assert report.token_count > 0
    assert report.bits_per_character > 0
    assert 0 < report.utilization < 1
    assert len(report.dead_token_ids) + len(report.firings) == len(vocab)
    assert report.singleton_count <= report.token_count
    assert report.tier_firings[Tier.PHRASE] >= 1
