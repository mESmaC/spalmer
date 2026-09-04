from __future__ import annotations

import math

import pytest
import torch

from spalmer.config import PLEConfig
from spalmer.embeddings import (
    AlternatingPLE,
    PLELayerEmbedding,
    qr_codebook_rows,
    qr_lane_moduli,
)


def test_qr_moduli_are_deterministic_distinct_primes_above_sqrt_vocab() -> None:
    assert qr_lane_moduli(8_192, 4) == (97, 101, 103, 107)
    assert qr_lane_moduli(8_192, 4) == qr_lane_moduli(8_192, 4)

    moduli = qr_lane_moduli(64, 3)
    assert moduli == (11, 13, 17)
    assert qr_codebook_rows(64, 3) == (41, 15)
    assert all(math.gcd(left, right) == 1 for left in moduli for right in moduli if left != right)


def test_qr_layer_zero_is_an_exact_unmixed_input_embedding() -> None:
    config = PLEConfig(
        vocab_size=23,
        d_model=8,
        n_layers=4,
        expansion_factor=3,
        backend="qr",
    )
    layer = PLELayerEmbedding(config, layer_index=0)

    assert isinstance(layer.input_embedding, torch.nn.Embedding)
    assert layer.input_embedding.weight.shape == (config.vocab_size, config.d_model)
    assert not hasattr(layer, "lane_logits")
    assert not hasattr(layer, "gate")
    assert not hasattr(layer, "remainder_embedding")

    token_ids = torch.tensor([[1, 2, 1]])
    expected = layer.input_embedding.weight[token_ids]
    torch.testing.assert_close(layer(token_ids), expected)
    layer(token_ids).square().mean().backward()
    assert layer.input_embedding.weight.grad is not None
    assert not layer.input_embedding.weight.grad.is_sparse


def test_qr_refresh_uses_complementary_product_lanes_and_dense_gradients() -> None:
    config = PLEConfig(
        vocab_size=64,
        d_model=4,
        n_layers=3,
        expansion_factor=3,
        backend="qr",
    )
    layer = PLELayerEmbedding(config, layer_index=1)

    assert layer.moduli.tolist() == [11, 13, 17]
    assert layer.remainder_offsets.tolist() == [0, 11, 24]
    assert layer.quotient_offsets.tolist() == [0, 6, 11]
    assert layer.remainder_embedding.num_embeddings == 41
    assert layer.quotient_embedding.num_embeddings == 15
    assert not hasattr(layer, "weight")

    token_ids = torch.tensor([[0, 12, 31], [63, 31, 4]])
    output = layer(token_ids)
    assert output.shape == (*token_ids.shape, config.d_model)
    torch.testing.assert_close(output[0, 2], output[1, 1], rtol=0, atol=0)

    output.square().mean().backward()
    for table in (layer.remainder_embedding, layer.quotient_embedding):
        assert table.weight.grad is not None
        assert not table.weight.grad.is_sparse
    assert layer.lane_logits.grad is not None
    assert layer.gate.grad is not None


def test_qr_composition_matches_product_softmax_and_gate() -> None:
    config = PLEConfig(
        vocab_size=17,
        d_model=2,
        n_layers=2,
        expansion_factor=2,
        gate_init=0.5,
        backend="qr",
    )
    layer = PLELayerEmbedding(config, layer_index=1)
    with torch.no_grad():
        layer.lane_logits.copy_(torch.tensor([0.0, math.log(3.0)]))
        layer.remainder_embedding.weight[: layer.remainder_offsets[1]].fill_(1.0)
        layer.remainder_embedding.weight[layer.remainder_offsets[1] :].fill_(2.0)
        layer.quotient_embedding.weight[: layer.quotient_offsets[1]].fill_(2.0)
        layer.quotient_embedding.weight[layer.quotient_offsets[1] :].fill_(3.0)

    # Lane products are 2 and 6; softmax weights are 1/4 and 3/4; gate is 1/2.
    expected = torch.full((1, 3, 2), 2.5)
    torch.testing.assert_close(layer(torch.tensor([[0, 8, 16]])), expected)


def test_lane_controls_do_not_promote_embedding_dtype() -> None:
    config = PLEConfig(
        vocab_size=17,
        d_model=4,
        n_layers=2,
        expansion_factor=2,
    )
    layer = PLELayerEmbedding(config, layer_index=1).to(dtype=torch.bfloat16)
    # CUDA softmax may retain these tiny stability-sensitive controls in FP32.
    layer.lane_logits = torch.nn.Parameter(layer.lane_logits.float())

    output = layer(torch.tensor([[1, 2, 3]]))

    assert output.dtype == torch.bfloat16
    output.float().square().mean().backward()
    assert layer.lane_logits.grad is not None
    assert layer.lane_logits.grad.dtype == torch.float32


def test_qr_refresh_uses_two_flat_embedding_gathers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PLEConfig(
        vocab_size=257,
        d_model=8,
        n_layers=2,
        expansion_factor=4,
        backend="qr",
    )
    layer = PLELayerEmbedding(config, layer_index=1)
    real_embedding = torch.nn.functional.embedding
    calls = 0

    def count_embedding(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return real_embedding(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "embedding", count_embedding)
    layer(torch.tensor([[1, 2, 256]]))

    assert calls == 2


def test_qr_bank_uses_only_exact_and_compositional_tables() -> None:
    config = PLEConfig(
        vocab_size=32,
        d_model=8,
        n_layers=3,
        expansion_factor=2,
    )
    embeddings = AlternatingPLE(config)

    token_ids = torch.tensor([[1, 2, 3]])
    assert embeddings(token_ids, layer_index=0).shape == (1, 3, config.d_model)
    assert embeddings(token_ids, layer_index=2).shape == (1, 3, config.d_model)
    assert hasattr(embeddings.layers[0], "input_embedding")
    assert hasattr(embeddings.layers[2], "remainder_embedding")
    assert hasattr(embeddings.layers[2], "quotient_embedding")


def test_qr_indices_are_computed_once_and_reused_across_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PLEConfig(
        vocab_size=257,
        d_model=8,
        n_layers=4,
        expansion_factor=4,
        backend="qr",
    )
    embeddings = AlternatingPLE(config)
    token_ids = torch.tensor([[1, 2, 256]])
    real_div = torch.div
    divisions = 0

    def count_div(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal divisions
        divisions += 1
        return real_div(*args, **kwargs)

    monkeypatch.setattr(torch, "div", count_div)
    qr_indices = embeddings.prepare_qr_indices(token_ids)
    assert divisions == 1

    reused = [
        embeddings(token_ids, layer_index=index, qr_indices=qr_indices)
        for index in range(1, config.n_layers)
    ]
    assert divisions == 1
    torch.testing.assert_close(
        reused[0],
        embeddings(token_ids, layer_index=1),
    )
    assert divisions == 2
