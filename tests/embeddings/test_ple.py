from __future__ import annotations

import math

import pytest
import torch

from spalmer.config import PLEConfig
from spalmer.embeddings import (
    AlternatingPLE,
    PLELayerEmbedding,
    fake_quantize_low_bit,
    qr_codebook_rows,
    qr_lane_moduli,
)


def test_fixed_alternation_owns_one_table_per_layer() -> None:
    config = PLEConfig(vocab_size=23, d_model=8, n_layers=4, expansion_factor=3)
    embeddings = AlternatingPLE(config)

    assert [embeddings.phase_for_layer(index) for index in range(4)] == ["A", "B", "A", "B"]
    assert len({id(layer.weight) for layer in embeddings.layers}) == config.n_layers

    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    assert embeddings(token_ids, layer_index=2).shape == (2, 3, config.d_model)


def test_eval_rounding_is_deterministic_and_training_keeps_gradients() -> None:
    config = PLEConfig(vocab_size=17, d_model=6, n_layers=2, expansion_factor=2)
    layer = PLELayerEmbedding(config, layer_index=0)
    token_ids = torch.tensor([[1, 2, 1]])

    layer.eval()
    first = layer(token_ids)
    second = layer(token_ids)
    torch.testing.assert_close(first, second)

    layer.train()
    layer(token_ids).square().mean().backward()
    assert layer.weight.grad is not None
    assert layer.lane_logits.grad is not None
    assert layer.gate.grad is not None


def test_repeated_tokens_share_one_stochastic_rounding_sample_per_forward() -> None:
    config = PLEConfig(vocab_size=17, d_model=6, n_layers=2, expansion_factor=2)
    layer = PLELayerEmbedding(config, layer_index=0).train()

    output = layer(torch.tensor([[3, 3, 3, 3]]))

    torch.testing.assert_close(output[:, :1].expand_as(output), output, rtol=0, atol=0)


def test_fake_quantization_uses_a_straight_through_gradient() -> None:
    values = torch.tensor([[0.12, -0.37, 0.91]], requires_grad=True)
    quantized = fake_quantize_low_bit(
        values,
        bits=4,
        stochastic=False,
        straight_through=True,
    )
    quantized.sum().backward()

    assert not torch.equal(values.detach(), quantized.detach())
    torch.testing.assert_close(values.grad, torch.ones_like(values))


def test_ple_config_rejects_non_low_bit_values() -> None:
    with pytest.raises(ValueError, match="quant_bits"):
        PLEConfig(vocab_size=8, d_model=4, n_layers=2, quant_bits=16)


def test_reference_backend_guards_accidental_large_shadow_allocation() -> None:
    config = PLEConfig(
        vocab_size=32,
        d_model=16,
        n_layers=4,
        expansion_factor=2,
        reference_max_numel=1_000,
    )

    with pytest.raises(MemoryError, match="fake-QAT PLE backend"):
        AlternatingPLE(config)


def test_sparse_lookup_mode_produces_sparse_table_gradients() -> None:
    config = PLEConfig(
        vocab_size=17,
        d_model=6,
        n_layers=1,
        expansion_factor=2,
        sparse_gradients=True,
    )
    layer = PLELayerEmbedding(config, layer_index=0).train()

    layer(torch.tensor([[1, 2, 1]])).sum().backward()

    assert layer.weight.grad is not None and layer.weight.grad.is_sparse


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


@pytest.mark.parametrize("backend", ["fake_qat", "qr"])
def test_lane_controls_do_not_promote_embedding_dtype(backend: str) -> None:
    config = PLEConfig(
        vocab_size=17,
        d_model=4,
        n_layers=2,
        expansion_factor=2,
        backend=backend,
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


def test_qr_backend_never_calls_fake_quantization_or_shadow_size_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PLEConfig(
        vocab_size=32,
        d_model=8,
        n_layers=3,
        expansion_factor=2,
        backend="qr",
        reference_max_numel=1,
    )

    def fail_fake_quant(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("QR-PLE must not enter the fake-quantization path")

    monkeypatch.setattr("spalmer.embeddings.ple.fake_quantize_low_bit", fail_fake_quant)
    embeddings = AlternatingPLE(config)

    token_ids = torch.tensor([[1, 2, 3]])
    assert embeddings(token_ids, layer_index=0).shape == (1, 3, config.d_model)
    assert embeddings(token_ids, layer_index=2).shape == (1, 3, config.d_model)
    with pytest.raises(ValueError, match="no A/B phase"):
        embeddings.phase_for_layer(0)


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
