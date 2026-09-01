from __future__ import annotations

import pytest
import torch

from spalmer.config import PLEConfig
from spalmer.embeddings import AlternatingPLE, PLELayerEmbedding, fake_quantize_low_bit


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
