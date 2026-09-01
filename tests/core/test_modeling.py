from __future__ import annotations

import torch
from torch import Tensor, nn

from spalmer import (
    ChannelMixerOutput,
    SPALMERBackbone,
    SPALMERBlock,
    SPALMERCausalLM,
    SPALMERConfig,
)

TOKENIZER_VERSION = 1
TOKENIZER_FINGERPRINT = "test-tokenizer-v1"


class TinyTokenMixer(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, hidden_states: Tensor, *, state: int | None = None, **_: object):
        next_state = 1 if state is None else state + 1
        return self.projection(hidden_states), next_state

    def step(self, hidden_states: Tensor, *, state: int | None = None, **_: object):
        next_state = 10 if state is None else state + 10
        return self.projection(hidden_states), next_state


class TinyChannelMixer(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.projection(hidden_states).tanh()


class RoutedChannelMixer(TinyChannelMixer):
    def forward(self, hidden_states: Tensor, *, state: int | None = None) -> ChannelMixerOutput:
        next_state = 1 if state is None else state + 1
        return ChannelMixerOutput(
            update=super().forward(hidden_states),
            state=next_state,
            auxiliary_loss=hidden_states.square().mean() * 0.01,
            metrics={"active_experts": 2},
        )


def _tiny_model() -> SPALMERCausalLM:
    config = SPALMERConfig(
        vocab_size=31,
        d_model=12,
        n_layers=4,
        tokenizer_version=TOKENIZER_VERSION,
        tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
        ple_expansion_factor=2,
    )
    blocks = [
        SPALMERBlock(
            config.d_model,
            TinyTokenMixer(config.d_model),
            TinyChannelMixer(config.d_model),
            norm_eps=config.norm_eps,
        )
        for _ in range(config.n_layers)
    ]
    backbone = SPALMERBackbone(config, blocks)
    return SPALMERCausalLM(config, backbone)


def test_causal_lm_composes_blocks_and_computes_next_token_loss() -> None:
    torch.manual_seed(7)
    model = _tiny_model().eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])

    output = model(input_ids, labels=input_ids)

    assert output.logits.shape == (2, 5, model.config.vocab_size)
    assert output.token_mixer_states == (1, 1, 1, 1)
    assert output.channel_mixer_states == (None, None, None, None)
    assert output.loss is not None and torch.isfinite(output.loss)

    output.loss.backward()
    assert model.backbone.embeddings.layers[0].weight.grad is not None
    assert model.backbone.embeddings.layers[-1].weight.grad is not None
    assert model.lm_head.weight.grad is not None


def test_token_mixer_states_round_trip_through_the_backbone() -> None:
    model = _tiny_model().eval()
    input_ids = torch.tensor([[1, 2, 3]])

    first = model(input_ids)
    second = model(input_ids, token_mixer_states=first.token_mixer_states)

    assert first.token_mixer_states == (1, 1, 1, 1)
    assert second.token_mixer_states == (2, 2, 2, 2)


def test_decode_mode_dispatches_explicit_token_mixer_step() -> None:
    model = _tiny_model().eval()

    output = model(torch.tensor([[3]]), execution_mode="decode")

    assert output.token_mixer_states == (10, 10, 10, 10)


def test_channel_mixer_output_preserves_auxiliary_routing_information() -> None:
    config = SPALMERConfig(
        vocab_size=19,
        d_model=8,
        n_layers=1,
        tokenizer_version=TOKENIZER_VERSION,
        tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
        ple_expansion_factor=2,
    )
    block = SPALMERBlock(
        config.d_model,
        TinyTokenMixer(config.d_model),
        RoutedChannelMixer(config.d_model),
    )
    model = SPALMERCausalLM(config, SPALMERBackbone(config, [block])).eval()

    first = model(torch.tensor([[1, 2, 3]]))
    second = model(
        torch.tensor([[4]]),
        execution_mode="decode",
        token_mixer_states=first.token_mixer_states,
        channel_mixer_states=first.channel_mixer_states,
    )

    assert first.channel_mixer_states == (1,)
    assert second.channel_mixer_states == (2,)
    assert first.auxiliary_loss is not None
    assert first.layer_metrics == ({"active_experts": 2},)


def test_model_config_fails_closed_on_tokenizer_identity() -> None:
    config = SPALMERConfig(
        vocab_size=8,
        d_model=4,
        n_layers=1,
        tokenizer_version=TOKENIZER_VERSION,
        tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
    )

    config.assert_tokenizer_compatible(
        version=TOKENIZER_VERSION,
        fingerprint=TOKENIZER_FINGERPRINT,
    )
    try:
        config.assert_tokenizer_compatible(version=2, fingerprint=TOKENIZER_FINGERPRINT)
    except ValueError as error:
        assert "tokenizer identity mismatch" in str(error)
    else:
        raise AssertionError("mismatched tokenizer identity was accepted")


def test_default_token_mixer_pattern_is_three_kda_to_one_mla() -> None:
    config = SPALMERConfig(
        vocab_size=8,
        d_model=4,
        n_layers=9,
        tokenizer_version=TOKENIZER_VERSION,
        tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
    )
    pattern = [config.token_mixer_for_layer(index) for index in range(config.n_layers)]

    assert pattern == ["kda", "kda", "kda", "mla", "kda", "kda", "kda", "mla", "kda"]
