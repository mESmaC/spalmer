"""Focused checks for the C16 lateral + active-silencing branch (no training).

Ported from ``agent/directional-v0`` and extended for the peer-summary
silencing input and the production factory gate.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from spalmer.attention import KDAConfig, MLAConfig
from spalmer.config import SPALMERConfig
from spalmer.directional import DirectionalConfig, LateralSilencingMixer
from spalmer.experts import MicroExpertsConfig
from spalmer.factory import build_spalmer_model
from spalmer.modeling import SPALMERBlock


def make_config(**overrides: object) -> DirectionalConfig:
    values: dict[str, object] = {
        "d_model": 12,
        "num_feature_groups": 4,
        "lateral_rank": 3,
        "enabled": True,
    }
    values.update(overrides)
    return DirectionalConfig(**values)


def make_mixer(**overrides: object) -> LateralSilencingMixer:
    torch.manual_seed(0)
    return LateralSilencingMixer(make_config(**overrides))


class TinyTokenMixer(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, hidden_states: Tensor, *, state: object = None, **_: object):
        return self.projection(hidden_states), state


class TinyChannelMixer(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, hidden_states: Tensor, *, state: object = None, **_: object):
        return self.projection(hidden_states)


def test_shape_preservation():
    for d_model, groups, rank in ((12, 4, 3), (16, 2, 8), (8, 8, 1)):
        mixer = make_mixer(d_model=d_model, num_feature_groups=groups, lateral_rank=rank)
        assert mixer.config.group_width == d_model // groups
        x = torch.randn(2, 7, d_model)
        update = mixer(x)
        assert update.shape == (2, 7, d_model)
        assert torch.isfinite(update).all()


def test_exact_noop_when_residual_gate_is_zero():
    mixer = make_mixer()
    assert mixer.residual_gate.item() == 0.0
    x = torch.randn(3, 5, 12)
    update = mixer(x)
    assert torch.equal(update, torch.zeros_like(update))

    torch.manual_seed(1)
    plain = SPALMERBlock(12, TinyTokenMixer(12), TinyChannelMixer(12))
    extended = SPALMERBlock(12, TinyTokenMixer(12), TinyChannelMixer(12), directional_mixer=mixer)
    extended.token_mixer.load_state_dict(plain.token_mixer.state_dict())
    extended.channel_mixer.load_state_dict(plain.channel_mixer.state_dict())
    x = torch.randn(2, 4, 12)
    assert torch.equal(extended(x).hidden_states, plain(x).hidden_states)
    assert "directional" in extended(x).metrics


def test_no_cross_token_influence():
    mixer = make_mixer(residual_gate_init=1.0)
    x = torch.randn(2, 6, 12)
    update_a = mixer(x)
    changed = x.clone()
    changed[:, 4] = torch.randn(2, 12) * 3.0
    update_b = mixer(changed)
    untouched = [i for i in range(6) if i != 4]
    assert torch.equal(update_a[:, untouched], update_b[:, untouched])
    assert not torch.equal(update_a[:, 4], update_b[:, 4])


def test_diagonal_removal_blocks_self_feed():
    mixer = make_mixer(residual_gate_init=1.0)
    x = torch.zeros(1, 1, 12)
    x[0, 0, :3] = torch.randn(3)  # only group 0 carries signal
    update = mixer(x)
    groups = update.reshape(1, 1, 4, 3)
    assert torch.equal(groups[0, 0, 0], torch.zeros(3))
    assert groups[0, 0, 1:].abs().sum() > 0
    mixing = mixer.group_mixing_matrix
    assert torch.equal(torch.diagonal(mixing), torch.zeros_like(torch.diagonal(mixing)))


def test_high_silence_suppresses_proposed_update():
    mixer = make_mixer(residual_gate_init=1.0)
    with torch.no_grad():
        mixer.silence_proj.bias.fill_(50.0)  # sigmoid(50) ~= 1
    x = torch.randn(2, 5, 12)
    update = mixer(x)
    assert update.abs().max().item() < 1e-12
    assert mixer.last_metrics["mean_silence"].item() > 0.999


def test_silencing_reads_the_peer_summary_not_only_its_own_group():
    mixer = make_mixer(residual_gate_init=1.0)
    width = mixer.config.group_width
    with torch.no_grad():
        # Silence depends only on the peer-summary half of the input.
        mixer.silence_proj.weight.zero_()
        mixer.silence_proj.weight[0, width:] = 20.0
    x = torch.zeros(1, 1, 12)
    x[0, 0, :3] = 1.0  # group 0 active, every other group silent
    mixer(x)
    quiet_peers = mixer.last_metrics["mean_silence"].item()
    x[0, 0, 3:] = 1.0  # now every peer is active too
    mixer(x)
    loud_peers = mixer.last_metrics["mean_silence"].item()
    assert loud_peers > quiet_peers


def test_gradients_reach_all_directional_parameters():
    torch.manual_seed(0)
    mixer = make_mixer(residual_gate_init=1.0)
    x = torch.randn(2, 5, 12)
    update = mixer(x)
    (update * torch.randn_like(update)).sum().backward()
    for name, parameter in mixer.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert set(dict(mixer.named_parameters())) == {
        "lateral_a",
        "lateral_b",
        "silence_proj.weight",
        "silence_proj.bias",
        "residual_gate",
    }
    assert sum(p.numel() for p in mixer.parameters()) == mixer.config.parameters_per_layer


def test_disabled_config_cannot_build_a_mixer():
    with pytest.raises(ValueError, match="enabled"):
        LateralSilencingMixer(make_config(enabled=False))


def _configs():
    model_config = SPALMERConfig(
        vocab_size=32,
        d_model=16,
        n_layers=4,
        tokenizer_version=1,
        tokenizer_fingerprint="directional-test",
        ple_expansion_factor=1,
    )
    kda_config = KDAConfig(hidden_size=16, num_heads=2, head_k_dim=8, backend="reference")
    mla_config = MLAConfig(
        hidden_size=16, num_heads=2, head_k_dim=8, q_latent_dim=8, kv_latent_dim=8
    )
    experts_config = MicroExpertsConfig(
        d_model=16, num_experts=4, expert_inter_dim=8, shared_inter_dim=8, active_experts=2
    )
    return model_config, kda_config, mla_config, experts_config


def test_factory_gate_is_off_by_default_and_on_when_enabled():
    model_config, kda_config, mla_config, experts_config = _configs()
    torch.manual_seed(0)
    plain = build_spalmer_model(model_config, kda_config, mla_config, experts_config).eval()
    for block in plain.backbone.blocks:
        assert block.directional_mixer is None
        assert block.directional_norm is None
    assert not any("directional" in key for key in plain.state_dict())
    assert plain.parameter_accounting().components["directional"] == 0

    disabled = DirectionalConfig(d_model=16, enabled=False)
    torch.manual_seed(0)
    still_plain = build_spalmer_model(
        model_config, kda_config, mla_config, experts_config, directional_config=disabled
    )
    assert not any("directional" in key for key in still_plain.state_dict())

    enabled = DirectionalConfig(d_model=16, num_feature_groups=4, lateral_rank=4, enabled=True)
    torch.manual_seed(0)
    routed = build_spalmer_model(
        model_config, kda_config, mla_config, experts_config, directional_config=enabled
    ).eval()
    for block in routed.backbone.blocks:
        assert isinstance(block.directional_mixer, LateralSilencingMixer)
    assert (
        routed.parameter_accounting().components["directional"] == 4 * enabled.parameters_per_layer
    )

    # Zero residual gates make the enabled model bitwise identical to the
    # plain one on the shared parameters.
    routed.load_state_dict(plain.state_dict(), strict=False)
    input_ids = torch.randint(0, 32, (2, 5))
    torch.testing.assert_close(routed(input_ids).logits, plain(input_ids).logits)
    assert "directional" in routed(input_ids).layer_metrics[0]

    with pytest.raises(ValueError, match="component widths"):
        build_spalmer_model(
            model_config,
            kda_config,
            mla_config,
            experts_config,
            directional_config=DirectionalConfig(d_model=8, enabled=True),
        )
