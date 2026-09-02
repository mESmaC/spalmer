"""Minimal smoke coverage for the micro-expert channel mixer.

Covers exactly the requested checks: shapes, selected-expert count and
validity, gradients reaching the router and the selected experts, plus the
shared-router and 200-expert requirements.
"""

from __future__ import annotations

import torch

from spalmer.experts import (
    MicroExpertChannelMixer,
    MicroExpertsConfig,
    SurpriseRouter,
)


def make_mixer(**overrides: object) -> MicroExpertChannelMixer:
    config = MicroExpertsConfig(
        d_model=16,
        num_experts=6,
        expert_inter_dim=8,
        **overrides,
    )
    torch.manual_seed(0)
    return MicroExpertChannelMixer(config)


def test_output_shapes():
    mixer = make_mixer()
    x = torch.randn(2, 5, 16)
    out = mixer(x)
    assert out.update.shape == (2, 5, 16)
    assert out.state is None
    assert out.auxiliary_loss is not None and out.auxiliary_loss.ndim == 0
    assert out.metrics["expert_ids"].shape == (2, 5, 2)
    assert out.metrics["routing_weights"].shape == (2, 5, 2)
    assert out.metrics["router_scores"].shape == (2, 5, 6)
    assert out.metrics["expert_utilization"].shape == (6,)
    # Kept on device so the forward pass never synchronizes with the host.
    assert int(out.metrics["num_active_experts"]) >= 1


def test_selected_expert_count_and_validity():
    for active in (2, 3):
        mixer = make_mixer(active_experts=active)
        x = torch.randn(4, 7, 16)
        result = mixer(x)
        ids = result.metrics["expert_ids"]
        weights = result.metrics["routing_weights"]
        assert ids.shape == (4, 7, active)
        assert (ids >= 0).all() and (ids < 6).all()
        # bottom-k selection returns unique experts per token
        assert (ids.sort(dim=-1).values.diff(dim=-1) != 0).all()
        assert weights.dtype == torch.float32
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(4, 7))
        assert (weights > 0).all()


def test_least_surprise_is_preferred():
    mixer = make_mixer()
    x = torch.randn(3, 4, 16)
    result = mixer(x)
    scores = result.metrics["router_scores"]
    ids = result.metrics["expert_ids"]
    weights = result.metrics["routing_weights"]
    expected_ids = scores.topk(2, dim=-1, largest=False).indices
    assert torch.isfinite(scores).all()
    assert (scores >= 0).all()
    torch.testing.assert_close(ids, expected_ids)
    # ids are ordered by ascending predicted surprise, weights descending
    selected_surprise = scores.gather(-1, ids)
    assert (selected_surprise[..., :-1] <= selected_surprise[..., 1:] + 1e-6).all()
    assert (weights[..., :-1] >= weights[..., 1:] - 1e-6).all()


def test_two_hundred_experts_supported():
    config = MicroExpertsConfig(d_model=16, num_experts=200, expert_inter_dim=8)
    torch.manual_seed(0)
    mixer = MicroExpertChannelMixer(config)
    x = torch.randn(2, 3, 16)
    result = mixer(x)
    assert result.update.shape == (2, 3, 16)
    assert result.metrics["expert_ids"].shape == (2, 3, 2)
    assert result.metrics["expert_utilization"].shape == (200,)
    torch.testing.assert_close(
        result.metrics["expert_utilization"].sum(), torch.tensor(1.0)
    )
    assert 1 <= int(result.metrics["num_active_experts"]) <= 2 * 3 * 2


def test_gradients_reach_router_and_selected_experts():
    mixer = make_mixer()
    x = torch.randn(4, 6, 16)
    result = mixer(x)
    (result.update.sum() + result.auxiliary_loss).backward()

    router_grad = mixer.router.proj.weight.grad
    assert router_grad is not None
    assert torch.isfinite(router_grad).all()
    assert router_grad.abs().sum() > 0

    selected = set(result.metrics["expert_ids"].reshape(-1).tolist())
    for expert in range(6):
        grad = mixer.experts.down_proj.grad[expert]
        if expert in selected:
            assert grad.abs().sum() > 0, f"selected expert {expert} got no gradient"
        else:
            assert grad is None or grad.abs().sum() == 0


def test_shared_router_serves_multiple_banks():
    config = MicroExpertsConfig(d_model=16, num_experts=6, expert_inter_dim=8)
    torch.manual_seed(0)
    shared_router = SurpriseRouter(config)
    layer_one = MicroExpertChannelMixer(config, router=shared_router)
    layer_two = MicroExpertChannelMixer(config, router=shared_router)
    assert layer_one.router is layer_two.router
    assert layer_one.experts is not layer_two.experts

    x = torch.randn(2, 4, 16)
    out_one = layer_one(x)
    out_two = layer_two(x)
    # identical inputs and shared router select identical experts...
    torch.testing.assert_close(
        out_one.metrics["expert_ids"], out_two.metrics["expert_ids"]
    )
    # ...but the two banks stay layer-local
    assert not torch.allclose(out_one.update, out_two.update)

    (out_one.update.sum() + out_two.update.sum()).backward()
    router_grad = shared_router.proj.weight.grad
    assert router_grad is not None and router_grad.abs().sum() > 0
    assert layer_one.experts.down_proj.grad.abs().sum() > 0
    assert layer_two.experts.down_proj.grad.abs().sum() > 0


def test_shared_router_rejects_mismatched_bank():
    config = MicroExpertsConfig(d_model=16, num_experts=6, expert_inter_dim=8)
    router = SurpriseRouter(config)
    other = MicroExpertsConfig(d_model=16, num_experts=8, expert_inter_dim=8)
    try:
        MicroExpertChannelMixer(other, router=router)
    except ValueError as exc:
        assert "8" in str(exc)
    else:
        raise AssertionError("expected a ValueError for mismatched num_experts")
