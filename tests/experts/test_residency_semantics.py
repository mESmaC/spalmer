"""Non-training checks of resident expert identity, routing masks, and accounting.

Every test here constructs a model and runs forward passes only; nothing
optimizes model weights.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from spalmer.attention import KDAConfig, MLAConfig
from spalmer.config import SPALMERConfig
from spalmer.experts import (
    ExpertResidency,
    MicroExpertChannelMixer,
    MicroExpertsConfig,
    load_balance_loss,
    select_least_surprised_experts,
)
from spalmer.factory import build_spalmer_model

D_MODEL = 16
NUM_EXPERTS = 12


def _experts(**overrides: object) -> MicroExpertsConfig:
    values: dict[str, object] = {
        "d_model": D_MODEL,
        "num_experts": NUM_EXPERTS,
        "expert_inter_dim": 4,
        "shared_inter_dim": 8,
        "active_experts": 2,
        "max_resident_experts": 6,
        "potentiation_budget": 0,
    }
    values.update(overrides)
    return MicroExpertsConfig(**values)


def _model(experts: MicroExpertsConfig | None = None, n_layers: int = 4, vocab_size: int = 40):
    experts = experts or _experts()
    config = SPALMERConfig(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_layers=n_layers,
        tokenizer_version=1,
        tokenizer_fingerprint="residency-semantics",
        ple_expansion_factor=1,
    )
    kda = KDAConfig(hidden_size=D_MODEL, num_heads=2, head_k_dim=8, backend="reference")
    mla = MLAConfig(hidden_size=D_MODEL, num_heads=2, head_k_dim=8, q_latent_dim=8, kv_latent_dim=4)
    torch.manual_seed(0)
    return build_spalmer_model(config, kda, mla, experts).eval()


def _selected_ids(output) -> set[int]:
    ids: set[int] = set()
    for metrics in output.layer_metrics:
        ids |= set(metrics["expert_ids"].reshape(-1).tolist())
    return ids


# --- identity coherence -----------------------------------------------------


def test_one_residency_object_is_shared_by_every_layer_and_the_model() -> None:
    model = _model()
    assert isinstance(model.residency, ExpertResidency)
    for block in model.backbone.blocks:
        assert block.channel_mixer.residency is model.residency
    assert model.resident_expert_ids == tuple(range(NUM_EXPERTS))
    # Residency is request-level state, not checkpoint state.
    assert not any("resident" in key for key in model.state_dict())


def test_every_layer_routes_inside_the_same_resident_set() -> None:
    model = _model()
    input_ids = torch.randint(0, 40, (3, 9))
    model.residency.set([1, 4, 7])
    output = model(input_ids)
    for metrics in output.layer_metrics:
        selected = set(metrics["expert_ids"].reshape(-1).tolist())
        assert selected <= {1, 4, 7}
        assert torch.equal(metrics["resident_experts"], model.residency.resident_mask)
    model.residency.reset()


# --- routing-mask enforcement -----------------------------------------------


def test_selection_never_picks_a_non_resident_expert() -> None:
    torch.manual_seed(1)
    scores = torch.rand(5, 7, NUM_EXPERTS)
    resident = torch.zeros(NUM_EXPERTS, dtype=torch.bool)
    resident[[2, 3, 9]] = True
    ids, weights = select_least_surprised_experts(scores, 2, resident)
    assert set(ids.reshape(-1).tolist()) <= {2, 3, 9}
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(5, 7))
    # Among residents the least surprised still wins.
    expected = scores.masked_fill(~resident, torch.inf).topk(2, dim=-1, largest=False).indices
    torch.testing.assert_close(ids, expected)


def test_non_resident_weights_cannot_influence_the_routed_update() -> None:
    model = _model()
    input_ids = torch.randint(0, 40, (2, 6))
    model.residency.set([0, 5, 6])
    with torch.no_grad():
        reference = model(input_ids).logits.clone()
        for block in model.backbone.blocks:
            bank = block.channel_mixer.experts
            for expert in range(NUM_EXPERTS):
                if expert not in (0, 5, 6):
                    bank.gate_proj[expert].fill_(3.0)
                    bank.up_proj[expert].fill_(-2.0)
                    bank.down_proj[expert].fill_(5.0)
        perturbed = model(input_ids).logits
    torch.testing.assert_close(perturbed, reference)
    model.residency.reset()


def test_resident_set_smaller_than_topk_is_rejected() -> None:
    model = _model()
    with pytest.raises(ValueError, match="at least active_experts"):
        model.residency.set([3])
    model.residency.set([3, 4])
    with pytest.raises(ValueError, match="cannot be served"):
        model.set_active_experts(4)
    assert model.active_experts == 2
    model.residency.expand([5, 6])
    model.set_active_experts(4)
    with pytest.raises(ValueError, match="at least active_experts=4"):
        model.residency.set([3, 4])
    model.set_active_experts(None)
    model.residency.reset()


# --- expansion monotonicity -------------------------------------------------


def test_expansion_preserves_existing_ids_and_adds_only_new_explicit_ids() -> None:
    residency = ExpertResidency(_experts())
    residency.set([2, 5])
    added = residency.expand([9, 0])
    assert added == (0, 9)
    assert residency.ids == (0, 2, 5, 9)
    with pytest.raises(ValueError, match="already resident"):
        residency.expand([2])
    assert residency.ids == (0, 2, 5, 9)
    with pytest.raises(ValueError, match="lie in"):
        residency.expand([NUM_EXPERTS])
    snapshot = residency.snapshot()
    state = residency.snapshot_state()
    residency.expand([1])
    residency.set_active_experts(4)
    assert set(snapshot) < set(residency.ids)
    residency.restore_state(state)
    assert residency.ids == snapshot
    assert residency.active_experts == 2
    assert residency.active_experts_override is None
    assert torch.equal(residency.resident_ids, torch.tensor(snapshot, dtype=torch.long))
    assert residency.resident_mask.sum() == len(snapshot)


def test_session_restores_the_previous_set_even_after_expansion() -> None:
    model = _model()
    with model.residency_session([1, 2]):
        model.residency.expand([8])
        model.set_active_experts(3)
        assert model.resident_expert_ids == (1, 2, 8)
        assert model.active_experts == 3
    assert model.resident_expert_ids == tuple(range(NUM_EXPERTS))
    assert model.active_experts_override is None
    assert model.active_experts == 2
    model.residency.set([3, 4])
    model.set_active_experts(2)
    with model.residency_session():
        model.residency.expand([10])
        model.set_active_experts(3)
    assert model.resident_expert_ids == (3, 4)
    assert model.active_experts == 2
    model.residency.reset()
    model.set_active_experts(None)


def test_topk_override_does_not_change_residency() -> None:
    model = _model()
    model.residency.set([0, 1, 2, 3])
    model.set_active_experts(3)
    assert model.resident_expert_ids == (0, 1, 2, 3)
    output = model(torch.randint(0, 40, (1, 5)))
    assert output.layer_metrics[0]["expert_ids"].shape[-1] == 3
    assert _selected_ids(output) <= {0, 1, 2, 3}
    model.set_active_experts(None)
    assert model.resident_expert_ids == (0, 1, 2, 3)
    model.residency.reset()


# --- exact parameter accounting ---------------------------------------------


def test_accounting_matches_tensor_sums_and_moves_by_whole_experts() -> None:
    model = _model()
    accounting = model.parameter_accounting()
    assert accounting.total == sum(p.numel() for p in model.parameters())
    per_layer = 3 * D_MODEL * 4
    assert accounting.parameters_per_expert == per_layer * 4
    assert accounting.expert_pool == NUM_EXPERTS * accounting.parameters_per_expert
    assert accounting.router == D_MODEL * NUM_EXPERTS
    assert accounting.shared_channel == 4 * 3 * D_MODEL * 8
    assert accounting.vocab_head == 40 * D_MODEL
    assert accounting.shared_base == (
        accounting.total - accounting.expert_pool - accounting.vocab_head - accounting.router
    )
    assert accounting.resident_parameters == accounting.total
    assert accounting.per_token_active_parameters == (
        accounting.total - accounting.expert_pool + 2 * accounting.parameters_per_expert
    )

    model.residency.set([0, 1])
    two = model.parameter_accounting()
    assert two.resident_expert_ids == (0, 1)
    assert two.resident_parameters == (
        accounting.total - accounting.expert_pool + 2 * accounting.parameters_per_expert
    )
    model.residency.expand([7])
    three = model.parameter_accounting()
    assert three.resident_parameters - two.resident_parameters == accounting.parameters_per_expert
    # Persistent storage uses the configured BF16 expert masters; execution
    # remains four-bit fake QAT and is reported separately.
    assert three.nominal_bits["expert_pool"] == 16
    assert three.execution_bits["expert_pool"] == 4
    assert three.nominal_bits["embeddings"] == 32
    assert three.execution_bits["embeddings"] == 4
    assert three.actual_parameter_bytes == sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    assert three.resident_bytes() < accounting.resident_bytes()
    assert "resident=" in three.summary()
    model.residency.reset()


# --- shared path presence ---------------------------------------------------


def test_shared_path_is_present_by_default_and_carries_signal_without_experts() -> None:
    model = _model()
    for block in model.backbone.blocks:
        assert block.channel_mixer.has_shared_path
    input_ids = torch.randint(0, 40, (2, 5))
    with torch.no_grad():
        for block in model.backbone.blocks:
            bank = block.channel_mixer.experts
            bank.gate_proj.zero_()
            bank.up_proj.zero_()
            bank.down_proj.zero_()
    hidden = model.backbone.embeddings(input_ids, layer_index=0)
    block = model.backbone.blocks[0]
    update = block.channel_mixer(block.channel_norm(hidden)).update
    assert update.abs().sum() > 0
    accounting = model.parameter_accounting()
    assert accounting.shared_channel > 0
    assert (
        accounting.shared_channel
        == 4 * model.backbone.blocks[0].channel_mixer.shared.parameter_count
    )


def test_shared_path_can_be_budgeted_away_explicitly() -> None:
    model = _model(_experts(shared_inter_dim=0))
    for block in model.backbone.blocks:
        assert not block.channel_mixer.has_shared_path
    assert model.parameter_accounting().shared_channel == 0
    assert model(torch.randint(0, 40, (1, 4))).logits.shape == (1, 4, 40)


def test_config_exposes_the_shared_routed_split() -> None:
    config = _experts()
    assert config.shared_parameters_per_layer == 3 * D_MODEL * 8
    assert config.expert_parameters_per_layer == 3 * D_MODEL * 4
    assert config.expert_pool_parameters_per_layer == NUM_EXPERTS * 3 * D_MODEL * 4
    assert config.resident_cap == 6
    assert MicroExpertsConfig(d_model=8, num_experts=4, active_experts=3).min_resident_experts == 3


# --- grouped execution ------------------------------------------------------


@pytest.mark.parametrize("resident", [None, [0, 2, 5, 9, 11]])
@pytest.mark.parametrize("promoted", [False, True])
def test_grouped_execution_matches_the_loop_reference(resident, promoted) -> None:
    base = _experts(active_experts=3, max_resident_experts=8, potentiation_budget=2)
    torch.manual_seed(3)
    grouped = MicroExpertChannelMixer(replace(base, expert_execution="grouped")).eval()
    loop = MicroExpertChannelMixer(replace(base, expert_execution="loop")).eval()
    loop.load_state_dict(grouped.state_dict())
    loop.router.load_state_dict(grouped.router.state_dict())
    for mixer in (grouped, loop):
        if resident is not None:
            mixer.residency.set(resident)
        if promoted:
            mixer.potentiation_controller.promoted_mask[[0, 5]] = True
    x = torch.randn(3, 7, D_MODEL)
    out_grouped = grouped(x)
    out_loop = loop(x)
    torch.testing.assert_close(out_grouped.update, out_loop.update, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out_grouped.metrics["expert_ids"], out_loop.metrics["expert_ids"])
    torch.testing.assert_close(
        out_grouped.metrics["expert_quantization_error"],
        out_loop.metrics["expert_quantization_error"],
    )
    (out_grouped.update.sum() + out_grouped.auxiliary_loss).backward()
    (out_loop.update.sum() + out_loop.auxiliary_loss).backward()
    torch.testing.assert_close(
        grouped.experts.down_proj.grad, loop.experts.down_proj.grad, rtol=1e-4, atol=1e-6
    )


def test_grouped_execution_touches_only_resident_experts_in_the_backward_pass() -> None:
    mixer = MicroExpertChannelMixer(_experts()).train()
    mixer.residency.set([1, 2, 3])
    x = torch.randn(2, 5, D_MODEL)
    mixer(x).update.sum().backward()
    grad = mixer.experts.down_proj.grad
    assert grad is not None
    for expert in range(NUM_EXPERTS):
        if expert in (1, 2, 3):
            continue
        assert grad[expert].abs().sum() == 0


# --- auxiliary loss normalization -------------------------------------------


@pytest.mark.parametrize("residents", [3, 6, 12])
def test_load_balance_loss_is_one_at_uniform_use_for_any_resident_count(residents) -> None:
    mask = torch.zeros(NUM_EXPERTS, dtype=torch.bool)
    mask[:residents] = True
    scores = torch.zeros(4, residents * 5, NUM_EXPERTS)
    # Uniform predicted surprise among residents, each resident used equally.
    ids = torch.arange(residents).repeat(4 * 5).reshape(4, residents * 5, 1)
    loss = load_balance_loss(scores, ids, mask)
    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_load_balance_loss_ignores_sequence_length_and_backbone_averages_layers() -> None:
    torch.manual_seed(5)
    mixer = MicroExpertChannelMixer(_experts()).eval()
    short = mixer(torch.zeros(1, 4, D_MODEL)).auxiliary_loss
    long = mixer(torch.zeros(1, 64, D_MODEL)).auxiliary_loss
    torch.testing.assert_close(short, long)

    four = _model(n_layers=4)
    eight = _model(n_layers=8)
    input_ids = torch.randint(0, 40, (2, 6))
    loss_four = float(four(input_ids).auxiliary_loss)
    loss_eight = float(eight(input_ids).auxiliary_loss)
    per_layer_eight = [
        float(load_balance_loss(m["router_scores"], m["expert_ids"]))
        for m in eight(input_ids).layer_metrics
    ]
    assert loss_eight == pytest.approx(sum(per_layer_eight) / 8, rel=1e-5)
    assert abs(loss_four - loss_eight) < 0.5  # no growth with depth


# --- forward shapes ---------------------------------------------------------


def test_forward_shapes_in_prefill_and_decode_under_restricted_residency() -> None:
    model = _model()
    model.residency.set([2, 3, 4])
    prompt = torch.randint(0, 40, (2, 6))
    prefill = model(prompt, labels=prompt)
    assert prefill.logits.shape == (2, 6, 40)
    assert prefill.token_nll.shape == (2, 5)
    assert prefill.predictive_entropy.shape == (2,)
    step = model(
        prompt[:, :1],
        execution_mode="decode",
        token_mixer_states=prefill.token_mixer_states,
        channel_mixer_states=prefill.channel_mixer_states,
    )
    assert step.logits.shape == (2, 1, 40)
    assert _selected_ids(step) <= {2, 3, 4}
    model.residency.reset()


# --- request lifecycle ------------------------------------------------------


def test_controller_start_size_honors_a_topk_override_and_reports_effective_k() -> None:
    from spalmer.experts import choose_inference_residency

    model = _model()
    model.set_active_experts(4)  # larger than min_resident_experts (2)
    prompt = torch.randint(0, 40, (1, 6))
    decision = choose_inference_residency(model, prompt, min_gain=10.0)
    assert decision.active_experts == 4
    assert decision.resident_count == 4
    assert model.residency.request_open
    # Changing top-k does not silently tear down request residency.
    model.set_active_experts(3)
    assert model.residency.request_open
    assert model.resident_expert_ids == decision.resident_ids
    # Explicit request exit restores both the prior full set and top-k=4.
    model.end_residency_request()
    assert not model.residency.request_open
    assert model.resident_expert_ids == tuple(range(NUM_EXPERTS))
    assert model.active_experts == 4
    model.set_active_experts(None)


def test_controller_restores_the_prior_set_when_evaluation_fails(monkeypatch) -> None:
    from spalmer.experts import residency as residency_module

    model = _model()
    model.residency.set([3, 4, 5])
    calls = {"count": 0}
    real_evaluate = residency_module._evaluate

    def failing_evaluate(model_, prompt_ids):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated device failure")
        return real_evaluate(model_, prompt_ids)

    monkeypatch.setattr(residency_module, "_evaluate", failing_evaluate)
    with pytest.raises(RuntimeError, match="simulated"):
        residency_module.choose_inference_residency(
            model, torch.randint(0, 40, (1, 6)), min_gain=-1.0
        )
    assert model.resident_expert_ids == (3, 4, 5)
    assert model.active_experts == 2
    assert not model.residency.request_open
    model.residency.reset()


def test_residency_set_under_inference_mode_still_allows_a_grad_forward() -> None:
    model = _model()
    input_ids = torch.randint(0, 40, (1, 5))
    with torch.inference_mode():
        model.residency.set([0, 1, 2])
        model(input_ids)
    model.train()
    output = model(input_ids, labels=input_ids)
    output.loss.backward()  # no optimizer step: a gradient probe only
    assert model.backbone.blocks[0].channel_mixer.experts.down_proj.grad is not None
    model.residency.reset()


def test_nested_request_end_returns_to_the_outermost_prior_set() -> None:
    residency = ExpertResidency(_experts())
    residency.set([6, 7, 8])
    residency.set_active_experts(3)
    with pytest.raises(ValueError, match="at least active_experts=3"):
        residency.begin_request([0, 1])
    assert not residency.request_open
    residency.begin_request([0, 1], active_experts=2)
    residency.begin_request([2, 3, 4, 5], active_experts=4)
    assert residency.ids == (2, 3, 4, 5)
    assert residency.active_experts == 4
    residency.end_request()
    assert residency.ids == (6, 7, 8)
    assert residency.active_experts == 3
    residency.end_request()  # no open request: a no-op
    assert residency.ids == (6, 7, 8)
    assert residency.active_experts == 3


def test_grouped_execution_skips_experts_without_pairs() -> None:
    mixer = MicroExpertChannelMixer(_experts(active_experts=2, router_bias=True)).eval()
    x = torch.randn(1, 3, D_MODEL)
    # Force every token onto experts 4 and 9 by making them the least surprised.
    with torch.no_grad():
        mixer.router.proj.weight.zero_()
        mixer.router.proj.bias.zero_()
        mixer.router.proj.bias[[4, 9]] = -20.0
    out = mixer(x)
    assert set(out.metrics["expert_ids"].reshape(-1).tolist()) == {4, 9}
    assert int(out.metrics["num_active_experts"]) == 2
    assert torch.isfinite(out.update).all()
