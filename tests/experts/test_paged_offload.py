"""Correctness checks for unrestricted, layer-local expert paging.

All numerical tests use tiny CPU banks. Model placement/lifecycle checks use
the allocation-free ``meta`` device; no test trains or builds a large model.
"""

from __future__ import annotations

import pytest
import torch

from spalmer.attention import KDAConfig, MLAConfig
from spalmer.config import RecurrenceConfig, SPALMERConfig
from spalmer.experts import (
    MicroExpertBank,
    MicroExpertChannelMixer,
    MicroExpertsConfig,
    choose_inference_residency,
)
from spalmer.factory import build_spalmer_model


def _experts(**overrides: object) -> MicroExpertsConfig:
    values: dict[str, object] = {
        "d_model": 8,
        "num_experts": 6,
        "expert_inter_dim": 3,
        "shared_inter_dim": 4,
        "active_experts": 2,
        "max_active_experts": 4,
        "max_resident_experts": 4,
        "potentiation_budget": 0,
        "expert_execution": "grouped",
        "expert_weight_format": "float32",
        "expert_activation_format": "float32",
        "expert_master_dtype": "float32",
        "expert_promotion_format": "float32",
    }
    values.update(overrides)
    return MicroExpertsConfig(**values)


def _tiny_model():
    model_config = SPALMERConfig(
        vocab_size=16,
        d_model=8,
        n_layers=2,
        tokenizer_version=1,
        tokenizer_fingerprint="paged-offload",
        ple_expansion_factor=1,
    )
    kda = KDAConfig(hidden_size=8, num_heads=2, head_k_dim=4, backend="reference")
    mla = MLAConfig(
        hidden_size=8,
        num_heads=2,
        head_k_dim=4,
        q_latent_dim=4,
        kv_latent_dim=4,
    )
    return build_spalmer_model(model_config, kda, mla, _experts()).eval()


def _prepare_paged(bank: MicroExpertBank, capacity: int) -> None:
    bank.eval()
    bank._prepare_expert_offload(
        torch.device("cpu"),
        capacity=capacity,
        non_blocking=False,
        pin_memory=False,
        paging=True,
    )


@pytest.mark.parametrize("execution", ["grouped", "loop"])
def test_paged_prefill_matches_unrestricted_when_unique_experts_exceed_capacity(
    execution: str,
) -> None:
    config = _experts(expert_execution=execution)
    torch.manual_seed(31)
    reference = MicroExpertBank(config).eval()
    paged = MicroExpertBank(config).eval()
    paged.load_state_dict(reference.state_dict())
    _prepare_paged(paged, capacity=2)
    paged._stage_expert_rows((0, 3))

    hidden = torch.randn(9, config.d_model)
    token_index = torch.arange(9).repeat_interleave(2)
    # Six non-contiguous/global identities with deliberately uneven loads,
    # forcing multiple count buckets and more selected ids than cache rows.
    expert_index = torch.tensor((5, 1, 4, 2, 0, 5, 3, 1, 4, 5, 2, 3, 0, 5, 1, 4, 3, 5))
    routing_weights = torch.rand(expert_index.numel()).reshape(9, 2)
    routing_weights = (routing_weights / routing_weights.sum(dim=-1, keepdim=True)).reshape(-1)
    expected = reference.execute_routing(
        hidden,
        token_index,
        expert_index,
        routing_weights,
    )
    actual = paged.execute_routing(
        hidden,
        token_index,
        expert_index,
        routing_weights,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert len(paged.cached_expert_ids) <= 2
    assert paged.expert_cache_bytes <= (
        2 * paged.parameters_per_expert * paged.gate_proj.element_size()
    )

    counters_before_metrics = paged.expert_offload_counters
    torch.testing.assert_close(
        paged.quantization_error(expert_index),
        reference.quantization_error(expert_index),
    )
    # Quantization telemetry was collected while each execution page was live;
    # reading it must not page the bank a second time.
    assert paged.expert_offload_counters == counters_before_metrics


def test_paged_mixer_routes_full_pool_and_honors_an_explicit_logical_mask() -> None:
    config = _experts(router_bias=True)
    torch.manual_seed(7)
    reference = MicroExpertChannelMixer(config).eval()
    paged = MicroExpertChannelMixer(config).eval()
    paged.load_state_dict(reference.state_dict())
    with torch.no_grad():
        for mixer in (reference, paged):
            mixer.router.proj.weight.zero_()
            assert mixer.router.proj.bias is not None
            mixer.router.proj.bias.zero_()
            mixer.router.proj.bias[[4, 5]] = -20.0
    _prepare_paged(paged.experts, capacity=2)
    paged.experts._stage_expert_rows((0, 1))
    hidden = torch.randn(2, 5, config.d_model)

    expected = reference(hidden)
    actual = paged(hidden)
    assert set(actual.metrics["expert_ids"].reshape(-1).tolist()) == {4, 5}
    torch.testing.assert_close(actual.metrics["expert_ids"], expected.metrics["expert_ids"])
    torch.testing.assert_close(actual.update, expected.update, rtol=1e-6, atol=1e-7)
    assert actual.metrics["routing_full_pool"]
    assert actual.metrics["expert_paging"]

    paged.residency.set((0, 1))
    restricted = paged(hidden)
    assert set(restricted.metrics["expert_ids"].reshape(-1).tolist()) == {0, 1}
    assert not restricted.metrics["routing_full_pool"]


def test_end_to_end_paged_logits_and_router_choices_match_unrestricted_model() -> None:
    torch.manual_seed(19)
    reference = _tiny_model()
    paged = _tiny_model()
    paged.load_state_dict(reference.state_dict())
    for bank in _banks(paged):
        _prepare_paged(bank, capacity=2)
    prompt = torch.randint(0, reference.config.vocab_size, (3, 11))

    expected = reference(prompt)
    actual = paged(prompt)

    torch.testing.assert_close(actual.logits, expected.logits, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.logits.argmax(dim=-1), expected.logits.argmax(dim=-1))
    for actual_metrics, expected_metrics in zip(
        actual.layer_metrics,
        expected.layer_metrics,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_metrics["expert_ids"],
            expected_metrics["expert_ids"],
        )
    assert any(
        metrics["expert_ids"].unique().numel() > 2 for metrics in actual.layer_metrics
    )
    assert all(len(bank.cached_expert_ids) <= 2 for bank in _banks(paged))


def test_paged_recurrent_decode_matches_unrestricted_across_steps() -> None:
    torch.manual_seed(23)
    reference = _tiny_model()
    paged = _tiny_model()
    paged.load_state_dict(reference.state_dict())
    for bank in _banks(paged):
        _prepare_paged(bank, capacity=1)
    prompt = torch.randint(0, reference.config.vocab_size, (1, 13))

    with torch.inference_mode():
        expected = reference(prompt)
        actual = paged(prompt)
        torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
        for _step in range(6):
            expected_token = expected.logits[:, -1].argmax(dim=-1, keepdim=True)
            actual_token = actual.logits[:, -1].argmax(dim=-1, keepdim=True)
            torch.testing.assert_close(actual_token, expected_token, rtol=0, atol=0)
            expected = reference(
                expected_token,
                token_mixer_states=expected.token_mixer_states,
                channel_mixer_states=expected.channel_mixer_states,
                execution_mode="decode",
            )
            actual = paged(
                actual_token,
                token_mixer_states=actual.token_mixer_states,
                channel_mixer_states=actual.channel_mixer_states,
                execution_mode="decode",
            )
            torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
            for actual_metrics, expected_metrics in zip(
                actual.layer_metrics,
                expected.layer_metrics,
                strict=True,
            ):
                torch.testing.assert_close(
                    actual_metrics["expert_ids"],
                    expected_metrics["expert_ids"],
                    rtol=0,
                    atol=0,
                )

    assert all(len(bank.cached_expert_ids) <= 1 for bank in _banks(paged))


def test_layer_caches_page_independently_and_retain_hits() -> None:
    config = _experts(expert_execution="loop")
    first = MicroExpertBank(config).eval()
    second = MicroExpertBank(config).eval()
    for bank in (first, second):
        _prepare_paged(bank, capacity=2)
        bank._stage_expert_rows((0, 1))

    first._stage_execution_page((1, 4))
    second._stage_execution_page((2, 5))
    assert first.cached_expert_ids == (1, 4)
    assert second.cached_expert_ids == (2, 5)
    first_rows = first.expert_offload_counters[1]

    # Reusing both rows is a cache hit; revisiting an evicted identity loads
    # only that row while retaining the still-hot row 4.
    first._stage_execution_page((1, 4))
    assert first.expert_offload_counters[1] == first_rows
    first._stage_execution_page((0, 4))
    assert first.cached_expert_ids == (0, 4)
    assert first.expert_offload_counters[1] == first_rows + 1


def test_transactional_page_failure_keeps_previous_cache_executable(monkeypatch) -> None:
    bank = MicroExpertBank(_experts()).eval()
    _prepare_paged(bank, capacity=2)
    bank._stage_expert_rows((1, 4))
    cached_before = (
        bank._cached_gate_proj.clone(),
        bank._cached_up_proj.clone(),
        bank._cached_down_proj.clone(),
    )
    counters_before = bank.expert_offload_counters
    real_empty = torch.empty
    calls = 0

    def fail_second_allocation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated page allocation failure")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", fail_second_allocation)
    with pytest.raises(RuntimeError, match="simulated page allocation failure"):
        bank._stage_expert_rows((2, 4))

    assert bank.cached_expert_ids == (1, 4)
    assert bank.expert_offload_counters == counters_before
    assert bank._cached_gate_proj is not None
    assert bank._cached_up_proj is not None
    assert bank._cached_down_proj is not None
    for actual, expected in zip(
        (bank._cached_gate_proj, bank._cached_up_proj, bank._cached_down_proj),
        cached_before,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_model_paging_defaults_to_full_logical_pool_and_reports_layer_caches() -> None:
    model = _tiny_model()
    model.residency.set((1, 2))
    telemetry = model.enable_expert_offload(
        "meta",
        cache_size=2,
        resident_ids=(4,),
        pin_memory=False,
    )

    assert telemetry.paging
    assert telemetry.mode == "paged"
    assert telemetry.resident_expert_ids == ()
    assert telemetry.cached_expert_ids_by_layer == ((4,), (4,))
    assert telemetry.occupancy == 0.5
    assert model.residency.is_full
    assert model.resident_expert_ids == tuple(range(6))
    assert all(bank.expert_paging_enabled for bank in _banks(model))
    with pytest.raises(RuntimeError, match="disabled while paged"):
        choose_inference_residency(model, torch.tensor([[1, 2]]))


def test_paged_load_hook_refreshes_each_independent_cache() -> None:
    source = _tiny_model()
    model = _tiny_model()
    model.enable_expert_offload(
        "meta",
        cache_size=2,
        resident_ids=(1, 4),
        pin_memory=False,
    )
    before = model.expert_offload_telemetry()
    assert before is not None
    with pytest.warns(UserWarning, match="meta parameter"):
        model.load_state_dict(source.state_dict())
    after = model.expert_offload_telemetry()
    assert after is not None
    assert after.cached_expert_ids_by_layer == ((1, 4), (1, 4))
    assert after.stage_operations == before.stage_operations + len(_banks(model))
    assert after.transferred_expert_rows == (
        before.transferred_expert_rows + 2 * len(_banks(model))
    )


def test_enable_rejects_open_request_without_mutation_and_restores_on_failure(
    monkeypatch,
) -> None:
    model = _tiny_model()
    model.residency.begin_request((1, 2))
    with pytest.raises(RuntimeError, match="open residency request"):
        model.enable_expert_offload("meta", cache_size=2, pin_memory=False)
    assert model.residency.request_open
    assert model.resident_expert_ids == (1, 2)
    model.residency.end_request()

    model.residency.set((2, 3))
    first = _banks(model)[0]

    def fail_prepare(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("simulated preparation failure")

    monkeypatch.setattr(first, "_prepare_expert_offload", fail_prepare)
    with pytest.raises(RuntimeError, match="simulated preparation failure"):
        model.enable_expert_offload("meta", cache_size=2, pin_memory=False)
    assert model.resident_expert_ids == (2, 3)
    assert not model.expert_offload_enabled


def _banks(model) -> tuple[MicroExpertBank, ...]:
    return tuple(block.channel_mixer.experts for block in model.backbone.blocks)


def _recurrent_tiny_model():
    model_config = SPALMERConfig(
        vocab_size=16,
        d_model=8,
        n_layers=3,
        tokenizer_version=1,
        tokenizer_fingerprint="paged-offload-recurrent",
        ple_expansion_factor=1,
        recurrence=RecurrenceConfig(1, 1, 1, default_steps=3),
    )
    kda = KDAConfig(hidden_size=8, num_heads=2, head_k_dim=4, backend="reference")
    mla = MLAConfig(
        hidden_size=8,
        num_heads=2,
        head_k_dim=4,
        q_latent_dim=4,
        kv_latent_dim=4,
    )
    return build_spalmer_model(model_config, kda, mla, _experts()).eval()


def test_paged_core_bank_counters_scale_with_iterations() -> None:
    """Paged expert banks in the recurrent core are re-staged once per iteration."""

    torch.manual_seed(37)
    model = _recurrent_tiny_model()
    banks = _banks(model)
    for bank in banks:
        _prepare_paged(bank, capacity=2)
    prompt = torch.randint(0, model.config.vocab_size, (1, 7))

    routing_calls = [0, 0, 0]
    for index, bank in enumerate(banks):
        real_execute = bank.execute_routing

        def counting_execute(*args, _index=index, _real=real_execute, **kwargs):
            routing_calls[_index] += 1
            return _real(*args, **kwargs)

        bank.execute_routing = counting_execute

    def stages() -> tuple[int, ...]:
        return tuple(bank.expert_offload_counters[0] for bank in banks)

    def rows() -> tuple[int, ...]:
        return tuple(bank.expert_offload_counters[1] for bank in banks)

    measured: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    with torch.inference_mode():
        for steps in (1, 3, 5):
            routing_calls[:] = [0, 0, 0]
            before_stages, before_rows = stages(), rows()
            output = model(prompt, recurrence_steps=steps)
            measured[steps] = (
                tuple(a - b for a, b in zip(stages(), before_stages, strict=True)),
                tuple(a - b for a, b in zip(rows(), before_rows, strict=True)),
            )
            assert output.recurrence_steps == steps
            # The prelude and coda banks route once; the core bank routes once
            # per iteration, so its paged cache is re-staged r times.
            assert routing_calls == [1, steps, 1]
            # Reading the per-expert quantization error stays valid: the mixer
            # collects it inside the same execute_routing call every iteration.
            for metrics in output.layer_metrics:
                error = metrics.get("expert_quantization_error")
                if error is not None:
                    assert torch.isfinite(error).all()

    core_stages = [measured[steps][0][1] for steps in (1, 3, 5)]
    assert core_stages[0] < core_stages[1] < core_stages[2]
    assert core_stages[2] <= 5 * core_stages[0] * 2
    # The prelude bank's paging traffic does not depend on the recurrence depth.
    assert {measured[steps][0][0] for steps in (1, 3, 5)} == {measured[1][0][0]}
    assert measured[5][1][1] > measured[5][1][0] + measured[5][1][2]
    assert all(len(bank.cached_expert_ids) <= 2 for bank in banks)
