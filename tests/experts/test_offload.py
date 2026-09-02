"""Static/mocked checks for physical inference expert offload.

The model-level placement checks use PyTorch's allocation-free ``meta`` device;
the numerical check uses one tiny CPU bank.  No test starts training or builds
a production-sized model.
"""

from __future__ import annotations

import pytest
import torch

from spalmer.attention import KDAConfig, MLAConfig
from spalmer.config import SPALMERConfig
from spalmer.experts import MicroExpertBank, MicroExpertsConfig
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
        "potentiation_budget": 1,
        "expert_weight_format": "legacy_int",
        "expert_activation_format": "bfloat16",
        "expert_master_dtype": "float32",
        "expert_promotion_format": "bfloat16",
    }
    values.update(overrides)
    return MicroExpertsConfig(**values)


def _tiny_model():
    model_config = SPALMERConfig(
        vocab_size=16,
        d_model=8,
        n_layers=2,
        tokenizer_version=1,
        tokenizer_fingerprint="offload-static",
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


def _banks(model) -> tuple[MicroExpertBank, ...]:
    return tuple(block.channel_mixer.experts for block in model.backbone.blocks)


def test_model_api_keeps_full_masters_on_cpu_and_only_resident_rows_in_cache() -> None:
    model = _tiny_model()
    state_keys = tuple(model.state_dict())
    telemetry = model.enable_expert_offload(
        "meta",
        cache_size=4,
        resident_ids=(1, 4),
        pin_memory=False,
    )

    assert model.execution_device == torch.device("meta")
    assert model.expert_offload_enabled
    assert telemetry.masters_on_cpu
    assert telemetry.caches_on_target
    assert not telemetry.masters_pinned
    assert telemetry.resident_expert_ids == (1, 4)
    assert telemetry.cached_expert_ids_by_layer == ((1, 4), (1, 4))
    assert telemetry.occupancy == 0.5
    assert telemetry.cache_bytes == (
        len(_banks(model))
        * len(telemetry.resident_expert_ids)
        * _banks(model)[0].parameters_per_expert
        * _banks(model)[0].gate_proj.element_size()
    )
    assert telemetry.master_bytes == sum(bank.expert_master_bytes for bank in _banks(model))
    assert all(
        parameter.device.type == "cpu"
        for bank in _banks(model)
        for parameter in bank.parameters()
    )
    assert tuple(model.state_dict()) == state_keys
    assert not any("cache" in key or "offload" in key for key in model.state_dict())
    with pytest.raises(RuntimeError, match="disabled while expert offload is active"):
        model.to("cpu")
    with pytest.raises(RuntimeError, match="inference-only"):
        model.train()


def test_default_offload_residency_is_bounded_for_fixed_inference() -> None:
    model = _tiny_model()
    telemetry = model.enable_expert_offload("meta", pin_memory=False)

    assert telemetry.capacity == model.residency.config.resident_cap
    assert telemetry.resident_expert_ids == (0, 1)
    assert model.resident_expert_ids == (0, 1)
    assert telemetry.cached_expert_ids_by_layer == ((0, 1),) * len(_banks(model))
    assert telemetry.cache_bytes < telemetry.master_bytes


def test_residency_commits_stage_and_evict_the_same_global_ids_in_every_layer() -> None:
    model = _tiny_model()
    initial = model.enable_expert_offload(
        "meta", cache_size=3, resident_ids=(0, 1), pin_memory=False
    )
    layer_count = len(_banks(model))
    assert initial.transferred_expert_rows == 2 * layer_count

    model.residency.set((1, 5))
    replaced = model.expert_offload_telemetry()
    assert replaced is not None
    assert replaced.cached_expert_ids_by_layer == ((1, 5),) * layer_count
    assert replaced.transferred_expert_rows == 3 * layer_count
    assert replaced.evicted_expert_rows == layer_count

    snapshot = model.residency.snapshot_state()
    model.residency.expand((3,))
    expanded = model.expert_offload_telemetry()
    assert expanded is not None
    assert expanded.cached_expert_ids_by_layer == ((1, 3, 5),) * layer_count
    assert expanded.transferred_expert_rows == 4 * layer_count
    model.residency.restore_state(snapshot)
    rolled_back = model.expert_offload_telemetry()
    assert rolled_back is not None
    assert rolled_back.cached_expert_ids_by_layer == ((1, 5),) * layer_count
    assert rolled_back.transferred_expert_rows == 4 * layer_count
    assert rolled_back.evicted_expert_rows == 2 * layer_count


def test_capacity_failure_and_partial_layer_failure_leave_prior_state_executable(
    monkeypatch,
) -> None:
    model = _tiny_model()
    model.enable_expert_offload(
        "meta", cache_size=3, resident_ids=(0, 1), pin_memory=False
    )
    before = model.expert_offload_telemetry()
    assert before is not None
    with pytest.raises(ValueError, match="physical cache capacity"):
        model.residency.set((0, 1, 2, 3))
    assert model.resident_expert_ids == (0, 1)
    assert all(bank.cached_expert_ids == (0, 1) for bank in _banks(model))

    second = _banks(model)[1]

    def fail_stage(expert_ids, *, force=False):
        del expert_ids, force
        raise RuntimeError("simulated layer transfer failure")

    monkeypatch.setattr(second, "_stage_expert_rows", fail_stage)
    with pytest.raises(RuntimeError, match="simulated layer transfer failure"):
        model.residency.set((2, 3))
    after = model.expert_offload_telemetry()
    assert after is not None
    assert model.resident_expert_ids == (0, 1)
    assert all(bank.cached_expert_ids == (0, 1) for bank in _banks(model))
    assert after.stage_operations == before.stage_operations
    assert after.transfer_bytes == before.transfer_bytes


def test_state_dict_load_refreshes_detached_resident_caches() -> None:
    source = _tiny_model()
    state = source.state_dict()
    model = _tiny_model()
    model.enable_expert_offload(
        "meta", cache_size=3, resident_ids=(2, 4), pin_memory=False
    )
    before = model.expert_offload_telemetry()
    assert before is not None
    with pytest.warns(UserWarning, match="meta parameter"):
        model.load_state_dict(state)
    after = model.expert_offload_telemetry()
    assert after is not None
    assert after.cached_expert_ids_by_layer == ((2, 4),) * len(_banks(model))
    assert after.stage_operations == before.stage_operations + 1
    assert after.transferred_expert_rows == (
        before.transferred_expert_rows + 2 * len(_banks(model))
    )


def test_cached_global_id_mapping_preserves_qat_and_potentiation_numerics() -> None:
    config = _experts(expert_execution="grouped")
    torch.manual_seed(7)
    reference = MicroExpertBank(config).eval()
    cached = MicroExpertBank(config).eval()
    cached.load_state_dict(reference.state_dict())
    cached._prepare_expert_offload(
        torch.device("cpu"),
        capacity=3,
        non_blocking=False,
        pin_memory=False,
    )
    cached._stage_expert_rows((1, 4, 5))

    hidden = torch.randn(5, config.d_model)
    token_index = torch.tensor((0, 0, 1, 2, 3, 4))
    expert_index = torch.tensor((1, 4, 5, 1, 4, 5))
    routing_weights = torch.tensor((0.4, 0.6, 1.0, 1.0, 1.0, 1.0))
    promoted = torch.zeros(config.num_experts, dtype=torch.bool)
    promoted[4] = True
    resident_ids = torch.tensor((1, 4, 5))

    expected = reference.execute_routing(
        hidden,
        token_index,
        expert_index,
        routing_weights,
        promoted_mask=promoted,
        resident_ids=resident_ids,
    )
    actual = cached.execute_routing(
        hidden,
        token_index,
        expert_index,
        routing_weights,
        promoted_mask=promoted,
        resident_ids=resident_ids,
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        cached.quantization_error(expert_index, resident_ids),
        reference.quantization_error(expert_index, resident_ids),
    )
    assert tuple(cached.state_dict()) == tuple(reference.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_forward_uses_cpu_masters_and_bounded_resident_caches() -> None:
    model = _tiny_model()
    resident_ids = (1, 4)
    model.residency.set(resident_ids)
    input_ids = torch.tensor([[1, 3, 5, 7]], dtype=torch.long)
    with torch.inference_mode():
        expected = model(input_ids).logits

    telemetry = model.enable_expert_offload(
        "cuda",
        cache_size=4,
        resident_ids=resident_ids,
        pin_memory=True,
    )
    with torch.inference_mode():
        actual = model(input_ids.to(model.execution_device)).logits.cpu()

    assert telemetry.masters_on_cpu
    assert telemetry.caches_on_target
    assert telemetry.resident_expert_ids == resident_ids
    assert telemetry.cache_bytes < telemetry.master_bytes
    assert all(
        parameter.device.type == "cpu"
        for bank in _banks(model)
        for parameter in bank.parameters(recurse=False)
    )
    assert model.lm_head.weight.device.type == "cuda"
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
