from __future__ import annotations

import pytest
import torch
from torch import nn

from spalmer.training import (
    BF16MasterAdamW,
    OptimizerBundle,
    TrainingConfig,
    build_optimizers,
    classify_parameters,
)
from spalmer.training.engine import _validate_optimizer_precision_contract
from spalmer.training.optim import _stochastic_round_bfloat16


class _SparseConfig:
    sparse_gradients = True


class _SparseTable(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _SparseConfig()
        self.weight = nn.Parameter(torch.empty(8, 4))


class _Fixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.table = _SparseTable()
        self.projection = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)


def test_parameter_partition_is_complete_and_disjoint() -> None:
    model = _Fixture()
    groups = classify_parameters(model)
    grouped = [*groups.decay, *groups.no_decay, *groups.sparse]

    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in model.parameters()
    }
    assert groups.sparse == (model.table.weight,)
    assert id(model.projection.weight) in {id(parameter) for parameter in groups.decay}
    assert id(model.projection.bias) in {id(parameter) for parameter in groups.no_decay}
    assert id(model.norm.weight) in {id(parameter) for parameter in groups.no_decay}


def test_sparse_parameters_receive_a_sparse_optimizer() -> None:
    model = _Fixture()
    config = TrainingConfig(
        max_steps=2,
        micro_batch_size=1,
        sequence_length=8,
        warmup_steps=0,
        device="cpu",
        require_cuda=False,
        compute_dtype="float32",
        parameter_dtype="float32",
        fused_adamw="off",
    )

    bundle = build_optimizers(model, config)

    assert isinstance(bundle.dense, torch.optim.AdamW)
    assert isinstance(bundle.sparse, torch.optim.SparseAdam)


def test_bf16_master_selects_owned_adam_without_allocating_shadow_weights() -> None:
    model = nn.Linear(8, 8).to(dtype=torch.bfloat16)
    config = TrainingConfig(
        max_steps=2,
        micro_batch_size=1,
        sequence_length=8,
        warmup_steps=0,
        device="cpu",
        require_cuda=False,
        compute_dtype="float32",
        parameter_dtype="bfloat16",
        fused_adamw="off",
    )

    bundle = build_optimizers(model, config)

    assert isinstance(bundle.dense, BF16MasterAdamW)
    assert bundle.sparse is None
    assert not bundle.dense.state
    assert {parameter.dtype for parameter in model.parameters()} == {torch.bfloat16}


def test_bf16_optimizer_load_preserves_fp32_moment_values() -> None:
    parameter = nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    optimizer = BF16MasterAdamW(
        [parameter],
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        stochastic_rounding=True,
        update_chunk_size=2,
    )
    payload = optimizer.state_dict()
    expected_mean = torch.tensor([1.0001, -0.00012345], dtype=torch.float32)
    expected_variance = torch.tensor([0.00000017, 2.0003], dtype=torch.float32)
    parameter_id = payload["param_groups"][0]["params"][0]
    payload["state"][parameter_id] = {
        "step": 7,
        "exp_avg": expected_mean,
        "exp_avg_sq": expected_variance,
    }

    optimizer.load_state_dict(payload)

    assert optimizer.state[parameter]["exp_avg"].dtype == torch.float32
    assert optimizer.state[parameter]["exp_avg_sq"].dtype == torch.float32
    torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], expected_mean)
    torch.testing.assert_close(optimizer.state[parameter]["exp_avg_sq"], expected_variance)


def test_cpu_offloaded_bf16_moments_update_in_bounded_chunks_and_resume() -> None:
    parameter = nn.Parameter(torch.zeros(5, dtype=torch.bfloat16))
    optimizer = BF16MasterAdamW(
        [parameter],
        lr=1e-2,
        betas=(0.9, 0.99),
        eps=1e-8,
        stochastic_rounding=False,
        update_chunk_size=2,
        state_offload="cpu",
    )
    parameter.grad = torch.tensor([1, 2, 3, 4, 5], dtype=torch.bfloat16)

    optimizer.step()

    state = optimizer.state[parameter]
    assert parameter.dtype == torch.bfloat16
    assert parameter.device.type == "cpu"
    assert state["exp_avg"].device.type == "cpu"
    assert state["exp_avg_sq"].device.type == "cpu"
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32
    placement = optimizer.state_placement()
    assert placement["policy"] == "cpu"
    assert placement["configured_device"] == "cpu"
    assert placement["moment_tensors"] == 2
    assert placement["update_chunk_sizes"] == [2]
    assert placement["parameter_devices"] == ["cpu"]
    assert placement["parameter_dtypes"] == ["bfloat16"]

    payload = optimizer.state_dict()
    restored_parameter = nn.Parameter(parameter.detach().clone())
    restored = BF16MasterAdamW(
        [restored_parameter],
        lr=1e-2,
        betas=(0.9, 0.99),
        eps=1e-8,
        stochastic_rounding=False,
        update_chunk_size=2,
        state_offload="cpu",
    )
    restored.load_state_dict(payload)

    restored_state = restored.state[restored_parameter]
    assert restored_state["step"] == 1
    assert restored_state["exp_avg"].device.type == "cpu"
    assert restored_state["exp_avg_sq"].device.type == "cpu"
    torch.testing.assert_close(restored_state["exp_avg"], state["exp_avg"])
    torch.testing.assert_close(restored_state["exp_avg_sq"], state["exp_avg_sq"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_parameters_keep_fp32_moments_in_pinned_cpu_storage() -> None:
    parameter = nn.Parameter(torch.zeros(5, dtype=torch.bfloat16, device="cuda"))
    optimizer = BF16MasterAdamW(
        [parameter],
        lr=1e-2,
        betas=(0.9, 0.99),
        eps=1e-8,
        stochastic_rounding=False,
        update_chunk_size=2,
        state_offload="cpu",
    )
    parameter.grad = torch.ones_like(parameter)

    optimizer.step()

    state = optimizer.state[parameter]
    assert parameter.device.type == "cuda"
    assert parameter.dtype == torch.bfloat16
    assert state["exp_avg"].device.type == "cpu"
    assert state["exp_avg_sq"].device.type == "cpu"
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32
    assert state["exp_avg"].is_pinned()
    assert state["exp_avg_sq"].is_pinned()
    placement = optimizer.state_placement()
    assert placement["devices"] == ["cpu"]
    assert placement["cpu_pinned"] is True
    assert placement["parameter_devices"] == ["cuda:0"]


def test_cpu_offload_policy_is_selected_by_training_config() -> None:
    model = nn.Linear(4, 4).to(dtype=torch.bfloat16)
    config = TrainingConfig(
        max_steps=2,
        micro_batch_size=1,
        sequence_length=8,
        device="cpu",
        require_cuda=False,
        compute_dtype="float32",
        parameter_dtype="bfloat16",
        optimizer_state_offload="cpu",
        optimizer_update_chunk_size=3,
        fused_adamw="off",
    )

    bundle = build_optimizers(model, config)

    assert isinstance(bundle.dense, BF16MasterAdamW)
    assert bundle.dense.optimizer_state_offload == "cpu"
    assert bundle.dense.state_placement()["update_chunk_sizes"] == [3]
    assert bundle.state_placement()["dense"]["policy"] == "cpu"
    assert bundle.state_placement()["sparse"] is None


def test_bf16_stochastic_writeback_saturates_finite_overflow_symmetrically() -> None:
    values = torch.tensor([torch.finfo(torch.float32).max, -torch.finfo(torch.float32).max])
    rounded = _stochastic_round_bfloat16(values)

    assert rounded.dtype == torch.bfloat16
    assert torch.isfinite(rounded).all()
    expected = torch.finfo(torch.bfloat16).max
    torch.testing.assert_close(rounded.float(), torch.tensor([expected, -expected]))


def test_injected_optimizer_cannot_weaken_bf16_moment_contract() -> None:
    parameter = nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    incompatible = OptimizerBundle(
        dense=torch.optim.AdamW([parameter], lr=1e-3),
        sparse=None,
    )
    config = TrainingConfig(
        max_steps=1,
        micro_batch_size=1,
        sequence_length=8,
        warmup_steps=0,
        device="cpu",
        require_cuda=False,
        parameter_dtype="bfloat16",
    )

    with pytest.raises(ValueError, match="BF16MasterAdamW"):
        _validate_optimizer_precision_contract(incompatible, config)


def test_injected_bf16_optimizer_must_match_offload_policy() -> None:
    parameter = nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    incompatible = OptimizerBundle(
        dense=BF16MasterAdamW(
            [parameter],
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            stochastic_rounding=False,
            update_chunk_size=2,
            state_offload="none",
        ),
        sparse=None,
    )
    config = TrainingConfig(
        max_steps=1,
        micro_batch_size=1,
        sequence_length=8,
        warmup_steps=0,
        device="cpu",
        require_cuda=False,
        parameter_dtype="bfloat16",
        optimizer_state_offload="cpu",
    )

    with pytest.raises(ValueError, match="offload policy"):
        _validate_optimizer_precision_contract(incompatible, config)
