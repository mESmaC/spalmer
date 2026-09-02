from __future__ import annotations

import torch
from torch import nn

from spalmer.training import TrainingConfig, build_optimizers, classify_parameters


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
        fused_adamw="off",
    )

    bundle = build_optimizers(model, config)

    assert isinstance(bundle.dense, torch.optim.AdamW)
    assert isinstance(bundle.sparse, torch.optim.SparseAdam)
