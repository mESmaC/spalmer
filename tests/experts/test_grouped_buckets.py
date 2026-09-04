"""Correctness and memory-shape checks for grouped micro-expert execution."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from spalmer.experts import MicroExpertBank, MicroExpertsConfig
from spalmer.experts.bank import _power_of_two_count_buckets


def _config(*, execution: str, num_experts: int = 8) -> MicroExpertsConfig:
    return MicroExpertsConfig(
        d_model=8,
        num_experts=num_experts,
        expert_inter_dim=4,
        active_experts=2,
        min_active_experts=2,
        max_active_experts=min(8, num_experts),
        min_resident_experts=2,
        max_resident_experts=min(8, num_experts),
        expert_execution=execution,
        expert_weight_format="legacy_int",
        expert_fake_quantization=False,
        potentiation_budget=0,
    )


def test_bucketed_grouped_forward_and_backward_match_expert_loop() -> None:
    torch.manual_seed(41)
    grouped = MicroExpertBank(_config(execution="grouped"))
    loop = MicroExpertBank(_config(execution="loop"))
    loop.load_state_dict(grouped.state_dict())

    # Five non-empty experts span capacities 1, 2, 4, 8, and 16.
    counts = (1, 2, 3, 5, 9)
    expert_index = torch.cat(
        [torch.full((count,), expert, dtype=torch.long) for expert, count in enumerate(counts)]
    )
    token_index = torch.arange(expert_index.numel()) % 11
    grouped_input = torch.randn(11, 8, requires_grad=True)
    loop_input = grouped_input.detach().clone().requires_grad_(True)
    grouped_weights = torch.rand(expert_index.numel(), requires_grad=True)
    loop_weights = grouped_weights.detach().clone().requires_grad_(True)

    grouped_output = grouped.execute_routing(
        grouped_input,
        token_index,
        expert_index,
        grouped_weights,
    )
    loop_output = loop.execute_routing(
        loop_input,
        token_index,
        expert_index,
        loop_weights,
    )
    torch.testing.assert_close(grouped_output, loop_output, rtol=1e-5, atol=1e-6)

    output_scale = torch.linspace(0.5, 1.5, grouped_output.numel()).reshape_as(grouped_output)
    (grouped_output * output_scale).sum().backward()
    (loop_output * output_scale).sum().backward()
    torch.testing.assert_close(grouped_input.grad, loop_input.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        grouped_weights.grad,
        loop_weights.grad,
        rtol=1e-5,
        atol=1e-6,
    )
    for grouped_parameter, loop_parameter in zip(
        grouped.parameters(), loop.parameters(), strict=True
    ):
        torch.testing.assert_close(
            grouped_parameter.grad,
            loop_parameter.grad,
            rtol=1e-5,
            atol=1e-6,
        )


@pytest.mark.parametrize("execution", ["loop", "grouped"])
@pytest.mark.parametrize("hidden_dtype", [torch.bfloat16, torch.float32])
def test_mixed_precision_routes_reduce_in_hidden_dtype(
    execution: str,
    hidden_dtype: torch.dtype,
) -> None:
    """Autocast expert GEMMs and FP32 probabilities follow the residual dtype."""

    torch.manual_seed(43)
    bank = MicroExpertBank(_config(execution=execution)).to(dtype=torch.bfloat16)
    hidden_states = torch.randn(7, 8, dtype=hidden_dtype, requires_grad=True)
    token_index = torch.arange(7).repeat_interleave(2)
    expert_index = torch.tensor((0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7))
    routing_weights = torch.rand(7, 2, dtype=torch.float32)
    routing_weights = (
        routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    ).reshape(-1)
    routing_weights.requires_grad_()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = bank.execute_routing(
            hidden_states,
            token_index,
            expert_index,
            routing_weights,
        )

    assert output.dtype == hidden_dtype
    output.float().square().mean().backward()
    assert routing_weights.grad is not None
    assert routing_weights.grad.dtype == torch.float32
    assert torch.isfinite(routing_weights.grad).all()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()


def test_power_of_two_bucket_padding_is_strictly_bounded() -> None:
    counts = [1, 3, 5, 9, 17, 33, 65, 129, 257, 4097]
    buckets = _power_of_two_count_buckets(counts)
    padded = sum(capacity * len(positions) for capacity, positions in buckets)
    assert padded < 2 * sum(counts)
    assert {capacity for capacity, _positions in buckets} == {
        1,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        8192,
    }


def test_severe_imbalance_never_builds_a_global_maximum_block(monkeypatch) -> None:
    num_experts = 200
    config = replace(
        _config(execution="grouped", num_experts=num_experts),
        d_model=4,
        expert_inter_dim=2,
        max_active_experts=20,
        max_resident_experts=20,
    )
    bank = MicroExpertBank(config)
    counts = [4096, *([1] * (num_experts - 1))]
    expert_index = torch.cat(
        [torch.full((count,), expert, dtype=torch.long) for expert, count in enumerate(counts)]
    )
    token_index = torch.cat(
        [torch.arange(counts[0]), torch.zeros(num_experts - 1, dtype=torch.long)]
    )
    hidden_states = torch.randn(counts[0], config.d_model)
    routing_weights = torch.ones(expert_index.numel())

    real_bmm = torch.bmm
    input_shapes: list[tuple[int, ...]] = []

    def recording_bmm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        input_shapes.append(tuple(left.shape))
        return real_bmm(left, right)

    monkeypatch.setattr(torch, "bmm", recording_bmm)
    output = bank.execute_routing(
        hidden_states,
        token_index,
        expert_index,
        routing_weights,
    )

    # Three GEMMs per bucket; inspect the input to the first one in each set.
    bucket_shapes = input_shapes[::3]
    padded_slots = sum(groups * capacity for groups, capacity, _width in bucket_shapes)
    assert bucket_shapes == [(199, 1, 4), (1, 4096, 4)]
    assert padded_slots == sum(counts)
    assert padded_slots < 2 * expert_index.numel()
    assert (num_experts, counts[0], config.d_model) not in bucket_shapes
    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()
