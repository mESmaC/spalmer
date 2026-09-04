from __future__ import annotations

import pytest
import torch

from spalmer.experts import MicroExpertChannelMixer, MicroExpertsConfig
from spalmer.experts.mixer import _expert_execution_metrics


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


@pytest.mark.parametrize("execution", ["grouped", "loop"])
def test_expert_execution_reports_exact_bounded_padding_scalars(execution: str) -> None:
    torch.manual_seed(7)
    mixer = MicroExpertChannelMixer(
        MicroExpertsConfig(
            d_model=4,
            num_experts=4,
            expert_inter_dim=2,
            active_experts=2,
            max_active_experts=4,
            max_resident_experts=4,
            expert_execution=execution,
            expert_weight_format="bfloat16",
            expert_activation_format="bfloat16",
            expert_master_dtype="bfloat16",
            expert_qat_backend="auto",
            expert_promotion_format="bfloat16",
            potentiation_budget=0,
        )
    ).eval()

    output = mixer(torch.randn(2, 5, 4))
    metrics = output.metrics
    ids = metrics["expert_ids"].reshape(-1)
    counts = torch.bincount(ids, minlength=4)
    active_groups = int((counts > 0).sum())
    max_group_load = int(counts.max())
    real_pairs = ids.numel()
    bucket_padded_pairs = sum(
        _next_power_of_two(int(count)) for count in counts if int(count) > 0
    )
    padded_pairs = bucket_padded_pairs if execution == "grouped" else real_pairs
    global_max_counterfactual_pairs = active_groups * max_group_load

    assert metrics["expert_execution"] == execution
    assert int(metrics["expert_group_count"]) == active_groups
    assert int(metrics["expert_group_max_load"]) == max_group_load
    assert int(metrics["expert_group_real_pairs"]) == real_pairs
    assert int(metrics["expert_group_padded_pairs"]) == padded_pairs
    assert float(metrics["expert_group_padding_amplification"]) == pytest.approx(
        padded_pairs / real_pairs
    )
    assert int(
        metrics["expert_group_global_max_counterfactual_padded_pairs"]
    ) == global_max_counterfactual_pairs
    assert float(
        metrics[
            "expert_group_global_max_counterfactual_padding_amplification"
        ]
    ) == pytest.approx(global_max_counterfactual_pairs / real_pairs)
    assert float(metrics["expert_group_padding_amplification"]) < 2.0
    for key in (
        "expert_group_count",
        "expert_group_max_load",
        "expert_group_real_pairs",
        "expert_group_padded_pairs",
        "expert_group_padding_amplification",
        "expert_group_global_max_counterfactual_padded_pairs",
        "expert_group_global_max_counterfactual_padding_amplification",
    ):
        assert metrics[key].numel() == 1
        assert not metrics[key].requires_grad


def test_severe_imbalance_reports_bounded_actual_and_huge_legacy_padding() -> None:
    counts = [4096, *([1] * 199)]
    expert_ids = torch.repeat_interleave(
        torch.arange(len(counts), dtype=torch.long),
        torch.tensor(counts, dtype=torch.long),
    )

    metrics = _expert_execution_metrics(
        expert_ids,
        num_experts=len(counts),
        execution="grouped",
    )

    assert int(metrics["expert_group_real_pairs"]) == 4295
    assert int(metrics["expert_group_padded_pairs"]) == 4295
    assert float(metrics["expert_group_padding_amplification"]) < 2.0
    assert int(
        metrics["expert_group_global_max_counterfactual_padded_pairs"]
    ) == 819_200
    assert float(
        metrics[
            "expert_group_global_max_counterfactual_padding_amplification"
        ]
    ) > 190.0
