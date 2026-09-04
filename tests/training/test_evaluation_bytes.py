from __future__ import annotations

import pytest
import torch
from torch import nn

from spalmer.modeling import CausalLMOutput
from spalmer.training.engine import CausalBatch
from spalmer.training.evaluation import _target_bytes, evaluate_model_batches


def test_bpb_uses_only_explicit_exact_target_byte_counts() -> None:
    valid = torch.tensor([[True, True], [False, True]])
    exact = CausalBatch(
        torch.tensor([[1, 2, 3], [4, 5, 6]]),
        token_utf8_bytes=((9, 1, 2), (9, 3, 4)),
    )
    unavailable = CausalBatch(torch.tensor([[1, 2, 3], [4, 5, 6]]))

    assert _target_bytes(exact, valid) == 7
    assert _target_bytes(unavailable, valid) == 0


class _EvaluationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.calls: list[dict[str, object]] = []

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> CausalLMOutput:
        self.calls.append(dict(kwargs))
        targets = input_ids.shape[1] - 1
        return CausalLMOutput(
            logits=torch.zeros((*input_ids.shape, 4)),
            token_mixer_states=(),
            channel_mixer_states=(),
            token_nll=torch.zeros((input_ids.shape[0], targets)),
            layer_metrics=(),
        )


def _evaluate(model: nn.Module, **kwargs: object) -> None:
    evaluate_model_batches(
        model,  # type: ignore[arg-type]
        [CausalBatch(torch.tensor([[1, 2, 3]]), strata=("english",))],
        tokenizer=None,  # type: ignore[arg-type]
        device="cpu",
        compute_dtype=torch.float32,
        **kwargs,  # type: ignore[arg-type]
    )


def test_evaluation_threads_recurrence_steps_only_when_requested() -> None:
    model = _EvaluationModel()

    _evaluate(model)
    assert model.calls[-1].get("recurrence_steps") is None
    assert "recurrence_steps" not in model.calls[-1]

    _evaluate(model, recurrence_steps=3)
    assert model.calls[-1]["recurrence_steps"] == 3

    with pytest.raises(ValueError, match="recurrence_steps must be positive"):
        _evaluate(model, recurrence_steps=0)
