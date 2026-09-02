from __future__ import annotations

import pytest
import torch

from spalmer.runtime import _validated_trainable_parameters


class _ParameterSource:
    def __init__(self, parameters: list[tuple[str, torch.nn.Parameter]]) -> None:
        self._parameters = parameters

    def named_parameters(self):
        return iter(self._parameters)


def test_legacy_runtime_validates_all_parameters_before_selecting_optimizer() -> None:
    bf16 = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    another_bf16 = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    parameters, dtype = _validated_trainable_parameters(  # type: ignore[arg-type]
        _ParameterSource([("first", bf16), ("second", another_bf16)])
    )
    assert parameters == (bf16, another_bf16)
    assert dtype == torch.bfloat16

    fp32 = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
    with pytest.raises(ValueError, match="share one dtype"):
        _validated_trainable_parameters(  # type: ignore[arg-type]
            _ParameterSource([("first", bf16), ("second", fp32)])
        )
