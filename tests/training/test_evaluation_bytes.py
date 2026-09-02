from __future__ import annotations

import torch

from spalmer.training.engine import CausalBatch
from spalmer.training.evaluation import _target_bytes


def test_bpb_uses_only_explicit_exact_target_byte_counts() -> None:
    valid = torch.tensor([[True, True], [False, True]])
    exact = CausalBatch(
        torch.tensor([[1, 2, 3], [4, 5, 6]]),
        token_utf8_bytes=((9, 1, 2), (9, 3, 4)),
    )
    unavailable = CausalBatch(torch.tensor([[1, 2, 3], [4, 5, 6]]))

    assert _target_bytes(exact, valid) == 7
    assert _target_bytes(unavailable, valid) == 0
