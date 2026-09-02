from __future__ import annotations

import pytest
import torch

from spalmer.training import CausalBatch


def test_causal_batch_counts_only_non_ignored_targets() -> None:
    batch = CausalBatch(
        input_ids=torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        labels=torch.tensor([[1, 2, -100, 4], [5, 6, 7, -100]]),
        strata=("english", "python"),
        token_utf8_bytes=((0, 4, 4, 4), (0, 8, 8, 0)),
    )

    assert batch.target_tokens == 4


def test_causal_batch_rejects_row_metadata_mismatch() -> None:
    with pytest.raises(ValueError, match="strata"):
        CausalBatch(torch.tensor([[1, 2], [3, 4]]), strata=("english",))


def test_host_batch_transfer_preserves_metadata() -> None:
    batch = CausalBatch(
        torch.tensor([[1, 2, 3]]),
        strata=("cpp",),
        token_utf8_bytes=((0, 4, 5),),
    )

    moved = batch.to(torch.device("cpu"))

    assert moved.input_ids.device.type == "cpu"
    assert moved.labels is not None
    assert moved.strata == batch.strata
    assert moved.token_utf8_bytes == batch.token_utf8_bytes
