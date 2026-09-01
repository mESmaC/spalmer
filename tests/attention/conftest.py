"""Make the worktree's ``src`` layout importable without installing."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest

from spalmer.attention.config import KDAConfig


@pytest.fixture()
def small_config() -> KDAConfig:
    return KDAConfig(
        hidden_size=32,
        num_heads=4,
        head_k_dim=8,
        head_v_dim=8,
        conv_width=4,
        backend="reference",
    )
