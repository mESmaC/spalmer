"""Shared fixtures for KDA smoke tests."""

from __future__ import annotations

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
