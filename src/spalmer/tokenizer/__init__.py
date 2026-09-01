"""SPALMER C01: Ranked-Precedence Distributional Tokenizer, v0 slice.

`src/spalmer` is an implicit namespace package; this package owns only
`spalmer.tokenizer`.
"""

from __future__ import annotations

from .encoder import Encoder, build_tier_tables, longest_match_in
from .tiers import TIER_LABELS, TIER_ORDER, Tier
from .trainer import Sample, TrainerConfig, lexer_surfaces, split_identifier, train
from .vocab import (
    FORMAT_NAME,
    FORMAT_VERSION,
    TokenEntry,
    VersionRecord,
    Vocab,
    append_byte_backstop,
    byte_surface,
)

__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "Encoder",
    "Sample",
    "TIER_LABELS",
    "TIER_ORDER",
    "Tier",
    "TokenEntry",
    "TrainerConfig",
    "Vocab",
    "VersionRecord",
    "append_byte_backstop",
    "build_tier_tables",
    "byte_surface",
    "lexer_surfaces",
    "longest_match_in",
    "split_identifier",
    "train",
]
