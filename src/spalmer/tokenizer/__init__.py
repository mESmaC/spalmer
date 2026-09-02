"""SPALMER C01: Ranked-Precedence Distributional Tokenizer, v0 slice.

`src/spalmer` is an implicit namespace package; this package owns only
`spalmer.tokenizer`.
"""

from __future__ import annotations

from .backends import (
    CompatibleTokenizerAdapter,
    ExactByteMappingUnavailable,
    ExactByteTokenizerBackend,
    HFTokenizerAdapter,
    LocalSerializedTokenizerAdapter,
    RPDTokenizerAdapter,
    SpecialTokenIds,
    TokenizerBackend,
    TokenizerIdentity,
    fingerprint_local_artifact,
)
from .encoder import Encoder, build_tier_tables, longest_match_in
from .evaluation import VocabEvaluation, evaluate_vocab
from .proxy import ProxyCharModel
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
    "CompatibleTokenizerAdapter",
    "ExactByteMappingUnavailable",
    "ExactByteTokenizerBackend",
    "Encoder",
    "HFTokenizerAdapter",
    "LocalSerializedTokenizerAdapter",
    "ProxyCharModel",
    "RPDTokenizerAdapter",
    "Sample",
    "SpecialTokenIds",
    "TIER_LABELS",
    "TIER_ORDER",
    "Tier",
    "TokenEntry",
    "TokenizerBackend",
    "TokenizerIdentity",
    "TrainerConfig",
    "Vocab",
    "VocabEvaluation",
    "VersionRecord",
    "append_byte_backstop",
    "build_tier_tables",
    "byte_surface",
    "evaluate_vocab",
    "fingerprint_local_artifact",
    "lexer_surfaces",
    "longest_match_in",
    "split_identifier",
    "train",
]
