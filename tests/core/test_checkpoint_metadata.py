from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from spalmer.checkpoint import (
    _load_checkpoint_tokenizer,
    _load_model_parameter_dtype,
    _migrate_experts_config,
    _model_parameter_dtype,
    _serialize_tokenizer_binding,
    _validate_checkpoint_parameter_state_dtype,
)
from spalmer.experts import MicroExpertsConfig
from spalmer.tokenizer import HFTokenizerAdapter


class _Tokenizer:
    vocab_size = 4
    bos_token_id = 0
    eos_token_id = 1
    eod_token_id = None
    pad_token_id = None
    init_kwargs: dict[str, object] = {}
    special_tokens_map: dict[str, object] = {}

    def __len__(self) -> int:
        return self.vocab_size

    def get_vocab(self) -> dict[str, int]:
        return {"<bos>": 0, "<eos>": 1, "hello": 2, "world": 3}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del text, add_special_tokens
        return [2]

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        del ids, skip_special_tokens
        return "hello"


def _experts(*, master_dtype: str = "bfloat16") -> MicroExpertsConfig:
    legacy = master_dtype == "float32"
    return MicroExpertsConfig(
        d_model=8,
        num_experts=2,
        expert_inter_dim=4,
        active_experts=2,
        expert_weight_format="legacy_int" if legacy else "mxfp4",
        expert_activation_format="bfloat16" if legacy else "mxfp8",
        expert_master_dtype=master_dtype,
        expert_qat_backend="reference" if legacy else "auto",
        expert_promotion_format="bfloat16" if legacy else "mxfp8",
    )


def test_model_parameter_dtype_checks_every_trainable_tensor() -> None:
    parameters = [
        ("first", torch.zeros(2, dtype=torch.bfloat16, requires_grad=True)),
        ("frozen", torch.zeros(2, dtype=torch.float32)),
        ("second", torch.zeros(2, dtype=torch.bfloat16, requires_grad=True)),
    ]
    assert _model_parameter_dtype(parameters) == "bfloat16"

    parameters[-1] = ("second", torch.zeros(2, dtype=torch.float32, requires_grad=True))
    with pytest.raises(ValueError, match="mixed trainable parameter dtypes"):
        _model_parameter_dtype(parameters)


def test_v5_model_dtype_is_authoritative_and_must_match_expert_recipe() -> None:
    assert (
        _load_model_parameter_dtype(
            {"model_parameter_dtype": "bfloat16"},
            5,
            _experts(),
        )
        == "bfloat16"
    )
    with pytest.raises(ValueError, match="disagrees with expert_master_dtype"):
        _load_model_parameter_dtype(
            {"model_parameter_dtype": "float32"},
            5,
            _experts(),
        )
    assert _load_model_parameter_dtype({}, 4, _experts(master_dtype="float32")) == "float32"


def test_saved_parameter_payload_must_match_declared_model_dtype() -> None:
    parameter = torch.zeros(2, dtype=torch.float32, requires_grad=True)
    with pytest.raises(ValueError, match="payload disagrees"):
        _validate_checkpoint_parameter_state_dtype(
            [("weight", parameter)],
            {"weight": parameter},
            "bfloat16",
        )


def test_hf_checkpoint_binding_accepts_only_the_same_tokenizer_identity() -> None:
    adapter = HFTokenizerAdapter.from_tokenizer(
        _Tokenizer(),
        source="Qwen/example-tokenizer",
        revision="pinned-revision",
    )
    payload = {"tokenizer_binding": _serialize_tokenizer_binding(adapter)}
    assert _load_checkpoint_tokenizer(payload, 5, tokenizer_override=adapter) is adapter

    changed = HFTokenizerAdapter.from_tokenizer(
        _Tokenizer(),
        source="local-copy",
        revision="pinned-revision",
        artifact_fingerprint="different",
    )
    with pytest.raises(ValueError, match="different tokenizer identity"):
        _load_checkpoint_tokenizer(payload, 5, tokenizer_override=changed)


def test_in_memory_hf_tokenizer_requires_a_reconstructible_source() -> None:
    adapter = HFTokenizerAdapter.from_tokenizer(_Tokenizer())
    with pytest.raises(ValueError, match="cannot be reconstructed"):
        _serialize_tokenizer_binding(adapter)


def test_local_hf_binding_persists_an_absolute_source(tmp_path: Path) -> None:
    local = tmp_path / "tokenizer"
    local.mkdir()
    adapter = HFTokenizerAdapter.from_tokenizer(_Tokenizer(), source=str(local))
    binding = _serialize_tokenizer_binding(adapter)
    assert binding["identity"]["source"] == str(local.resolve())


def test_legacy_top1_and_decoupled_residency_configs_migrate_to_valid_capacity() -> None:
    v1 = {
        "d_model": 8,
        "num_experts": 4,
        "expert_inter_dim": 4,
        "active_experts": 1,
        "min_active_experts": 1,
        "max_active_experts": 1,
        "router_bias": False,
        "initializer_range": 0.02,
    }
    migrated_v1 = _migrate_experts_config(v1, 1)
    assert migrated_v1["min_resident_experts"] == 1
    MicroExpertsConfig(**migrated_v1)

    v4 = _experts(master_dtype="float32")
    raw_v4 = asdict(v4)
    for field in (
        "expert_weight_format",
        "expert_activation_format",
        "expert_master_dtype",
        "expert_qat_backend",
        "expert_promotion_format",
    ):
        raw_v4.pop(field)
    raw_v4.update(num_experts=8, max_active_experts=2, min_resident_experts=4)
    migrated_v4 = _migrate_experts_config(raw_v4, 4)
    assert migrated_v4["min_resident_experts"] == 4
    MicroExpertsConfig(**migrated_v4)
