from __future__ import annotations

from pathlib import Path

import pytest
import torch

from spalmer.attention import KDAConfig, MLAConfig
from spalmer.checkpoint import (
    FORMAT_VERSION,
    _load_checkpoint_tokenizer,
    _load_model_parameter_dtype,
    _model_parameter_dtype,
    _serialize_tokenizer_binding,
    _validate_checkpoint_parameter_state_dtype,
    load_checkpoint,
    save_checkpoint,
)
from spalmer.config import RecurrenceConfig, SPALMERConfig
from spalmer.experiment.state import tensor_state_sha256
from spalmer.experts import MicroExpertsConfig
from spalmer.factory import build_spalmer_model
from spalmer.modeling import LatentRecurrence
from spalmer.tokenizer import HFTokenizerAdapter, Sample, train


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


def _experts() -> MicroExpertsConfig:
    return MicroExpertsConfig(
        d_model=8,
        num_experts=2,
        expert_inter_dim=4,
        active_experts=2,
        expert_weight_format="bfloat16",
        expert_activation_format="bfloat16",
        expert_master_dtype="bfloat16",
        expert_qat_backend="auto",
        expert_promotion_format="bfloat16",
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


_RECURRENCE = RecurrenceConfig(1, 2, 1, default_steps=5, latent_init_std=0.5)


def _bundle_inputs(
    vocab,
    recurrence: RecurrenceConfig | None,
):
    model_config = SPALMERConfig(
        vocab_size=len(vocab),
        d_model=8,
        n_layers=4,
        tokenizer_version=vocab.version,
        tokenizer_fingerprint=vocab.fingerprint,
        ple_expansion_factor=1,
        recurrence=recurrence,
    )
    kda_config = KDAConfig(
        hidden_size=8, num_heads=2, head_k_dim=4, head_v_dim=4, backend="reference"
    )
    mla_config = MLAConfig(
        hidden_size=8,
        num_heads=2,
        head_k_dim=4,
        head_v_dim=4,
        q_latent_dim=4,
        kv_latent_dim=4,
    )
    experts_config = MicroExpertsConfig(
        d_model=8,
        num_experts=4,
        expert_inter_dim=3,
        active_experts=2,
        expert_weight_format="bfloat16",
        expert_activation_format="bfloat16",
        expert_master_dtype="bfloat16",
        expert_qat_backend="auto",
        expert_promotion_format="bfloat16",
    )
    return model_config, kda_config, mla_config, experts_config


def _save_bundle(
    path: Path,
    recurrence: RecurrenceConfig | None,
):
    vocab = train([Sample("alpha beta gamma delta " * 8)])
    model_config, kda_config, mla_config, experts_config = _bundle_inputs(
        vocab,
        recurrence,
    )
    torch.manual_seed(0)
    model = build_spalmer_model(
        model_config, kda_config, mla_config, experts_config
    ).to(dtype=torch.bfloat16).eval()
    save_checkpoint(
        path,
        model,
        vocab,
        kda_config=kda_config,
        mla_config=mla_config,
        experts_config=experts_config,
    )
    return model, vocab


def _rewrite_payload(source: Path, destination: Path, mutate) -> Path:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    mutate(payload)
    torch.save(payload, destination)
    return destination


def _declare_v7_qr_schema(payload: dict) -> None:
    payload["format_version"] = 7
    payload["model_config"].update(
        ple_quant_bits=4,
        ple_stochastic_rounding=True,
        ple_sparse_gradients=False,
        ple_alternation_policy="fixed_layer_parity",
        ple_reference_max_numel=100_000_000,
    )
    payload["experts_config"].update(
        expert_quant_bits=4,
        expert_fake_quantization=False,
        expert_stochastic_rounding=False,
    )


def test_v5_fake_qat_bundle_is_rejected_before_model_construction(tmp_path: Path) -> None:
    _save_bundle(tmp_path / "current-flat.pt", None)
    assert (
        torch.load(tmp_path / "current-flat.pt", weights_only=False)["format_version"]
        == FORMAT_VERSION
        == 8
    )

    def to_v5(payload: dict) -> None:
        payload["format_version"] = 5
        payload["model_config"].pop("recurrence")
        payload["model_config"]["ple_backend"] = "fake_qat"

    legacy = _rewrite_payload(tmp_path / "current-flat.pt", tmp_path / "v5-flat.pt", to_v5)
    with pytest.raises(ValueError, match="retired fake-QAT PLE"):
        load_checkpoint(legacy)


def test_current_recurrent_bundle_round_trips_adapter_and_default_steps(tmp_path: Path) -> None:
    path = tmp_path / "current-recurrent.pt"
    original = _save_bundle(path, _RECURRENCE)[0]
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert payload["format_version"] == FORMAT_VERSION == 8
    assert payload["model_config"]["recurrence"] == {
        "prelude_layers": 1,
        "core_layers": 2,
        "coda_layers": 1,
        "default_steps": 5,
        "latent_init_std": 0.5,
        "adapter": "linear_concat",
        "adapter_init": "identity_mix",
    }
    assert {key for key in payload["model_state"] if ".recurrence." in key} == {
        "backbone.recurrence.injection_norm.weight",
        "backbone.recurrence.adapter.weight",
        "backbone.recurrence.latent_norm.weight",
    }

    model = load_checkpoint(path)[0].eval()
    assert model.config.recurrence == _RECURRENCE
    assert model.recurrence_default_steps == 5
    assert isinstance(model.backbone.recurrence, LatentRecurrence)
    torch.testing.assert_close(
        model.backbone.recurrence.adapter.weight,
        original.backbone.recurrence.adapter.weight,
    )

    prompt = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        reloaded = model(
            prompt, recurrence_steps=3, latent_generator=torch.Generator().manual_seed(4)
        )
        expected = original(
            prompt, recurrence_steps=3, latent_generator=torch.Generator().manual_seed(4)
        )
    torch.testing.assert_close(reloaded.logits, expected.logits)
    assert reloaded.recurrence_steps == 3

    # The binding digest covers the recurrence parameters like any other weight.
    resaved = tmp_path / "resaved.pt"
    _save_bundle(resaved, _RECURRENCE)
    assert tensor_state_sha256(
        torch.load(path, map_location="cpu", weights_only=False)["model_state"]
    ) == tensor_state_sha256(
        torch.load(resaved, map_location="cpu", weights_only=False)["model_state"]
    )


def test_pre_v6_payloads_tolerate_a_null_recurrence_but_not_a_populated_one(
    tmp_path: Path,
) -> None:
    flat = tmp_path / "current-flat.pt"
    _save_bundle(flat, None)

    def keep_null_recurrence(payload: dict) -> None:
        payload["format_version"] = 5
        assert payload["model_config"]["recurrence"] is None
        payload["model_config"]["ple_backend"] = "fake_qat"

    tolerated = _rewrite_payload(flat, tmp_path / "v5-null.pt", keep_null_recurrence)
    with pytest.raises(ValueError, match="retired fake-QAT PLE"):
        load_checkpoint(tolerated)

    recurrent = tmp_path / "current-recurrent.pt"
    _save_bundle(recurrent, _RECURRENCE)

    def backdate(payload: dict) -> None:
        payload["format_version"] = 5

    with pytest.raises(ValueError, match="predates the recurrent core"):
        load_checkpoint(_rewrite_payload(recurrent, tmp_path / "v5-recurrent.pt", backdate))


def test_recurrence_schema_is_exact(tmp_path: Path) -> None:
    path = tmp_path / "current-recurrent.pt"
    _save_bundle(path, _RECURRENCE)

    def add_key(payload: dict) -> None:
        payload["model_config"]["recurrence"]["exit_threshold"] = 0.03

    extra = _rewrite_payload(path, tmp_path / "extra-key.pt", add_key)
    with pytest.raises(ValueError, match="invalid recurrence configuration fields"):
        load_checkpoint(extra)

    def drop_key(payload: dict) -> None:
        payload["model_config"]["recurrence"].pop("default_steps")

    missing = _rewrite_payload(path, tmp_path / "missing-key.pt", drop_key)
    with pytest.raises(ValueError, match="invalid recurrence configuration fields"):
        load_checkpoint(missing)


def test_pre_v7_missing_ple_backend_is_rejected_as_fake_qat(
    tmp_path: Path,
) -> None:
    current = tmp_path / "current-flat.pt"
    _save_bundle(current, None)

    def to_v6_without_backend(payload: dict) -> None:
        payload["format_version"] = 6
        payload["model_config"].pop("ple_backend")

    legacy = _rewrite_payload(current, tmp_path / "v6-no-backend.pt", to_v6_without_backend)
    with pytest.raises(ValueError, match="retired fake-QAT PLE"):
        load_checkpoint(legacy)


def test_v7_fake_qat_ple_metadata_is_rejected_before_construction(tmp_path: Path) -> None:
    current = tmp_path / "current-flat.pt"
    _save_bundle(current, None)

    def request_fake_qat(payload: dict) -> None:
        _declare_v7_qr_schema(payload)
        payload["model_config"]["ple_backend"] = "fake_qat"

    invalid = _rewrite_payload(current, tmp_path / "v7-fake-ple.pt", request_fake_qat)
    with pytest.raises(ValueError, match="retired fake-QAT PLE"):
        load_checkpoint(invalid)


@pytest.mark.parametrize(
    "override",
    [
        {"expert_fake_quantization": True},
        {"expert_stochastic_rounding": True},
        {"expert_qat_backend": "reference"},
        {"expert_weight_format": "legacy_int"},
        {"expert_master_dtype": "float32"},
    ],
)
def test_v8_emulated_expert_metadata_is_rejected_before_construction(
    tmp_path: Path,
    override: dict[str, object],
) -> None:
    current = tmp_path / "current-flat.pt"
    _save_bundle(current, None)

    def request_emulation(payload: dict) -> None:
        payload["experts_config"].update(override)

    invalid = _rewrite_payload(current, tmp_path / "v8-emulated-experts.pt", request_emulation)
    with pytest.raises(ValueError, match="retired emulated expert precision"):
        load_checkpoint(invalid)


def test_pre_v7_checkpoint_cannot_claim_qr_weights(tmp_path: Path) -> None:
    current = tmp_path / "current-flat.pt"
    _save_bundle(current, None)

    def backdate_qr(payload: dict) -> None:
        payload["format_version"] = 6
        payload["model_config"]["ple_backend"] = "qr"

    invalid = _rewrite_payload(current, tmp_path / "v6-qr.pt", backdate_qr)
    with pytest.raises(ValueError, match="predates QR-PLE"):
        load_checkpoint(invalid)


def test_v7_checkpoint_requires_explicit_ple_backend(tmp_path: Path) -> None:
    current = tmp_path / "current-flat.pt"
    _save_bundle(current, None)

    def remove_backend(payload: dict) -> None:
        _declare_v7_qr_schema(payload)
        payload["model_config"].pop("ple_backend")

    invalid = _rewrite_payload(current, tmp_path / "v7-no-backend.pt", remove_backend)
    with pytest.raises(ValueError, match=r"missing=\['ple_backend'\]"):
        load_checkpoint(invalid)


def test_v7_qr_checkpoint_round_trips_exact_layout_and_logits(tmp_path: Path) -> None:
    current = tmp_path / "current-qr.pt"
    original, _ = _save_bundle(current, None)
    path = _rewrite_payload(current, tmp_path / "v7-qr.pt", _declare_v7_qr_schema)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert payload["format_version"] == 7
    assert payload["model_config"]["ple_backend"] == "qr"
    assert "backbone.embeddings.layers.0.input_embedding.weight" in payload["model_state"]
    assert "backbone.embeddings.layers.1.remainder_embedding.weight" in payload["model_state"]
    assert "backbone.embeddings.layers.1.quotient_embedding.weight" in payload["model_state"]

    loaded = load_checkpoint(path)[0].eval()
    assert loaded.config.ple_backend == "qr"
    prompt = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        torch.testing.assert_close(loaded(prompt).logits, original(prompt).logits)


def test_adapter_presence_must_match_config(tmp_path: Path) -> None:
    vocab = train([Sample("alpha beta gamma delta " * 8)])
    recurrent_inputs = _bundle_inputs(vocab, _RECURRENCE)
    flat_inputs = _bundle_inputs(vocab, None)

    torch.manual_seed(0)
    recurrent = build_spalmer_model(*recurrent_inputs).to(dtype=torch.bfloat16).eval()
    recurrent.backbone.recurrence = None
    with pytest.raises(ValueError, match="recurrence configuration disagrees"):
        save_checkpoint(
            tmp_path / "no-adapter.pt",
            recurrent,
            vocab,
            kda_config=recurrent_inputs[1],
            mla_config=recurrent_inputs[2],
            experts_config=recurrent_inputs[3],
        )

    torch.manual_seed(0)
    flat = build_spalmer_model(*flat_inputs).to(dtype=torch.bfloat16).eval()
    flat.backbone.recurrence = LatentRecurrence(flat.config.d_model)
    with pytest.raises(ValueError, match="recurrence configuration disagrees"):
        save_checkpoint(
            tmp_path / "stray-adapter.pt",
            flat,
            vocab,
            kda_config=flat_inputs[1],
            mla_config=flat_inputs[2],
            experts_config=flat_inputs[3],
        )
