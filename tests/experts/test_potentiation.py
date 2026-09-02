from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from spalmer.attention import KDAConfig, MLAConfig
from spalmer.checkpoint import load_checkpoint, save_checkpoint
from spalmer.config import SPALMERConfig
from spalmer.experts import ExpertPotentiationController, MicroExpertBank, MicroExpertsConfig
from spalmer.factory import build_spalmer_model
from spalmer.modeling import _attach_surprise_telemetry
from spalmer.nn import fake_quantize_low_bit
from spalmer.tokenizer import Encoder, Sample, train


def _config(**overrides: object) -> MicroExpertsConfig:
    values: dict[str, object] = {
        "d_model": 4,
        "num_experts": 4,
        "expert_inter_dim": 2,
        "expert_weight_format": "legacy_int",
        "expert_activation_format": "bfloat16",
        "expert_master_dtype": "float32",
        "expert_qat_backend": "reference",
        "expert_promotion_format": "bfloat16",
        "potentiation_budget": 2,
        "potentiation_ema_decay": 0.0,
        "potentiation_warmup_steps": 1,
        "potentiation_hold_steps": 1,
        "potentiation_hysteresis": 0.0,
    }
    values.update(overrides)
    return MicroExpertsConfig(**values)


def test_surprise_calibration_matches_weighted_mixture_and_shifted_nll() -> None:
    scores = torch.tensor(
        [[[0.2, 0.8, 1.4], [0.3, 0.9, 1.5], [0.4, 1.0, 1.6]]],
        requires_grad=True,
    )
    weight_logits = torch.tensor(
        [[[1.0, -1.0], [0.5, -0.5], [0.25, -0.25]]],
        requires_grad=True,
    )
    weights = weight_logits.softmax(dim=-1)
    expert_ids = torch.tensor([[[0, 1], [1, 2], [0, 2]]])
    target_nll = torch.tensor([[1.25, 2.5]], requires_grad=True)
    valid = torch.tensor([[True, False]])

    enriched, calibration = _attach_surprise_telemetry(
        (
            {
                "router_scores": scores,
                "expert_ids": expert_ids,
                "routing_weights": weights,
            },
        ),
        target_nll,
        valid,
    )

    selected = scores.gather(-1, expert_ids)[:, :-1]
    predicted = (selected * weights[:, :-1]).sum(dim=-1)
    expected = F.smooth_l1_loss(predicted[valid], target_nll.detach()[valid])
    torch.testing.assert_close(calibration, expected)
    assert enriched[0]["expert_attributed_nll"].shape == (3,)
    assert enriched[0]["potentiation_utilization"].shape == (3,)

    calibration.backward()
    assert scores.grad is not None and scores.grad.abs().sum() > 0
    assert weight_logits.grad is not None and weight_logits.grad.abs().sum() > 0
    assert target_nll.grad is None


def test_promoted_expert_bypasses_low_bit_substrate_and_demotion_restores_it() -> None:
    config = _config(num_experts=2, potentiation_budget=1)
    bank = MicroExpertBank(config).eval()
    with torch.no_grad():
        values = torch.linspace(-0.37, 0.43, bank.gate_proj.numel())
        bank.gate_proj.copy_(values.reshape_as(bank.gate_proj))
        bank.up_proj.copy_((values * 0.7).reshape_as(bank.up_proj))
        down = torch.linspace(-0.29, 0.41, bank.down_proj.numel())
        bank.down_proj.copy_(down.reshape_as(bank.down_proj))
    hidden = torch.tensor([[0.2, -0.4, 0.6, -0.8]])
    unpromoted = torch.zeros(2, dtype=torch.bool)
    promoted = unpromoted.clone()
    promoted[0] = True

    low_output = bank.expert_forward(hidden, 0, unpromoted)
    high_output = bank.expert_forward(hidden, 0, promoted)
    demoted_output = bank.expert_forward(hidden, 0, unpromoted)

    def quantized(parameter: torch.Tensor) -> torch.Tensor:
        return fake_quantize_low_bit(
            parameter[0],
            bits=config.expert_quant_bits,
            stochastic=False,
            straight_through=False,
        )

    manual_low = (
        F.silu(hidden @ quantized(bank.gate_proj)) * (hidden @ quantized(bank.up_proj))
    ) @ quantized(bank.down_proj)
    manual_high = (
        F.silu(hidden @ bank.gate_proj[0]) * (hidden @ bank.up_proj[0])
    ) @ bank.down_proj[0]
    torch.testing.assert_close(low_output, manual_low)
    torch.testing.assert_close(high_output, manual_high)
    torch.testing.assert_close(demoted_output, low_output)
    assert not torch.allclose(low_output, high_output)

    bank.train()
    bank.zero_grad(set_to_none=True)
    bank.expert_forward(hidden, 0, unpromoted).sum().backward()
    assert bank.gate_proj.grad is not None and bank.gate_proj.grad[0].abs().sum() > 0
    assert bank.up_proj.grad is not None and bank.up_proj.grad[0].abs().sum() > 0
    assert bank.down_proj.grad is not None and bank.down_proj.grad[0].abs().sum() > 0


def test_controller_keeps_separate_signals_and_persists_exact_promoted_set() -> None:
    config = _config()
    controller = ExpertPotentiationController(config)
    promoted = controller.observe(
        utilization=torch.tensor([0.6, 0.3, 0.1, 0.0]),
        attributed_nll=torch.tensor([2.0, 4.0, 9.0, 100.0]),
        quantization_error=torch.tensor([0.5, 0.5, 0.2, 100.0]),
    )

    # Precision pressure is utilization * quantization error: [0.3, 0.15, 0.02, 0.0].
    # Attributed NLL is recorded but plays no part in the promotion decision.
    assert set(promoted) == {0, 1}
    assert controller.promoted_mask.sum() == 2
    torch.testing.assert_close(
        controller.scores,
        torch.tensor([0.3, 0.15, 0.02, 0.0]),
    )
    torch.testing.assert_close(
        controller.attributed_nll_ema,
        torch.tensor([2.0, 4.0, 9.0, 100.0]),
    )

    restored = ExpertPotentiationController(config)
    restored.load_state_dict(controller.state_dict())
    assert restored.promoted_ids() == controller.promoted_ids()
    torch.testing.assert_close(restored.utilization_ema, controller.utilization_ema)
    torch.testing.assert_close(restored.attributed_nll_ema, controller.attributed_nll_ema)
    torch.testing.assert_close(
        restored.quantization_error_ema,
        controller.quantization_error_ema,
    )


def test_attributed_nll_does_not_decide_promotion() -> None:
    controller = ExpertPotentiationController(_config(potentiation_budget=1))
    (promoted,) = controller.observe(
        utilization=torch.tensor([0.5, 0.5, 0.0, 0.0]),
        attributed_nll=torch.tensor([9.0, 1.0, 0.0, 0.0]),
        quantization_error=torch.tensor([0.1, 0.4, 0.0, 0.0]),
    )

    # Expert 0 is the most surprised, but expert 1 is the one that is
    # precision-limited, so expert 1 is promoted (ledger C10 signal roles).
    assert promoted == 1


def test_controller_does_not_promote_without_observed_precision_pressure() -> None:
    controller = ExpertPotentiationController(_config())
    zeros = torch.zeros(4)
    assert controller.observe(zeros, zeros, zeros) == ()
    assert not controller.promoted_mask.any()


def test_controller_ema_remains_fp32_when_model_is_cast_to_bfloat16() -> None:
    controller = ExpertPotentiationController(_config()).to(dtype=torch.bfloat16)
    assert controller.utilization_ema.dtype == torch.float32
    assert controller.attributed_nll_ema.dtype == torch.float32
    assert controller.quantization_error_ema.dtype == torch.float32


def test_checkpoint_restores_one_authoritative_controller_and_effective_logits(
    tmp_path,
) -> None:
    vocab = train([Sample("alpha beta gamma delta " * 8)])
    model_config = SPALMERConfig(
        vocab_size=len(vocab),
        d_model=8,
        n_layers=4,
        tokenizer_version=vocab.version,
        tokenizer_fingerprint=vocab.fingerprint,
        ple_expansion_factor=1,
    )
    kda_config = KDAConfig(
        hidden_size=8,
        num_heads=2,
        head_k_dim=4,
        head_v_dim=4,
        backend="reference",
    )
    mla_config = MLAConfig(
        hidden_size=8,
        num_heads=2,
        head_k_dim=4,
        head_v_dim=4,
        q_latent_dim=4,
        kv_latent_dim=4,
    )
    experts_config = _config(d_model=8, expert_inter_dim=3)
    model = build_spalmer_model(
        model_config,
        kda_config,
        mla_config,
        experts_config,
    ).eval()
    controller = model.potentiation_controller
    assert controller is not None
    controller.observe(
        torch.tensor([0.7, 0.2, 0.1, 0.0]),
        torch.tensor([3.0, 4.0, 2.0, 0.0]),
        torch.tensor([0.4, 0.3, 0.1, 0.0]),
    )
    for block in model.backbone.blocks:
        assert block.channel_mixer.potentiation_controller is controller

    input_ids = torch.tensor([Encoder(vocab).encode("alpha beta")])
    expected_logits = model(input_ids).logits
    path = tmp_path / "potentiated.pt"
    save_checkpoint(
        path,
        model,
        vocab,
        kda_config=kda_config,
        mla_config=mla_config,
        experts_config=experts_config,
    )

    with pytest.raises(ValueError, match="does not match"):
        save_checkpoint(
            tmp_path / "mismatched.pt",
            model,
            vocab,
            kda_config=kda_config,
            mla_config=mla_config,
            experts_config=replace(experts_config, expert_quant_bits=5),
        )
    with pytest.raises(ValueError, match="KDA config does not match"):
        save_checkpoint(
            tmp_path / "mismatched-kda.pt",
            model,
            vocab,
            kda_config=replace(kda_config, backend="auto"),
            mla_config=mla_config,
            experts_config=experts_config,
        )

    restored, _, _ = load_checkpoint(path)
    restored.eval()

    assert restored.promoted_expert_ids() == model.promoted_expert_ids()
    for block in restored.backbone.blocks:
        assert block.channel_mixer.potentiation_controller is restored.potentiation_controller
    torch.testing.assert_close(restored(input_ids).logits, expected_logits)

    over_budget_payload = torch.load(path, weights_only=False)
    over_budget_payload["model_state"]["potentiation_controller.promoted_mask"].fill_(True)
    over_budget = tmp_path / "over-budget.pt"
    torch.save(over_budget_payload, over_budget)
    with pytest.raises(ValueError, match="exceeds budget"):
        load_checkpoint(over_budget)

    payload = torch.load(path, weights_only=False)
    payload["model_state"].pop("potentiation_controller.promoted_mask")
    damaged = tmp_path / "damaged.pt"
    torch.save(payload, damaged)
    with pytest.raises(RuntimeError, match="promoted_mask"):
        load_checkpoint(damaged)

    missing_config_payload = torch.load(path, weights_only=False)
    missing_config_payload["experts_config"].pop("expert_quant_bits")
    missing_config = tmp_path / "missing-config.pt"
    torch.save(missing_config_payload, missing_config)
    with pytest.raises(ValueError, match="expert_quant_bits"):
        load_checkpoint(missing_config)

    legacy_payload = torch.load(path, weights_only=False)
    legacy_payload["format_version"] = 1
    legacy_payload["model_config"].pop("surprise_ema_decay")
    legacy_payload["model_state"].pop("surprise_ema")
    legacy_payload["model_state"].pop("surprise_observations")
    for name in list(legacy_payload["model_state"]):
        if name.startswith("potentiation_controller."):
            legacy_payload["model_state"].pop(name)
    for name in (
        "expert_fake_quantization",
        "expert_quant_bits",
        "expert_stochastic_rounding",
        "expert_weight_format",
        "expert_activation_format",
        "expert_master_dtype",
        "expert_qat_backend",
        "expert_promotion_format",
        "potentiation_budget",
        "potentiation_ema_decay",
        "potentiation_warmup_steps",
        "potentiation_hold_steps",
        "potentiation_hysteresis",
        "router_score_transform",
        "residency_increment",
        "residency_min_gain",
        # Fields introduced with the shared-path / residency architecture.
        "shared_inter_dim",
        "min_resident_experts",
        "max_resident_experts",
        "expert_execution",
    ):
        legacy_payload["experts_config"].pop(name, None)
    legacy_path = tmp_path / "legacy-v1.pt"
    torch.save(legacy_payload, legacy_path)
    legacy_model, _, _ = load_checkpoint(legacy_path)
    legacy_controller = legacy_model.potentiation_controller
    assert legacy_controller is not None
    assert legacy_controller.config.router_score_transform == "identity"
    assert not legacy_controller.config.expert_fake_quantization
    assert legacy_controller.config.expert_weight_format == "legacy_int"
    assert legacy_controller.config.expert_master_dtype == "float32"
    assert legacy_controller.config.potentiation_budget == 0

    damaged_legacy_payload = torch.load(legacy_path, weights_only=False)
    damaged_legacy_payload["experts_config"].pop("active_experts")
    damaged_legacy_path = tmp_path / "damaged-legacy-v1.pt"
    torch.save(damaged_legacy_payload, damaged_legacy_path)
    with pytest.raises(ValueError, match="active_experts"):
        load_checkpoint(damaged_legacy_path)
