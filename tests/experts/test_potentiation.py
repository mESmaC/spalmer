from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from spalmer.checkpoint import _validate_potentiation_state
from spalmer.experts import ExpertPotentiationController, MicroExpertBank, MicroExpertsConfig
from spalmer.modeling import _attach_surprise_telemetry
from spalmer.precision import detect_precision_capabilities


def _config(**overrides: object) -> MicroExpertsConfig:
    values: dict[str, object] = {
        "d_model": 4,
        "num_experts": 4,
        "expert_inter_dim": 2,
        "expert_execution": "loop",
        "expert_weight_format": "mxfp8",
        "expert_activation_format": "mxfp8",
        "expert_master_dtype": "bfloat16",
        "expert_qat_backend": "native",
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


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not detect_precision_capabilities("cuda").supports("mxfp8", "mxfp8"),
    reason="verified native MXFP8 forward/backward is unavailable",
)
def test_promoted_expert_switches_complete_identity_to_native_bf16() -> None:
    config = _config(num_experts=2, potentiation_budget=1)
    bank = MicroExpertBank(config).to(device="cuda", dtype=torch.bfloat16).eval()
    hidden = torch.tensor(
        [[0.2, -0.4, 0.6, -0.8]],
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    unpromoted = torch.zeros(2, dtype=torch.bool, device="cuda")
    promoted = unpromoted.clone()
    promoted[0] = True

    low_output = bank.expert_forward(hidden, 0, unpromoted)
    high_output = bank.expert_forward(hidden, 0, promoted)
    assert low_output.shape == high_output.shape == hidden.shape
    assert torch.isfinite(low_output).all()
    assert torch.isfinite(high_output).all()

    (low_output.float().sum() + high_output.float().sum()).backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert bank.gate_proj.grad is not None and torch.isfinite(bank.gate_proj.grad).all()
    assert bank.up_proj.grad is not None and torch.isfinite(bank.up_proj.grad).all()
    assert bank.down_proj.grad is not None and torch.isfinite(bank.down_proj.grad).all()


def test_controller_keeps_separate_signals_and_persists_exact_promoted_set() -> None:
    config = _config()
    controller = ExpertPotentiationController(config)
    promoted = controller.observe(
        utilization=torch.tensor([0.6, 0.3, 0.1, 0.0]),
        attributed_nll=torch.tensor([2.0, 4.0, 9.0, 100.0]),
        quantization_error=torch.tensor([0.5, 0.5, 0.2, 100.0]),
    )

    assert set(promoted) == {0, 1}
    assert controller.promoted_mask.sum() == 2
    torch.testing.assert_close(controller.scores, torch.tensor([0.3, 0.15, 0.02, 0.0]))
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


def test_native_checkpoint_validation_accepts_a_promoted_expert_set() -> None:
    config = _config(potentiation_budget=1)
    controller = ExpertPotentiationController(config)
    controller.promoted_mask[1] = True

    _validate_potentiation_state(
        SimpleNamespace(potentiation_controller=controller),
        config,
    )
