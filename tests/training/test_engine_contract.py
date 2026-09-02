from __future__ import annotations

import weakref

import pytest
import torch
from torch import nn

from spalmer.modeling import CausalLMOutput
from spalmer.training import CausalBatch, ExperimentTrainer, OptimizerBundle, TrainingConfig


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


class _AccumulationBatchSource:
    def __init__(self) -> None:
        self._batches = iter(
            (
                CausalBatch(torch.tensor([[1, 2, 3]])),
                CausalBatch(
                    torch.tensor([[4, 5, 6]]),
                    labels=torch.tensor([[4, -100, 6]]),
                ),
            )
        )

    def next_batch(self, *, batch_size: int, sequence_length: int) -> CausalBatch:
        assert batch_size == 1
        assert sequence_length == 2
        return next(self._batches)

    def state_dict(self) -> dict[str, int]:
        return {}

    def load_state_dict(self, state: object) -> None:
        del state


class _AccumulationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.backbone = nn.Module()
        self.backbone.blocks = nn.ModuleList()
        self.average_surprise = 7.5
        self.calls = 0
        self.full_output_refs: list[weakref.ReferenceType[torch.Tensor]] = []
        self.aggregated_metrics: tuple[dict[str, torch.Tensor], ...] | None = None

    def forward(self, input_ids: torch.Tensor, **_kwargs: object) -> CausalLMOutput:
        values = (
            {
                "loss": 2.0,
                "auxiliary": 4.0,
                "calibration": 6.0,
                "entropy": 8.0,
                "utilization": [0.75, 0.25],
                "nll": [2.0, 4.0],
                "quantization": [0.1, 0.0],
            },
            {
                "loss": 4.0,
                "auxiliary": 8.0,
                "calibration": 10.0,
                "entropy": 12.0,
                "utilization": [0.0, 1.0],
                "nll": [0.0, 10.0],
                "quantization": [0.0, 0.3],
            },
        )[self.calls]
        self.calls += 1
        anchor = self.weight * 0.0
        logits = anchor + torch.zeros((*input_ids.shape, 32), device=input_ids.device)
        router_scores = anchor + torch.zeros((*input_ids.shape, 2), device=input_ids.device)
        self.full_output_refs.extend((weakref.ref(logits), weakref.ref(router_scores)))

        def scalar(name: str) -> torch.Tensor:
            return anchor + float(values[name])

        def vector(name: str) -> torch.Tensor:
            return anchor + torch.tensor(values[name], device=input_ids.device)

        return CausalLMOutput(
            logits=logits,
            token_mixer_states=(),
            channel_mixer_states=(),
            loss=scalar("loss"),
            auxiliary_loss=scalar("auxiliary"),
            surprise_calibration_loss=scalar("calibration"),
            predictive_entropy=scalar("entropy"),
            layer_metrics=(
                {
                    "router_scores": router_scores,
                    "potentiation_utilization": vector("utilization"),
                    "expert_attributed_nll": vector("nll"),
                    "expert_quantization_error": vector("quantization"),
                },
            ),
        )

    def update_potentiation(self, layer_metrics):
        assert all(reference() is None for reference in self.full_output_refs)
        self.aggregated_metrics = tuple(dict(metrics) for metrics in layer_metrics)
        return (1,)


def test_gradient_accumulation_aggregates_potentiation_without_retaining_outputs() -> None:
    model = _AccumulationModel()
    optimizer = OptimizerBundle(
        dense=torch.optim.SGD(model.parameters(), lr=1e-3),
        sparse=None,
    )
    trainer = ExperimentTrainer(
        model,
        _AccumulationBatchSource(),
        TrainingConfig(
            max_steps=1,
            micro_batch_size=1,
            sequence_length=2,
            gradient_accumulation_steps=2,
            auxiliary_loss_weight=0.1,
            surprise_calibration_weight=0.2,
            gradient_clip=None,
            device="cpu",
            require_cuda=False,
            compute_dtype="float32",
            parameter_dtype="float32",
        ),
        optimizer=optimizer,
    )

    metrics = next(trainer.run())

    assert metrics.tokens_seen == 3
    assert metrics.objective == pytest.approx(5.2)
    assert metrics.model_loss == pytest.approx(3.0)
    assert metrics.auxiliary_loss == pytest.approx(6.0)
    assert metrics.surprise_calibration_loss == pytest.approx(8.0)
    assert metrics.predictive_entropy == pytest.approx(10.0)
    assert metrics.promoted_experts == (1,)
    assert model.aggregated_metrics is not None
    layer = model.aggregated_metrics[0]
    torch.testing.assert_close(layer["potentiation_utilization"], torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(layer["expert_attributed_nll"], torch.tensor([2.0, 8.0]))
    torch.testing.assert_close(layer["expert_quantization_error"], torch.tensor([0.1, 0.3]))
    assert all(not value.requires_grad for value in layer.values())
