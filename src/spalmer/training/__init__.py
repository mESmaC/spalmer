"""GPU-first training infrastructure (construction only until explicitly run)."""

from spalmer.training.config import TrainingConfig
from spalmer.training.data import MMapBatchSource
from spalmer.training.device import RuntimeDevice, resolve_runtime_device, seed_everything
from spalmer.training.engine import (
    CausalBatch,
    ExperimentTrainer,
    StatefulBatchSource,
    TrainerProgress,
    TrainStepMetrics,
    initialize_model,
)
from spalmer.training.evaluation import ModelEvaluationReport, evaluate_model_batches
from spalmer.training.optim import (
    OptimizerBundle,
    ParameterGroups,
    build_optimizers,
    classify_parameters,
    gradients_are_finite,
)
from spalmer.training.resume import (
    build_artifact_hashes,
    canonical_sha256,
    capture_trainer_run_state,
    restore_trainer_run_state,
)

__all__ = [
    "OptimizerBundle",
    "ParameterGroups",
    "RuntimeDevice",
    "MMapBatchSource",
    "ModelEvaluationReport",
    "CausalBatch",
    "ExperimentTrainer",
    "StatefulBatchSource",
    "TrainStepMetrics",
    "TrainerProgress",
    "TrainingConfig",
    "build_optimizers",
    "build_artifact_hashes",
    "canonical_sha256",
    "capture_trainer_run_state",
    "classify_parameters",
    "evaluate_model_batches",
    "gradients_are_finite",
    "initialize_model",
    "resolve_runtime_device",
    "restore_trainer_run_state",
    "seed_everything",
]
