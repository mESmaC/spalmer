"""Construction adapters for total-parameter :class:`ScalePlan` objects.

The scale planner is the sole source of architecture shapes and parameter
budgets. A name such as ``10M`` denotes a *total* target, including the PLE
tables and untied vocabulary head; this module contains no second set of
hard-coded, non-embedding shapes.

No model is allocated by :func:`build_configs`. The returned bundle exposes
an explicit :meth:`ConfigBundle.build` boundary for callers that intentionally
want to construct the planned architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from spalmer.attention import KDAConfig, MLAConfig
from spalmer.config import SPALMERConfig
from spalmer.directional import DirectionalConfig
from spalmer.experiment.planning import (
    ATXYScaleConfig,
    DirectionalScaleConfig,
    RecurrenceScaleConfig,
    ScalePlan,
    ScaleSearchSpace,
    count_parameters,
    plan_configuration,
)
from spalmer.experts import MicroExpertsConfig
from spalmer.memory import ATXYConfig

if TYPE_CHECKING:
    from spalmer.experts import ParameterAccounting
    from spalmer.modeling import SPALMERCausalLM


TOTAL_PARAMETER_TARGETS: dict[str, int] = {
    "10M": 10_000_000,
    "50M": 50_000_000,
    "100M": 100_000_000,
}


def plan_named_scale(
    name: str,
    *,
    vocab_size: int,
    search_space: ScaleSearchSpace | None = None,
    expert_weight_format: Literal["mxfp4", "nvfp4"] = "mxfp4",
    directional: DirectionalScaleConfig | None = None,
    atxy: ATXYScaleConfig | None = None,
    recurrence: RecurrenceScaleConfig | None = None,
    ple_backend: Literal["fake_qat", "qr"] = "fake_qat",
) -> ScalePlan:
    """Plan a conventional experiment rung as an inclusive total budget."""

    try:
        target = TOTAL_PARAMETER_TARGETS[name]
    except KeyError as error:
        raise KeyError(
            f"unknown scale {name!r}; choose from {sorted(TOTAL_PARAMETER_TARGETS)}"
        ) from error
    kwargs: dict[str, Any] = {
        "expert_weight_format": expert_weight_format,
        "directional": directional,
        "atxy": atxy,
        "recurrence": recurrence,
        "ple_backend": ple_backend,
    }
    if search_space is not None:
        kwargs["search_space"] = search_space
    return plan_configuration(target, vocab_size, **kwargs)


@dataclass(frozen=True, slots=True)
class ConfigBundle:
    """Configuration objects consumed by ``build_spalmer_model``."""

    plan: ScalePlan
    model: SPALMERConfig
    kda: KDAConfig
    mla: MLAConfig
    experts: MicroExpertsConfig
    directional: DirectionalConfig | None = None
    atxy: ATXYConfig | None = None

    def build(self) -> SPALMERCausalLM:
        """Construct the model and establish the plan's persistent master dtype."""

        import torch

        from spalmer.factory import build_spalmer_model

        model = build_spalmer_model(
            self.model,
            self.kda,
            self.mla,
            self.experts,
            directional_config=self.directional,
            atxy_config=self.atxy,
        )
        dtype = (
            torch.bfloat16
            if self.experts.expert_master_dtype == "bfloat16"
            else torch.float32
        )
        return model.to(dtype=dtype)


_MODEL_SHAPE_FIELDS = {
    "vocab_size",
    "d_model",
    "n_layers",
    "ple_expansion_factor",
    "ple_quant_bits",
    "ple_backend",
    "token_mixer_pattern",
    "recurrence",
}
_EXPERT_SHAPE_FIELDS = {
    "d_model",
    "num_experts",
    "expert_inter_dim",
    "shared_inter_dim",
    "active_experts",
    "min_active_experts",
    "max_active_experts",
    "min_resident_experts",
    "max_resident_experts",
    "router_bias",
    "expert_weight_format",
    "expert_activation_format",
    "expert_master_dtype",
    "expert_fake_quantization",
}


def _reject_plan_overrides(kind: str, overrides: dict[str, Any], protected: set[str]) -> None:
    conflicts = sorted(protected.intersection(overrides))
    if conflicts:
        names = ", ".join(conflicts)
        raise ValueError(f"{kind} overrides cannot change ScalePlan fields: {names}")


def build_configs(
    plan: ScalePlan,
    *,
    tokenizer_version: int,
    tokenizer_fingerprint: str,
    attention_backend: str = "auto",
    experts_overrides: dict[str, Any] | None = None,
    model_overrides: dict[str, Any] | None = None,
) -> ConfigBundle:
    """Convert one internally consistent :class:`ScalePlan` into configs.

    Architecture/count-changing overrides are rejected so the plan remains
    authoritative. Runtime controls such as potentiation policy, QAT backend,
    and surprise calibration may still be supplied through the two override
    mappings.
    """

    expected = count_parameters(plan.config)
    if expected != plan.parameters:
        raise ValueError("ScalePlan parameter breakdown does not match its config")
    config = plan.config

    model_extra = dict(model_overrides or {})
    expert_extra = dict(experts_overrides or {})
    _reject_plan_overrides("model", model_extra, _MODEL_SHAPE_FIELDS)
    _reject_plan_overrides("expert", expert_extra, _EXPERT_SHAPE_FIELDS)

    model_fields: dict[str, Any] = {
        "vocab_size": config.vocab_size,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "tokenizer_version": tokenizer_version,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "ple_expansion_factor": config.ple_expansion,
        "ple_quant_bits": 4,
        "token_mixer_pattern": ("kda", "kda", "kda", "mla"),
    }
    if config.ple_backend != "fake_qat":
        # Legacy plans keep constructing the exact historical config; only a
        # QR-PLE plan names its backend.
        model_fields["ple_backend"] = config.ple_backend
    if config.recurrence is not None:
        # Imported lazily so a non-recurrent build never depends on the
        # recurrence config type being present.
        from spalmer.config import RecurrenceConfig

        model_fields["recurrence"] = RecurrenceConfig(
            prelude_layers=config.recurrence.prelude_layers,
            core_layers=config.recurrence.core_layers(config.n_layers),
            coda_layers=config.recurrence.coda_layers,
            default_steps=config.recurrence.default_steps,
            latent_init_std=config.recurrence.latent_init_std,
        )
    model_fields.update(model_extra)
    model = SPALMERConfig(**model_fields)
    kda = KDAConfig(
        hidden_size=config.d_model,
        num_heads=config.num_heads,
        head_k_dim=config.head_dim,
        head_v_dim=config.head_dim,
        conv_width=config.conv_width,
        conv_bias=config.conv_bias,
        backend=attention_backend,
    )
    mla = MLAConfig(
        hidden_size=config.d_model,
        num_heads=config.num_heads,
        head_k_dim=config.head_dim,
        head_v_dim=config.head_dim,
        q_latent_dim=config.resolved_q_latent_dim,
        kv_latent_dim=config.resolved_kv_latent_dim,
    )
    expert_fields: dict[str, Any] = {
        "d_model": config.d_model,
        "num_experts": config.num_experts,
        "expert_inter_dim": config.expert_width,
        "shared_inter_dim": config.resolved_shared_width,
        "active_experts": config.active_experts,
        "max_active_experts": min(config.max_resident_experts, config.num_experts),
        "min_resident_experts": config.min_resident_experts,
        "max_resident_experts": config.max_resident_experts,
        "router_bias": config.router_bias,
        "expert_weight_format": config.expert_weight_format,
        "expert_activation_format": config.expert_activation_format,
        "expert_master_dtype": config.expert_master_dtype,
    }
    expert_fields.update(expert_extra)
    experts = MicroExpertsConfig(**expert_fields)

    directional_config = None
    if config.directional is not None:
        directional_config = DirectionalConfig(
            d_model=config.d_model,
            num_feature_groups=config.directional.num_feature_groups,
            lateral_rank=config.directional.lateral_rank,
            enabled=True,
        )
    atxy_config = None
    if config.atxy is not None:
        atxy_config = ATXYConfig(
            d_model=config.d_model,
            value_dim=config.atxy.value_dim,
            a_cardinality=config.atxy.a_cardinality,
            t_cardinality=config.atxy.t_cardinality,
            x_cardinality=config.atxy.x_cardinality,
            y_cardinality=config.atxy.y_cardinality,
            injection_layer=config.atxy.injection_layer,
        )
    return ConfigBundle(
        plan=plan,
        model=model,
        kda=kda,
        mla=mla,
        experts=experts,
        directional=directional_config,
        atxy=atxy_config,
    )


def assert_accounting_matches_plan(
    plan: ScalePlan,
    accounting: ParameterAccounting,
) -> None:
    """Fail closed when measured ``account_parameters`` differs from a plan."""

    expected = plan.parameters.components
    actual = dict(accounting.components)
    mismatches = {
        name: (expected.get(name, 0), actual.get(name, 0))
        for name in sorted(set(expected) | set(actual))
        if expected.get(name, 0) != actual.get(name, 0)
    }
    expected_per_expert = (
        plan.config.n_layers
        * 3
        * plan.config.d_model
        * plan.config.expert_width
    )
    if accounting.parameters_per_expert != expected_per_expert:
        mismatches["parameters_per_expert"] = (
            expected_per_expert,
            accounting.parameters_per_expert,
        )
    if mismatches:
        detail = ", ".join(
            f"{name}: planned={planned:,} measured={measured:,}"
            for name, (planned, measured) in mismatches.items()
        )
        raise ValueError(f"measured parameter accounting does not match ScalePlan ({detail})")


__all__ = [
    "TOTAL_PARAMETER_TARGETS",
    "ConfigBundle",
    "assert_accounting_matches_plan",
    "build_configs",
    "plan_named_scale",
]
