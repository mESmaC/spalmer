"""Shared configuration contracts for the first SPALMER prototype."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class PLEConfig:
    """Configuration for fixed alternating per-layer lookup embeddings."""

    vocab_size: int
    d_model: int
    n_layers: int
    expansion_factor: int = 4
    quant_bits: int = 4
    stochastic_rounding: bool = True
    sparse_gradients: bool = False
    gate_init: float | None = None
    alternation_policy: Literal["fixed_layer_parity"] = "fixed_layer_parity"
    backend: Literal["fake_qat"] = "fake_qat"
    reference_max_numel: int | None = 100_000_000
    quant_eps: float = 1e-8
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        _require_positive("vocab_size", self.vocab_size)
        _require_positive("d_model", self.d_model)
        _require_positive("n_layers", self.n_layers)
        _require_positive("expansion_factor", self.expansion_factor)
        if not 2 <= self.quant_bits <= 8:
            raise ValueError("quant_bits must be between 2 and 8")
        if self.quant_eps <= 0:
            raise ValueError("quant_eps must be positive")
        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be positive")
        if self.reference_max_numel is not None and self.reference_max_numel <= 0:
            raise ValueError("reference_max_numel must be positive or None")
        if self.alternation_policy != "fixed_layer_parity":
            raise ValueError(f"unsupported alternation policy: {self.alternation_policy}")
        if self.backend != "fake_qat":
            raise ValueError(f"unsupported PLE backend: {self.backend}")

    @property
    def resolved_gate_init(self) -> float:
        """Scale repeated lexical injections without erasing them at startup."""

        if self.gate_init is not None:
            return self.gate_init
        return 1.0 / math.sqrt(self.n_layers)

    def phase_for_layer(self, layer_index: int) -> str:
        if layer_index < 0:
            raise ValueError("layer_index must be non-negative")
        if self.alternation_policy != "fixed_layer_parity":
            raise ValueError(f"unsupported alternation policy: {self.alternation_policy}")
        return "A" if layer_index % 2 == 0 else "B"

    @property
    def reference_shadow_numel(self) -> int:
        """Floating shadow-weight count used by the fake-QAT reference bank."""

        return self.n_layers * self.vocab_size * self.expansion_factor * self.d_model

    def reference_shadow_bytes(self, *, bytes_per_element: int = 4) -> int:
        if bytes_per_element <= 0:
            raise ValueError("bytes_per_element must be positive")
        return self.reference_shadow_numel * bytes_per_element

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SPALMERConfig:
    """Model-wide values whose ownership crosses component boundaries."""

    vocab_size: int
    d_model: int
    n_layers: int
    tokenizer_version: int
    tokenizer_fingerprint: str
    ple_expansion_factor: int = 4
    ple_quant_bits: int = 4
    ple_stochastic_rounding: bool = True
    ple_sparse_gradients: bool = False
    ple_gate_init: float | None = None
    ple_alternation_policy: Literal["fixed_layer_parity"] = "fixed_layer_parity"
    ple_backend: Literal["fake_qat"] = "fake_qat"
    ple_reference_max_numel: int | None = 100_000_000
    norm_eps: float = 1e-6
    initializer_range: float = 0.02
    token_mixer_pattern: tuple[str, ...] = ("kda", "kda", "kda", "mla")
    # Averaging window of the shared "average surprise" signal (ledger C08/C13):
    # an exponential moving average of realized next-token NLL seen in training.
    surprise_ema_decay: float = 0.99

    def __post_init__(self) -> None:
        _require_positive("vocab_size", self.vocab_size)
        _require_positive("d_model", self.d_model)
        _require_positive("n_layers", self.n_layers)
        _require_positive("tokenizer_version", self.tokenizer_version)
        if not self.tokenizer_fingerprint.strip():
            raise ValueError("tokenizer_fingerprint cannot be empty")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")
        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be positive")
        if not 0 <= self.surprise_ema_decay < 1:
            raise ValueError("surprise_ema_decay must be in [0, 1)")
        if not self.token_mixer_pattern:
            raise ValueError("token_mixer_pattern cannot be empty")
        unsupported = set(self.token_mixer_pattern) - {"kda", "mla"}
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"unsupported token mixer names: {names}")
        # Constructing this also validates every PLE-specific field.
        self.ple_config()

    def ple_config(self) -> PLEConfig:
        return PLEConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            expansion_factor=self.ple_expansion_factor,
            quant_bits=self.ple_quant_bits,
            stochastic_rounding=self.ple_stochastic_rounding,
            sparse_gradients=self.ple_sparse_gradients,
            gate_init=self.ple_gate_init,
            alternation_policy=self.ple_alternation_policy,
            backend=self.ple_backend,
            reference_max_numel=self.ple_reference_max_numel,
            quant_eps=1e-8,
            initializer_range=self.initializer_range,
        )

    def token_mixer_for_layer(self, layer_index: int) -> str:
        if not 0 <= layer_index < self.n_layers:
            raise IndexError(f"layer index {layer_index} is outside [0, {self.n_layers})")
        return self.token_mixer_pattern[layer_index % len(self.token_mixer_pattern)]

    def assert_tokenizer_compatible(self, *, version: int, fingerprint: str) -> None:
        """Fail closed when a model is paired with a different tokenizer artifact."""

        if version != self.tokenizer_version or fingerprint != self.tokenizer_fingerprint:
            raise ValueError(
                "tokenizer identity mismatch: model expects "
                f"version={self.tokenizer_version}, fingerprint={self.tokenizer_fingerprint!r}; "
                f"received version={version}, fingerprint={fingerprint!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
