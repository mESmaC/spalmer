"""Shared configuration contracts for the first SPALMER prototype."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class PLEConfig:
    """Configuration for exact-input plus QR-compositional layer embeddings."""

    vocab_size: int
    d_model: int
    n_layers: int
    expansion_factor: int = 4
    gate_init: float | None = None
    backend: Literal["qr"] = "qr"
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        _require_positive("vocab_size", self.vocab_size)
        _require_positive("d_model", self.d_model)
        _require_positive("n_layers", self.n_layers)
        _require_positive("expansion_factor", self.expansion_factor)
        if self.backend != "qr":
            raise ValueError(
                "fake-QAT PLE is retired; use backend='qr'. "
                "There is no executable emulation fallback."
            )
        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be positive")

    @property
    def resolved_gate_init(self) -> float:
        """Scale repeated lexical injections without erasing them at startup."""

        if self.gate_init is not None:
            return self.gate_init
        return 1.0 / math.sqrt(self.n_layers)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RecurrenceConfig:
    """Depth-recurrent core split of a physical block stack (Huginn-style).

    ``prelude_layers`` blocks run once, ``core_layers`` blocks are iterated
    ``r`` times over a latent state, and ``coda_layers`` blocks run once on the
    final latent. The three counts describe *physical* blocks and must sum to
    :attr:`SPALMERConfig.n_layers`; ``n_layers`` keeps meaning physical blocks
    everywhere (PLE tables, token-mixer pattern, checkpoint validators).
    """

    prelude_layers: int
    core_layers: int
    coda_layers: int
    default_steps: int = 8
    latent_init_std: float = 1.0
    adapter: Literal["linear_concat"] = "linear_concat"
    adapter_init: Literal["identity_mix", "random"] = "identity_mix"

    def __post_init__(self) -> None:
        _require_positive("prelude_layers", self.prelude_layers)
        _require_positive("core_layers", self.core_layers)
        _require_positive("coda_layers", self.coda_layers)
        _require_positive("default_steps", self.default_steps)
        if self.latent_init_std < 0 or not math.isfinite(self.latent_init_std):
            raise ValueError("latent_init_std must be finite and non-negative")
        if self.adapter != "linear_concat":
            raise ValueError(f"unsupported recurrence adapter: {self.adapter}")
        if self.adapter_init not in {"identity_mix", "random"}:
            raise ValueError(f"unsupported recurrence adapter_init: {self.adapter_init}")

    @property
    def total_layers(self) -> int:
        return self.prelude_layers + self.core_layers + self.coda_layers

    def core_range(self) -> range:
        return range(self.prelude_layers, self.prelude_layers + self.core_layers)

    def role(self, layer_index: int) -> Literal["prelude", "core", "coda"]:
        if not 0 <= layer_index < self.total_layers:
            raise IndexError(f"layer index {layer_index} is outside [0, {self.total_layers})")
        if layer_index < self.prelude_layers:
            return "prelude"
        if layer_index < self.prelude_layers + self.core_layers:
            return "core"
        return "coda"

    def effective_depth(self, steps: int) -> int:
        """Block passes one token makes at ``steps`` recurrence iterations."""

        _require_positive("steps", steps)
        return self.prelude_layers + steps * self.core_layers + self.coda_layers

    def parameter_count(self, d_model: int) -> int:
        """Adapter projection plus the injection and latent RMSNorm weights."""

        return 2 * d_model * d_model + 2 * d_model

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
    ple_gate_init: float | None = None
    ple_backend: Literal["qr"] = "qr"
    norm_eps: float = 1e-6
    initializer_range: float = 0.02
    token_mixer_pattern: tuple[str, ...] = ("kda", "kda", "kda", "mla")
    # Averaging window of the shared "average surprise" signal (ledger C08/C13):
    # an exponential moving average of realized next-token NLL seen in training.
    surprise_ema_decay: float = 0.99
    # Optional depth-recurrent core (Huginn-style latent reasoning). ``None``
    # keeps the model a plain physical stack, bit-for-bit as before.
    recurrence: RecurrenceConfig | None = None

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
        if self.ple_backend != "qr":
            raise ValueError(
                "fake-QAT PLE is retired; use ple_backend='qr'. "
                "There is no executable emulation fallback."
            )
        if isinstance(self.recurrence, Mapping):
            # Checkpoint round-trip: ``to_dict`` is ``asdict``, so the nested
            # dataclass arrives as a plain mapping on reconstruction.
            object.__setattr__(self, "recurrence", RecurrenceConfig(**self.recurrence))
        if self.recurrence is not None and self.recurrence.total_layers != self.n_layers:
            raise ValueError(
                "recurrence layers prelude+core+coda="
                f"{self.recurrence.total_layers} must equal n_layers={self.n_layers}"
            )
        # Constructing this also validates every PLE-specific field.
        self.ple_config()

    def ple_config(self) -> PLEConfig:
        return PLEConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            expansion_factor=self.ple_expansion_factor,
            gate_init=self.ple_gate_init,
            backend=self.ple_backend,
            initializer_range=self.initializer_range,
        )

    def token_mixer_for_layer(self, layer_index: int) -> str:
        if not 0 <= layer_index < self.n_layers:
            raise IndexError(f"layer index {layer_index} is outside [0, {self.n_layers})")
        return self.token_mixer_pattern[layer_index % len(self.token_mixer_pattern)]

    def block_role(self, layer_index: int) -> Literal["prelude", "core", "coda", "stack"]:
        """Role of one physical block; ``"stack"`` when the model is not recurrent."""

        if not 0 <= layer_index < self.n_layers:
            raise IndexError(f"layer index {layer_index} is outside [0, {self.n_layers})")
        if self.recurrence is None:
            return "stack"
        return self.recurrence.role(layer_index)

    @property
    def core_layer_indices(self) -> tuple[int, ...]:
        """Physical indices of the iterated core blocks (empty when not recurrent)."""

        if self.recurrence is None:
            return ()
        return tuple(self.recurrence.core_range())

    def effective_depth(self, steps: int | None = None) -> int:
        """Block passes per token; ``n_layers`` when the model is not recurrent."""

        if self.recurrence is None:
            return self.n_layers
        return self.recurrence.effective_depth(steps or self.recurrence.default_steps)

    @property
    def core_mixer_counts(self) -> dict[str, int]:
        """Token-mixer composition of the recurrent core, e.g. ``{"kda": 3, "mla": 1}``."""

        counts = {"kda": 0, "mla": 0}
        for index in self.core_layer_indices:
            counts[self.token_mixer_for_layer(index)] += 1
        return counts

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
