"""ATXY exact-memory path (ledger C03/C04).

Ported from ``agent/atxy-v0`` (7ea07a1) and reconciled with the ledger's
attention coupling, which places the factorized address in the ordinary
residual stream *before* the controller boundary::

    factorized ATXY address at an ADDR/READ position
            -> ordinary residual stream -> unchanged KDA/MLA projections
            -> controller at one selected block boundary
            -> exact, version-checked external lookup
            -> bounded value encoder
            -> gated residual injection at the same active position
            -> remaining unchanged KDA/MLA blocks

Accordingly the module has two entry points the backbone calls separately:

- :meth:`ATXYInjection.embed_addresses` adds ``Project(E_A[A] + E_T[T] +
  E_X[X] + E_Y[Y])`` at mask-marked positions. The backbone applies it at the
  input, so every KDA/MLA projection sees the semantic coordinate (C03).
- :meth:`ATXYInjection.inject_values` performs the exact, immutable,
  version-checked lookup and adds ``value_gate * tanh(Linear(value))`` at the
  marked positions that hit. The backbone applies it after the configured
  block boundary. ``value_gate`` is initialized exactly to zero so the retrofit
  starts as an identity on the value path while retaining a live gradient.

Calling the module directly runs both steps at once (the 7ea07a1 behavior).
Addresses are ``LongTensor[..., 4]`` ordered exactly ``A, T, X, Y``; a
separate boolean ``address_mask`` marks address/read positions, and
coordinate values at unmarked positions are ignored. Retrieved values are
never expanded into tokens: the sequence length is unchanged.

Out of scope for v0: host offload, persistence, cache eviction, retrieval
ranking, the optional MLA factor-match bias, quantization, and generalized
plugin interfaces.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class ATXYConfig:
    """Configuration of the ATXY address embedding and value injection.

    Args:
        d_model: Residual-stream width the injection operates on.
        value_dim: Width of the value vectors stored in the external store.
        a_cardinality: Number of distinct ``A`` coordinates.
        t_cardinality: Number of distinct ``T`` coordinates.
        x_cardinality: Number of distinct ``X`` coordinates.
        y_cardinality: Number of distinct ``Y`` coordinates.
        injection_layer: Block index after which the value injection runs.
            Must be non-negative; the backbone validates the upper bound
            against its layer count.
        initializer_range: Std-dev of the normal init for address embeddings
            and projections. The value gate is initialized exactly to zero.
    """

    d_model: int
    value_dim: int
    a_cardinality: int
    t_cardinality: int
    x_cardinality: int
    y_cardinality: int
    injection_layer: int = 0
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        _require_positive("d_model", self.d_model)
        _require_positive("value_dim", self.value_dim)
        _require_positive("a_cardinality", self.a_cardinality)
        _require_positive("t_cardinality", self.t_cardinality)
        _require_positive("x_cardinality", self.x_cardinality)
        _require_positive("y_cardinality", self.y_cardinality)
        if self.injection_layer < 0:
            raise ValueError(f"injection_layer must be >= 0, got {self.injection_layer}")
        if self.initializer_range <= 0:
            raise ValueError(f"initializer_range must be positive, got {self.initializer_range}")

    @property
    def cardinalities(self) -> tuple[int, int, int, int]:
        """Address coordinate bounds in canonical A, T, X, Y order."""

        return (
            self.a_cardinality,
            self.t_cardinality,
            self.x_cardinality,
            self.y_cardinality,
        )

    @property
    def parameter_count(self) -> int:
        """Exact parameter count of the module built from this config."""

        d_model = self.d_model
        return sum(self.cardinalities) * d_model + d_model * d_model + self.value_dim * d_model + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ATXYRequest:
    """The optional ATXY inputs of one forward call, supplied together.

    Attributes:
        addresses: ``[batch, seq, 4]`` ``LongTensor`` ordered ``A, T, X, Y``.
        mask: ``[batch, seq]`` boolean, ``True`` at address/read positions.
        store: The :class:`ATXYStore` to look values up in.
        expected_store_version: Version the caller expects; a mismatch raises
            before any injection.
    """

    addresses: Tensor
    mask: Tensor
    store: ATXYStore
    expected_store_version: int


def _normalize_address(address: Sequence[int] | Tensor) -> tuple[int, int, int, int]:
    if isinstance(address, Tensor):
        parts = address.detach().reshape(-1).tolist()
    else:
        parts = list(address)
    if len(parts) != 4:
        raise ValueError(f"ATXY addresses have exactly 4 coordinates, got {len(parts)}")
    normalized: list[int] = []
    for coordinate in parts:
        if isinstance(coordinate, bool) or not isinstance(coordinate, int):
            raise ValueError(f"ATXY coordinates must be integers, got {coordinate!r}")
        if coordinate < 0:
            raise ValueError(f"ATXY coordinates must be non-negative, got {coordinate}")
        normalized.append(coordinate)
    return (normalized[0], normalized[1], normalized[2], normalized[3])


class ATXYStore:
    """Immutable, versioned, exact lookup from ``(A, T, X, Y)`` to value vectors.

    Lookups are exact: a missing address returns ``None`` and is never
    replaced by a nearby value. Duplicate addresses are rejected while
    building the store, and the store exposes no write API afterwards.
    """

    def __init__(
        self,
        *,
        store_id: str,
        version: int,
        value_dim: int,
        entries: Iterable[tuple[Sequence[int] | Tensor, Tensor]] = (),
    ) -> None:
        if not isinstance(store_id, str) or not store_id.strip():
            raise ValueError("store_id must be a non-empty string")
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise ValueError(f"version must be a non-negative integer, got {version!r}")
        _require_positive("value_dim", value_dim)
        records: dict[tuple[int, int, int, int], Tensor] = {}
        for address, value in entries:
            key = _normalize_address(address)
            if key in records:
                raise ValueError(f"duplicate ATXY address: {key}")
            value = torch.as_tensor(value)
            if value.shape != (value_dim,):
                raise ValueError(f"value for address {key} must have shape ({value_dim},)")
            records[key] = value.detach().clone()
        self._store_id = store_id
        self._version = version
        self._value_dim = value_dim
        self._records = MappingProxyType(records)

    @property
    def store_id(self) -> str:
        return self._store_id

    @property
    def version(self) -> int:
        return self._version

    @property
    def value_dim(self) -> int:
        return self._value_dim

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, address: object) -> bool:
        try:
            key = _normalize_address(address)  # type: ignore[arg-type]
        except ValueError:
            return False
        return key in self._records

    def lookup(self, address: Sequence[int] | Tensor) -> Tensor | None:
        """Exact lookup; ``None`` means not found (no substitution)."""

        return self._records.get(_normalize_address(address))


class ATXYInjection(nn.Module):
    """Factorized address embedding plus gated exact-value injection.

    Shape contract (``B`` batch, ``S`` sequence, ``D`` model width):

    - ``hidden_states [B, S, D]``;
    - ``addresses [B, S, 4]`` ``LongTensor`` ordered ``A, T, X, Y``;
    - ``address_mask [B, S]`` boolean, ``True`` at address/read positions;
    - ``store`` :class:`ATXYStore`, ``expected_store_version`` int.

    A version mismatch raises before any injection. Unmasked positions never
    read the address tensor and never receive an update.
    """

    def __init__(self, config: ATXYConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_a = nn.Embedding(config.a_cardinality, config.d_model)
        self.embed_t = nn.Embedding(config.t_cardinality, config.d_model)
        self.embed_x = nn.Embedding(config.x_cardinality, config.d_model)
        self.embed_y = nn.Embedding(config.y_cardinality, config.d_model)
        self.address_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value_proj = nn.Linear(config.value_dim, config.d_model, bias=False)
        self.value_gate = nn.Parameter(torch.zeros(()))
        for embedding in (self.embed_a, self.embed_t, self.embed_x, self.embed_y):
            nn.init.normal_(embedding.weight, mean=0.0, std=config.initializer_range)
        nn.init.normal_(self.address_proj.weight, mean=0.0, std=config.initializer_range)
        nn.init.normal_(self.value_proj.weight, mean=0.0, std=config.initializer_range)

    def forward(
        self,
        hidden_states: Tensor,
        addresses: Tensor,
        address_mask: Tensor,
        store: ATXYStore,
        expected_store_version: int,
    ) -> Tensor:
        """Address embedding and value injection at once (standalone use)."""

        hidden_states = self.embed_addresses(hidden_states, addresses, address_mask)
        return self.inject_values(
            hidden_states, addresses, address_mask, store, expected_store_version
        )

    def embed_addresses(
        self, hidden_states: Tensor, addresses: Tensor, address_mask: Tensor
    ) -> Tensor:
        """``h_t = h_t + address_mask_t * Project(E_A[A], E_T[T], E_X[X], E_Y[Y])``."""

        marked_index, marked = self._marked(hidden_states, addresses, address_mask)
        if marked is None:
            return hidden_states
        factorized = (
            self.embed_a(marked[:, 0])
            + self.embed_t(marked[:, 1])
            + self.embed_x(marked[:, 2])
            + self.embed_y(marked[:, 3])
        )
        address_update = torch.zeros_like(hidden_states)
        address_update[marked_index[:, 0], marked_index[:, 1]] = self.address_proj(factorized).to(
            hidden_states.dtype
        )
        return hidden_states + address_update

    def inject_values(
        self,
        hidden_states: Tensor,
        addresses: Tensor,
        address_mask: Tensor,
        store: ATXYStore,
        expected_store_version: int,
    ) -> Tensor:
        """Exact lookup at marked positions; gated bounded value residual on hits."""

        if not isinstance(store, ATXYStore):
            raise TypeError(f"store must be an ATXYStore, got {type(store).__name__}")
        if store.value_dim != self.config.value_dim:
            raise ValueError(
                f"store value_dim={store.value_dim} does not match config "
                f"value_dim={self.config.value_dim}"
            )
        if store.version != expected_store_version:
            raise ValueError(
                "ATXY store version mismatch: expected "
                f"{expected_store_version}, store {store.store_id!r} is version {store.version}"
            )
        marked_index, marked = self._marked(hidden_states, addresses, address_mask)
        if marked is None:
            return hidden_states

        found_values: list[Tensor] = []
        hit_rows: list[int] = []
        for row, address in enumerate(marked.tolist()):
            value = store.lookup(address)
            if value is not None:
                found_values.append(value)
                hit_rows.append(row)
        if not found_values:
            return hidden_states

        values = torch.stack(found_values).to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        bounded = torch.tanh(self.value_proj(values))
        value_update = torch.zeros_like(hidden_states)
        hit_index = marked_index[torch.tensor(hit_rows, device=marked_index.device)]
        value_update[hit_index[:, 0], hit_index[:, 1]] = (self.value_gate * bounded).to(
            hidden_states.dtype
        )
        return hidden_states + value_update

    def _marked(
        self, hidden_states: Tensor, addresses: Tensor, address_mask: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        if hidden_states.dim() != 3 or hidden_states.shape[-1] != self.config.d_model:
            raise ValueError(
                f"expected hidden_states [B, S, {self.config.d_model}], "
                f"got {tuple(hidden_states.shape)}"
            )
        if addresses.shape != (*hidden_states.shape[:2], 4):
            raise ValueError(
                f"expected addresses {tuple(hidden_states.shape[:2]) + (4,)}, "
                f"got {tuple(addresses.shape)}"
            )
        if addresses.dtype != torch.long:
            raise ValueError(f"addresses must be a LongTensor, got {addresses.dtype}")
        if address_mask.shape != hidden_states.shape[:2]:
            raise ValueError(
                f"expected address_mask {tuple(hidden_states.shape[:2])}, "
                f"got {tuple(address_mask.shape)}"
            )
        marked_index = address_mask.bool().nonzero()
        if marked_index.shape[0] == 0:
            return marked_index, None
        marked = addresses[marked_index[:, 0], marked_index[:, 1]]
        for coordinate, cardinality in enumerate(self.config.cardinalities):
            column = marked[:, coordinate]
            if bool((column < 0).any()) or bool((column >= cardinality).any()):
                raise ValueError(
                    f"ATXY address coordinate {coordinate} outside cardinality {cardinality}"
                )
        return marked_index, marked

    def extra_repr(self) -> str:
        c = self.config
        return (
            f"d_model={c.d_model}, value_dim={c.value_dim}, "
            f"cardinalities={c.cardinalities}, injection_layer={c.injection_layer}"
        )


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


__all__ = ["ATXYConfig", "ATXYInjection", "ATXYRequest", "ATXYStore"]
