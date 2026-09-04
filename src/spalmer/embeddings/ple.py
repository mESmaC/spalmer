"""Exact-input and expand-before-compress per-layer lookup embeddings."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from spalmer.config import PLEConfig
from spalmer.qr import qr_codebook_rows, qr_lane_moduli

QRIndices = tuple[Tensor, Tensor]


class PLELayerEmbedding(nn.Module):
    """One layer's lexical input or per-layer embedding refresh.

    Layer zero is an exact token embedding. Every later lane is the elementwise
    product of quotient and remainder codewords. QR tables deliberately use
    ordinary dense gradients: they are small, and many token IDs contribute to
    each row.
    """

    def __init__(
        self,
        config: PLEConfig,
        layer_index: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not 0 <= layer_index < config.n_layers:
            raise IndexError(f"layer index {layer_index} is outside [0, {config.n_layers})")

        self.config = config
        self.layer_index = layer_index

        factory_kwargs = {"device": device, "dtype": dtype}
        if layer_index == 0:
            # Layer zero is the model's sole input embedding.  Preserve exact
            # token identity here rather than subjecting it to shared QR rows.
            self.input_embedding = nn.Embedding(
                config.vocab_size,
                config.d_model,
                sparse=False,
                **factory_kwargs,
            )
        else:
            modulus_values = qr_lane_moduli(config.vocab_size, config.expansion_factor)
            quotient_sizes = tuple(
                (config.vocab_size + modulus - 1) // modulus for modulus in modulus_values
            )
            remainder_rows, quotient_rows = qr_codebook_rows(
                config.vocab_size,
                config.expansion_factor,
            )
            self._modulus_values = modulus_values
            self.register_buffer(
                "moduli",
                torch.tensor(modulus_values, dtype=torch.int64, device=device),
            )
            self.register_buffer(
                "remainder_offsets",
                torch.tensor(_exclusive_offsets(modulus_values), dtype=torch.int64, device=device),
            )
            self.register_buffer(
                "quotient_offsets",
                torch.tensor(_exclusive_offsets(quotient_sizes), dtype=torch.int64, device=device),
            )
            # Flatten the lane codebooks so a refresh performs two vectorized
            # gathers rather than launching two gathers for every lane.
            self.remainder_embedding = nn.Embedding(
                remainder_rows,
                config.d_model,
                sparse=False,
                **factory_kwargs,
            )
            self.quotient_embedding = nn.Embedding(
                quotient_rows,
                config.d_model,
                sparse=False,
                **factory_kwargs,
            )
            self.lane_logits = nn.Parameter(
                torch.zeros(config.expansion_factor, **factory_kwargs)
            )
            self.gate = nn.Parameter(torch.tensor(config.resolved_gate_init, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.layer_index == 0:
            nn.init.normal_(
                self.input_embedding.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            return

        nn.init.normal_(
            self.remainder_embedding.weight,
            mean=0.0,
            std=self.config.initializer_range,
        )
        # R * Q therefore begins as an ordinary hashed embedding with a small
        # learned multiplicative perturbation rather than near zero.
        nn.init.normal_(
            self.quotient_embedding.weight,
            mean=1.0,
            std=self.config.initializer_range,
        )
        nn.init.zeros_(self.lane_logits)
        with torch.no_grad():
            self.gate.fill_(self.config.resolved_gate_init)

    def forward(self, input_ids: Tensor, *, qr_indices: QRIndices | None = None) -> Tensor:
        if self.layer_index == 0:
            return self.input_embedding(input_ids)
        return self._forward_qr(input_ids, qr_indices=qr_indices)

    def prepare_qr_indices(self, input_ids: Tensor) -> QRIndices:
        """Resolve every lane address once for reuse by all physical layers."""

        if self.layer_index == 0:
            raise ValueError("QR indices require a QR-PLE refresh layer")
        lane_ids = input_ids.unsqueeze(-1)
        quotients = torch.div(lane_ids, self.moduli, rounding_mode="floor")
        remainders = lane_ids - quotients * self.moduli
        return remainders + self.remainder_offsets, quotients + self.quotient_offsets

    def _forward_qr(
        self,
        input_ids: Tensor,
        *,
        qr_indices: QRIndices | None = None,
    ) -> Tensor:
        remainder_ids, quotient_ids = (
            self.prepare_qr_indices(input_ids) if qr_indices is None else qr_indices
        )
        expanded = self.remainder_embedding(remainder_ids) * self.quotient_embedding(
            quotient_ids
        )
        lane_weights = self.lane_logits.softmax(dim=0).to(dtype=expanded.dtype)
        view_shape = (1,) * (expanded.ndim - 2) + (self.config.expansion_factor, 1)
        compressed = (expanded * lane_weights.view(view_shape)).sum(
            dim=-2,
            dtype=expanded.dtype,
        )
        return compressed * self.gate.to(dtype=compressed.dtype)

    def inject(
        self,
        hidden_states: Tensor,
        input_ids: Tensor,
        *,
        qr_indices: QRIndices | None = None,
    ) -> Tensor:
        lexical_update = (
            self(input_ids)
            if qr_indices is None
            else self(input_ids, qr_indices=qr_indices)
        )
        if lexical_update.shape != hidden_states.shape:
            raise ValueError(
                "hidden_states and the PLE update must have identical shapes; "
                f"got {tuple(hidden_states.shape)} and {tuple(lexical_update.shape)}"
            )
        return hidden_states + lexical_update

    def extra_repr(self) -> str:
        detail = (
            "exact_input=True"
            if self.layer_index == 0
            else f"moduli={self._modulus_values}"
        )
        return (
            f"layer={self.layer_index}, backend=qr, {detail}, "
            f"vocab_size={self.config.vocab_size}, "
            f"d_model={self.config.d_model}, expansion_factor={self.config.expansion_factor}"
        )


class AlternatingPLE(nn.Module):
    """Per-layer embedding bank; name retained for checkpoint/API compatibility."""

    def __init__(
        self,
        config: PLEConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            PLELayerEmbedding(config, index, device=device, dtype=dtype)
            for index in range(config.n_layers)
        )

    def forward(
        self,
        input_ids: Tensor,
        layer_index: int,
        *,
        qr_indices: QRIndices | None = None,
    ) -> Tensor:
        layer = self._layer(layer_index)
        if qr_indices is None:
            return layer(input_ids)
        return layer(input_ids, qr_indices=qr_indices)

    def inject(
        self,
        hidden_states: Tensor,
        input_ids: Tensor,
        layer_index: int,
        *,
        qr_indices: QRIndices | None = None,
    ) -> Tensor:
        layer = self._layer(layer_index)
        if qr_indices is None:
            return layer.inject(hidden_states, input_ids)
        return layer.inject(hidden_states, input_ids, qr_indices=qr_indices)

    def prepare_qr_indices(self, input_ids: Tensor) -> QRIndices | None:
        """Compute shared QR addresses once; one-layer banks need none."""

        if len(self.layers) < 2:
            return None
        return self.layers[1].prepare_qr_indices(input_ids)

    def _layer(self, layer_index: int) -> PLELayerEmbedding:
        if not 0 <= layer_index < len(self.layers):
            raise IndexError(f"layer index {layer_index} is outside [0, {len(self.layers)})")
        return self.layers[layer_index]

def _exclusive_offsets(sizes: tuple[int, ...]) -> tuple[int, ...]:
    offsets = []
    running = 0
    for size in sizes:
        offsets.append(running)
        running += size
    return tuple(offsets)
