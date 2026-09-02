"""Reference expand-before-compress per-layer lookup embeddings."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from spalmer.config import PLEConfig
from spalmer.nn import fake_quantize_low_bit


class PLELayerEmbedding(nn.Module):
    """One layer's wide low-bit lookup and inexpensive learned lane reduction."""

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
        self.phase = config.phase_for_layer(layer_index)
        _guard_reference_size(config, layer_count=1)

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(
                config.vocab_size,
                config.expansion_factor * config.d_model,
                **factory_kwargs,
            )
        )
        self.lane_logits = nn.Parameter(torch.zeros(config.expansion_factor, **factory_kwargs))
        self.gate = nn.Parameter(torch.tensor(config.resolved_gate_init, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=self.config.initializer_range)
        nn.init.zeros_(self.lane_logits)
        with torch.no_grad():
            self.gate.fill_(self.config.resolved_gate_init)

    def forward(self, input_ids: Tensor) -> Tensor:
        flat_ids = input_ids.reshape(-1)
        unique_ids, inverse = torch.unique(flat_ids, sorted=False, return_inverse=True)
        unique_rows = F.embedding(
            unique_ids,
            self.weight,
            sparse=self.config.sparse_gradients,
        ).unflatten(
            -1,
            (self.config.expansion_factor, self.config.d_model),
        )
        quantized_unique = fake_quantize_low_bit(
            unique_rows,
            bits=self.config.quant_bits,
            stochastic=self.training and self.config.stochastic_rounding,
            straight_through=self.training,
            eps=self.config.quant_eps,
        )
        quantized = quantized_unique[inverse].reshape(
            *input_ids.shape,
            self.config.expansion_factor,
            self.config.d_model,
        )
        lane_weights = self.lane_logits.softmax(dim=0)
        view_shape = (1,) * (quantized.ndim - 2) + (self.config.expansion_factor, 1)
        compressed = (quantized * lane_weights.view(view_shape)).sum(dim=-2)
        return compressed * self.gate.to(dtype=compressed.dtype)

    def inject(self, hidden_states: Tensor, input_ids: Tensor) -> Tensor:
        lexical_update = self(input_ids)
        if lexical_update.shape != hidden_states.shape:
            raise ValueError(
                "hidden_states and the PLE update must have identical shapes; "
                f"got {tuple(hidden_states.shape)} and {tuple(lexical_update.shape)}"
            )
        return hidden_states + lexical_update

    def extra_repr(self) -> str:
        return (
            f"layer={self.layer_index}, phase={self.phase}, vocab_size={self.config.vocab_size}, "
            f"d_model={self.config.d_model}, expansion_factor={self.config.expansion_factor}, "
            f"quant_bits={self.config.quant_bits}"
        )


class AlternatingPLE(nn.Module):
    """Per-layer embedding bank using the fixed A/B layer alternation baseline."""

    def __init__(
        self,
        config: PLEConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        _guard_reference_size(config, layer_count=config.n_layers)
        self.layers = nn.ModuleList(
            PLELayerEmbedding(config, index, device=device, dtype=dtype)
            for index in range(config.n_layers)
        )

    def forward(self, input_ids: Tensor, layer_index: int) -> Tensor:
        return self._layer(layer_index)(input_ids)

    def inject(self, hidden_states: Tensor, input_ids: Tensor, layer_index: int) -> Tensor:
        return self._layer(layer_index).inject(hidden_states, input_ids)

    def phase_for_layer(self, layer_index: int) -> str:
        return self._layer(layer_index).phase

    def _layer(self, layer_index: int) -> PLELayerEmbedding:
        if not 0 <= layer_index < len(self.layers):
            raise IndexError(f"layer index {layer_index} is outside [0, {len(self.layers)})")
        return self.layers[layer_index]


def _guard_reference_size(config: PLEConfig, *, layer_count: int) -> None:
    limit = config.reference_max_numel
    if limit is None:
        return
    requested = layer_count * config.vocab_size * config.expansion_factor * config.d_model
    if requested <= limit:
        return
    gib = requested * 4 / (1024**3)
    raise MemoryError(
        "the fake-QAT PLE backend would allocate "
        f"{requested:,} floating shadow parameters (about {gib:.2f} GiB in FP32) before "
        "gradients or optimizer state; choose a smaller prototype or explicitly set "
        "reference_max_numel=None while accepting that cost"
    )
