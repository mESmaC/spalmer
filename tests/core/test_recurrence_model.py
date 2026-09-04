"""Depth-recurrent core: config, state containers, forward algorithm, telemetry.

Every model here is a tiny CPU double. The contract under test is that a
``recurrence=None`` model is unchanged and a recurrent model iterates the core
blocks with per-(block, iteration) mixer state, truncated backprop, last-
iteration metrics, and a well-defined decode/exit state budget.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest
import torch
from torch import Tensor, nn

from spalmer import (
    AdaptiveExit,
    ChannelMixerOutput,
    RecurrenceConfig,
    RecurrentMixerStates,
    SPALMERBackbone,
    SPALMERBlock,
    SPALMERCausalLM,
    SPALMERConfig,
)
from spalmer.embeddings.ple import PLELayerEmbedding
from spalmer.experts.accounting import classify_parameter
from spalmer.training.optim import classify_parameters

TOKENIZER_FINGERPRINT = "test-recurrence-v1"


class TinyTokenMixer(nn.Module):
    """States are integers so per-iteration threading is directly readable."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, d_model, bias=False)
        self.backward_calls = 0

    def _count(self, grad: Tensor) -> Tensor:
        self.backward_calls += 1
        return grad

    def _update(self, hidden_states: Tensor) -> Tensor:
        update = self.projection(hidden_states)
        if update.requires_grad:
            update.register_hook(self._count)
        return update

    def forward(self, hidden_states: Tensor, *, state: int | None = None, **_: object):
        return self._update(hidden_states), 1 if state is None else state + 1

    def step(self, hidden_states: Tensor, *, state: int | None = None, **_: object):
        return self._update(hidden_states), 10 if state is None else state + 10


class RecordingChannelMixer(nn.Module):
    """Reports the call index as metrics and an auxiliary loss per call."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, d_model, bias=False)
        self.calls = 0
        self.auxiliary_values: list[Tensor] = []

    def forward(self, hidden_states: Tensor, *, state: int | None = None) -> ChannelMixerOutput:
        self.calls += 1
        auxiliary = hidden_states.square().mean() * 0.01
        self.auxiliary_values.append(auxiliary)
        return ChannelMixerOutput(
            update=self.projection(hidden_states).tanh(),
            state=1 if state is None else state + 1,
            auxiliary_loss=auxiliary,
            metrics={"call_index": self.calls},
        )


def _config(
    recurrence: RecurrenceConfig | None = None,
    *,
    n_layers: int = 4,
    ple_backend: Literal["qr"] = "qr",
) -> SPALMERConfig:
    return SPALMERConfig(
        vocab_size=31,
        d_model=12,
        n_layers=n_layers,
        tokenizer_version=1,
        tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
        ple_expansion_factor=2,
        ple_backend=ple_backend,
        recurrence=recurrence,
    )


def _tiny_model(
    recurrence: RecurrenceConfig | None = None,
    *,
    channel_mixer: type[nn.Module] = RecordingChannelMixer,
    n_layers: int = 4,
    seed: int = 0,
    ple_backend: Literal["qr"] = "qr",
) -> SPALMERCausalLM:
    torch.manual_seed(seed)
    config = _config(recurrence, n_layers=n_layers, ple_backend=ple_backend)
    blocks = [
        SPALMERBlock(
            config.d_model,
            TinyTokenMixer(config.d_model),
            channel_mixer(config.d_model),
            norm_eps=config.norm_eps,
        )
        for _ in range(config.n_layers)
    ]
    # eval(): deterministic behavior so repeated forwards are comparable.
    return SPALMERCausalLM(config, SPALMERBackbone(config, blocks)).eval()


def _recurrent_model(**kwargs: Any) -> SPALMERCausalLM:
    return _tiny_model(RecurrenceConfig(1, 2, 1, default_steps=3), **kwargs)


# --------------------------------------------------------------------------
# Configuration contract
# --------------------------------------------------------------------------


def test_recurrence_config_validates_counts_and_exposes_roles() -> None:
    recurrence = RecurrenceConfig(1, 2, 1, default_steps=5)
    assert recurrence.total_layers == 4
    assert list(recurrence.core_range()) == [1, 2]
    assert [recurrence.role(index) for index in range(4)] == [
        "prelude",
        "core",
        "core",
        "coda",
    ]
    assert recurrence.effective_depth(5) == 1 + 5 * 2 + 1
    assert recurrence.parameter_count(12) == 2 * 12 * 12 + 2 * 12
    assert recurrence.to_dict()["adapter"] == "linear_concat"

    for kwargs in ({"prelude_layers": 0}, {"core_layers": 0}, {"coda_layers": 0}):
        with pytest.raises(ValueError, match="must be positive"):
            RecurrenceConfig(**{"prelude_layers": 1, "core_layers": 1, "coda_layers": 1, **kwargs})
    with pytest.raises(ValueError, match="default_steps must be positive"):
        RecurrenceConfig(1, 1, 1, default_steps=0)
    with pytest.raises(ValueError, match="latent_init_std"):
        RecurrenceConfig(1, 1, 1, latent_init_std=-1.0)
    with pytest.raises(ValueError, match="latent_init_std"):
        RecurrenceConfig(1, 1, 1, latent_init_std=float("inf"))
    with pytest.raises(ValueError, match="adapter"):
        RecurrenceConfig(1, 1, 1, adapter="mlp")
    with pytest.raises(ValueError, match="adapter_init"):
        RecurrenceConfig(1, 1, 1, adapter_init="zeros")


def test_model_config_coerces_a_mapping_and_checks_the_layer_split() -> None:
    config = _config(RecurrenceConfig(1, 2, 1))
    assert config.block_role(0) == "prelude"
    assert config.block_role(2) == "core"
    assert config.block_role(3) == "coda"
    assert config.core_layer_indices == (1, 2)
    assert config.effective_depth() == 1 + 8 * 2 + 1
    assert config.effective_depth(3) == 1 + 3 * 2 + 1
    # The default pattern is 3 KDA to 1 MLA by physical index: the core is KDA-only.
    assert config.core_mixer_counts == {"kda": 2, "mla": 0}

    round_tripped = SPALMERConfig(**config.to_dict())
    assert round_tripped.recurrence == config.recurrence
    assert round_tripped == config

    flat = _config()
    assert flat.block_role(0) == "stack"
    assert flat.core_layer_indices == ()
    assert flat.effective_depth() == flat.n_layers
    assert flat.core_mixer_counts == {"kda": 0, "mla": 0}
    assert flat.to_dict()["recurrence"] is None

    with pytest.raises(ValueError, match="must equal n_layers"):
        _config(RecurrenceConfig(1, 1, 1))


def test_adaptive_exit_validates_and_clamps_min_steps() -> None:
    assert AdaptiveExit("latent_diff").resolved_threshold == pytest.approx(0.03)
    assert AdaptiveExit("kl").resolved_threshold == pytest.approx(1e-3)
    assert AdaptiveExit("latent_diff", min_steps=1).min_steps == 2

    with pytest.raises(ValueError, match="criterion"):
        AdaptiveExit("entropy")
    with pytest.raises(ValueError, match="threshold"):
        AdaptiveExit("latent_diff", threshold=0.0)
    with pytest.raises(ValueError, match="threshold"):
        AdaptiveExit("latent_diff", threshold=float("inf"))
    with pytest.raises(ValueError, match="min_steps"):
        AdaptiveExit("latent_diff", min_steps=0)
    with pytest.raises(ValueError, match="check_every"):
        AdaptiveExit("latent_diff", check_every=0)
    with pytest.raises(ValueError, match="state policy"):
        AdaptiveExit("latent_diff", state_policy="drop")


# --------------------------------------------------------------------------
# Forward algorithm
# --------------------------------------------------------------------------


def test_recurrent_prefill_returns_per_iteration_core_states() -> None:
    model = _recurrent_model()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    output = model(input_ids, recurrence_steps=3)

    assert output.recurrence_steps == 3
    assert output.token_mixer_states[0] == 1
    assert output.token_mixer_states[3] == 1
    for slot in output.token_mixer_states[1:3]:
        assert isinstance(slot, RecurrentMixerStates)
        assert len(slot) == 3
        assert tuple(slot) == (1, 1, 1)
    for slot in output.channel_mixer_states[1:3]:
        assert isinstance(slot, RecurrentMixerStates)
        assert len(slot) == 3
    assert len(output.layer_metrics) == 4
    assert output.latent_deltas.shape == (3,)
    assert output.latent_deltas.dtype == torch.float32
    assert output.latent_states.shape == (1, 5, model.config.d_model)
    assert output.latent_states.dtype == torch.float32
    assert not output.latent_states.requires_grad
    assert output.latent_position_correlation is not None
    assert output.latent_position_correlation.shape == ()
    # Every core block runs once per iteration; prelude and coda run once.
    assert [block.channel_mixer.calls for block in model.backbone.blocks] == [1, 3, 3, 1]


def test_single_token_forward_has_no_position_correlation() -> None:
    model = _recurrent_model()
    output = model(torch.tensor([[4]]), recurrence_steps=2)
    assert output.latent_position_correlation is None
    assert output.latent_deltas.shape == (2,)


def test_recurrent_decode_threads_iteration_states_and_keeps_budget() -> None:
    model = _recurrent_model()
    prefill = model(torch.tensor([[1, 2, 3, 4]]), recurrence_steps=4)
    assert [len(slot) for slot in prefill.token_mixer_states[1:3]] == [4, 4]

    full = model(
        torch.tensor([[5]]),
        execution_mode="decode",
        recurrence_steps=4,
        token_mixer_states=prefill.token_mixer_states,
        channel_mixer_states=prefill.channel_mixer_states,
    )
    assert full.recurrence_steps == 4
    assert [len(slot) for slot in full.token_mixer_states[1:3]] == [4, 4]
    assert tuple(full.token_mixer_states[1]) == (11, 11, 11, 11)

    # Fewer steps than the budget: the deeper iterations are filled by
    # advancing only the token mixer, so their state still moves forward.
    filled = model(
        torch.tensor([[5]]),
        execution_mode="decode",
        recurrence_steps=2,
        token_mixer_states=prefill.token_mixer_states,
        channel_mixer_states=prefill.channel_mixer_states,
    )
    assert filled.recurrence_steps == 2
    assert [len(slot) for slot in filled.token_mixer_states[1:3]] == [4, 4]
    assert tuple(filled.token_mixer_states[1]) == (11, 11, 11, 11)
    # The core channel mixer ran only for the two executed iterations.
    assert model.backbone.blocks[1].channel_mixer.calls == 4 + 4 + 2

    # "skip" leaves the deeper iteration states exactly as they arrived.
    with torch.no_grad():
        skipped = model(
            torch.tensor([[5]]),
            execution_mode="decode",
            recurrence_steps=4,
            adaptive_exit=AdaptiveExit(
                "latent_diff", threshold=1e9, min_steps=2, state_policy="skip"
            ),
            token_mixer_states=prefill.token_mixer_states,
            channel_mixer_states=prefill.channel_mixer_states,
        )
    assert skipped.recurrence_steps == 2
    assert tuple(skipped.token_mixer_states[1]) == (11, 11, 1, 1)


def test_decode_rejects_more_steps_than_state_budget() -> None:
    model = _recurrent_model()
    prefill = model(torch.tensor([[1, 2, 3, 4]]), recurrence_steps=3)

    with pytest.raises(ValueError, match="prefill deeper or reduce recurrence_steps"):
        model(
            torch.tensor([[5]]),
            execution_mode="decode",
            recurrence_steps=5,
            token_mixer_states=prefill.token_mixer_states,
            channel_mixer_states=prefill.channel_mixer_states,
        )


def test_core_state_slots_must_share_one_budget() -> None:
    model = _recurrent_model()
    prefill = model(torch.tensor([[1, 2, 3, 4]]), recurrence_steps=3)
    states = list(prefill.token_mixer_states)
    states[2] = RecurrentMixerStates(tuple(states[2]) + (7,))

    with pytest.raises(ValueError, match="share one iteration budget"):
        model(
            torch.tensor([[5]]),
            execution_mode="decode",
            recurrence_steps=3,
            token_mixer_states=states,
            channel_mixer_states=prefill.channel_mixer_states,
        )


def test_adaptive_exit_is_decode_only_and_respects_min_steps() -> None:
    model = _recurrent_model()
    always = AdaptiveExit("latent_diff", threshold=1e9, min_steps=3)
    prefill = model(torch.tensor([[1, 2, 3, 4]]), recurrence_steps=5)

    with pytest.raises(ValueError, match="inference-only policy"):
        model(torch.tensor([[5]]), execution_mode="decode", adaptive_exit=always)

    with torch.no_grad():
        with pytest.raises(ValueError, match="only available in decode mode"):
            model(torch.tensor([[1, 2, 3, 4]]), recurrence_steps=5, adaptive_exit=always)

        exited = model(
            torch.tensor([[5]]),
            execution_mode="decode",
            recurrence_steps=5,
            adaptive_exit=always,
            token_mixer_states=prefill.token_mixer_states,
            channel_mixer_states=prefill.channel_mixer_states,
        )
    assert exited.recurrence_steps == 3
    assert exited.latent_deltas.shape == (3,)

    with torch.no_grad():
        never = model(
            torch.tensor([[5]]),
            execution_mode="decode",
            recurrence_steps=5,
            adaptive_exit=AdaptiveExit("latent_diff", threshold=1e-12, min_steps=2),
            token_mixer_states=prefill.token_mixer_states,
            channel_mixer_states=prefill.channel_mixer_states,
        )
    assert never.recurrence_steps == 5


def test_prefill_exit_requires_an_explicit_opt_in() -> None:
    model = _recurrent_model()
    policy = AdaptiveExit("latent_diff", threshold=1e9, min_steps=2, apply_in_prefill=True)

    with torch.no_grad():
        output = model(torch.tensor([[1, 2, 3, 4]]), recurrence_steps=5, adaptive_exit=policy)

    assert output.recurrence_steps == 2
    # The state budget still covers every requested iteration so a following
    # decode has memory for all of them; the unrun tail is filled.
    assert [len(slot) for slot in output.token_mixer_states[1:3]] == [5, 5]
    assert tuple(output.token_mixer_states[1]) == (1, 1, 1, 1, 1)


def test_kl_exit_probes_head_without_mutating_coda_states() -> None:
    model = _recurrent_model()
    prompt = torch.tensor([[1, 2, 3, 4]])
    prefill = model(prompt, recurrence_steps=5)

    readout_calls = 0
    real_head = model.lm_head

    class CountingHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = real_head

        def forward(self, hidden_states: Tensor) -> Tensor:
            nonlocal readout_calls
            readout_calls += 1
            return self.inner(hidden_states)

    model.lm_head = CountingHead()
    with torch.no_grad():
        exited = model(
            torch.tensor([[5]]),
            execution_mode="decode",
            recurrence_steps=5,
            adaptive_exit=AdaptiveExit("kl", threshold=1e9, min_steps=2),
            latent_generator=torch.Generator().manual_seed(3),
            token_mixer_states=prefill.token_mixer_states,
            channel_mixer_states=prefill.channel_mixer_states,
        )
        # Two probes plus the real head call on the returned hidden states.
        probe_and_head_calls = readout_calls
        # Iteration 2 seeds the reference distribution, iteration 3 exits.
        reference = model(
            torch.tensor([[5]]),
            execution_mode="decode",
            recurrence_steps=3,
            latent_generator=torch.Generator().manual_seed(3),
            token_mixer_states=prefill.token_mixer_states,
            channel_mixer_states=prefill.channel_mixer_states,
        )

    assert exited.recurrence_steps == 3
    assert probe_and_head_calls == 3
    # Coda states are pure probe inputs: the real coda pass is undisturbed.
    assert exited.token_mixer_states[3] == reference.token_mixer_states[3]
    assert exited.channel_mixer_states[3] == reference.channel_mixer_states[3]
    assert len(exited.layer_metrics) == len(reference.layer_metrics) == 4
    torch.testing.assert_close(exited.logits, reference.logits)


def test_per_sequence_exit_freezes_latent_rows() -> None:
    model = _recurrent_model()
    prompt = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    prefill = model(prompt, recurrence_steps=6)

    recurrence = model.backbone.recurrence
    real_normalize = recurrence.normalize_latent
    calls = {"count": 0}
    frozen: dict[str, Tensor] = {}

    def freezing_normalize(hidden_states: Tensor) -> Tensor:
        normalized = real_normalize(hidden_states)
        calls["count"] += 1
        if calls["count"] >= 2:
            frozen.setdefault("row", normalized[0].detach().clone())
            normalized = normalized.clone()
            normalized[0] = frozen["row"]
        return normalized

    recurrence.normalize_latent = freezing_normalize
    try:
        with torch.no_grad():
            output = model(
                torch.tensor([[5], [6]]),
                execution_mode="decode",
                recurrence_steps=6,
                adaptive_exit=AdaptiveExit("latent_diff", threshold=1e-6, min_steps=2),
                token_mixer_states=prefill.token_mixer_states,
                channel_mixer_states=prefill.channel_mixer_states,
            )
    finally:
        del recurrence.normalize_latent

    # Row 0 stops changing at iteration 3, row 1 never does, so the batch runs
    # the full budget while row 0's latent stays at its exit value.
    assert output.recurrence_steps == 6
    torch.testing.assert_close(output.latent_states[0], frozen["row"].float())
    assert not torch.allclose(output.latent_states[1], output.latent_states[0])
    assert [len(slot) for slot in output.token_mixer_states[1:3]] == [6, 6]


def test_latent_init_replaces_noise_and_generator_is_honoured() -> None:
    model = _recurrent_model()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    first = model(
        input_ids,
        recurrence_steps=2,
        latent_generator=torch.Generator().manual_seed(11),
    )
    second = model(
        input_ids,
        recurrence_steps=2,
        latent_generator=torch.Generator().manual_seed(11),
    )
    third = model(
        input_ids,
        recurrence_steps=2,
        latent_generator=torch.Generator().manual_seed(12),
    )
    torch.testing.assert_close(first.logits, second.logits)
    assert not torch.allclose(first.logits, third.logits)

    warm = torch.full((1, 4, model.config.d_model), 0.25)
    warm_first = model(input_ids, recurrence_steps=2, latent_init=warm)
    warm_second = model(input_ids, recurrence_steps=2, latent_init=warm)
    torch.testing.assert_close(warm_first.logits, warm_second.logits)

    with pytest.raises(ValueError, match="latent_init must have shape"):
        model(input_ids, recurrence_steps=2, latent_init=torch.zeros(1, 3, 12))

    zeroed = _tiny_model(RecurrenceConfig(1, 2, 1, latent_init_std=0.0))
    left = zeroed(input_ids, recurrence_steps=2)
    right = zeroed(input_ids, recurrence_steps=2)
    torch.testing.assert_close(left.logits, right.logits)
    torch.testing.assert_close(
        zeroed(input_ids, recurrence_steps=2, latent_init=torch.zeros(1, 4, 12)).logits,
        left.logits,
    )


def test_truncated_backprop_grad_flow() -> None:
    model = _recurrent_model().train()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    output = model(input_ids, labels=input_ids, recurrence_steps=4, backprop_steps=2)
    recorded = [block.channel_mixer.auxiliary_values for block in model.backbone.blocks]
    expected_terms = [
        recorded[0][0],
        torch.stack(recorded[1][-2:]).mean(),
        torch.stack(recorded[2][-2:]).mean(),
        recorded[3][0],
    ]
    torch.testing.assert_close(
        output.auxiliary_loss, torch.stack(expected_terms).mean(), rtol=0, atol=0
    )
    output.loss.backward()

    assert model.backbone.recurrence.adapter.weight.grad is not None
    assert model.backbone.recurrence.injection_norm.weight.grad is not None
    assert model.backbone.recurrence.latent_norm.weight.grad is not None
    # Only the last two iterations retain a graph.
    assert model.backbone.blocks[1].token_mixer.backward_calls == 2
    assert model.backbone.blocks[2].token_mixer.backward_calls == 2
    assert model.backbone.blocks[0].token_mixer.backward_calls == 1
    # Prelude weights and every PLE table, core included, still learn.
    assert model.backbone.blocks[0].token_mixer.projection.weight.grad is not None
    for layer in model.backbone.embeddings.layers:
        tables = (
            [layer.input_embedding]
            if layer.layer_index == 0
            else [layer.remainder_embedding, layer.quotient_embedding]
        )
        for table in tables:
            assert table.weight.grad is not None
            assert table.weight.grad.abs().sum().item() > 0

    full = _recurrent_model().train()
    full(input_ids, labels=input_ids, recurrence_steps=4).loss.backward()
    assert full.backbone.blocks[1].token_mixer.backward_calls == 4

    with pytest.raises(ValueError, match="backprop_steps must be at least 1"):
        model(input_ids, recurrence_steps=2, backprop_steps=0)
    with pytest.raises(ValueError, match="recurrence_steps must be at least 1"):
        model(input_ids, recurrence_steps=0)


def test_core_layer_metrics_are_last_iteration_and_constant_length() -> None:
    model = _recurrent_model()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    output = model(input_ids, labels=input_ids, recurrence_steps=3)

    assert len(output.layer_metrics) == 4
    assert [metrics["call_index"] for metrics in output.layer_metrics] == [1, 3, 3, 1]

    deeper = model(input_ids, labels=input_ids, recurrence_steps=5)
    assert len(deeper.layer_metrics) == 4
    assert [metrics["call_index"] for metrics in deeper.layer_metrics] == [2, 8, 8, 2]


def test_core_ple_is_looked_up_once_per_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _recurrent_model()
    lookups: list[int] = []
    real_forward = PLELayerEmbedding.forward

    def counting_forward(
        self: PLELayerEmbedding,
        input_ids: Tensor,
        **kwargs: object,
    ) -> Tensor:
        lookups.append(self.layer_index)
        return real_forward(self, input_ids, **kwargs)

    monkeypatch.setattr(PLELayerEmbedding, "forward", counting_forward)
    model.train()
    output = model(torch.tensor([[1, 2, 3, 4]]), recurrence_steps=5)

    assert output.recurrence_steps == 5
    # One lookup per physical block, not P + r*C + K.
    assert sorted(lookups) == [0, 1, 2, 3]


def test_core_ple_tensor_is_reused_across_iterations() -> None:
    model = _recurrent_model()
    seen: list[Tensor] = []
    real_forward = model.backbone.embeddings.forward

    def recording_forward(
        input_ids: Tensor,
        layer_index: int,
        **kwargs: object,
    ) -> Tensor:
        result = real_forward(input_ids, layer_index, **kwargs)
        seen.append(result)
        return result

    model.backbone.embeddings.forward = recording_forward
    try:
        model(torch.tensor([[1, 2, 3, 4]]), recurrence_steps=4)
    finally:
        del model.backbone.embeddings.forward

    # Layer 0's embedding plus one lookup for each of the two core blocks.
    assert len(seen) == 3


def test_qr_core_refreshes_are_looked_up_once_and_reused_across_iterations() -> None:
    model = _recurrent_model(ple_backend="qr").train()
    seen: list[int] = []
    embeddings = model.backbone.embeddings
    real_forward = embeddings.forward

    def recording_forward(
        input_ids: Tensor,
        layer_index: int,
        **kwargs: object,
    ) -> Tensor:
        seen.append(layer_index)
        return real_forward(input_ids, layer_index, **kwargs)

    embeddings.forward = recording_forward
    try:
        output = model(
            torch.tensor([[1, 2, 3, 4]]),
            labels=torch.tensor([[1, 2, 3, 4]]),
            recurrence_steps=5,
        )
    finally:
        del embeddings.forward

    # Layer 0 is the sole input embedding. Each physical core layer is looked
    # up once outside the recurrence loop, then its tensor is reused five times.
    assert seen == [0, 1, 2]
    assert output.loss is not None
    output.loss.backward()
    assert embeddings.layers[0].input_embedding.weight.grad is not None
    for layer in embeddings.layers[1:]:
        table_parameters = [
            parameter
            for name, parameter in layer.named_parameters()
            if name.endswith("weight")
        ]
        assert table_parameters
        assert all(parameter.grad is not None for parameter in table_parameters)


def test_recurrence_kwargs_rejected_without_recurrence() -> None:
    model = _tiny_model()
    input_ids = torch.tensor([[1, 2, 3]])

    supplied: list[dict[str, Any]] = [
        {"recurrence_steps": 2},
        {"backprop_steps": 2},
        {"latent_init": torch.zeros(1, 3, 12)},
        {"adaptive_exit": AdaptiveExit("latent_diff")},
        {"latent_generator": torch.Generator()},
    ]
    for kwargs in supplied:
        with pytest.raises(ValueError, match="model has no recurrent core"):
            model(input_ids, **kwargs)

    output = model(input_ids)
    assert output.recurrence_steps is None
    assert output.latent_states is None
    assert output.latent_deltas is None
    assert output.latent_position_correlation is None


def test_recurrent_mixer_states_container_validation_and_to() -> None:
    with pytest.raises(ValueError, match="at least one iteration"):
        RecurrentMixerStates(())

    states = RecurrentMixerStates((torch.ones(2, dtype=torch.float32), None, 7))
    assert len(states) == 3
    assert states[2] == 7
    assert list(states)[1] is None

    moved = states.to(dtype=torch.float16)
    assert moved.iterations[0].dtype == torch.float16
    assert moved.iterations[1] is None
    assert moved.iterations[2] == 7
    assert isinstance(moved, RecurrentMixerStates)

    # A list is accepted and normalized to a tuple.
    assert RecurrentMixerStates([1, 2]).iterations == (1, 2)


def test_state_containers_are_checked_against_the_block_role() -> None:
    model = _recurrent_model()
    prefill = model(torch.tensor([[1, 2, 3, 4]]), recurrence_steps=3)
    states = list(prefill.token_mixer_states)

    bad_core = list(states)
    bad_core[1] = 5
    with pytest.raises(TypeError, match="core block 1 token mixer state"):
        model(
            torch.tensor([[5]]),
            execution_mode="decode",
            recurrence_steps=3,
            token_mixer_states=bad_core,
        )

    bad_flat = list(states)
    bad_flat[0] = RecurrentMixerStates((1, 2, 3))
    with pytest.raises(TypeError, match="outside the recurrent core"):
        model(
            torch.tensor([[5]]),
            execution_mode="decode",
            recurrence_steps=3,
            token_mixer_states=bad_flat,
        )

    flat = _tiny_model()
    with pytest.raises(TypeError, match="outside the recurrent core"):
        flat(torch.tensor([[1, 2, 3]]), token_mixer_states=[RecurrentMixerStates((1,))] * 4)


# --------------------------------------------------------------------------
# Model-level defaults, module shape, accounting
# --------------------------------------------------------------------------


def test_model_level_recurrence_defaults_resolve_below_explicit_kwargs() -> None:
    model = _recurrent_model()
    assert model.is_recurrent
    assert model.recurrence_default_steps == 3

    assert model(torch.tensor([[1, 2, 3, 4]])).recurrence_steps == 3
    model.set_recurrence_defaults(steps=5)
    assert model(torch.tensor([[1, 2, 3, 4]])).recurrence_steps == 5
    # An explicit kwarg still wins.
    assert model(torch.tensor([[1, 2, 3, 4]]), recurrence_steps=2).recurrence_steps == 2

    prefill = model(torch.tensor([[1, 2, 3, 4]]))
    model.set_recurrence_defaults(
        steps=5, adaptive_exit=AdaptiveExit("latent_diff", threshold=1e9, min_steps=2)
    )
    # The exit default is a decode policy; prefill still runs the full budget.
    assert model(torch.tensor([[1, 2, 3, 4]])).recurrence_steps == 5
    with torch.no_grad():
        decoded = model(
            torch.tensor([[5]]),
            execution_mode="decode",
            token_mixer_states=prefill.token_mixer_states,
            channel_mixer_states=prefill.channel_mixer_states,
        )
    assert decoded.recurrence_steps == 2

    model.clear_recurrence_defaults()
    assert model(torch.tensor([[1, 2, 3, 4]])).recurrence_steps == 3

    with pytest.raises(ValueError, match="at least 1"):
        model.set_recurrence_defaults(steps=0)

    flat = _tiny_model()
    assert not flat.is_recurrent
    assert flat.recurrence_default_steps is None
    with pytest.raises(ValueError, match="model has no recurrent core"):
        flat.set_recurrence_defaults(steps=2)


def test_latent_recurrence_parameter_names_shapes_and_identity_mix_init() -> None:
    model = _recurrent_model()
    names = {name for name, _ in model.named_parameters() if ".recurrence." in name}
    assert names == {
        "backbone.recurrence.injection_norm.weight",
        "backbone.recurrence.adapter.weight",
        "backbone.recurrence.latent_norm.weight",
    }
    d_model = model.config.d_model
    assert model.backbone.recurrence.adapter.weight.shape == (d_model, 2 * d_model)
    assert model.backbone.recurrence.adapter.bias is None
    assert sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".recurrence." in name
    ) == model.config.recurrence.parameter_count(d_model)

    identity = torch.eye(d_model)
    reference = torch.cat([identity, identity], dim=1) / (2.0**0.5)
    difference = model.backbone.recurrence.adapter.weight.detach() - reference
    assert difference.abs().max().item() < 0.2

    random_init = _tiny_model(RecurrenceConfig(1, 2, 1, adapter_init="random"))
    assert (
        (random_init.backbone.recurrence.adapter.weight.detach() - reference).abs().max().item()
        > 0.2
    )

    # Weight decay and no-decay grouping follow the parameter names.
    assert "norm" not in "backbone.recurrence.adapter.weight"
    assert all(
        "norm" in name
        for name in names
        if name != "backbone.recurrence.adapter.weight"
    )

    flat = _tiny_model()
    assert flat.backbone.recurrence is None
    assert not any(".recurrence." in name for name, _ in flat.named_parameters())


def test_accounting_classifies_the_recurrence_module_and_reports_depth() -> None:
    assert classify_parameter("backbone.recurrence.adapter.weight") == "recurrence"
    assert classify_parameter("backbone.recurrence.injection_norm.weight") == "recurrence"
    assert classify_parameter("backbone.recurrence.latent_norm.weight") == "recurrence"
    assert classify_parameter("backbone.blocks.0.token_norm.weight") == "norms"

    model = _recurrent_model()
    accounting = model.parameter_accounting()
    assert accounting.components["recurrence"] == model.config.recurrence.parameter_count(
        model.config.d_model
    )
    assert accounting.effective_depth == model.config.effective_depth()
    assert accounting.per_token_block_passes == accounting.effective_depth
    assert accounting.is_recurrent
    assert "effective_depth=" in accounting.summary()

    flat = _tiny_model()
    flat_accounting = flat.parameter_accounting()
    assert flat_accounting.components["recurrence"] == 0
    assert flat_accounting.effective_depth == flat.config.n_layers
    assert not flat_accounting.is_recurrent
    assert "effective_depth=" not in flat_accounting.summary()


def test_recurrence_parameters_land_in_the_expected_weight_decay_groups() -> None:
    model = _recurrent_model()
    groups = classify_parameters(model)
    decay = {groups.names[id(parameter)] for parameter in groups.decay}
    no_decay = {groups.names[id(parameter)] for parameter in groups.no_decay}

    assert "backbone.recurrence.adapter.weight" in decay
    assert "backbone.recurrence.injection_norm.weight" in no_decay
    assert "backbone.recurrence.latent_norm.weight" in no_decay


def test_latent_carry_and_telemetry_stay_float32_under_bf16_autocast() -> None:
    model = _recurrent_model().train()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(input_ids, labels=input_ids, recurrence_steps=3, backprop_steps=2)

    assert output.logits.dtype == torch.bfloat16
    assert output.latent_states.dtype == torch.float32
    assert output.latent_deltas.dtype == torch.float32
    assert output.latent_position_correlation.dtype == torch.float32
    output.loss.float().backward()
    assert model.backbone.recurrence.adapter.weight.grad is not None
