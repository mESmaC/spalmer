"""ATXY exact-memory path checks (ledger C03/C04), forward-only.

Ported from ``agent/atxy-v0`` and extended for the ``ATXYRequest`` forward
interface, the address-at-input placement, and the production factory gate.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from spalmer import (
    RecurrenceConfig,
    SPALMERBackbone,
    SPALMERBlock,
    SPALMERCausalLM,
    SPALMERConfig,
)
from spalmer.attention import KDAConfig, MLAConfig
from spalmer.experts import MicroExpertsConfig
from spalmer.factory import build_spalmer_model
from spalmer.memory import ATXYConfig, ATXYInjection, ATXYRequest, ATXYStore

ADDRESS = (1, 2, 3, 0)


def make_config(**overrides) -> ATXYConfig:
    base = dict(
        d_model=12,
        value_dim=5,
        a_cardinality=7,
        t_cardinality=6,
        x_cardinality=5,
        y_cardinality=4,
        injection_layer=1,
        initializer_range=0.08,
    )
    base.update(overrides)
    return ATXYConfig(**base)


@pytest.fixture()
def config() -> ATXYConfig:
    return make_config()


@pytest.fixture()
def stored_value() -> Tensor:
    torch.manual_seed(0)
    return torch.randn(5)


@pytest.fixture()
def store(stored_value: Tensor) -> ATXYStore:
    return ATXYStore(
        store_id="unit-store", version=3, value_dim=5, entries=[(ADDRESS, stored_value)]
    )


@pytest.fixture()
def injection(config: ATXYConfig) -> ATXYInjection:
    torch.manual_seed(1)
    return ATXYInjection(config)


def _marked_inputs(d_model: int) -> tuple[Tensor, Tensor, Tensor]:
    torch.manual_seed(2)
    hidden = torch.randn(1, 4, d_model)
    addresses = torch.zeros(1, 4, 4, dtype=torch.long)
    addresses[0, 1] = torch.tensor(ADDRESS, dtype=torch.long)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    mask[0, 1] = True
    return hidden, addresses, mask


def test_exact_lookup_and_version_mismatch(
    config: ATXYConfig, injection: ATXYInjection, store: ATXYStore, stored_value: Tensor
) -> None:
    torch.testing.assert_close(store.lookup(ADDRESS), stored_value, rtol=0, atol=0)
    assert store.lookup((1, 2, 3, 1)) is None
    assert store.lookup((0, 0, 0, 0)) is None
    assert ADDRESS in store and (9, 9, 9, 9) not in store

    with pytest.raises(ValueError, match="duplicate"):
        ATXYStore(
            store_id="s",
            version=1,
            value_dim=config.value_dim,
            entries=[(ADDRESS, stored_value), (ADDRESS, torch.zeros(config.value_dim))],
        )

    hidden, addresses, mask = _marked_inputs(config.d_model)
    with pytest.raises(ValueError, match="version mismatch"):
        injection(hidden, addresses, mask, store, expected_store_version=99)


def test_unaddressed_tokens_unchanged(
    config: ATXYConfig, injection: ATXYInjection, store: ATXYStore
) -> None:
    hidden, _addresses, _mask = _marked_inputs(config.d_model)
    garbage = torch.full((1, 4, 4), -7, dtype=torch.long)
    no_marks = torch.zeros(1, 4, dtype=torch.bool)
    torch.testing.assert_close(
        injection(hidden, garbage, no_marks, store, store.version), hidden, rtol=0, atol=0
    )

    _, addresses, mask = _marked_inputs(config.d_model)
    out = injection(hidden, addresses, mask, store, store.version)
    unmarked = ~mask
    torch.testing.assert_close(out[unmarked], hidden[unmarked], rtol=0, atol=0)


def test_missing_address_injects_no_value(config: ATXYConfig, injection: ATXYInjection) -> None:
    empty = ATXYStore(store_id="empty", version=3, value_dim=config.value_dim)
    hidden, addresses, mask = _marked_inputs(config.d_model)

    injection.value_gate.data.fill_(1.0)
    opened = injection(hidden, addresses, mask, empty, 3)
    injection.value_gate.data.zero_()
    gated_off = injection(hidden, addresses, mask, empty, 3)
    torch.testing.assert_close(opened, gated_off, rtol=0, atol=0)


def test_zero_gate_is_value_identity_but_address_present(
    config: ATXYConfig, injection: ATXYInjection, store: ATXYStore
) -> None:
    assert injection.value_gate.item() == 0.0
    empty = ATXYStore(store_id="empty", version=store.version, value_dim=config.value_dim)
    hidden, addresses, mask = _marked_inputs(config.d_model)

    with_hit = injection(hidden, addresses, mask, store, store.version)
    without_hit = injection(hidden, addresses, mask, empty, store.version)
    torch.testing.assert_close(with_hit, without_hit, rtol=0, atol=0)
    assert not torch.allclose(with_hit, hidden)


def test_nonzero_gate_injects_only_at_marked_positions(
    config: ATXYConfig, injection: ATXYInjection, store: ATXYStore, stored_value: Tensor
) -> None:
    injection.value_gate.data.fill_(1.0)
    hidden, addresses, mask = _marked_inputs(config.d_model)
    out = injection(hidden, addresses, mask, store, store.version)

    torch.testing.assert_close(out[~mask], hidden[~mask], rtol=0, atol=0)
    marked_address = torch.tensor([ADDRESS], dtype=torch.long)
    factorized = (
        injection.embed_a(marked_address[:, 0])
        + injection.embed_t(marked_address[:, 1])
        + injection.embed_x(marked_address[:, 2])
        + injection.embed_y(marked_address[:, 3])
    )
    expected = (
        hidden[0, 1]
        + injection.address_proj(factorized)[0]
        + torch.tanh(injection.value_proj(stored_value.to(hidden.dtype)))
    )
    torch.testing.assert_close(out[0, 1], expected, rtol=1e-5, atol=1e-6)
    # The split entry points compose to the same result.
    staged = injection.inject_values(
        injection.embed_addresses(hidden, addresses, mask), addresses, mask, store, store.version
    )
    torch.testing.assert_close(staged, out, rtol=0, atol=0)


def test_gradients_reach_address_embeddings_value_projection_and_gate(
    config: ATXYConfig, injection: ATXYInjection, store: ATXYStore
) -> None:
    hidden, addresses, mask = _marked_inputs(config.d_model)
    out = injection(hidden, addresses, mask, store, store.version)
    out.sum().backward()
    assert injection.value_gate.grad is not None
    assert injection.value_gate.grad.item() != 0.0

    injection.zero_grad()
    injection.value_gate.data.fill_(1.0)
    out = injection(hidden, addresses, mask, store, store.version)
    out.sum().backward()
    for embedding in (injection.embed_a, injection.embed_t, injection.embed_x, injection.embed_y):
        assert embedding.weight.grad is not None
        assert embedding.weight.grad.abs().sum().item() > 0
    assert injection.address_proj.weight.grad.abs().sum().item() > 0
    assert injection.value_proj.weight.grad.abs().sum().item() > 0
    assert injection.value_gate.grad.abs().sum().item() > 0
    assert sum(p.numel() for p in injection.parameters()) == config.parameter_count


class _StubTokenMixer(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, hidden_states: Tensor, *, state: int | None = None, **_: object):
        return self.projection(hidden_states), 1

    def step(self, hidden_states: Tensor, *, state: int | None = None, **_: object):
        return self.projection(hidden_states), 1


class _RecordingChannelMixer(nn.Module):
    """Remembers its input so tests can see what each block observed."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, d_model, bias=False)
        self.seen: Tensor | None = None

    def forward(self, hidden_states: Tensor) -> Tensor:
        self.seen = hidden_states.detach().clone()
        return self.projection(hidden_states).tanh()


def _tiny_model(atxy: ATXYInjection | None) -> SPALMERCausalLM:
    config = SPALMERConfig(
        vocab_size=31,
        d_model=12,
        n_layers=4,
        tokenizer_version=1,
        tokenizer_fingerprint="test-atxy-v0",
        ple_expansion_factor=2,
    )
    blocks = [
        SPALMERBlock(
            config.d_model,
            _StubTokenMixer(config.d_model),
            _RecordingChannelMixer(config.d_model),
            norm_eps=config.norm_eps,
        )
        for _ in range(config.n_layers)
    ]
    # eval(): deterministic PLE rounding so forward passes are comparable.
    return SPALMERCausalLM(config, SPALMERBackbone(config, blocks, atxy=atxy)).eval()


def _request(store: ATXYStore) -> ATXYRequest:
    addresses = torch.zeros(2, 5, 4, dtype=torch.long)
    addresses[0, 2] = torch.tensor(ADDRESS, dtype=torch.long)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[0, 2] = True
    return ATXYRequest(addresses=addresses, mask=mask, store=store, expected_store_version=1)


def test_atxy_is_optional_and_inert_without_a_request() -> None:
    torch.manual_seed(3)
    plain = _tiny_model(atxy=None)
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    output = plain(input_ids, labels=input_ids)
    assert output.logits.shape == (2, 5, 31)
    assert output.loss is not None

    torch.manual_seed(3)
    wired = _tiny_model(atxy=ATXYInjection(make_config()))
    wired.load_state_dict(plain.state_dict(), strict=False)
    omitted = wired(input_ids)
    torch.testing.assert_close(omitted.logits, output.logits)

    store = ATXYStore(store_id="model-store", version=1, value_dim=5)
    injected = wired(input_ids, atxy=_request(store))
    assert injected.logits.shape == (2, 5, 31)
    assert not torch.allclose(injected.logits, output.logits)

    with pytest.raises(ValueError, match="no ATXY module"):
        plain(input_ids, atxy=_request(store))
    bad = ATXYRequest(
        addresses=torch.zeros(2, 3, 4, dtype=torch.long),
        mask=torch.zeros(2, 5, dtype=torch.bool),
        store=store,
        expected_store_version=1,
    )
    with pytest.raises(ValueError, match="atxy.addresses"):
        wired(input_ids, atxy=bad)


def test_address_enters_the_residual_stream_before_the_injection_boundary() -> None:
    torch.manual_seed(4)
    wired = _tiny_model(atxy=ATXYInjection(make_config(injection_layer=2)))
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    store = ATXYStore(store_id="model-store", version=1, value_dim=5)
    wired(input_ids)
    plain_seen = [block.channel_mixer.seen.clone() for block in wired.backbone.blocks]
    wired(input_ids, atxy=_request(store))
    marked_seen = [block.channel_mixer.seen for block in wired.backbone.blocks]
    # Block 0 already sees the address at the marked position (C03) ...
    assert not torch.allclose(plain_seen[0][0, 2], marked_seen[0][0, 2])
    # ... and no unmarked position anywhere in the stack is disturbed.
    unmarked = torch.ones(2, 5, dtype=torch.bool)
    unmarked[0, 2] = False
    torch.testing.assert_close(marked_seen[0][unmarked], plain_seen[0][unmarked])


def test_value_injection_happens_only_at_the_configured_boundary(
    stored_value: Tensor,
) -> None:
    torch.manual_seed(5)
    atxy = ATXYInjection(make_config(injection_layer=2))
    atxy.value_gate.data.fill_(1.0)
    wired = _tiny_model(atxy=atxy)
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    empty = ATXYStore(store_id="empty", version=1, value_dim=5)
    hit = ATXYStore(store_id="hit", version=1, value_dim=5, entries=[(ADDRESS, stored_value)])
    wired(input_ids, atxy=_request(empty))
    miss_seen = [block.channel_mixer.seen.clone() for block in wired.backbone.blocks]
    wired(input_ids, atxy=_request(hit))
    hit_seen = [block.channel_mixer.seen for block in wired.backbone.blocks]
    # Blocks up to and including the boundary block observe identical inputs;
    # the block after the boundary is the first to see the retrieved value.
    for layer in range(3):
        torch.testing.assert_close(hit_seen[layer], miss_seen[layer])
    assert not torch.allclose(hit_seen[3][0, 2], miss_seen[3][0, 2])


def _recurrent_model(
    atxy: ATXYInjection | None,
    recurrence: RecurrenceConfig,
) -> SPALMERCausalLM:
    config = SPALMERConfig(
        vocab_size=31,
        d_model=12,
        n_layers=4,
        tokenizer_version=1,
        tokenizer_fingerprint="test-atxy-v0",
        ple_expansion_factor=2,
        recurrence=recurrence,
    )
    blocks = [
        SPALMERBlock(
            config.d_model,
            _StubTokenMixer(config.d_model),
            _RecordingChannelMixer(config.d_model),
            norm_eps=config.norm_eps,
        )
        for _ in range(config.n_layers)
    ]
    return SPALMERCausalLM(config, SPALMERBackbone(config, blocks, atxy=atxy)).eval()


def test_atxy_injection_inside_core_is_rejected() -> None:
    torch.manual_seed(6)
    # The exact-value boundary must not be re-applied once per iteration.
    for injection_layer in (1, 2):
        with pytest.raises(ValueError, match="inside the recurrent core"):
            _recurrent_model(
                ATXYInjection(make_config(injection_layer=injection_layer)),
                RecurrenceConfig(1, 2, 1),
            )


def test_atxy_outside_core_fires_once_per_forward() -> None:
    torch.manual_seed(7)
    atxy = ATXYInjection(make_config(injection_layer=3))
    calls = 0
    real_inject = atxy.inject_values

    def counting_inject(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_inject(*args, **kwargs)

    atxy.inject_values = counting_inject
    model = _recurrent_model(atxy, RecurrenceConfig(1, 2, 1, default_steps=3))
    store = ATXYStore(store_id="model-store", version=1, value_dim=5)

    output = model(torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]), atxy=_request(store))

    assert output.recurrence_steps == 3
    # The coda boundary fires exactly once, not once per core iteration.
    assert calls == 1
    # A prelude boundary is equally valid and equally single-shot.
    torch.manual_seed(7)
    prelude_atxy = ATXYInjection(make_config(injection_layer=0))
    prelude_model = _recurrent_model(prelude_atxy, RecurrenceConfig(1, 2, 1, default_steps=3))
    assert prelude_model(torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])).recurrence_steps == 3


def test_factory_gate_and_decode_step_with_atxy() -> None:
    model_config = SPALMERConfig(
        vocab_size=32,
        d_model=16,
        n_layers=4,
        tokenizer_version=1,
        tokenizer_fingerprint="atxy-factory",
        ple_expansion_factor=1,
    )
    kda = KDAConfig(hidden_size=16, num_heads=2, head_k_dim=8, backend="reference")
    mla = MLAConfig(hidden_size=16, num_heads=2, head_k_dim=8, q_latent_dim=8, kv_latent_dim=8)
    experts = MicroExpertsConfig(d_model=16, num_experts=4, expert_inter_dim=8, active_experts=2)
    torch.manual_seed(0)
    plain = build_spalmer_model(model_config, kda, mla, experts).eval()
    assert plain.backbone.atxy is None
    assert not any("atxy" in key for key in plain.state_dict())

    atxy_config = make_config(d_model=16, injection_layer=3)
    torch.manual_seed(0)
    wired = build_spalmer_model(model_config, kda, mla, experts, atxy_config=atxy_config).eval()
    assert isinstance(wired.backbone.atxy, ATXYInjection)
    assert wired.parameter_accounting().components["atxy"] == atxy_config.parameter_count

    store = ATXYStore(store_id="s", version=1, value_dim=5)
    input_ids = torch.randint(0, 32, (2, 5))
    prefill = wired(input_ids, atxy=_request(store))
    step_request = ATXYRequest(
        addresses=torch.zeros(2, 1, 4, dtype=torch.long),
        mask=torch.zeros(2, 1, dtype=torch.bool),
        store=store,
        expected_store_version=1,
    )
    step = wired(
        input_ids[:, :1],
        execution_mode="decode",
        token_mixer_states=prefill.token_mixer_states,
        channel_mixer_states=prefill.channel_mixer_states,
        atxy=step_request,
    )
    assert step.logits.shape == (2, 1, 32)

    with pytest.raises(ValueError, match="injection_layer"):
        build_spalmer_model(
            model_config, kda, mla, experts, atxy_config=make_config(d_model=16, injection_layer=4)
        )
    with pytest.raises(ValueError, match="component widths"):
        build_spalmer_model(model_config, kda, mla, experts, atxy_config=make_config(d_model=8))
