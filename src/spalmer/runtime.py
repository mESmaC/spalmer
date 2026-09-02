"""Minimal training and cached-generation loops for executable prototypes."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from spalmer.modeling import SPALMERCausalLM


@dataclass(slots=True)
class TrainResult:
    losses: list[float]
    tokens_seen: int
    elapsed_seconds: float


def train_token_stream(
    model: SPALMERCausalLM,
    token_ids: Tensor,
    *,
    steps: int,
    batch_size: int,
    sequence_length: int,
    learning_rate: float = 3e-4,
    auxiliary_loss_weight: float = 0.01,
    gradient_clip: float | None = 1.0,
    seed: int = 0,
    log: Callable[[int, float], None] | None = None,
) -> TrainResult:
    """Train directly from one one-dimensional token stream."""

    if token_ids.ndim != 1:
        raise ValueError("token_ids must be one-dimensional")
    if len(token_ids) <= sequence_length:
        raise ValueError("token stream must be longer than sequence_length")
    if min(steps, batch_size, sequence_length) <= 0:
        raise ValueError("steps, batch_size, and sequence_length must be positive")

    device = next(model.parameters()).device
    stream = token_ids.to(device=device, dtype=torch.long)
    generator = torch.Generator(device=device).manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    started = time.perf_counter()
    model.train()

    max_start = len(stream) - sequence_length
    for step in range(1, steps + 1):
        starts = torch.randint(
            0,
            max_start,
            (batch_size,),
            generator=generator,
            device=device,
        )
        offsets = torch.arange(sequence_length + 1, device=device)
        batch = stream[starts[:, None] + offsets[None, :]]

        optimizer.zero_grad(set_to_none=True)
        output = model(batch, labels=batch)
        if output.loss is None:
            raise RuntimeError("model did not return its next-token loss")
        objective = output.loss
        if output.auxiliary_loss is not None:
            objective = objective + auxiliary_loss_weight * output.auxiliary_loss
        objective.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        value = float(objective.detach())
        losses.append(value)
        if log is not None:
            log(step, value)

    return TrainResult(
        losses=losses,
        tokens_seen=steps * batch_size * sequence_length,
        elapsed_seconds=time.perf_counter() - started,
    )


@torch.inference_mode()
def generate_tokens(
    model: SPALMERCausalLM,
    prompt_ids: Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    seed: int = 0,
) -> Tensor:
    """Generate with one prefill followed by explicit recurrent decode steps."""

    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] == 0:
        raise ValueError("prompt_ids must describe one non-empty sequence")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if temperature < 0:
        raise ValueError("temperature cannot be negative")

    device = next(model.parameters()).device
    generated = prompt_ids.to(device=device, dtype=torch.long)
    if max_new_tokens == 0:
        return generated
    was_training = model.training
    model.eval()
    try:
        output = model(generated)
        token_states = output.token_mixer_states
        channel_states = output.channel_mixer_states
        generator = torch.Generator(device=device).manual_seed(seed)

        for token_index in range(max_new_tokens):
            next_token = _select_token(
                output.logits[:, -1],
                temperature=temperature,
                top_k=top_k,
                generator=generator,
            )
            generated = torch.cat((generated, next_token), dim=1)
            if token_index + 1 == max_new_tokens:
                break
            output = model(
                next_token,
                token_mixer_states=token_states,
                channel_mixer_states=channel_states,
                execution_mode="decode",
            )
            token_states = output.token_mixer_states
            channel_states = output.channel_mixer_states
    finally:
        model.train(was_training)
    return generated


def _select_token(
    logits: Tensor,
    *,
    temperature: float,
    top_k: int | None,
    generator: torch.Generator,
) -> Tensor:
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        count = min(top_k, logits.shape[-1])
        cutoff = logits.topk(count, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
    probabilities = logits.float().softmax(dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


__all__ = ["TrainResult", "generate_tokens", "train_token_stream"]
