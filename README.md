# SPALMER

**S**urprise-**P**otentiated **A**ddressed **L**inear
**M**icro-**E**xpert **R**outing Transformer

SPALMER is an open, MIT-licensed research implementation of a decoder-only
language-model architecture built around per-layer low-bit embeddings, hybrid
linear attention, addressed memory, many small dynamically routed experts, and
expert-level adaptive potentiation.

The repository is at the first executable-prototype stage. Components are kept
small, independently testable, and feature-gated so architectural ideas can be
changed without treating an early implementation as a frozen specification.

## Current implementation lanes

- shared configuration and decoder assembly;
- per-layer expand-before-compress embeddings;
- ranked-precedence distributional tokenization;
- Kimi Delta Attention integration;
- surprise-routed micro-experts and adaptive potentiation.

The project favors the simplest faithful implementation of each drafted
component and records experimental limitations explicitly.

## Reference-backend boundary

The initial PLE implementation is a fake-quantization correctness backend. It
uses stochastic low-bit values in the forward pass while retaining floating
shadow weights for autograd, so it does **not** yet provide packed-table memory
savings. A hard allocation guard prevents accidentally scaling that backend to
model-sized tables. Packed/native low-bit storage is a separate backend milestone.
