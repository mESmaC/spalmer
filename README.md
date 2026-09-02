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

## Run the prototype

```shell
python -m pip install -e .
spalmer --smoke --steps 100 --output runs/prototype.pt
spalmer path/to/your-corpus.txt --steps 100 --output runs/prototype.pt
```

That command trains a tokenizer, assembles the 3:1 KDA/MLA model with
surprise-routed micro-experts, trains next-token prediction, saves the exact
tokenizer and construction configuration with the weights, and generates a
short sample. Model width, layer count, head count, expert count, active expert
count, sequence length, batch size, and training steps are command-line knobs.

The current optimized KDA adapter uses `fla-core` when it is installed and
otherwise runs the plain-PyTorch correctness backend.

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
