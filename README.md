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
spalmer path/to/source.py --kind code --steps 100 --output runs/code.pt
spalmer generate runs/prototype.pt --prompt "SPALMER routes"
```

That command trains a tokenizer, assembles the 3:1 KDA/MLA model with
surprise-routed micro-experts, trains next-token prediction, saves the exact
tokenizer and construction configuration with the weights, and generates a
short sample. Model width, layer count, head count, expert count, active expert
count, sequence length, batch size, and training steps are command-line knobs.
Training telemetry stored in the checkpoint includes the separate LM loss,
routing load-balance loss, surprise-calibration loss, predictive entropy, the
promoted expert identities, and the model's average surprise, alongside the
corpus SHA-256 and training knobs. `spalmer generate ... --dynamic-residency`
lets the inference residency controller grow the active expert set from the
configured minimum while the prompt's surprise stays above that average.
The corpus `--kind` is saved and reused for prompt tokenization; pass
`spalmer generate ... --kind prose|code|mixed` to override it.

The current optimized KDA adapter uses `fla-core` when it is installed and
otherwise runs the plain-PyTorch correctness backend.

## Current implementation lanes

- shared configuration and decoder assembly;
- per-layer expand-before-compress embeddings;
- ranked-precedence distributional tokenization, with a character n-gram
  proxy model recursively scoring phrase and within-word splits, PMI-scored
  salvage merges, and Python-aware routing of code, comments/docstrings, and
  out-of-distribution string contents;
- Kimi Delta Attention integration;
- non-negative surprise-routed micro-experts calibrated against causally
  aligned realized next-token NLL;
- checkpointed expert-wide adaptive potentiation using one coherent promotion
  mask across every layer-local expert bank, decided by precision pressure
  (selection frequency times quantization reconstruction error) alone;
- a request-level inference residency controller that starts at the two-expert
  minimum and expands by bounded increments, recomputing effective NLL after
  each step and rolling back expansions that do not pay.

The project favors the simplest faithful implementation of each drafted
component and records experimental limitations explicitly.

## Reference-backend boundary

The initial PLE and expert implementations are fake-quantization correctness
backends. They use stochastic low-bit values in the training forward pass while
retaining floating shadow weights for autograd, so they do **not** yet provide
packed-storage memory savings. Expert potentiation reversibly promotes a whole
expert identity to those shadow weights; it establishes the controller semantics
but is not yet packed FP4 plus an FP8 residual. A hard allocation guard prevents
accidentally scaling the PLE shadow tables beyond prototype size.

The surprise-calibration target is mixture-level realized NLL attributed by
routing responsibility. It is not a measured counterfactual NLL for every
individual or unselected expert. The active expert count is fixed during
training; at inference the residency controller changes it per request at
prefill, not per decoded token, and does not yet move experts between devices.
The current override is model-global and therefore assumes serial generation.
Legacy v1/v2 checkpoints have no calibrated average-surprise state unless one
was recorded in metadata, so dynamic residency on those bundles uses the
documented zero-baseline cold start.
