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
spalmer generate runs/prototype.pt --prompt "SPALMER routes"
```

That command trains a tokenizer, assembles the 3:1 KDA/MLA model with
surprise-routed micro-experts, trains next-token prediction, saves the exact
tokenizer and construction configuration with the weights, and generates a
short sample. Model width, layer count, head count, expert count, active expert
count, sequence length, batch size, and training steps are command-line knobs.
Training telemetry stored in the checkpoint includes the separate LM loss,
routing load-balance loss, surprise-calibration loss, predictive entropy, and
the promoted expert identities, alongside the corpus SHA-256 and training knobs.

The current optimized KDA adapter uses `fla-core` when it is installed and
otherwise runs the plain-PyTorch correctness backend.

## Current implementation lanes

- shared configuration and decoder assembly;
- per-layer expand-before-compress embeddings;
- ranked-precedence distributional tokenization;
- Kimi Delta Attention integration;
- non-negative surprise-routed micro-experts calibrated against causally
  aligned realized next-token NLL;
- checkpointed expert-wide adaptive potentiation using one coherent promotion
  mask across every layer-local expert bank.

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
individual or unselected expert. Active expert count is still fixed during one
run; iterative NLL/entropy-driven 2–20 expert expansion is a later milestone.
