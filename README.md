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
corpus SHA-256 and training knobs. CUDA generation defaults to full-pool
logical routing with bounded layer-local expert paging. The legacy experimental
`spalmer generate ... --dynamic-residency` mode instead grows one prompt-global
resident candidate set while preserving the configured per-token top-k.
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
- full-pool per-token routing with bounded layer-local expert paging:
  `model.enable_expert_offload(device)` keeps every complete layer-local expert
  master bank on CPU while independently caching only the rows selected in each
  layer. Long prefills are tiled when their distinct routes exceed cache
  capacity, so physical residency does not alter the trained top-k decision.
  The older prompt-global `model.residency` controller remains available with
  `paging=False` for explicit experiments, and `model.parameter_accounting()`
  reports logical resident and per-token active parameter counts;
- an always-on shared SwiGLU channel path beside the routed micro-experts
  (`shared_inter_dim`), with grouped batched expert execution and a per-expert
  loop reference path;
- routed-expert W4A8 QAT with selectable MXFP4 or NVFP4 forward weights,
  MXFP8 expert inputs, one BF16 persistent/master weight payload, FP32 Adam
  moments, and expert-wide promotion to MXFP8 or BF16 execution;
- feature-gated C16 lateral mixing with peer-aware active silencing
  (`directional_config`) and feature-gated ATXY exact memory (`atxy_config`,
  acting only on forward calls that pass an `ATXYRequest`);
- `spalmer.presets` shape hooks for inclusive 10M/50M/100M total-parameter budgets
  at any vocabulary size, with analytic parameter estimates that match the
  measured accounting.

The project favors the simplest faithful implementation of each drafted
component and records experimental limitations explicitly.

## Plan scales and prepare data

Model size and vocabulary size are independent experiment inputs. A ladder can
therefore increase vocabulary with model capacity without constructing a model.
The default search keeps the architecture's 200 expert identities fixed and
changes expert width and the dense shape to approach each target:

```shell
spalmer plan 10m 50m 100m --vocab-size 4096 8192 16384
```

Approved JSONL exports can be filtered to English plus the six most represented
code languages, exactly/normalization deduplicated, split by content identity,
and encoded into immutable mmap-backed uint32 shards with a pre-existing RPD or
Hugging Face tokenizer:

```shell
python -m pip install -e ".[hf]"
spalmer prepare-data path/to/approved-jsonl \
  --name first-experiment \
  --tokenizer-hf path/to/tokenizer \
  --local-files-only \
  --output-directory data/first-experiment
```

The generated manifest, tokenizer identity, shard checksums, deterministic
weighted-sampler state, RNG state, optimizer/controller payloads, and evaluation
accumulators provide the non-model pieces needed for bound resume and held-out
English/per-language NLL, perplexity, repetition, and router telemetry. Exact
bits-per-byte is reported only when a prepared shard carries exact target-byte
metadata; it is never inferred from a lossy token decode. RPD preparation gives
EOD a reserved model-only ID after the content vocabulary rather than relabeling
an ordinary token. The training engine moves only the current mmap sample batch
to the GPU. Base pretraining keeps one BF16 parameter/master payload and BF16
gradients; Adam moments remain independent FP32 state. Stochastic BF16 writeback
preserves small updates without creating a duplicate FP32 master weight copy.
CPU optimizer-state offload is available as an explicit training policy: FP32
Adam moments remain in pinned host memory when available, while bounded moment
chunks visit the accelerator for each update. Same-device moments remain the
default.

## Reference-backend boundary

PLE and routed-expert low-precision paths currently remain quantize/dequantize
QAT correctness backends, so they do **not** yet provide packed-storage or
low-bit GEMM speedups. The routed experts nevertheless exercise the requested
MXFP4-or-NVFP4 weight and MXFP8 activation numerics from a BF16 master. A strict
native request fails instead of silently using BF16 matmul. The installed SM120
software stack has no mixed W4A8 grouped kernel; native execution is a separate
backend milestone. Backward currently uses BF16 autograd through the QAT STE;
an explicit MXFP8 backward kernel is also a later backend milestone. Expert
potentiation changes a complete expert's derived
forward precision to MXFP8 or BF16 without adding another master copy. A hard
allocation guard prevents accidentally scaling the PLE tables beyond prototype
size.

The surprise-calibration target is mixture-level realized NLL attributed by
routing responsibility. It is not a measured counterfactual NLL for every
individual or unselected expert. The active expert count is fixed during
training. CUDA generation enables physical expert offload by default: full
expert masters remain on CPU, the router scores the complete trained pool, and
selected rows are paged through an independent bounded cache in each layer.
`--expert-cache-size N` sets that physical per-layer ceiling without masking
router choices; long prefills tile all required experts through it. Cache
replacement is transactional and keeps at most `N` published rows per layer,
though the brief allocate-before-publish window can hold roughly `2N` rows.
Pass `--no-expert-offload` for all-resident CUDA inference.
`--dynamic-residency` explicitly selects the older prompt-global masked policy
and is experimental for checkpoints trained with full-pool routing. The cache
runtime assumes serial generation.
Legacy v1/v2 checkpoints have no calibrated average-surprise state unless one
was recorded in metadata, so dynamic residency on those bundles uses the
documented zero-baseline cold start.
