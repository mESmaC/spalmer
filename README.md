# SPALMER

**S**urprise-**P**otentiated **A**ddressed **L**inear
**M**icro-**E**xpert **R**outing Transformer

SPALMER is an open, MIT-licensed research implementation of a decoder-only
language-model architecture built around exact-plus-compositional per-layer
embeddings, hybrid linear attention, addressed memory, many small dynamically
routed experts, and expert-level adaptive potentiation.

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
- QR-PLE with one exact input table followed by multiplicative
  quotient/remainder refresh lanes trained directly in BF16; checkpoints that
  require the retired full-table fake-QAT topology fail closed on load;
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
  dense loop path;
- native-only routed-expert precision selection, one BF16 persistent/master
  weight payload, FP32 Adam moments, and expert-wide promotion to MXFP8 or BF16
  execution when the selected real kernel supports those operands;
- feature-gated C16 lateral mixing with peer-aware active silencing
  (`directional_config`) and feature-gated ATXY exact memory (`atxy_config`,
  acting only on forward calls that pass an `ATXYRequest`);
- `spalmer.presets` shape hooks for inclusive 10M/50M/100M total-parameter budgets
  at any vocabulary size, with analytic parameter estimates that match the
  measured accounting.

The project favors the simplest faithful implementation of each drafted
component and records experimental limitations explicitly.

The [architecture research plan](docs/architecture-research-plan.md) records
candidate improvements and their evidence. The selected next study is the
[micro-expert granularity and NLL-calibration experiment](docs/micro-expert-nll-experiment.md),
followed by selective precision promotion. These are proposed experiments,
not measured performance improvements.

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

## Native precision boundary

New runs never use quantize/dequantize simulation. QR-PLE does not simulate
quantization: layer 0 is one exact table, and later layers use two vectorized
BF16 codebook gathers, multiplicative lanes, scalar softmax reduction, and a
gated residual. Its codebook size scales approximately with the square root of
vocabulary size. The full-table fake-QAT PLE and historical expert-emulation
checkpoint formats are rejected before model construction; there is no
emulated compatibility fallback.

Run `spalmer precision --json` to inspect the exact weight/activation pairs
available on the selected device. A pair is selectable only when a SPALMER
provider verifies both its real forward and backward kernels; package presence
or advertised hardware dtype support alone is insufficient. `auto` therefore
selects a real provider or fails. BF16/BF16 is the portable default. MXFP4,
NVFP4, MXFP6, and MXFP8 are exposed only when the corresponding native pair is
integrated. In particular, an NVFP4/NVFP4 W4A4 kernel is not mislabeled as a
mixed NVFP4/MXFP8 W4A8 kernel. Expert potentiation changes a complete expert's
native execution precision without adding another master copy.

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
