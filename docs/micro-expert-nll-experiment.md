# Micro-expert granularity and NLL calibration experiment

Recorded: 2026-09-04. Status: selected next ablation; specification only. No training results or measured performance gains are claimed. This trial takes priority over the plasticity experiment in the [architecture research plan](architecture-research-plan.md).

## Hypothesis and scope

Tiny experts may work particularly well with SPALMER's predicted-surprise routing if calibration helps distribute recurring, homogeneous learning problems among them. Each expert could learn a narrower transformation, while combinations provide broader capacity. SPALMER's tokenizer may reinforce useful regularities, but hold the tokenizer fixed initially: the first trial tests granularity and calibration, not tokenizer synergy.

[PEER](https://arxiv.org/html/2407.04153v1) demonstrates single-neuron experts with product-key retrieval, including language-model compute comparisons. [Scaling Laws for Fine-Grained Mixture of Experts](https://proceedings.mlr.press/v235/ludziejewski24a.html) studies granularity as a scaling variable. These motivate the experiment; neither establishes SPALMER-specific NLL synergy or consumer-GPU speed.

In the current implementation, causal hidden states predict surprise before seeing the next-token target. Training subsequently calibrates the selected mixture's prediction against detached next-token NLL (`src/spalmer/modeling.py`, `_attach_surprise_telemetry`). The realized target is feedback, not an input to that token's routing decision. Inference must retain this causal boundary.

The supervision is mixture-level. Responsibility-attributed NLL is not each expert's counterfactual usefulness. Increasing k shares the observed outcome among more contributors, making individual credit less identifiable. High attributed loss may indicate difficult examples rather than an ineffective expert.

## Primary factorial trial

| Arm | Expert granularity | Surprise calibration |
| --- | --- | --- |
| A | Coarse | Off |
| B | Coarse | On |
| C | Fine | Off |
| D | Fine | On |

Disable calibration only by setting `TrainingConfig.surprise_calibration_weight = 0`; retain learned routing, its score transform, language-model loss, load-balancing loss, and all other objectives. Use the same positive calibration weight in B and D; the current default is 0.05. The off arms still learn routing through the remaining objectives. Record any calibration computation retained for telemetry and include its runtime.

Match expert-pool parameters and active expert parameters per token. For bias-free gated experts, per-layer counts are `P_pool = 3*d*h*E` and `P_active = 3*d*h*k`:

| Quantity | Coarse example | Fine example |
| --- | --- | --- |
| Residual width d | 512 | 512 |
| Expert hidden width h | 64 | 1 |
| Expert count E | 256 | 16,384 |
| Active experts k | 2 | 128 |
| Expert-pool weights per layer | 25,165,824 | 25,165,824 |
| Active expert weights per token per layer | 196,608 | 196,608 |
| Flat router weights, without bias | 131,072 | 8,388,608 |

These examples exclude router and shared-path weights from expert counts. SPALMER shares its router across layers; count its parameters once, but charge scoring at every invocation, including recurrence. This is not an equal-total-model-parameter or equal-total-compute comparison. Record full model counts, router activations, top-k selection, dispatch, and optimizer costs. Keep the same router family initially; product-key retrieval is a separate intervention.

The extreme fine example is an arithmetic reference, not a validated runnable configuration. Explicitly configure active/resident limits to accommodate k and validate actual executor shapes. Start at a scale where all four arms fit. Keep all experts resident and precision fixed, preferably the existing supported BF16 path. Keep backbone, shared path, recurrence, tokenizer, data order, optimizer, and token budget fixed. Pair seeds and initial shared weights. Record routed-update and gradient norms; predeclare any initialization/normalization rule needed to avoid confusing changed output scale with calibration benefit.

## Measurements and decision

After feasibility, use three paired seeds and preserve an untouched final test split. Report held-out NLL, exact bits per byte when target-byte metadata is available, domain/language quality, rare-pattern retention, tokens/s, time-to-quality, peak memory by device, and actual gradient/moment bytes. Report both fixed-token and fixed-time outcomes. Measure expert exposure, routing entropy, selection-weight concentration, dead experts, and unions across batches and recurrent iterations.

Let `G_fine = L_C - L_D` and `G_coarse = L_A - L_B`, using held-out NLL at the same budget. The primary interaction is `G_fine - G_coarse`: a reproducibly positive value supports stronger calibration benefit for fine experts. Fine experts winning both calibration settings would support granularity without establishing special NLL synergy. Require useful end-to-end quality/time behavior as well; predeclare tolerances before confirmation. Report negative results and seed variation.

Keep routing, plasticity, and precision decisions distinct: routing selects contributors; plasticity allocates updates; precision controls numerical fidelity. Retain the same exploration policy in all arms and measure underexposed experts before adding a new policy. On a small diagnostic subset, ablate or replace one selected expert while holding other routing decisions and weights fixed, then measure downstream loss. Define removal versus replacement explicitly. These conditional marginal-contribution probes do not capture every interaction; charge their cost and never tune on the final test split.

## Precision follow-up

Only after the primary trial, compare fixed native precision against selective promotion for the chosen granularity. SPALMER currently promotes using utilization times reconstruction error (`experts/potentiation.py`), a precision-pressure proxy. Validate it with sampled output-distortion and loss probes using fixed routes, then measure effects with ordinary routing. Higher execution precision can reduce forward/backward numerical distortion; it does not change BF16 optimizer-write precision and cannot generally replace missing capacity, training exposure, or useful routing.

Use only verified native weight/activation formats with supported forward/backward shapes. No quantize/dequantize emulation may stand in for native performance. If unsupported, record that limitation and retain the supported baseline. Persistent BF16 masters and optimizer state remain distinct from low-bit execution.

Hypothetically, equal-sized experts with 90% at 4 bits and 10% at 8 bits average 4.4 weight bits. If promoted experts receive 50% of accesses, the access-weighted average is 6 bits. Neither includes scales/metadata, activation formats, masters, or moments; neither predicts throughput. Measure promotion frequency, access distribution, transfer bytes, and quality.

## Separate offload evaluation

Tiny experts alone do not reduce transfer volume at matched active parameters. For the examples above, cold BF16 expert weights cost 384 KiB per token per layer in either case. Under independent uniform selection of k distinct experts per token, expected batch union is `U = E*(1-(1-k/E)**T)`. For T=512, fetching each distinct expert once costs approximately 47.135 MiB per layer for either granularity.

These calculations assume empty caches, reusable weights within the batch, and no repeated eviction; they exclude metadata and all other model traffic. Actual routing locality and cache capacity determine transfers. Evaluate inference offload separately with identical logical routing, cache bytes, and measured misses/evictions. Current expert paging is inference-only; dense training gradients and moments do not become sparse by subdividing experts.

A later training-offload implementation must retain or reload the exact forward-pass weight versions for backward, accumulate gradients by expert ID, and delay updates until all required contributors finish. Ordinary autograd may retain staged GPU weights, defeating eviction. Specify optimizer placement, inactive-expert momentum/decay semantics, and cache invalidation explicitly; pack tiny expert transfers into larger buffers and measure the full forward/backward/update traffic.
