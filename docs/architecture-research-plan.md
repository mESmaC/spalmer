# SPALMER architecture research plan

Assessment date: 2026-09-04. Source revision: `b8fbe5c28f2c74d816eeee7bb69f1903d712790d`.

Following discussion of this assessment, the selected next experiment is [micro-expert granularity with NLL calibration](micro-expert-nll-experiment.md), followed by selective precision promotion. The plasticity trial below remains a separate proposed experiment. Establish training-state and routing-union measurements before interpreting either trial.

The most direct next hypothesis is to separate total learned capacity from the amount of capacity that is plastic during an update. SPALMER already separates total experts from per-token execution. Extending that separation to gradients and optimizer state could make larger models trainable locally. Whether this preserves convergence speed is an experiment, not an established result.

This document records a source review, primary research, and arithmetic checks. It does not report a training benchmark. The target research hardware and strongest baseline run must be recorded before experiments; no model execution or model tests were performed for this assessment.

The current implementation already includes several relevant mechanisms:

| Mechanism | What the code does | Consequence for this research |
| --- | --- | --- |
| Sparse micro-experts plus a shared SwiGLU path | `src/spalmer/experts/mixer.py` dispatches top-k routes; `experts/bank.py` owns layer-local weights | Extra experts need not proportionally increase useful expert matmul work, but training state still grows |
| Expert precision potentiation | `experts/potentiation.py` ranks precision pressure from utilization and reconstruction error | This controls execution precision; it does not yet allocate learning capacity or consolidate experts |
| Recurrent depth | `modeling.py` and `training/recurrence.py` support repeated core execution and a no-grad prefix | Truncated backprop limits the gradient horizon; all requested forward iterations still cost time |
| Lateral silencing | `directional/mixer.py` computes a lateral update, then gates it | Present inhibition changes representation; it does not skip the computation it inhibits |
| Exact addressed memory | `memory/atxy.py` uses an explicit request and immutable external values | This is different from an automatically accessed, trainable local-pattern memory |
| QR per-layer embeddings | `embeddings/ple.py` combines small lexical codebooks | Compositional lexical storage is already present; another embedding proposal needs to add context-dependent value |

The first bottleneck to measure is training-state density. `experts/bank.py:52` creates three dense parameter stacks covering the entire expert pool. `training/optim.py:192` initializes full-sized FP32 moments, and its update loop at line 221 traverses every element of a parameter with a gradient. A zero gradient in an unused row does not prevent updates from existing momentum or weight decay. Expert weight paging is explicitly restricted to inference (`experts/bank.py:206`).

For the current BF16 lane, weights, gradients, and two FP32 Adam moments require approximately `2P + 2P + 8P = 12P` bytes once gradients and optimizer state exist. This excludes activations, caches, logits, temporary buffers, and allocator overhead. CPU moment offload changes the distribution to approximately `4P` bytes on the accelerator and `8P` on the host, plus staging buffers; the current implementation transfers both moments to and from the accelerator for each updated chunk. Low-bit execution does not change the persistent BF16 master contract.

Token sparsity also needs to be distinguished from batch sparsity. Under an illustrative model in which each token independently selects k distinct experts uniformly from E experts, the expected expert union over T tokens is:

`U = E * (1 - (1 - k/E)**T)`

| E | k | Tokens across the batch | Expected distinct experts |
| --- | --- | --- | --- |
| 200 | 2 | 128 | 144.75 |
| 200 | 2 | 512 | 198.84 |
| 200 | 2 | 2048 | Almost 200 |

These are calculated examples, not measurements of SPALMER routing. Correlated routing can change the result substantially. Measure unions across tokens, accumulation microbatches, and recurrent iterations separately. Reducing top-k alone does not guarantee a small union or small optimizer state.

The initial assessment identified the following candidate experiments. The linked micro-expert study records the subsequent selection of the next trial:

1. **A bounded pool of plastic experts with staged consolidation.** Keep a shared pathway learning while only a configured subset of expert weights receives updates in a stage. Keep consolidated experts available for forward routing. Eventually train, consolidate, and if necessary reactivate different experts. This targets a larger body of learned weights over the training run; it does not mean updating all weights simultaneously. There is direct precedent in [Lifelong-MoE](https://proceedings.mlr.press/v202/chen23aq.html), which grows and freezes experts during lifelong language pretraining. The SPALMER-specific hypothesis is that its data selection and plasticity policy can improve quality per hour on one consumer GPU.

   With the current dense-gradient optimizer, separate frozen and plastic parameter stacks are the straightforward implementation. A custom indexed-gradient and optimizer representation could instead retain shared master storage. Multiplying gradients by a mask on the existing dense stacks will retain full gradient and moment allocations. Frozen branches must still propagate gradients to their inputs when upstream representations are trainable; wrapping those entire branches in `no_grad()` would alter that learning path. Decide explicitly whether consolidation discards optimizer moments or preserves them on the host. Resetting moments on reactivation is a different optimization algorithm from resuming them. If moments are preserved, count their host memory and transfer time.

   With BF16 weights for all parameters and gradients/moments only for plastic parameters, an idealized state budget becomes `2*P_total + 10*P_plastic` bytes. For one billion total parameters and 100 million plastic parameters, that is 2.79 GiB, compared with 11.18 GiB if all are plastic. Always-trainable backbone, router, embeddings, and output head must be included in `P_plastic`. The calculation excludes saved moments for frozen weights and all activation/temporary costs; it is not a claim that a billion-parameter model will train in 4 GiB VRAM.

   Stable weights alone do not guarantee stable behavior: the shared representation and router can drift. Test replay or output regularization, retention on earlier data, and new-expert exploration. Start with a deterministic rotation policy as the control before adding a surprise-dependent controller.

2. **Causal expert groups to reduce the batch union.** Select a small candidate group from prior context, then apply token-level routing within it. Hold the group for a short span; provide a shared fallback and an exploration path. Use prefix information or metadata available at inference. Selecting a group's identity from future tokens in a training document would leak information. [DEMix](https://aclanthology.org/2022.naacl-main.407/) supplies precedent for domain-conditioned experts and domain-homogeneous batches; single-GPU union control is an extrapolation.

   Local specialization and global coverage need different timescales. Encourage broad coverage across training while permitting locality within a span. Track switching, missed useful experts, mixed-domain quality, and utilization. Small unions only reduce training-state costs if the gradient/optimizer representation exploits them; merely restricting routes in today's dense stacks is insufficient. Merely initializing sparse state lazily can also grow to the full pool over time.

3. **Learned causal pattern memory alongside ATXY.** Retrieve trainable vectors using causal token bigrams/trigrams, then inject them through a context-dependent gate at one early layer. This provides a separate place to store recurring linguistic or code patterns. Preserve exact token distinctions for code in the first experiment. [Engram](https://arxiv.org/html/2601.07372v1) supports conditional memory as complementary to MoE in controlled parameter/compute comparisons. Its small reported host-offload overhead is an inference result; training uses distributed table sharding. It does not establish equivalent single-GPU training performance.

   Compare allocating an equal parameter budget to extra experts versus memory, and separately measure actual active work. A trainable table requires an explicit sparse-update and optimizer-state design; sparse gathers alone do not solve state storage. Keep ATXY's exact-value semantics distinct from learned table values. Measure hash collisions, row coverage, rare-pattern retention, and generalization beyond memorized phrases.

4. **Product-key expert retrieval once flat routing becomes expensive.** SPALMER currently scores every expert (`experts/router.py:33`). [PEER](https://arxiv.org/html/2407.04153v1) demonstrates learned product-key retrieval from over a million tiny experts with better language-model performance per compute in its experiments. It reduces expert search cost from exhaustive dependence on E to approximately square-root dependence, plus candidate selection. It factorizes routing keys/search, not the full expert-value storage. Benchmark only after scaling the expert pool enough for flat scoring to matter; there is no assumed speed advantage at 200 experts.

The proposed plasticity trial should isolate selective updates before combining them with other interventions:

| Arm | Total expert pool | Plastic expert budget | Purpose |
| --- | --- | --- | --- |
| A | Current baseline E | All E | Reference quality, state memory, and speed |
| B | Same E | E/4 per stage, rotating | Isolate the effect of selective updates |
| C | 2E | Same absolute plastic budget as B | Test additional stored capacity at bounded update cost |

Keep width, shared path, top-k, recurrence, precision, tokenizer, and training sequence fixed for the architecture comparison. Report the additional router cost in C. Define rotation and moment lifecycle before the run. If replay is used, give the controls the same replay opportunities and count every replay token and its runtime. Begin at a scale where all three arms fit, using identical initial shared weights and matched expert initialization where possible. Use three paired seeds for confirmation after a feasibility run. Do not infer that the full larger pool was learned unless each expert's exposure and update coverage supports it.

Evaluate both a fixed-token budget and a fixed-wall-clock budget. Record held-out NLL and exact bits per byte overall and by language/domain, retention on earlier data, and the quality of small composition/retrieval tasks. Preserve a final untouched test split. If comparing tokenizers later, use identical original held-out bytes and context budgets measured in bytes as well as tokens.

Provisional success criteria are: C doubles the expert pool, retains at least 90% of baseline training throughput, stays within the target machine's measured peak memory budget, and reaches the baseline's validation quality sooner or achieves better quality at equal elapsed time. The 90% threshold is a proposed engineering tolerance, not a research result. Set quality non-inferiority tolerances before confirmation runs and report seed variation and per-domain regressions. Reject the intervention if reduced updates or transfer overhead erase its memory advantage through worse time-to-quality. Increasing stored parameters alone is not success.

Before that trial, extend measurements beyond the existing static planner. `training/engine.py:103` reports elapsed time and some expert/recurrence telemetry, while `experiment/planning.py:529` explicitly excludes activations and caches from memory estimates. A run should record total parameters; parameters eligible for updates; expert unions per token, batch, and optimizer step; actual weight/gradient/moment bytes by device; peak allocated and reserved VRAM; host RAM; synchronized forward/backward/optimizer timings; end-to-end training throughput; and quality versus elapsed time and unique data consumed. Benchmark timing overhead separately.

Several existing details can otherwise confound these experiments:

- `experts/accounting.py:102` describes per-token active parameters as distinct weights, but recurrent core routing can select new experts each iteration. The fixed k-based expert term describes one pass, not the full recurrent union. A recurrent bank can visit up to `min(E, k*R)` distinct experts for one token over R iterations. Runtime union accounting is needed. `modeling.py:812` retains only the final iteration's core metrics, so those metrics cannot reconstruct earlier routes.
- `modeling.py:1484` explicitly calibrates mixture surprise against downstream NLL. Responsibility-attributed NLL is not a counterfactual measure of each expert's usefulness. Harder examples can give a useful expert higher loss. Treat surprise as a candidate selection signal; evaluate growth/consolidation using measured learning progress and occasional matched ablation or replacement probes. Include probe cost and avoid tuning on the final test split.
- `data/sampling.py:163` excludes documents shorter than `sequence_length + 1`; its eligibility report exposes the count. First measure the excluded data by stratum. Compare shorter fixed-window runs on matched data, then investigate length buckets. KDA and MLA currently reject padded batches and in-chunk state resets, so document-safe packing requires attention changes. Variable-length accumulation also needs token-weighted loss normalization; the current trainer averages microbatch objectives equally.
- `training/engine.py:412` accepts an activation-checkpointing option only if the model exposes a hook; the checked-in SPALMER model has no such hook. Do not count this option as an available memory saving in the baseline.
- The recurrent path collects token/channel states for every iteration (`modeling.py:765`, `809`). Detaching a state removes its gradient graph, not its storage. Profile a training-only path that avoids returning unused caches before interpreting recurrence memory scaling. Correct inference state behavior must remain testable independently.

The biological bit-width observation is relevant inspiration with a narrower interpretation than an arithmetic format. [Samavat et al. (2024)](https://papers.cnl.salk.edu/PDFs/Synaptic%20Information%20Storage%20Capacity%20Measured%20with%20Information%20Theory%202024-4652.pdf) estimate approximately 4.1 bits of observed entropy and 4.59 bits of maximum entropy from distinguishable spine-size states in sampled rat hippocampal synapses. These are information-storage estimates from a biological proxy and a particular sample, not a universal numerical precision optimum for transformer weights. The transferable research question is how to allocate finite precision and plasticity according to utility and stability; the numerical optimum still needs model-specific evidence.

The immediate implementation priority is accurate training-state and routing-union measurement, followed by the selected [micro-expert granularity and NLL-calibration trial](micro-expert-nll-experiment.md). Plasticity, causal group routing, and learned memory remain independent ablations until those measurements show which resource actually limits quality per training hour.
