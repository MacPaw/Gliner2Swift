# GLiNER2Swift performance baseline

Append a new section at each phase boundary — **never overwrite** earlier numbers.

Reproduce with:

```sh
xcodebuild test -scheme GLiNER2Swift -destination 'platform=macOS' \
  -only-testing:GLiNER2SwiftTests/PerfBenchmarkTests
# fp32 comparison (note the TEST_RUNNER_ prefix):
TEST_RUNNER_GLINER2_MODEL=<fp32-weights-dir> xcodebuild test -scheme GLiNER2Swift \
  -destination 'platform=macOS' -only-testing:GLiNER2SwiftTests/PerfBenchmarkTests
```

---

## Phase 0 baseline — 2026-07-20

| | |
|---|---|
| Machine | Apple M3 Pro, 12 cores (6P/6E), 36 GB, plugged in |
| OS / toolchain | macOS 26.5.2 (25F84), Xcode 26.3, Swift 6.2.4 |
| mlx-swift | 0.30.3 |
| Branch / base commit | `perf/decode-sync-and-encoder-hoists` @ `8f08bc5` |
| Working tree | includes the uncommitted decode-sync changes (bulk `asArray` in SpanDecoder / classification / `getInputIds`, redundant-eval removal) |
| Iterations | 100 timed + 5 warmup (20 for long-text, 5 for batch) |

### fp16 — canonical model (`macpaw-research/gliner2_mlx`)

| scenario | min | p50 | p90 | mean | max | peak MB |
|---|---|---|---|---|---|---|
| 1-ner-8-labels | 30.61 | **32.18** | 33.07 | 32.21 | 34.44 | 433.2 |
| 2-mixed-schema | 40.63 | **41.89** | 42.67 | 41.92 | 44.14 | 436.4 |
| 3-long-text | 74.67 | **76.25** | 77.57 | 76.26 | 78.25 | 656.5 |
| 4-batch-32 | 338.60 | **340.85** | 342.49 | 340.78 | 342.49 | 475.6 |

5-cold-start: **413.5 ms** (`fromPretrained` + first inference).
Steady state after the run: active 397.7 MB, cache 484.8 MB.

### fp32 — converted weights (same architecture, `convert_weights.py` without `--dtype`)

| scenario | min | p50 | p90 | mean | max | peak MB |
|---|---|---|---|---|---|---|
| 1-ner-8-labels | 32.38 | **33.24** | 34.13 | 33.34 | 34.86 | 866.1 |
| 2-mixed-schema | 41.51 | **42.81** | 44.02 | 42.91 | 44.90 | 882.3 |
| 3-long-text | 84.71 | **86.28** | 88.11 | 86.44 | 88.99 | 1322.2 |
| 4-batch-32 | 351.86 | **354.89** | 356.69 | 354.26 | 356.69 | 957.5 |

### fp16 vs fp32 — and what it implies for the plan

| scenario | fp32 p50 | fp16 p50 | speedup | peak memory |
|---|---|---|---|---|
| 1-ner-8-labels | 33.24 | 32.18 | **3.2 %** | 2.00× lower |
| 2-mixed-schema | 42.81 | 41.89 | **2.1 %** | 2.02× lower |
| 3-long-text | 86.28 | 76.25 | **11.6 %** | 2.01× lower |
| 4-batch-32 | 354.89 | 340.85 | **4.0 %** | 2.01× lower |

**fp16 halves memory but buys only 2–12 % latency — far below the 1.5–2× the static
audit predicted.** The audit's reasoning (encoder matmuls are memory-bandwidth-bound, so
halving weight bytes should nearly halve encoder time) does not hold at these shapes. The
gradient across scenarios is the tell: the win grows with sequence length (11.6 % on the
~450-word document, 2–4 % on short text), which is what you would expect if bandwidth
matters only once the tensors get big, while short-input latency is dominated by a
length-independent fixed cost.

That fixed cost is the real target. A single 8-label NER call over one sentence takes
**32 ms**, which is far more than a 12-layer base encoder over ~60 tokens should cost on
an M3 Pro. The plan's own inventory names the mechanisms: hundreds of per-subword kernel
launches in pooling (§3.5), ~500–600 tiny GPU dispatches per `countEmbed` in the GRU
(§3.4), 24 redundant `[512,768]×[768,768]` rel-pos matmuls per call (§3.1), and the
remaining per-field/per-schema sync stalls (§2.1, §2.2).

Practical consequence — **reorder the work relative to the plan as written**: Phase 4's
dtype policy should be treated as a memory optimisation (genuinely valuable for iOS and
for the 2× peak-memory drop) rather than the headline latency win, and Phases 2 and 3
should be executed first. Phase 4.4 (8-bit `QuantizedLinear`) also looks less promising
than the audit assumed, since it targets the same bandwidth bound that fp16 shows is not
the limiter — measure it before investing.

### Test-suite state at this baseline

`xcodebuild test -scheme GLiNER2Swift -destination 'platform=macOS'`

| configuration | executed | skipped | failures |
|---|---|---|---|
| before Phase 0 | 177 | 80 | **66** (+ process crash) |
| after Phase 0, no env vars | 197 | 90 | 0 |
| after Phase 0, models wired | 197 | 2 | 0 |

Prediction-parity gate (`PredictionParityTests`, 58 cases vs Python `fastino/gliner2-base-v1`):
**35 passing, 23 expected failures, 0 unexpected passes, 0 failures.** Every expected
failure is mapped to the Phase 1 item that fixes it; the list must reach empty by the end
of Phase 1.

---

## After PR5 (Phase 2 partial: 2.3 CPU input IDs, 2.4 tokenizer memo) — 2026-07-20

Same machine and protocol as the Phase 0 baseline; fp16 canonical model.

| scenario | baseline p50 | PR5 p50 | change |
|---|---|---|---|
| 1-ner-8-labels | 32.18 | 31.91 | -0.8 % |
| 2-mixed-schema | 41.89 | 42.03 | +0.3 % (noise) |
| 3-long-text | 76.25 | 75.62 | -0.8 % |
| 4-batch-32 | 340.85 | 347.39 | +1.9 % (noise) |

**Both changes measure at roughly 1 %, i.e. within noise** — even though the plan rated
2.4 ("schema prompt tokenization is redone per text and per call") as *high* impact. The
memo demonstrably works (the schema half of the prompt is byte-identical every call, so
it is nearly all cache hits), and the GPU round-trip for input IDs is genuinely gone. The
CPU work they eliminate simply is not on the critical path.

This is the second independent confirmation of the baseline's conclusion: per-inference
latency is dominated by GPU-side kernel-launch and synchronization overhead, not by CPU
preprocessing and not by memory bandwidth. The remaining Phase 2/3 items that reduce
*kernel and sync counts* — 2.1 bulk score readback, 2.2 countPred sync batching, 3.1/3.2
rel-pos hoists, 3.4 GRU restructuring, 3.5 vectorized pooling — are where the headroom
should be, and each should be measured individually rather than assumed.

---

## After PR6 (Phase 3.1 + 3.2: relative-position hoists) — 2026-07-20

Same machine and protocol; fp16 canonical model.

| scenario | baseline p50 | PR6 p50 | improvement |
|---|---|---|---|
| 1-ner-8-labels | 32.18 | **24.03** | **25.3 % faster** |
| 2-mixed-schema | 41.89 | **34.45** | **17.8 % faster** |
| 3-long-text | 76.25 | **61.97** | **18.7 % faster** |
| 4-batch-32 | 340.85 | **294.37** | **13.6 % faster** |

Bit-exact: all 200 tests pass, including AttentionDebugTests' element-wise `< 1e-4` c2p/p2c
gates and `< 1e-3` layer-0 gate, and prediction parity is unchanged at 55/58. This is pure
memoization of computations that were producing byte-identical results.

Costs: peak memory +11 MB (the cached [heads, buckets, headDim] projections, ~19 MB across
12 layers at fp16), and cold start 413 → 465 ms since the first inference now materializes
the caches.

**This validates the baseline's hypothesis.** The win is largest on the shortest input
(25 % on a single sentence, 14 % on the 32-text batch) — exactly the signature of removing
a fixed, length-independent per-call cost. Two failed attempts at the other two candidate
bottlenecks (fp16 for bandwidth, tokenizer memo for CPU) plus this success localize the
remaining headroom firmly in **GPU kernel count and per-call redundant work**: 3.4 (GRU,
~500-600 dispatches per countEmbed), 3.5 (per-subword pooling launches), 2.1/2.2 (decode
sync stalls), 3.3 (fused SDPA).

---

## After PR7 (Phase 3.4: GRU restructuring) — 2026-07-20

| scenario | PR6 p50 | PR7 p50 | change |
|---|---|---|---|
| 1-ner-8-labels | 24.03 | 24.06 | ~0 % |
| 2-mixed-schema | 34.45 | **33.14** | **-3.8 %** |
| 3-long-text | 61.97 | 62.24 | ~0 % |
| 4-batch-32 | 294.37 | 294.08 | ~0 % |

The win lands only on the structure-bearing scenario, which is consistent with the
mechanism: entity-only schemas run `countEmbed` with a single count slot, so the
recurrence is one or two timesteps and there is little per-step overhead to remove. A
4-field structure runs the full predicted-count loop. Expect the gain to scale with
predicted instance counts, so structure- and relation-heavy workloads benefit most.

Cumulative since the Phase 0 baseline: 8-label NER **32.18 -> 24.06 ms (-25.2 %)**,
mixed schema **41.89 -> 33.14 (-20.9 %)**, long text **76.25 -> 62.24 (-18.4 %)**,
batch-32 **340.85 -> 294.08 (-13.7 %)**.

---

## After PR8 (Phase 3.5 + 3.6: vectorized gathers, no retained activations) — 2026-07-20

Same machine and protocol; fp16 canonical model. The "before" column is a fresh run of
`32e1d1a` measured in the same session as the "after" column, so the two are directly
comparable; it reads slightly differently from the PR7 table above purely from run-to-run
variation.

| scenario | before p50 | after p50 | improvement |
|---|---|---|---|
| 1-ner-8-labels | 23.98 | **21.04** | **12.3 % faster** |
| 2-mixed-schema | 32.84 | **29.32** | **10.7 % faster** |
| 3-long-text | 63.28 | **45.41** | **28.2 % faster** |
| 4-batch-32 | 293.10 | **254.00** | **13.3 % faster** |

All 199 tests pass; prediction parity is unchanged at 55/58 with the same three non-ASCII
expected failures. Cold start 428 -> 412 ms.

### The two items measured separately

Running 3.5 alone (3.6's flag flipped back on) gives p50 21.19 / 29.33 / 45.94 / 254.71 —
i.e. **the entire latency win is 3.5**, and 3.6 is within noise on every scenario. That is
what it should be: dropping the retained activations frees buffers, it does not remove
work. 3.6's effect shows up in peak memory instead, and it is modest:

| scenario | peak MB, 3.5 only | peak MB, 3.5 + 3.6 |
|---|---|---|
| 1-ner-8-labels | 445.9 | 444.6 |
| 2-mixed-schema | 454.7 | 451.6 |
| 3-long-text | 699.9 | 699.3 |
| 4-batch-32 | 512.6 | 500.4 |

The 12 MB it returns on batch-32 is real but far short of the "12 x [B, S, 768] buffers"
the plan estimated, so something else sets the high-water mark on the long-text and
single-text scenarios. Worth knowing before Phase 4 sells itself as the memory phase.
(Note the long-text peak is 699 MB here versus 656 MB at the Phase 0 baseline: that
regression arrived with PR6/PR7, which recorded no peak-memory column, not with this
change — `32e1d1a` measures 699.9 MB.)

### Why 3.5 is the biggest single win since the rel-pos hoists

The long document gains the most (28 %), which is the opposite gradient from PR6's
rel-pos hoists (largest on the *shortest* input). That is the expected signature: the
work removed here scales with token count. Pooling used to take one lazy row slice per
subword and one `stacked` + reduce per word — on a ~450-word document that is well over a
thousand kernel launches per sample, all to produce a `[words, 768]` matrix. It is now a
single `take` against index arrays computed during tokenization. Schema-embedding
extraction got the same treatment: the decode path no longer walks every position looking
each token id back up in the vocabulary, because the marker positions are recorded when
the prompt is tokenized, one `take` per schema.

Cumulative since the Phase 0 baseline: 8-label NER **32.18 -> 21.04 ms (-34.6 %)**,
mixed schema **41.89 -> 29.32 (-30.0 %)**, long text **76.25 -> 45.41 (-40.4 %)**,
batch-32 **340.85 -> 254.00 (-25.5 %)**.

Remaining Phase 2/3 kernel-count items are now 2.1 (bulk `spanScores` readback), 2.2
(`countPred` sync batching), 2.5/2.7 and 3.3 (fused SDPA).

---

## After PR9 (Phase 3.3: fused SDPA in the encoder) — 2026-07-20

Same machine and protocol; fp16 canonical model. Two consecutive runs are shown because
the deltas here are small enough that a single run would not distinguish them from noise.

| scenario | PR8 p50 | PR9 p50 (run 1 / run 2) | change |
|---|---|---|---|
| 1-ner-8-labels | 21.04 | 20.83 / 20.89 | -0.9 % |
| 2-mixed-schema | 29.32 | 29.46 / 29.55 | +0.6 % (noise) |
| 3-long-text | 45.41 | **43.05 / 43.40** | **-4.9 %** |
| 4-batch-32 | 254.00 | 250.11 / 248.92 | -1.4 % |

All 205 tests pass with **no tolerance loosened** — including `AttentionDebugTests`'
element-wise `< 1e-4` c2p/p2c gates and its `< 1e-3` layer-0 gate, which is where a
reassociation of this size would show up first. Prediction parity unchanged at 55/58.

The latency win is modest and concentrated on the long document, which fits: the manual
chain's extra kernels (a scaled-key copy, two bias adds, a separate softmax, a dropout
no-op) are cheap at seq ~128 and stop being cheap once the `[batch, heads, seq, seq]`
score matrix is large.

**The memory result is the more interesting one.** Long-text peak drops
**699.3 -> 644.2 MB (-55 MB)**, and it is reproducible across runs. That answers the
question PR8 left open about what sets the long-text high-water mark: it was the
materialized attention scores and probabilities, which the fused kernel never writes out.
Batch-32 peak moves the other way, 500.4 -> ~516.5 MB, presumably fused-kernel workspace
across 8 sequences at once. Net, this is a real memory improvement exactly where memory
was worst.

Cumulative since the Phase 0 baseline: 8-label NER **32.18 -> 20.89 ms (-35.1 %)**,
mixed schema **41.89 -> 29.55 (-29.5 %)**, long text **76.25 -> 43.40 (-43.1 %)**,
batch-32 **340.85 -> 248.92 (-27.0 %)**.

Phase 3 is complete. The remaining Phase 2 items (2.1 bulk `spanScores` readback, 2.2
`countPred` sync batching, 2.5 CPU string micro-fixes, 2.7 span-index construction) are
the next kernel-and-sync work, followed by Phase 4.

---

## After PR10 (Phase 2.1 + 2.2: one score readback, batched count syncs) — 2026-07-20

Same machine and protocol; fp16 canonical model. The "before" column is a control run of
`b9fc121` measured in the same session.

| scenario | before p50 | after p50 | improvement |
|---|---|---|---|
| 1-ner-8-labels | 20.95 | **20.38** | 2.7 % |
| 2-mixed-schema | 29.58 | **26.86** | **9.2 %** |
| 3-long-text | 44.78 | **43.53** | 2.8 % |
| 4-batch-32 | 249.22 | **239.25** | **4.0 %** |

All 205 tests pass — this is a tier-A change and nothing moved. Prediction parity
unchanged at 55/58.

### A trap worth recording: `asArray` on a strided array

The first working version of this change was a **9–36 % regression**, worst on
mixed-schema. Instrumenting it showed 6 ms of a 19.6 ms call inside the single
`asArray` — to copy 1984 floats, i.e. 8 KB. The copy was obviously not the cost.

The cause is in `MLXArray+Bytes.swift`: `asArray` uses `copyBytes` only when the source
backing is contiguous. Otherwise it falls into a Swift loop that walks the array chunk by
chunk, recomputing the source offset with a `zip(index, strides).reduce` per chunk. The
einsum that produces the span scores returns a strided view, so the full-array readback
took the slow path while the old per-field slices happened to take the fast one. Inserting
`MLX.contiguous(...)` — one GPU copy kernel — turned the regression into the table above.

Generalizable: **before reading an MLX array back to the CPU in bulk, make it contiguous.**
The cost is invisible in the shapes and shows up only under measurement. It is also
amplified by the Debug build the benchmark uses, but the fix is right in any configuration.

The distribution of the win is consistent with the mechanism. Mixed-schema gains most (it
has the most (instance, field) pairs, hence the most eliminated round trips), and
entity-only scenarios gain least. 2.2 contributes nothing measurable on its own — a
scenario with one span schema has one count sync either way — and was verified separately
by disabling it; it is kept because it is free and pays off for multi-schema calls.

Cumulative since the Phase 0 baseline: 8-label NER **32.18 -> 20.38 ms (-36.7 %)**,
mixed schema **41.89 -> 26.86 (-35.9 %)**, long text **76.25 -> 43.53 (-42.9 %)**,
batch-32 **340.85 -> 239.25 (-29.8 %)**.

---

## After PR11 (Phase 2.5 + 2.7: span-index construction, tokenizer bounds) — 2026-07-20

| scenario | PR10 p50 | PR11 p50 | improvement |
|---|---|---|---|
| 1-ner-8-labels | 20.38 | **19.77** | 3.0 % |
| 2-mixed-schema | 26.86 | **26.57** | 1.1 % |
| 3-long-text | 43.53 | **41.91** | 3.7 % |
| 4-batch-32 | 239.25 | **221.33** | **7.5 %** |

All 205 tests pass; tier-A clean; prediction parity unchanged at 55/58.

2.7 builds the span (start, end) pairs and their validity flags in the single CPU pass
that was already running, instead of writing (-1, -1) and recovering the same information
on the GPU with two `equal`s, a `logicalOr` and a `where`. That `where` compared against a
float32 `zeros`, which promoted the index tensor to float32 and made the span gather
float-indexed; indices are int32 end to end now. Four kernels per call become none, which
is why batch-32 — four batches per iteration, each paying the fixed cost per sample —
gains the most.

2.5's remaining items: the special-token probe in `preTokenize` now tests one set of first
characters before trying ~15 `hasPrefix` calls per character, and Viterbi's inner loop is
bounded by the real longest vocabulary piece instead of a hardcoded 50.

One trap here too. Deriving that bound with `token.count` over 128k vocabulary entries
cost **~45 ms of cold start** (403-442 ms became 472-478 ms), because `String.count` breaks
graphemes. `token.utf8.count` is O(1) on a native Swift string and can only over-estimate
the Character length, so the bound stays correct and cold start returned to 429 ms. Moving
the computation into the existing parse loop did *not* help — the cost was the 128k
`count` calls themselves, not the extra pass. Worth measuring cold start, not just p50,
whenever load-time work changes.

Phases 2 and 3 are complete. Cumulative since the Phase 0 baseline:
8-label NER **32.18 -> 19.77 ms (-38.6 %)**, mixed schema **41.89 -> 26.57 (-36.6 %)**,
long text **76.25 -> 41.91 (-45.0 %)**, batch-32 **340.85 -> 221.33 (-35.1 %)**.
Cold start 413 -> 429 ms and peak memory is roughly flat (433 -> 450 MB on short text,
656 -> 644 MB on the long document).

---

## After PR12 (Phase 4.1-4.3: dtype policy and pinning) — 2026-07-20

| scenario | PR11 p50 | PR12 p50 | change |
|---|---|---|---|
| 1-ner-8-labels | 19.77 | 19.89 | ~0 |
| 2-mixed-schema | 26.57 | **25.43** | **-4.3 %** |
| 3-long-text | 41.91 | 41.78 | ~0 |
| 4-batch-32 | 221.33 | 221.72 | ~0 |

209 tests pass (4 new). Prediction parity unchanged at 55/58 against both the fp16
snapshot and the fp32 reference.

### What was actually running in fp32

The plan predicted this and it was right: `DownscaledTransformer` built its attention
divisor as `MLXArray(sqrt(Float(headDim)))`, a strongly typed float32 array. Array-to-array
promotion then made everything downstream of `countEmbed` float32 — including the span
score einsum and its sigmoid — so an fp16 checkpoint computed its final scores in fp32.
A Swift scalar divisor adopts the array's dtype instead. Two more of the same shape were
found by audit: the GRU's zero initial hidden state (`MLXArray.zeros` defaults to float32,
which promoted the whole recurrence) and three empty-guard `zeros`.

The gain lands on the structure-bearing scenario, which runs the count-aware path with the
most fields and instances. Peak memory does not move measurably — these are activations at
small shapes, not weights.

### A second, larger finding: `parameters()` was reporting untrained tensors

The new dtype-pinning test failed at first, reporting five float32 parameters under an
all-fp16 checkpoint: the four GRU weights and `encoder.relEmbeddings`. The forward pass
was using fp16 for all five. The discrepancy is the Appendix A §8 trap, and it is worse
than a dtype cosmetic issue: `Module` reflection captures each stored `MLXArray` property
**once, at init**, so the direct assignments in `GRU.loadWeights` and
`DeBERTaEncoder.loadWeights` swapped the properties while leaving reflection pointing at
the random initialization. `parameters()` therefore reported untrained tensors — for the
whole life of the model — while inference used the loaded ones.

Nothing on the current inference path reads `parameters()`, which is why this never
surfaced. It would have surfaced in Phase 4.4 (`quantize` traverses parameters and
modules) and in Phase 5.1 (`compile` captures module state through `inputs:`/`outputs:`).
Both loaders now go through `update(parameters:)`, which replaces the array's context in
place: reflection stays correct and array identity is preserved for compiled graphs.

### Compute-dtype cross-check

fp16 versus an fp32 cast of the *same* snapshot — the only comparison that isolates
compute dtype, since comparing against the fp32 reference checkpoint would fail on weight
rounding alone. Over 36 confidences across 3 texts x 3 thresholds: **identical extracted
spans**, max |fp16 - fp32| confidence delta **0.00088**, far inside tier C's ±0.02.

One nuance worth knowing: two spans can swap places in the output list. fp16 has enough
resolution that "Kenya" and "Rwanda" in one sentence round to the **exact same** confidence
(observed gap 0.0), and the sort that orders results by confidence then has nothing to
separate them. The extracted set is unaffected, which is what the parity gate is defined
on, but callers who care about list order should not treat it as stable across dtypes.
The test asserts set equality and requires any order flip to be a genuine near-tie.
---

## Phase 4.4 (measured, NOT enabled by default) — 2026-07-20

8-bit `QuantizedLinear` over the encoder, group size 64. Each configuration loaded,
measured and released on its own — holding two models at once makes memory meaningless,
and an early run that did exactly that reported a bogus `-19.3 %`.

| configuration | short p50 | long p50 | active MB | peak MB | corpus |
|---|---|---|---|---|---|
| fp16 (shipping) | 16.11 | 28.74 | 415 | 521 | **55/58** |
| int8, Linear only | 15.56 (-3.4 %) | 26.88 (-6.5 %) | 341 | 436 | 53/58 |
| int8, Linear + embeddings | 14.66 (-9.0 %) | 27.05 (-5.9 %) | **253** | **355** | 53/58 |

**The memory win is large and the latency win is small — and neither is free.** Active
memory drops 39 % when the `[128011, 768]` embedding table is included, which is the single
biggest tensor in the model and where most of the saving lives. Latency moves a few per
cent and bounces run to run, so it should not be the reason to enable this.

The cost is two corpus cases. `borderline_accept_ner_many_labels` loses an entity whose
confidence sits within a hair of the threshold, and `choices_list_dtype` moves a confidence
by 0.021 — just outside the ±0.02 tier C allows. Both are borderline decisions rather than
wholesale errors, and the corpus was deliberately built with ≥5 sub-0.05-margin cases per
threshold, so it is doing exactly the job it was designed for.

Per the plan's own rule — *ship only if predictions stay exact on the corpus* — this does
not ship on by default. It is available as `fromPretrained(..., quantization: .int8)` with
the accuracy cost documented on the type, and `PredictionParityTests` accepts
`GLINER2_QUANTIZE=int8` so anyone considering it can re-measure on their own corpus.

Worth noting for judgement: a four-sentence spot check agreed 12/12 under quantization.
The 58-case corpus did not. Small hand-picked samples cannot answer this question.

### Precondition that had to be fixed first

`MLXNN.quantize` replaces children via `update(modules:)`, which throws `needModuleInfo`
for any child not declared `@ModuleInfo`. The encoder's six `Linear` properties and
`DeBERTaEmbeddings.wordEmbeddings` were plain `let`s and are now `@ModuleInfo var`. This
is the correct declaration for a replaceable submodule regardless of quantization, and it
composes with the reflection fix in PR12.
