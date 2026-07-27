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
