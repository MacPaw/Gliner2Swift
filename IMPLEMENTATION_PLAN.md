# GLiNER2Swift — Performance & Parity Implementation Plan

Prepared 2026-07-20 for implementation with Claude Opus 4.8. Every file:line and API
signature below was verified against the working tree (branch
`perf/decode-sync-and-encoder-hoists`), the pinned `mlx-swift 0.30.3` checkout in
`.build/checkouts/`, and the Python reference (github.com/fastino-ai/GLiNER2).

## 0. Mission & ground rules

Two goals, in priority order:

1. **Prediction parity with Python GLiNER2** — after every change, extracted
   entities/classifications/structures (text, start, end, label) must equal what the
   Python package produces for the same text + schema + threshold.
2. **Speed on Apple Silicon** — eliminate GPU sync stalls, kernel-launch storms, and
   redundant compute; make fp16 actually effective end-to-end.

Ground rules for the implementer:

- **Never start a perf phase before Phase 0 is green.** The parity harness is the only
  thing that makes the perf work safe.
- One PR per numbered task group (see §11). Each PR: all tests green, benchmark table
  (p50/p90 before vs after, §10) in the description.
- Never loosen a numeric tolerance silently. Tolerance changes only per the policy in §9,
  with a comment in the test explaining which phase required it and why.
- When a task says "bit-exact", verify with the strict gates (§9 tier A) before and after.
- MLX-specific mechanics (Module reflection caching, compile state, mask dtypes, promotion
  rules) are in Appendix A. **Read Appendix A before writing any MLX code.**

## 0a. How to run the tests (READ FIRST — `swift test` does not work)

`swift test` **cannot run any MLX test**: SwiftPM's command-line build cannot compile Metal
shaders, so `mlx-swift` has no `default.metallib` and every GPU test aborts the process
with `Failed to load the default metallib`. This is documented upstream
(`.build/checkouts/mlx-swift/README.md:87`). Use `xcodebuild`:

```sh
# Full suite (models auto-discovered where possible)
xcodebuild test -scheme GLiNER2Swift -destination 'platform=macOS'

# With models wired up — note the TEST_RUNNER_ prefix, which is how xcodebuild
# forwards environment variables into the test runner process. Without the prefix
# the variables are silently ignored and everything skips.
TEST_RUNNER_GLINER2_WEIGHTS_PATH=<fp32-weights-dir> \
TEST_RUNNER_GLINER2_MODEL_PATH=<raw-fastino-dir> \
TEST_RUNNER_GLINER2_ADAPTER_PATH=<lora-adapter-dir> \
xcodebuild test -scheme GLiNER2Swift -destination 'platform=macOS'

# Single suite
xcodebuild test -scheme GLiNER2Swift -destination 'platform=macOS' \
  -only-testing:GLiNER2SwiftTests/LoRAParityTests
```

`swift build --build-tests` is still the fast way to check compilation.

Known-good local paths on this machine (all under `/Users/tmwstw/Documents/mnemos/GLiNER2`):
`weights` (converted fp32, 263 tensors), `gliner2-base-v1` (raw PyTorch, 255 tensors),
`gliner2-base-v1/final` (LoRA adapter, 92 pairs). Python generators run with that repo's
venv: `/Users/tmwstw/Documents/mnemos/GLiNER2/.venv/bin/python` (torch 2.10, transformers
5.10, editable `gliner2` 1.2.4).

## 1. Canonical test model & environment

- **Canonical fp16 model (the one we ship against):** local snapshot
  `/Users/tmwstw/.cache/huggingface/hub/models--macpaw-research--gliner2_mlx/snapshots/bdabbd0d3f3ac71d7a012283e2e49571d03a26c9`
  (= HF repo `macpaw-research/gliner2_mlx`). Facts (verified, Appendix B): 263 tensors,
  **all float16**, Swift camelCase keys; loads **as-is** through
  `Extractor.loadWeights` — 263/263 keys consumed, 0 missing, `sanitize` is a no-op for
  it. `config.json`: `counting_layer=count_lstm_v2`, `max_width=8`,
  `token_pooling=first`. Pass the snapshot **directory path** to `fromPretrained` (the
  local-path branch, `GLiNER2.swift:70-74`). Note: passing the repo id instead downloads to
  `~/Documents/huggingface/models/...` (not `~/.cache`) — the comment at `GLiNER2.swift:76`
  is wrong; do not rely on it.
- **fp32 reference model (parity ground truth):** `fastino/gliner2-base-v1` (raw PyTorch
  safetensors, F32). Python GLiNER2 runs this via torch. Use it (a) directly through the
  raw-weights loading path, and (b) converted to fp32 MLX weights via
  `convert_weights.py` when a `weights/` dir is needed.
- **Env vars used by tests** (all set-or-skip after Phase 0.2):
  `GLINER2_FP16_MODEL` (NEW in 0.2 — canonical fp16 snapshot dir),
  `GLINER2_WEIGHTS_PATH` (converted camelCase **fp32** weights dir — the fp32-hardcoded
  L1/component gates key off this; never point it at the fp16 snapshot),
  `GLINER2_MODEL_PATH` (raw fastino dir), `GLINER2_ADAPTER_PATH` (LoRA),
  `GLINER2_MODEL` (benchmark; id or path).
- **Old checkout with missing generator scripts:**
  `/Users/tmwstw/Documents/mnemos/GLiNER2/GLiNER2Swift/scripts/` contains
  `generate_component_fixtures.py`, `generate_attention_fixtures.py`,
  `generate_tokenizer_fixtures.py`, `generate_parity_fixtures.py`, `convert_weights.py`,
  `verify_all_weights.py` — copy the ones we need into this repo's `scripts/` (Phase 0.3).
  Python reference clone for generators: use a checkout of
  `github.com/fastino-ai/GLiNER2` (the generators do `sys.path.insert(0, '../..')` and
  expect to run from a GLiNER2Swift nested inside it, or adjust the path line).
- **Python environment for generators:** `torch`, `transformers`, `tokenizers`,
  `safetensors`, `numpy`, `huggingface_hub` + importable `gliner2` package.

## 2. Phase 0 — Stabilize the test bed (prerequisite for everything)

### 0.1 Make the test target compile
`Tests/GLiNER2SwiftTests/Attempt3LoadingTests.swift:218` uses
`.field("email", dtype:"str", validators:[...])` — `validators:` does not exist on
`StructureBuilder.field` (`GLiNER2.swift:1274-1280`). Comment out that argument with a
`// TODO(Phase 1.4)` marker (the parameter gets added for real in Phase 1.4). Also replace
the machine-specific hardcoded path at `Attempt3LoadingTests.swift:33` with the
`GLINER2_WEIGHTS_PATH` env-var pattern used elsewhere.
**Accept:** `swift test` compiles.

### 0.2 Convert error-outs to skips, wire the canonical snapshot
- Every weight-dependent test must **skip** (`XCTSkip`), never error/fail, when its model
  dir is absent. Known offenders: `RealWeightsTests.testWeightsDirectoryExists:66`
  (XCTAssert on missing dir), `RealWeightsTests` `testParity*` block (1496-2030, loads
  `Fixtures/parity/` with no guard), `TokenizerParityTests.setUp:55-67` (throws),
  `ComponentParityTests`/`RealWeightsTests` fromPretrained fallback that tries to
  hub-download a nonexistent-path string as a repo id.
- Add a shared test helper (one file, e.g. `TestModelLocator.swift`):
  `canonicalFP16ModelDir()` → **`GLINER2_FP16_MODEL`** if set, else the snapshot path
  above if it exists, else nil→skip. Point `PerfBenchmarkTests` and the new
  fp16-targeting tests (0.4, 4.3) at it. Do NOT reuse `GLINER2_WEIGHTS_PATH` for the
  fp16 snapshot: the fp32-hardcoded gates (RealWeightsTests L1 sums,
  ComponentParityTests activation L1s) key off that var and would fail on fp16 weights —
  their skip-on-fp16 machinery only arrives in 4.3.
**Accept:** `swift test` green on this machine with zero env vars set; green with
`GLINER2_FP16_MODEL` pointed at the snapshot (and `GLINER2_WEIGHTS_PATH` unset).

### 0.3 Restore missing fixtures & generators
- Copy the missing generator scripts from the old checkout (§1) into `scripts/`.
- Download the raw `fastino/gliner2-base-v1` snapshot (fp32 PyTorch safetensors) and set
  `GLINER2_MODEL_PATH` to it.
- Run `scripts/convert_weights.py --model fastino/gliner2-base-v1 --output <fp32-weights-dir>`
  (NO `--dtype` flag → fp32) and set `GLINER2_WEIGHTS_PATH` to the output dir.
- Run `verify_all_weights.py` (old checkout) to produce
  `Tests/GLiNER2SwiftTests/Fixtures/weight_reference.json`.
- Run `scripts/generate_inference_fixtures.py` and `generate_parity_fixtures.py` against
  `fastino/gliner2-base-v1` to materialize `Tests/.../Fixtures/inference/` and
  `Fixtures/parity/` (currently missing; several tests reference them).
  **Fixture-input hygiene:** Phase-0 generator inputs must be ASCII-clean and use
  order-neutral schemas (no description dicts whose order matters, no charsmap-affected
  characters) — otherwise these gates cannot go green before Phases 1.2/1.3 land.
  Conversely, the fixtures regenerated AFTER 1.3/6.5 MUST use deliberately
  non-alphabetical dict orders, or an ordering regression is undetectable.
- Delete or wire the orphaned fixtures (`attention_step_by_step.safetensors`,
  `deberta_attention_weights.safetensors`, `gru_weights.safetensors` — ~23 MB combined,
  referenced by no test).
**Accept:** `InferenceParityTests`, `RealWeightsTests` parity block, and
`ComponentParityTests.testWeightL1SumsMatchPython` actually run (not skip) locally.

### 0.4 Build the corpus-level prediction-parity harness (THE master gate)
New generator `scripts/generate_prediction_corpus.py` + new test file
`PredictionParityTests.swift`:
- The generator calls Python with `format_results=True, include_confidence=True,
  include_spans=True` (the default formatted output has neither spans nor confidences —
  the comparator needs both). Corpus inputs may only use API surface Swift already has:
  no `prompt`/`examples`/label-descriptions (6.5), no `max_len` (6.1).
- Corpus: ≥40 cases covering — plain NER (varied label counts), classification
  (single-label, multi-label, and multi-label where ALL probs are below threshold —
  Python emits the argmax label, `engine.py:370-375`; Swift has the same fallback,
  verify it survives refactors), structures (multi-field, dtype str/list, choices,
  descriptions, **predicted count ≥ 2**), relations (**count ≥ 2**, and a case where the
  head is found but the tail is below threshold — the pair must be dropped,
  `engine.py:563`), ONE case combining entities + structure + classification + relations
  in a single schema (exercises multi-schema stacking, marker gather, and prefix-offset
  span indexing), a dtype-str field where the positionally-FIRST above-threshold span
  (row-major (start, width) order — Python takes `spans[0]`, `engine.py:674-675,
  713-717`) is NOT the max-confidence span, choice fields with overlapping labels
  ("very positive"/"positive") and multi-word choices, non-ASCII text (NFD accents
  `é`/`e+U+0301`, ligatures `ﬁ`, fullwidth forms, µ vs μ, CJK, U+00A0/U+200B), texts
  near 512 tokens, thresholds 0.3/0.5/0.7, empty/no-hit cases.
- **Borderline-margin requirement:** the generator must emit, per decision, the margin
  `|confidence − threshold|`, and the corpus must include ≥5 cases with margin < 0.05 at
  each tested threshold (find them by scanning candidate texts). The §9 borderline list
  (< 0.02) is pre-populated from this generator output. Without these, Phase 3's
  "predictions exact" gate passes vacuously.
- Swift side asserts per case: entity/structure/relation sets equal on
  (text, start, end) exactly; labels exact; confidences within tier-C tolerance (§9).
- Cases currently known to diverge go into an explicit `expectedFailures` list in the
  test file, each tagged with the phase that fixes it: choices → 1.1, non-ASCII → 1.2,
  duplicate-mention dedup / relation `relation_extraction` grouping / description-dict
  entity order → 1.3, per-name thresholds / entity+field dtype-str / validators → 1.4.
  The list must shrink to empty by end of Phase 1; empty is enforced from then on.
**Accept:** harness runs against the fp16 snapshot (`GLINER2_FP16_MODEL`); every
non-expected-failure case green.

### 0.5 Capture the baseline
Run `PerfBenchmarkTests` (100 iters) against the fp16 snapshot on this machine, plugged
in. Commit the table (min/p50/p90/mean/max ms + MLX active/cache/peak memory) to
`benchmarks/BASELINE.md` together with commit hash, machine, macOS and mlx-swift versions.
Extend the benchmark first per §10 (multi-schema + structure + batch cases, not just
8-label NER).

## 3. Phase 1 — Parity bug fixes (Swift output ≠ Python output today)

### 1.1 Wire choice-field decoding (currently always nil)
`GLiNER2.swift:1078` calls `decodeChoiceField(..., textTokens: [])` so
`findChoiceIndex` (`SpanDecoder.swift:242-301`) never matches → every `choices:` field
decodes to nil and the key is dropped. Python (`engine.py:623-660`): scores choice tokens
against the **prefix region** of `span_scores` (`prefix_scores = span_scores[inst, fidx, :-text_len]`),
`_find_choice_idx` (`engine.py:761-768`) is a **single forward pass** returning the
FIRST prefix element where `tok.lower() == choice_lower OR choice_lower in tok.lower()`
— NOT exact-match-first-then-substring (two passes give a different index whenever a
substring hit precedes an exact hit, e.g. choices ["very positive", "positive"]: Python
resolves "positive" to the "very positive" element). It reads `prefix_scores[idx, 0]`;
dtype `list` keeps every distinct choice ≥ field threshold, dtype `str` keeps argmax if
≥ threshold else **keeps the key with None**.
- Thread the prefix ELEMENT list verbatim (whole choice strings, unsplit —
  `processor.py:558-562` appends each choice as ONE element; one element = one score
  position; word-splitting the prefix would make multi-word choices never match).
  It is built in `SchemaTransformer.buildClassificationPrefix`; carry it through
  `PreprocessedBatch` into the decode call.
- Match Python exactly: nil result must still set the key (`NSNull`/`Optional` in the
  result dict) — today the nil branch also drops the instance as "empty". BUT keep the
  instance-level content gate: Python drops the whole instance only when ALL values are
  `None`/`[]` (`engine.py:697-699`). So: set nil keys, treat `NSNull` as non-content in
  the Swift `hasContent` check, and keep the any-content gate. Corpus must include:
  (a) structure with all fields below threshold → NO instance emitted; (b) a span field
  with content + a below-threshold str-dtype choice → instance kept, choice key null.
- While there: fix the readback pattern (`prefixScores[idx,0].item(...)` per choice) —
  read the prefix column once via `asArray` (or reuse the Phase 2.1 bulk buffer).
**Tests:** choice cases in the corpus (0.4); move them out of `expectedFailures`. Add
Python-generated fixture cases for both dtypes and for below-threshold (None) results.

### 1.2 Tokenizer normalization — SUPERSEDED, see below

> **Corrected 2026-07-20 by measurement. Do NOT implement the charsmap.** The premise
> below (and in Appendix C) is wrong about which normalizer Python runs. `gliner2` calls
> `AutoTokenizer.from_pretrained(<repo>)`, which builds a fast `DebertaV2Tokenizer` from
> `spm.model` whose normalizer is `Sequence[Replace(Regex("\s{2,}|[\n\r\t]"), " "), NFC(),
> Strip(right)]` — verified live: `normalizer.normalize_str("Ａ") == "Ａ"` (unchanged) and
> `normalize_str("é") == "é"`. The `Strip -> Precompiled -> Replace` chain in the model
> directory's tokenizer.json is a weight-conversion artifact that transformers never
> reads. A faithful charsmap port was implemented, verified against SentencePiece
> semantics (`ﬁ`→`fi`, `Ａ`→`A`, U+00A0→space, `µ`→`μ`, `①`→`1`), and then **removed**,
> because applying it moves Swift *away* from Python.
>
> What shipped instead: NFC + whitespace-collapse + right-strip in
> `UnigramTokenizer.normalize`, and — the larger correctness win — **character offsets
> counted in Unicode scalars rather than Swift Characters**, matching Python's codepoint
> offsets. Swift's word segmentation and offsets now match Python exactly for fullwidth
> and NBSP/ZWSP inputs (they did not before).
>
> Still open (tracked as expected failures with accurate reasons):
> `nonascii_nfd_accents` segments differently because ICU's `\w` matches combining marks
> while Python's `re` does not, so `"zoe\u{301}"` is one word in Swift and two in Python;
> plus residual token-id divergence on fullwidth and zero-width inputs. Reconciling the
> two regex engines is separate work — scope it before starting.

### 1.2 (original, superseded) Implement the `Precompiled` charsmap normalizer
Swift implements only `Strip` (`UnigramTokenizer.swift:175-176`); the model's
`tokenizer.json` normalizer is `Strip → Precompiled(316,720 b64 chars) → Replace(/ {2,}/→" ")`,
and Python's fast tokenizer applies all three. Full port spec in **Appendix C** —
~150 LOC, one new file `Sources/GLiNER2Swift/Tokenizer/PrecompiledCharsmap.swift`:
- base64-decode `normalizer.normalizers[i].precompiled_charsmap` from tokenizer.json;
  parse header (u32 LE trie byte size), darts double-array (`[UInt32]`), normalized-blob.
- `commonPrefixSearch` + `transform` + grapheme-wise `normalize` exactly per Appendix C —
  including the two deliberate quirks: **shortest**-prefix match (`results[0]`), and the
  "grapheme < 6 UTF-8 bytes" whole-grapheme gate.
- Wire-up: parse the `normalizer` section in `UnigramTokenizer.init`
  (`UnigramTokenizer.swift:86-164`, currently ignored); apply in `tokenize(_:)` after
  Strip, before Metaspace; then split the normalized output on spaces (covers the
  `Replace` stage — Appendix C §"Replace"). **Critical:** the space-split pieces stay
  under the ORIGINAL word's `origIdx` — Python maps all subwords of a word to one
  `orig_idx` (`processor.py:1062-1066`) and the word count / start/end char maps come
  from the raw-text splitter; normalization must not change `textLen` or the word maps.
**Tests:** new fixture set generated with Python `tokenizers` (extend
`generate_tokenizer_fixtures.py` from the old checkout): NFD accents, ligatures,
fullwidth, CJK compat, U+200B, U+00A8-style space-injecting mappings, plus the existing
37 cases. Exact token-ID equality. Corpus non-ASCII cases move out of `expectedFailures`.

### 1.3 Output-shape parity: formatting pass, relation grouping, entity order
Three small divergences that make Swift output differ from Python for identical inputs:
- **Format/dedup pass** (`engine.py:865-927`): dedup entity/structure list values
  (plain strings by lowercased text; span dicts by lowercased text; tuples by
  (text.lower, start, end)), falsy struct scalars → nil, and support
  `formatResults: Bool = true` param with raw mode. Implement as a post-pass in
  `GLiNER2.swift` mirroring `_format_entity_dict`/`_format_struct`.
- **Relation grouping** (`engine.py:786-861`): relation results go under a top-level
  `relation_extraction` key, with **all requested relation names present even when
  empty**. Swift currently stores instances at top level (`GLiNER2.swift:1033`).
- **Entity ordering** (`schema.py:178-205`): Python preserves dict insertion order; Swift
  sorts alphabetically (`GLiNER2.swift:1195-1206`). Change `entities([String:String])` to
  an order-preserving representation (e.g. `KeyValuePairs` overload or ordered array of
  pairs) so the serialized prompt token order matches Python. **This changes model input**
  → regenerate any fixture that used a description dictionary, and the regenerated
  fixtures MUST use deliberately non-alphabetical orders (an alphabetical fixture cannot
  detect an ordering regression).
**Tests:** corpus cases with duplicate entity mentions, relations (incl. empty), ordered
description dicts; compare against Python output byte-for-byte on the JSON level.

### 1.4 SchemaMetadata plumbing: per-field/entity/relation dtype & threshold, validators
The decode side already reads `SchemaMetadata` — the actual reads are
`metadata.entityMetadata[...]?.threshold` at ~`GLiNER2.swift:920` and
`fieldMetadata[fieldKey]` threshold/dtype/choices at ~`:1060-1063` (main-branch
numbering; the struct *definitions* are at ~:1329-1352) — but nothing populates it:
`StructureBuilder.field(dtype:threshold:)` **silently discards both**
(`GLiNER2.swift:1274-1298`), `entities()`/`relations()` accept no per-name config.
- Populate metadata from the builder; honor per-name thresholds over the call-level
  threshold and entity `dtype:"str"` scalar output (Python: `engine.py:467-506, 529-533,
  615-621`; relations threshold validation raises on out-of-[0,1]).
- Add `validators: [RegexValidator]` to `field()` and apply at decode exactly like Python
  (`engine.py:668-669`: drop spans failing any validator, before formatting). This also
  un-blocks the Phase 0.1 TODO and the already-written Python fixture generators
  (`validator_filter_cases.json`, `regex_engine_matrix.json` — generated by
  `generate_inference_fixtures.py:386-552` but consumed by no Swift test yet: write the
  consumer test now).
**Tests:** per-field-threshold and dtype-str corpus cases; validator fixture consumer.

### 1.5 Per-sample error isolation
Python: failed record transform → dummy fallback record (`processor.py:366-371`); decode
exception → `{}` for that sample (`engine.py:276-280`). Swift has neither; a malformed
sample can crash the batch. Mirror both behaviors (Swift decode loop: `do/catch` per
sample → empty dict; transform: fallback record `( [P] dummy ( [E] entity ) )` over `"."`).
**Tests:** unit test with a pathological input (e.g. empty text after normalization).

## 4. Phase 2 — Sync-elimination perf (bit-exact; tier-A gates must not move)

Ordered by measured-impact expectation; all are independent, small, and semantically
neutral. Continue the pattern the branch already started (its diff: bulk `asArray` in
`SpanDecoder`/classification/`getInputIds`, removed redundant evals).

### 2.1 One bulk readback of `spanScores` per schema
After `MLX.eval(spanScores)` at `GLiNER2.swift:841`, each `findSpans` call still does
`scores.asArray(Float32.self)` on a fresh lazy slice (`SpanDecoder.swift:94`) — one eval
round trip per field × count-instance (`GLiNER2.swift:927-938, 987-996, 1089-1098`).
Replace: read `[count, fields, L, maxWidth]` back **once** with
`spanScores.asArray(Float32.self)`, pass an `ArraySlice<Float>` + strides (or a tiny
`ScoreView` struct) into `findSpans`/`decodeChoiceField`; delete all MLX calls from
`SpanDecoder`. (Also replaces the `MLX.eval` at 841 — `asArray` evals.)

### 2.2 Batch the `countPred` syncs (do NOT batch the matmul)
`GLiNER2.swift:814-815`: `model.countPred(...)` + `argmax(...).item(Int32.self)` per
schema = one blocking sync each. **Keep each schema's `[1, hidden]` countPred graph
unchanged** — stacking to `[numSchemas, hidden]` changes the matmul shape, Metal may
switch GEMV→GEMM with different accumulation order, and a near-tie count logit flips
`predCount` (a whole-schema output change; violates this phase's bit-exact contract).
Instead batch only the synchronization: build every schema's `argMax` array lazily
first, then one `MLX.eval(allArgmaxArrays)`, then read each with `.item(Int32.self)`
(item on an evaluated array is a cheap copy, not a graph eval). Same numerics, one sync
instead of numSchemas.

### 2.3 Keep input IDs on the CPU
`PreprocessedBatch.getInputIds` (`PreprocessedBatch.swift:145-152`) slices the GPU tensor
and syncs per sample, for data born as Swift `[Int]`
(`SchemaTransformer.swift:541-563`). Store `records.map { $0.inputIds }` as `[[Int]]` on
`PreprocessedBatch`; return directly. Read all of them **before** any eval (matters for
2.6/5.2 ordering).

### 2.4 Cache schema-side preprocessing
`SchemaTransformer.transform` (`SchemaTransformer.swift:110-157`) re-runs prefix
construction and full Viterbi tokenization of every schema token **per text per call**.
Memoize on the transformer: `schemaTokensList`, classification prefix + subword ids,
schema-section `inputIds`, schema-section `mappedIndices`, taskTypes — plus any
schema-derived fields later phases add to `TransformedRecord` (3.5's per-schema marker
positions extend THIS cache). **Cache key: the order-preserving serialized schema-token
string — NOT a `[String: Any]` dict hash** (Swift dictionary hashing/iteration is
order-blind and per-run randomized; after 1.3, entity order is model-input-significant,
so an order-blind key would silently serve the wrong prompt). On hit, tokenize only text
words and append; a text token's `origIdx` = (cached COMBINED prefix length — schema
tokens + separators + classification prefix, i.e. everything before the text words in
the combined array, cf. `processor.py:1020-1042`) + position. Also hoist the schema work
out of the per-text loop in `batchExtract` (`GLiNER2.swift:283-288`) so a cache miss
happens once per call, not per text.

### 2.5 CPU string micro-fixes (matter on long documents)
- `WhitespaceTokenSplitter.swift:82-89`: two `String.distance(from: startIndex, ...)` per
  match = O(n²). Keep a running `(lastIndex, lastOffset)` cursor — one O(n) pass.
- `SpanDecoder.findSpans` (`SpanDecoder.swift:114-117`): `text.count` + two
  `text.index(startIndex, offsetBy:)` per accepted candidate. Precompute per sample:
  `let chars = Array(text)` (or a `[String.Index]` prefix table) shared across all
  findSpans/formatSpans calls for that sample.
- `UnigramTokenizer.preTokenize` (`UnigramTokenizer.swift:287-288`): gate the
  special-token probe on `char == "["` before iterating all 15 `hasPrefix` candidates.
- `UnigramTokenizer` Viterbi (`UnigramTokenizer.swift:341-367`): compute the true max
  piece length at init (actual max is 32, hardcoded 50 today) and add a `[String: [Int]]`
  word→ids memo cache (Zipfian hit rates; the whole-word fast path at :341 already
  catches ~95% of common words — the memo mostly helps schema tokens & repeated OOV).
  Have Viterbi record `(id, score)` and return ids directly — `encode()` currently
  re-hashes every token string (`:205-208`).

### 2.6 `asyncEval` after the encoder
`GLiNER2.swift:395` blocking `MLX.eval(hiddenStates)` idles the CPU for the whole encoder
forward. Replace with `MLX.asyncEval(hiddenStates)` — but **first** materialize all
CPU-side inputs the decode path needs (input IDs via 2.3) so no accidental readback syncs
early. Do **not** simply delete the eval (no overlap would happen — MLX schedules nothing
until the first dependent readback).

### 2.7 Span-index construction cleanup (bit-exact but changes gather dtype)
`Extractor.computeSpanRep` (`Extractor.swift:140-186`): the Swift loop already writes
(-1,-1) padding, then launches 2×`MLX.equal` + `logicalOr` + `MLX.where` against
**float32** `zeros` — promoting the int32 index tensor to f32 (a float-indexed gather;
value-safe today since indices ≤ 511 are exactly representable in f32, but wasteful).
Emit indices + a `[Bool]` validity array in the existing loop, **substituting (0, 0) for
invalid spans** — exactly what today's `where(mask, zeros, idx)` produces; do NOT clamp
to `textLength-1` or invalid rows' spanRep tensors change and the tier-A
component-tensor gate fails. Build the mask with one `MLXArray(boolArray)`; keep indices
int32 end-to-end. Optionally cache the (indices, mask) pair keyed by `textLength` (fixed
`maxWidth`).

**Phase-2 accept:** tier-A gates (§9) byte-identical results; corpus harness green;
benchmark shows the decode-side p50 drop; no `.item(`/`asArray` calls left inside
per-field/per-instance/per-schema loops (grep-able acceptance).

## 5. Phase 3 — Vectorization & encoder hoists (fp-reassociation allowed, tier-B gates)

Numerics change at the reassociation level (~1e-6 fp32). Tier-A element-wise gates may
move within tier-B bounds (§9); prediction-level gates must stay green.

### 3.1 Hoist frozen relative-position projections (biggest single win)
`DisentangledAttention.swift:328` (`keyProj(relEmbeddings)`) and `:387`
(`queryProj(relEmbeddings)`) run inside every layer, every call: 24 identical
`[512,768]×[768,768]` matmuls ≈ 14.5 GFLOP/inference (~20-35% of encoder FLOPs at
seq 128-256). Precompute per layer after weight load: `posKey`/`posQuery`, reshaped to
`[numHeads, numBuckets, headDim]`, plus the shared
`relEmbeddingsLayerNorm(relEmbeddings)` (`DeBERTaEncoder.swift:197`).
**Cache storage:** wrap in a plain Swift class (`final class RelPosCache`) stored on the
module — NOT a stored `MLXArray` var (Module reflection would treat it as a parameter;
see Appendix A §8). Invalidate in `loadWeights` and `loadAdapter` (LoRA can change
`keyProj`/`queryProj`).

### 3.2 Compute relative-position indices once per forward
`makeLogBucketPosition` (`DisentangledAttention.swift:236-240`, math at 101-116) rebuilds
an identical `[S,S]` int32 matrix in each of the 12 layers, and c2p/p2c gather indices
are rebuilt per layer (`:347-363`, `:404-418`) though p2c's index is derivable from c2p.
Compute once in `DeBERTaEncoder.callAsFunction`, pass down through `DeBERTaLayer`; cache
across calls keyed by seqLen (plain-class cache again). Drop the no-op
`.asType(.int32)` casts.

### 3.3 Fused attention in the DeBERTa encoder
Replace the manual chain (`DisentangledAttention.swift:251-294`: scaledKey copy, QKᵀ,
two bias adds, mask add, `MLX.softmax`, dropout, PV matmul) with
`MLXFast.scaledDotProductAttention(queries:keys:values:scale:mask:)` from module **MLX**
(the `MLXFast`-module spelling is deprecated — Appendix A §1):
`bias = (c2pScores + p2cScores) / scaleFactor` (+ additive attention mask when present),
`scale: 1/scaleFactor` where `scaleFactor = sqrt(headDim*3)`. Head dim 64 ⇒ hits the
fused "full" kernel for seq>8 (verified dispatch conditions, Appendix A §1). Dropout is a
no-op (`train(false)`) — delete it from the path.
**fp16 gotcha (matters in Phase 4):** the mask/bias array's dtype must promote to the
q/k/v dtype — a float32 bias with fp16 q/k/v **throws**. Build the bias in the compute
dtype.
**Do NOT convert `DownscaledTransformer`'s attention to SDPA** — headDim=32 misses the
fused kernels entirely (falls back to composed ops; zero win). Only remove its f32
`scale` MLXArray (Phase 4.1).

### 3.4 Restructure the GRU
`GRU.swift:101-176`: per timestep re-slices/transposes 12 weight views and runs 6
matmuls + ~10 elementwise ops → ~500-600 tiny dispatches per `countEmbed`. Restructure
`callAsFunction` (keep the module's stored weights/keys unchanged — checkpoint
compatibility):
- Hoist `weightIH.transposed()`/`weightHH.transposed()` + bias splits out of the loop.
- Precompute ALL input projections in one matmul before the loop:
  `gi = matmul(x.reshaped([L*B, D]), wIHt) + biasIH → [L, B, 3H]`.
- Per step: one `gh = matmul(h, wHHt) + biasHH`; `MLX.split(_, parts: 3, axis: -1)` on
  `gi[t]`/`gh`; standard r/z/n combine. (~1 matmul + ~8 elementwise per step.)
- **Bonus (do in the same PR):** the GRU input is `posEmbedding(0..<count)` broadcast —
  schema-independent (`CountLSTM.swift:161-166`). Precompute
  `posGates = matmul(posEmbedding.weight, wIHt) + biasIH` (`[maxCount, 3H]`) once at
  load; slice `posGates[0..<count]` at inference. This removes the input matmuls from
  the loop entirely.
- Do **not** swap to `MLXNN.GRU` — its bias layout differs (PyTorch-split `b`+`bhn`),
  it still loops per step, and remapping checkpoint keys buys nothing.
**Gate:** `GRUParityTests` atol 1e-4 (fixtures restored in 0.3) + `CountLSTMv2` forward
atol 3e-2 must pass unchanged.

### 3.5 Vectorize pooling & schema-embedding extraction
- `poolTextEmbeddings` (`GLiNER2.swift:610-649`, `aggregateEmbeddings` 654-669): per
  subword lazy row-slices + per-word `stacked`+`mean/max` + final `stacked` → hundreds of
  kernels/sample. Replace: precompute per-sample index arrays at preprocessing time in
  `TransformedRecord` (word-start indices for `.first` — the shipped config's mode;
  segment ids for mean/max). Decode-time: `.first` = one
  `take(subwordEmbeddings, MLXArray(firstIdx), axis: 0)`; `.mean` = one matmul with a
  precomputed `[W, S]` normalized one-hot; `.max` = masked segment max.
- `extractSchemaEmbeddingsPerSchema` (`GLiNER2.swift:565-591` + `stacked` at 474): same
  treatment — record per-schema marker positions in `TransformedRecord` during
  `formatInputWithMapping` (`SchemaTransformer.swift:482` knows token string + schemaIdx
  when emitting), then one `take` per schema. This also deletes the decode-time
  `idToToken` dictionary walk entirely (and the 2.3 input-ids accessor's remaining use).
  **Note:** marker positions for the schema section are schema-derived — on a 2.4 cache
  hit `formatInputWithMapping`'s schema portion doesn't run, so these positions must be
  stored in (and served from) the 2.4 schema cache.

### 3.6 Stop retaining all 13 encoder layer activations
`DeBERTaEncoder.swift:200-209` accumulates `allHiddenStates`; only `lastHiddenState` is
consumed on the hot path (`GLiNER2.swift:392`). Add
`outputHiddenStates: Bool = false`; return empty unless requested (tests that need
per-layer outputs pass `true`). ~12 × [B,S,768] buffers of peak memory saved.

**Phase-3 accept:** corpus harness green; component gates within tier-B; benchmark: p50
single-text and encoder-portion (time-to-`hiddenStates`) drop recorded; kernel-count
sanity via `MLX.Memory` snapshots.

## 6. Phase 4 — fp16 end-to-end dtype policy

> **Measured 2026-07-20 (see `benchmarks/BASELINE.md`) — this phase is a MEMORY win, not
> the headline latency win.** Running the same architecture at fp16 vs fp32 gives only
> 2–12 % lower latency (11.6 % on a ~450-word document, 2–4 % on short text) while halving
> peak memory (433 MB vs 866 MB). The audit's 1.5–2× estimate assumed the encoder matmuls
> are bandwidth-bound; at these shapes they are not — short-input latency is dominated by
> a length-independent fixed cost (kernel launches, graph construction, sync stalls).
> **Do Phases 2 and 3 first**, and measure 4.4 (8-bit quantization) before investing in
> it: it targets the same bandwidth bound that fp16 has shown is not the limiter.

Current state (verified): weights load as f16 (safetensors dtype preserved;
`update(parameters:)` doesn't coerce), Swift scalar literals adopt the array's dtype, so
the encoder/spanRep/classifier/countPred already run f16 **but**
`DownscaledTransformer.swift:175` creates `MLXArray(sqrt(Float(headDim)))` — a strongly
typed f32 array — and array÷array promotes: everything downstream of `countEmbed`'s
transformer (incl. the final span-score einsum + sigmoid, `GLiNER2.swift:839-840`) runs
f32 today.

### 4.1 Kill accidental promotions
- `DownscaledTransformer.swift:174-181`: replace the `scale` MLXArray with a Swift
  scalar divide (`scores / sqrt(Float(headDim))` — scalar literals stay in the array's
  dtype) or fold into the matmul.
- Audit for other strongly-typed f32 array constants on the hot path
  (`MLXArray.zeros` defaults f32 — `Extractor.swift:166` handled in 2.7; grep for
  `MLXArray(` with float literals and `MLX.zeros/ones` in Sources).
- Attention-mask constant `-10000` (`DeBERTaEncoder.swift:224-232`) is f16-safe
  (f16 min ≈ -65504) — keep, but ensure the mask is created in the compute dtype
  (SDPA mask-dtype rule, Appendix A §1).

### 4.2 Explicit dtype policy at load
Add `fromPretrained(..., dtype: DTypePolicy = .auto)`:
`.auto` = keep checkpoint dtype (f16 snapshot → f16; f32 → f32); `.float16`/`.bfloat16`/
`.float32` = cast all floating weights at load
(`weights.mapValues { $0.dtype.isFloatingPoint ? $0.asType(...) : $0 }` before the module
loaders). The layerNorm eps=1e-7 is a **non-issue**: `MLXNN.LayerNorm` keeps `eps` as a
Swift `Float` argument to `MLXFast.layerNorm` (it never round-trips through f16), the
kernel accumulates fp32 internally, and 1e-7 is representable in f16's subnormal range
anyway (min ≈ 5.96e-8). bf16 remains the fallback only if f16 *activations* show
range/precision problems in practice — decide from corpus results, not from eps.

### 4.3 Dtype pinning + fp16 parity gates
- New test: load the canonical snapshot, assert every parameter dtype == f16 and the
  final spanScores dtype == f16 (this is the regression trap for reintroduced
  promotions).
- Corpus harness vs the fp16 snapshot: predictions exact, confidences within tier-C.
- Two separate cross-checks (do NOT compare the f32-cast snapshot against the fastino
  fp32 reference — the snapshot's weights are already fp16-rounded, so that comparison
  fails from weight quantization alone and proves nothing):
  (a) **compute-dtype effect**: f32-cast-of-snapshot vs f16-run-of-snapshot — same
  weights, only compute dtype differs; predictions must be equal, confidences within a
  stated bound (start at ±0.01, record observed max).
  (b) **code bugs**: fp32 reference weights (converted fastino) vs Python — the
  existing tier-A/B gates.
- **Escape hatch for fp16 flips**: if a case with Python margin ≥ 0.02 flips under f16
  while cross-check (a)'s f32-cast run matches Python (i.e. the code is exonerated,
  it's pure precision), add it to the borderline list WITH its Python confidence and
  margin recorded, and note it in the PR. If the f32-cast run ALSO diverges, it is a
  code bug — fix it.
- Existing fp32-specific gates (hardcoded weight L1 sums `RealWeightsTests:731-759`,
  `ComponentParityTests` weight-L1 <0.01% at 380-386 and l1Tolerance 1.0 activation
  gates, `AttentionDebugTests` 1e-4/1e-3 element gates) **remain fp32 tests**: they must
  run with `GLINER2_MODEL_PATH`/fp32-converted weights, and skip (not fail) for fp16
  runs. Do not loosen them to make fp16 pass.

### 4.4 (Experiment, keep behind a flag) 8-bit QuantizedLinear encoder
`MLXNN.quantize(model:groupSize:bits:filter:)` with
`filter: { _, m in m is Linear }` over the encoder only, bits 8, groupSize 64 (all dims
÷64 ✓). **Precondition:** every Linear child replaced must be declared `@ModuleInfo` in
its parent or `update(modules:)` throws (Appendix A §5) — audit
DisentangledAttention/DeBERTaLayer property declarations first and add wrappers where
missing. Measure p50 + corpus accuracy vs fp16; ship only if predictions stay exact on
the corpus.

**Phase-4 accept:** dtype-pin test green; corpus green on fp16; benchmark shows the
encoder p50 drop (expect the biggest absolute win of the whole plan on M-series);
`benchmarks/BASELINE.md` updated with an fp16 column.

## 7. Phase 5 — Throughput: compile, pipelining, parallel preprocessing, memory

### 5.1 `MLX.compile` the encoder forward
Wrap in `GLiNER2` init:
`let compiledEncode = compile(inputs: [model], outputs: [model]) { ... }` (state via
`inputs:`/`outputs:` — there is no `state:` label in Swift; Appendix A §2). Rely on the
**per-shape cache** — do NOT use `shapeless: true` (the encoder reads dims via `.dim()`
into Swift constants; shapeless would silently bake them). To make shapes repeat,
bucket-pad batch seq length in `collateBatch` to e.g. {64, 128, 256, 384, 512} and mask
the tail (attention mask already exists; verify padding rows can't produce spans —
`end <= textLen` guard in `SpanDecoder.swift` already covers decode).
Compile AFTER 3.1-3.3 so the compiled tape is minimal. Measure with/without: if the
per-shape cache thrashes in real workloads, keep compile behind a config flag.

### 5.2 Pipeline `batchExtract`
`GLiNER2.swift:293-307` runs collate → encode → eval → decode strictly serially.
**Precondition: 2.1/2.2/2.6 landed** (decode must not contain early blocking syncs, or
they absorb the next batch's encode on MLX's single FIFO stream — verified semantics,
Appendix A §3). Then: after decoding batch k's *scores are read back*, immediately
`asyncEval` batch k+1's `hiddenStates` before doing batch k's CPU-side span/string work.
Expected: multi-batch throughput → `max(encode, decode)` instead of sum.

### 5.3 Parallel CPU preprocessing
Per-text `processor.transform` is pure Swift and independent per text. Use
`DispatchQueue.concurrentPerform(iterations: texts.count)` writing into a preallocated
array (schema cache from 2.4 is shared read-only after first build — make it
thread-safe: build once before the parallel loop). Keep ALL MLXArray creation in
`collateBatch` on the calling thread (MLX eval lock serializes anyway; array creation
off-thread buys nothing and risks stream confusion).

### 5.4 Memory policy
In `fromPretrained` (after load): `MLX.Memory.cacheLimit = <configurable>` — note
`MLX.GPU.set(cacheLimit:)` is deprecated in 0.30.3; use the `Memory` enum
(Appendix A §4). Default: no limit on macOS, 256 MB on iOS. Expose via `ExtractorConfig`.
Bucket padding (5.1) keeps the allocator shape-stable, which is what actually bounds
cache growth.

**Phase-5 accept:** corpus green; benchmark adds a batch-throughput table (8/32/128
texts) and, for 5.1, cold-vs-warm call latency.

## 8. Phase 6 (optional, parallel-safe) — API-surface parity features

Each is independent; implement in any order after Phase 1. Python references verified:

- **6.1 `maxLen` truncation** — every extract API accepts it; truncate word
  tokens/start/end maps to `maxLen` after word splitting, before prefix/schema encoding
  (`processor.py:408-410`); char maps still index the original string. Contract test:
  `tests/test_inference_max_len.py`.
- **6.2 Long-document APIs** — `extractLong`/`batchExtractLong`/
  `extractEntitiesLong` + `splitTextIntoChunks` (384-word chunks, 64 overlap) +
  `mergeChunkResults` (remap char spans by chunk start; entities merged with overlap
  removal, relations exact-dedup, classification max-confidence/most-frequent; strip
  span metadata per requested flags). Port `chunking.py:44,110` + `engine.py:938-1108`
  faithfully. Depends on 6.1 (chunk extraction passes `maxLen = chunkSize`).
- **6.3 Schema ingestion** — `Schema.fromDict`/`fromJSON`/`toDict` with the pydantic
  validation rules (`schema_model.py:12-191`), raw-dict shorthand normalization, and
  per-text schema lists in `batchExtract` (`engine.py:121-146`).
- **6.4 `extractJson` field-spec mini-language** (`'name::dtype::[a|b]::desc'`,
  `engine.py:1139-1220`).
- **6.5 Classification extras** — `prompt:`, label-description dicts
  (`' [DESCRIPTION] label: desc'`), few-shot `examples` (`' [EXAMPLE] inp [OUTPUT] out'`);
  serialization per `processor.py:865-898` (the special tokens already exist in the
  tokenizer, ids 128008/128009/128010). `classifyText` multi-task dict form
  (`engine.py:1110-1137`). These change model input → add Python-fixture parity cases.
- **6.6 Structure-parent grouping** (`processor.py:639-651`): merge repeated
  `.structure(name)` into one schema with the union of fields.
- **6.7 Loader robustness** — `pytorch_model.bin` fallback, vocab-mismatch repair
  (`model.py:673-756`), HF token pass-through for gated repos
  (`HubApi` token param — `WeightLoader.swift:59-72` currently swallows auth errors),
  and a strict-load mode (`update(parameters:verify: .all)`) so missing keys fail loudly
  instead of silently keeping random init (today every loader `if let`-skips).
- **6.8 Remaining convenience wrappers** — `batchExtractEntities`, `batchClassifyText`,
  `batchExtractJson`, `batchExtractRelations` (`engine.py:1074, 1124, 1151, 1171`):
  thin call-throughs to `batchExtract` with the matching schema; port for API
  completeness, no new logic.

Intentionally NOT planned (dispositions, so nobody hunts for them): the span-score
einsum itself (`sigmoid(einsum('lkd,cpd->cplk'))`) is already a single fused op — its
only planned change is dtype (Phase 4); swapping the Viterbi substring probe for a full
darts trie is superseded by 2.5's max-piece-length + memo cache (revisit only if
profiling still shows Viterbi hot after 2.4/2.5).

## 9. Tolerance & gate policy

| Tier | What | Bound | Applies |
|---|---|---|---|
| A (bit-exact) | Phase 2 refactors | Byte-identical predictions AND unchanged component tensors (existing atol gates untouched) | fp32 reference weights |
| B (reassociation) | Phase 3 | Element-wise: existing atol gates may need at most one order of magnitude (1e-5→1e-4, 1e-4→1e-3); L1-relative gates (1%/5%) unchanged; predictions exact | fp32 reference weights |
| C (fp16) | Phase 4+ | Predictions exact on corpus; confidences within ±0.02 of Python fp32; component-tensor gates NOT applied to fp16 (fp32-only tests skip) | fp16 snapshot |

Rules: a tier-B tolerance bump requires a comment naming the PR and the observed maxDiff.
If predictions ever differ from Python on the corpus, that is a bug — never an acceptable
tolerance. Borderline-confidence flips (|conf − threshold| < 0.02 in Python) may be
excluded from exactness ONLY by adding the case to a documented borderline list with its
Python confidence recorded.

## 10. Benchmark protocol

Extend `PerfBenchmarkTests` into scenarios (all env-selectable, default canonical fp16
snapshot; machine plugged in; 5 warmup + 100 timed):
1. single text × 8-label NER (current case), seq ~128
2. single text × mixed schema (entities + 1 structure w/ 4 fields + classification)
3. single long text (~450 words)
4. `batchExtract` 32 texts, batchSize 8
5. cold-start: `fromPretrained` + first call (startup / compile cost)
Report min/p50/p90/mean/max + `MLX.Memory.snapshot()` before/after. Every PR pastes the
table for scenarios it plausibly affects; `benchmarks/BASELINE.md` is updated at each
phase boundary, never overwritten (append columns).

## 11. PR breakdown & sequencing

| # | Contents | Depends on |
|---|---|---|
| PR1 | Phase 0 complete (0.1-0.5) | — |
| PR2 | 1.1 choices + 1.4 metadata/validators (shared plumbing) | PR1 |
| PR3 | 1.2 charsmap | PR1 |
| PR4 | 1.3 output-shape parity + 1.5 error isolation | PR1 |
| PR5 | Phase 2 complete (2.1-2.7; small commits per item) | PR1 |
| PR6 | 3.1 + 3.2 (rel-pos hoists) | PR5 |
| PR7 | 3.3 (SDPA) | PR6 |
| PR8 | 3.4 (GRU + posGates) | PR5 |
| PR9 | 3.5 + 3.6 (pooling/schema gather, hidden-states) | PR5 |
| PR10 | Phase 4 (4.1-4.3; 4.4 separate follow-up) | PR6-9 |
| PR11 | Phase 5 (5.1-5.4; 5.2 requires PR5) | PR10 |
| PR12+ | Phase 6 items, one per PR, any time after PR4 | PR2-4 |

Current branch `perf/decode-sync-and-encoder-hoists` (uncommitted: bulk asArray in
SpanDecoder/classification/getInputIds, redundant-eval removal, SpanMarker print removal)
should be finished as the seed of PR5: its changes are all Phase-2-category and verified
sync-reduction-only.

PR6/PR8's dependency on PR5 is not technical (they touch encoder/GRU code Phase 2 never
modifies) — it exists so each PR's benchmark delta is attributable to one change set. If
calendar time matters more than clean attribution, they may run in parallel after PR1.

## Appendix A — mlx-swift 0.30.3 API facts (verified in the pinned checkout)

1. **SDPA**: `MLXFast.scaledDotProductAttention(queries:keys:values:scale:mask:)` lives
   in module **MLX** (enum `MLXFast`); the separate MLXFast-module free function is
   deprecated. Shapes rank-4 `[B, H, T, D]`. Additive float mask up to 4-D broadcasts to
   `[B, H, Tq, Tkv]`. **Mask dtype must promote to result_type(q,k,v): f32 mask + f16
   q/k/v throws.** Fused "full" kernel: Tq>8 and D ∈ {64, 80, 128} (D=64 ✓ for the
   encoder; D=32 DownscaledTransformer falls back — don't bother). Softmax internally
   fp32 in the fused kernel.
2. **compile**: `compile(inputs: [any Updatable], outputs: [any Updatable], shapeless:
   Bool = false, _ f:)`. Module state goes through `inputs:`/`outputs:` (no `state:`
   label). Per-shape cache keyed on (shapes, ndim, dtype, constants); shapeless still
   recompiles on ndim/dtype change and silently bakes Swift-side `.dim()` reads — do not
   use it here. Cache lives until the compiled closure deallocates.
3. **asyncEval**: `MLX.asyncEval(_:)` schedules and returns; same FIFO stream as `eval`,
   so any earlier blocking readback absorbs the queued work. Eval calls serialize on a
   global recursive lock across threads.
4. **Memory**: use `MLX.Memory.cacheLimit` / `.memoryLimit` / `.snapshot()` /
   `.clearCache()`; `MLX.GPU.set(cacheLimit:)` is deprecated in 0.30.3.
5. **quantize**: `MLXNN.quantize(model:groupSize:bits:mode:filter:)`; `Linear` and
   `Embedding` are Quantizable; filter `{ _, m in m is Linear }`. Replacement uses
   `update(modules:)` → **every replaced child must be declared `@ModuleInfo`** or it
   throws `needModuleInfo`. groupSize must evenly divide the weight's last dim
   (768 % 64 == 0 ✓; QuantizedEmbedding works for the 128011×768 table).
6. **Dtype/promotion**: `f16 array op f32 array → f32`. But Swift **scalar** literals
   adopt the array's dtype (`f16Array * 2.5` stays f16; `f16Array * MLXArray(2.5)`
   promotes to f32). Int scalars → int32. No global compute-dtype; cast weights at load.
7. **Vectorized ops**: `take(_:_:axis:)`, `takeAlong(_:_:axis:)`,
   `split(_:parts:axis:)`, `stacked`, `concatenated`, `cumsum`, `padded`,
   advanced-indexing subscripts with `MLXArray` indices, and scatter-accumulate via
   `a.at[idx].add(v)`. No dedicated segmented reduce — use one-hot matmul or scatter-add.
8. **Module mechanics (critical for the caches in 3.1/3.2)**: any stored `MLXArray`
   property (even `private var`) is captured by Module reflection **once at init** as a
   parameter; reassigning it later leaves stale reflection state, and
   `update(parameters:)` mutates arrays **in place** (identity preserved — good for
   compiled state). Two safe cache patterns: leading-underscore property name (filtered
   from `parameters()`, still seen by `innerState`), or — **preferred** — a plain Swift
   class wrapper holding the MLXArrays (reflection ignores non-Module classes entirely).
   `freeze()` only affects `trainableParameters()`, not the forward pass.
9. **`MLXNN.GRU` exists** but uses PyTorch-split bias layout (`b`[3H] + `bhn`[H]) and a
   per-step Swift loop — do not adopt; restructure our own (Phase 3.4).

## Appendix B — Canonical snapshot facts (verified)

- 263 tensors, all F16; keys already Swift-style; `Extractor.sanitize` no-op
  (`isRawPyTorchFormat` checks `span_rep.` prefix, absent here); 263/263 consumed,
  0 missing, 0 unconsumed. Embedding `[128011, 768]`.
- `weight_mapping.json` in the snapshot is read by nothing and is stale (255 entries,
  lists pre-split `in_proj` keys that don't exist) — ignore it.
- Special-token ids (pin in tests): `[PAD]`=0 `[CLS]`=1 `[SEP]`=2 `[UNK]`=3
  `[MASK]`=128000 `[SEP_STRUCT]`=128001 `[SEP_TEXT]`=128002 `[P]`=128003 `[C]`=128004
  `[E]`=128005 `[R]`=128006 `[L]`=128007 `[EXAMPLE]`=128008 `[OUTPUT]`=128009
  `[DESCRIPTION]`=128010. The 11 ids ≥128000 exist only in `added_tokens` (all
  `special: true`), not `model.vocab` (128000 entries).
- `SchemaTransformer.createFromLocalDirectory` reads **tokenizer.json only**; `spm.model`
  and `tokenizer_config.json` are unused.
- Known dead spot: `UnigramTokenizer.tokenToId(_:)` ignores `specialTokenToId` (returns
  nil for added-only tokens); no current caller — leave or fix opportunistically.

## Appendix C — Precompiled charsmap port spec (verified against HF `spm_precompiled`)

Data (from `tokenizer.json` `normalizer.normalizers[1].precompiled_charsmap`, base64 →
237,539 bytes):
- Header: `u32 LE trie_size_bytes` (= 177,152 for this model).
- Trie: `trie_size_bytes / 4` little-endian u32 darts-clone double-array units.
- Normalized blob: remaining bytes — concatenated NUL-terminated UTF-8 replacement
  strings.

Unit accessors (u32 `unit`):
`hasLeaf = (unit >> 8) & 1 == 1`; `value = unit & 0x7FFF_FFFF`;
`label = unit & ((1 << 31) | 0xFF)`; `offset = (unit >> 10) << ((unit & (1 << 9)) >> 6)`.

`commonPrefixSearch(key: [UInt8]) -> [Int]`:
```
nodePos = 0; unit = array[0]; nodePos ^= offset(unit); results = []
for c in key:
    if c == 0: break
    nodePos ^= Int(c); unit = array[nodePos]
    if label(unit) != Int(c): return results
    nodePos ^= offset(unit)
    if hasLeaf(unit): results.append(value(array[nodePos]))
return results          // shortest-prefix first
```

`transform(chunk) -> String?`: run the search on the chunk's UTF-8 bytes; empty → nil;
else take **`results[0]`** (the SHORTEST match — HF ships this quirk deliberately;
"seems broken but the original code is exactly like this") as byte offset into the
normalized blob; replacement = bytes up to the next NUL.

`normalize(text)`: iterate extended grapheme clusters (Swift `Character` == UAX#29
grapheme): if `grapheme.utf8.count < 6` AND `transform(wholeGrapheme)` hits → replace the
whole grapheme; else per Unicode scalar in the grapheme: hit → replace, miss → copy.

**Replace stage** (`/ {2,}/ → " "`): text words never contain spaces (splitter regex), so
it only matters when a charsmap replacement *injects* a space (e.g. U+00A8 → `" ̈"`).
Handle by re-splitting the normalized word on spaces before Metaspace/Viterbi — the
split pieces keep the original word's `origIdx` and must not alter word count or the
start/end char maps (see 1.2).

**Risks:** only grapheme-segmentation drift between rust `unicode-segmentation` and Swift
stdlib (both UAX#29) — covered by Python-generated fixtures; otherwise a pure function.

## Appendix D — Which existing gates break when (so failures don't surprise you)

- Phase 2 (bit-exact): nothing may move. If AttentionDebugTests 1e-4 gates move, the
  change was not bit-exact — find the bug.
- Phase 3: may move `AttentionDebugTests` c2p/p2c `<1e-4` (447, 560) and layer0 `<1e-3`
  (759), `ParityTests` atol 1e-5/1e-4, borderline `ComponentParityTests` l1Tolerance 1.0
  absolute (~0.003% relative on ~35k L1 sums). Predictions must not move.
- Phase 4 (fp16): breaks by design — `RealWeightsTests` hardcoded fp32 L1s (731-759),
  `ComponentParityTests` weight-L1 <0.01% (380-386) and activation L1s,
  `AttentionDebugTests` element gates, `LoRAParityTests` synthetic merge <1e-6 if merge
  runs f16. These become fp32-only (skip on fp16), per 4.3. `LoRAParityTests` encoder
  mean<0.01 and confidence ±0.01 gates are expected to still pass.
- Tokenizer changes (Phase 1.2): `TokenizerParityTests` must be regenerated WITH the
  charsmap applied (current 37 fixtures were generated by HF with charsmap active, so
  they should now pass *better* — any case that changes indicates the port is wrong);
  `GLiNER2Tests.testWhitespaceTokenSplitter*` (always-on) must not change (splitter is
  upstream of the normalizer).
