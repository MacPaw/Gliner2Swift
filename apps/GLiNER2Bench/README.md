# GLiNER2 on-device benchmark

A minimal "speedtest"-style iOS app with two tabs:

- **Benchmark** — loads the model at a chosen precision (**fp16** or **int8**), runs four
  extraction scenarios (`ner-8-labels`, `mixed-schema`, `long-text`, `batch-16`), and shows
  per-scenario latency (p50 / p90 / mean / min / max / std / ms-per-token), throughput, and
  peak memory, plus model-load time, MLX active/peak/cache memory, the app's resident
  footprint, and the real device name (e.g. "iPhone 13 Pro Max · A15 Bionic"). Tap **Share**
  to send the results back as Markdown.
- **Predictions** — a displaCy-style entity visualizer: type or pick text, choose entity
  types, and see the recognized spans highlighted inline with their labels and confidences.

## 1. Produce a model directory

The app needs an **fp16** model directory (`model.safetensors` + `config.json` +
`tokenizer.json`). It runs int8 by quantizing that fp16 model at load, so one directory
serves both precisions — no separate int8 file is required on device.

From the repo root:

```sh
# Uses the GLiNER2 reference venv (torch + mlx + huggingface_hub)
GLINER2_VENV=/path/to/GLiNER2/.venv/bin/python

$GLINER2_VENV scripts/convert_weights.py \
    --model fastino/gliner2-base-v1 \
    --output out/gliner2_mlx_fp16 --dtype fp16
```

To also publish a genuinely int8-quantized model to the Hub (usable by MLX-Python and for
distribution — see the note in §4 about Swift loading it directly):

```sh
$GLINER2_VENV scripts/convert_weights.py \
    --model fastino/gliner2-base-v1 \
    --output out/gliner2_mlx_int8 --quantize int8 \
    --push-to-hub your-org/gliner2_mlx_int8 --private
```

## 2. Get the model onto the device — pick one

**A. Bundle it (simplest, ~400 MB heavier app).**
Copy the produced fp16 directory to `apps/GLiNER2Bench/Model` (the folder is gitignored):

```sh
cp -R out/gliner2_mlx_fp16 apps/GLiNER2Bench/Model
```

`project.yml` already references `Model` as a bundled folder, and `ModelLocator` finds it
by that name — no manual dragging in Xcode. (If you skip this step, the project won't build
until the `Model` folder exists.)

**B. Download on first launch (small app).**
Push the fp16 directory to the Hub, then set `ModelLocator.hubRepoId` in
`Sources/ModelLocator.swift` to that repo id, and drop the `Model` line from `project.yml`.
The app downloads the weights into Application Support on first run.

## 3. Create and run the app

**With XcodeGen** (`brew install xcodegen`):

```sh
cd apps/GLiNER2Bench
xcodegen generate
open GLiNER2Bench.xcodeproj
```

**Or by hand in Xcode:**
1. File → New → Project → iOS → App (SwiftUI). Delete its `ContentView.swift`/`App.swift`.
2. Add the `Sources/*.swift` files here to the target.
3. File → Add Package Dependencies → Add Local… → select the repo root (`GLiNER2Swift`).
   Add the `GLiNER2Swift` and `Hub` products to the app target.
4. Add the model per §2.

`project.yml` sets a `DEVELOPMENT_TEAM` and automatic signing — **replace it with your own
team ID** (in `project.yml`, then regenerate, or override it in Xcode's Signing &
Capabilities). Select your iPhone and Run. First launch loads the model (a few hundred ms
once resident), after which **Run benchmark** fills the table. Leave both precisions checked
to get an fp16-vs-int8 comparison in one pass.

## 4. Notes / current limitations

- **The app runs int8 by quantizing the fp16 model at load** (`fromPretrained(…,
  quantization: .int8)`), which is exactly what was measured on macOS: encoder Linear +
  embedding to 8-bit, everything else fp16. On an M3 Pro this took MLX steady-state memory
  from ~416 MB (fp16) to ~254 MB (int8); expect the same shape on device. Accuracy on the
  parity corpus is 56/58 under int8 vs 58/58 at fp16 — two borderline cases move — so int8
  is a memory lever, not free.
- **The on-disk int8 directory the script produces is now loadable directly** — pass its
  path to `fromPretrained` and it loads quantized (detected via the `quantization` config
  block; verified numerically identical to load-time quantization in
  `OnDiskQuantizationTests`). It is ~234 MB vs the fp16 398 MB and skips the fp16→int8
  transient at load, so it is the better choice for a **shipped int8-only** app. For this
  *benchmark* app, though, point `ModelLocator` at the **fp16** directory: that lets the
  precision toggle compare fp16 against runtime-int8 from a single model. (If you point it
  at the int8 directory instead, both toggle positions report int8 — you can't un-quantize
  a packed model.)
- The whole app **compiles for iOS** (`arm64`, Metal shaders and all), and the benchmark
  **engine** (`BenchmarkEngine.swift`, `DeviceInfo.swift`) is additionally validated on macOS
  against the real library at both precisions. If a first build complains, it will be about
  signing or the `Model` folder, not the benchmark logic.
- Memory figures: **MLX active** is the steady-state unified memory held after a cache
  clear (the fairest number to compare precisions). **MLX peak** is the high-water mark
  during a scenario. **App RAM** is the process `phys_footprint` iOS's jetsam limit tracks.
