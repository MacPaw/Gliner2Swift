# GLiNER2 on-device benchmark

A minimal "speedtest"-style iOS app: it loads the GLiNER2 model at a chosen precision
(**fp16** or **int8**), runs a fixed set of extraction scenarios, and shows a table of
latency, throughput, and memory. Tap **Share** to send the results back as Markdown.

Screenshot of what it reports, per precision:

| scenario | p50 ms | tok/s | peak MB |
|---|---|---|---|
| ner-8-labels | … | … | … |
| mixed-schema | … | … | … |
| long-text | … | … | … |
| batch-16 | … | … | … |

plus model-load time, MLX active/peak memory, and the app's resident footprint.

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
Drag `out/gliner2_mlx_fp16` into the Xcode project as a **folder reference** (choose
"Create folder references" — the folder icon is *blue*, not yellow) and name it `Model`.
`ModelLocator` finds it automatically.

**B. Download on first launch (small app).**
Push the fp16 directory to the Hub, then set `ModelLocator.hubRepoId` in
`Sources/ModelLocator.swift` to that repo id. The app downloads it into Application Support
on first run.

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

Then set your signing team under Signing & Capabilities, select your iPhone, and Run.
First launch loads the model (a few hundred ms once resident), after which **Run
benchmark** fills the table. Leave both precisions checked to get an fp16-vs-int8
comparison in one pass.

## 4. Notes / current limitations

- **The app runs int8 by quantizing the fp16 model at load** (`fromPretrained(…,
  quantization: .int8)`), which is exactly what was measured on macOS: encoder Linear +
  embedding to 8-bit, everything else fp16. On an M3 Pro this took MLX steady-state memory
  from ~416 MB (fp16) to ~254 MB (int8); expect the same shape on device. Accuracy on the
  parity corpus is 53/58 under int8 vs 58/58 at fp16 — two borderline cases move — so int8
  is a memory lever, not free.
- **The on-disk int8 directory the script produces is a standard MLX quantized model**
  (packed `.weight`/`.scales`/`.biases` + a `quantization` config block). MLX-Python loads
  it directly. The Swift package currently loads the **fp16** directory and quantizes at
  load; loading a pre-quantized directory directly (smaller download) is a small follow-up
  in `Extractor` — not wired up yet.
- The benchmark **engine** (`BenchmarkEngine.swift`, `DeviceInfo.swift`) is validated on
  macOS against the real library, including both precisions. The iOS build itself was not
  compiled on the machine that wrote this (the iOS platform runtime wasn't installed
  there); if the first build complains, it will be about signing or the model folder, not
  the benchmark logic.
- Memory figures: **MLX active** is the steady-state unified memory held after a cache
  clear (the fairest number to compare precisions). **MLX peak** is the high-water mark
  during a scenario. **App RAM** is the process `phys_footprint` iOS's jetsam limit tracks.
