# GLiNER2Swift

<img width="2000" height="800" alt="Cover - white@2x" src="https://github.com/user-attachments/assets/a254dd02-88f1-41a4-a649-ce08b9aee00e" />

[![Swift](https://github.com/MacPaw/Gliner2Swift/actions/workflows/swift.yml/badge.svg)](https://github.com/MacPaw/Gliner2Swift/actions/workflows/swift.yml)
[![Platforms](https://img.shields.io/badge/platforms-macOS%2014%20%7C%20iOS%2017-blue)](https://github.com/MacPaw/Gliner2Swift)
[![Swift](https://img.shields.io/badge/swift-5.9%2B-orange)](https://swift.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Twitter](https://img.shields.io/static/v1?label=Twitter&message=@MacPaw&color=CA1F67)](https://twitter.com/MacPaw)

Swift/MLX implementation of [GLiNER2](https://github.com/fastino-ai/gliner2) - a unified schema-based information extraction framework.
([Article](https://research.macpaw.com/publications/gliner2-swift)) 
## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Available Models](#available-models)
- [API Reference](#api-reference)
- [LoRA Adapters](#lora-adapters)
- [Architecture](#architecture)
- [Performance](#performance)
- [Work in Progress](#work-in-progress)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Features

- Named Entity Recognition (NER)
- Text Classification (with optional prompts, label descriptions, and few-shot examples)
- Structured Data Extraction (JSON schemas, including the `name::dtype::[a|b]::desc` field-spec form)
- Relation Extraction
- Long-document extraction via automatic chunking, plus `maxLen` truncation
- Optional int8 encoder quantization (~40% less memory)
- LoRA adapter loading (merge at load time, zero runtime overhead)
- Native Apple Silicon support via MLX (Metal GPU)

## Requirements

- macOS 14.0+ or iOS 17.0+
- Swift 5.9+
- Apple Silicon (M1/M2/M3) on macOS; A-series (Metal) on iOS

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/MacPaw/Gliner2Swift", branch: "main"),
]
```

Or in Xcode: File → Add Package Dependencies → Enter the repository URL.

## Quick Start

```swift
import GLiNER2Swift

// Load model (downloads automatically from HuggingFace, or pass a local directory)
let model = try await GLiNER2.fromPretrained("fastino/gliner2-base-v1")

// Extract entities — results come back as a [String: Any] dictionary
let text = "Tim Cook is CEO of Apple in Cupertino."
let result = model.extractEntities(
    text: text,
    entityTypes: ["person", "company", "location"],
    includeSpans: true
)

if let entities = result["entities"] as? [String: [Any]] {
    for (label, spans) in entities {
        for case let span as [String: Any] in spans {
            print("\(label): \(span["text"]!) [\(span["start"]!)-\(span["end"]!)]")
        }
    }
}
// person: Tim Cook [0-8]
// company: Apple [23-28]
// location: Cupertino [32-41]
```

## Available Models

| Model | Parameters | HuggingFace ID |
|-------|------------|----------------|
| Base | 205M | `fastino/gliner2-base-v1` |

## API Reference

The extraction methods are synchronous (not `throws`) and return a `[String: Any]` result
dictionary; only `fromPretrained` is `async throws`.

### Entity Extraction

```swift
let result = model.extractEntities(
    text: "Your text here",
    entityTypes: ["person", "organization", "location"]
)
// result["entities"] is [String: [Any]] — label → list of matches
```

### Text Classification

```swift
let result = model.classifyText(
    text: "Great product, highly recommend!",
    task: "sentiment",
    labels: ["positive", "negative", "neutral"]
)
// result["sentiment"] == "positive"
```

### Structured Extraction

```swift
let schema = model.createSchema()
    .entities(["person", "company"])
    .classification(task: "sentiment", labels: ["positive", "negative"])

let result = model.extract(text: text, schema: schema)
```

### Long Documents

```swift
// Splits into overlapping word-chunks, remaps spans back to the original text, and merges.
let result = model.extractEntitiesLong(text: veryLongText, entityTypes: ["person", "company"])
```

## LoRA Adapters

GLiNER2Swift supports loading LoRA (Low-Rank Adaptation) adapters trained with the Python GLiNER2 framework. Adapters are merged into the base weights at load time, giving identical results to Python with zero runtime overhead.

### Loading an Adapter

```swift
// One-step: load base model + adapter together
let model = try await GLiNER2.fromPretrained(
    "fastino/gliner2-base-v1",
    adapterPath: "/path/to/adapter"
)

// Two-step: load base model first, then attach adapter
let model = try await GLiNER2.fromPretrained("fastino/gliner2-base-v1")
try model.loadAdapter(from: "/path/to/adapter")
```

### Adapter Directory Structure

The adapter directory must contain:
- `adapter_config.json` - LoRA configuration (rank, alpha, target modules)
- `adapter_weights.safetensors` - LoRA weight matrices

All parameters (rank, alpha, dropout, target modules) are read from `adapter_config.json` - any valid LoRA configuration is supported.

### How It Works

Instead of maintaining separate LoRA modules at runtime, weights are merged at load time:

```
W_merged = W_base + (lora_B @ lora_A) * (alpha / r)
```

This produces numerically identical results to Python's `model.load_adapter()` + `model.merge_lora()` pipeline.

## Architecture

GLiNER2Swift is a direct port of the Python GLiNER2 implementation, achieving numerical parity with the reference implementation:

- **Encoder**: DeBERTa v3 with disentangled attention
- **Span Marker**: MLP-based span representation
- **Count LSTM**: For predicting entity counts
- **Downscaled Transformer**: For schema embedding

## Performance

On Apple Silicon (M3 Pro, fp16):
- Model loading: ~0.4 seconds
- Inference: ~20ms for a single sentence, scaling with length

Opt-in int8 encoder quantization roughly halves steady-state memory (~415 MB → ~253 MB)
for a small accuracy trade-off; pass `quantization: .int8` to `fromPretrained`.

## Work in Progress

This is an active port of the [Python GLiNER2](https://github.com/fastino-ai/gliner2) implementation. Inference is at full prediction parity with the reference. The following are **not yet implemented**:

- **Training loop** - Fine-tuning and training from scratch are not yet supported
- **PyTorch `.bin` checkpoints** - Only safetensors weights are loadable (MLX cannot read pickle)
- **Additional GLiNER models** - Currently only `deberta-v3-base` is supported; other model variants are not yet available

Contributions and PRs are welcome!

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on submitting PRs, branch naming conventions, and parity testing requirements.

## Acknowledgments

- [GLiNER2](https://github.com/fastino-ai/gliner2) - Original Python implementation
- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
