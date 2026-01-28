# GLiNER2Swift

Swift/MLX implementation of [GLiNER2](https://github.com/fastino-ai/gliner2) - a unified schema-based information extraction framework.

## Features

- Named Entity Recognition (NER)
- Text Classification
- Structured Data Extraction
- Relation Extraction
- Native Apple Silicon support via MLX
- CPU-first design - no GPU required

## Requirements

- macOS 14.0+
- Swift 5.9+
- Apple Silicon (M1/M2/M3)

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/anthropics/GLiNER2Swift", from: "1.0.0"),
]
```

Or in Xcode: File → Add Package Dependencies → Enter the repository URL.

## Quick Start

```swift
import GLiNER2Swift

// Load model (downloads automatically from HuggingFace)
let model = try await GLiNER2.fromPretrained("fastino/gliner2-base-v1")

// Extract entities
let text = "Tim Cook is CEO of Apple in Cupertino."
let entities = try model.extractEntities(
    from: text,
    labels: ["person", "company", "location"]
)

for entity in entities {
    print("\(entity.label): \(entity.text) [\(entity.start)-\(entity.end)]")
}
// Output:
// person: Tim Cook [0-8]
// company: Apple [23-28]
// location: Cupertino [32-41]
```

## Available Models

| Model | Parameters | HuggingFace ID |
|-------|------------|----------------|
| Base | 205M | `fastino/gliner2-base-v1` |
| Large | 340M | `fastino/gliner2-large-v1` |

## API Reference

### Entity Extraction

```swift
let entities = try model.extractEntities(
    from: "Your text here",
    labels: ["person", "organization", "location"]
)
```

### Text Classification

```swift
let classification = try model.classifyText(
    "Great product, highly recommend!",
    labels: ["positive", "negative", "neutral"]
)
```

### Structured Extraction

```swift
let schema = model.createSchema()
    .entities(["person", "company"])
    .classification(task: "sentiment", labels: ["positive", "negative"])

let result = try model.extract(from: text, schema: schema)
```

## Architecture

GLiNER2Swift is a direct port of the Python GLiNER2 implementation, achieving numerical parity with the reference implementation:

- **Encoder**: DeBERTa v3 with disentangled attention
- **Span Marker**: MLP-based span representation
- **Count LSTM**: For predicting entity counts
- **Downscaled Transformer**: For schema embedding

## Performance

On Apple Silicon (M1/M2/M3):
- Model loading: ~2 seconds
- Inference: ~50ms per sentence (varies by length)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [GLiNER2](https://github.com/fastino-ai/gliner2) - Original Python implementation
- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [swift-transformers](https://github.com/huggingface/swift-transformers) - Tokenizer support