#!/usr/bin/env python3
"""Generate L1 sums for ALL weights to verify Swift loading.

This script generates reference L1 sums and shapes for all weights in the
GLiNER2 model for verification that Swift weight loading is correct.

Usage:
    python verify_all_weights.py

Output:
    ../Tests/GLiNER2SwiftTests/Fixtures/weight_reference.json
"""

import json
import math
from pathlib import Path
from safetensors import safe_open
from huggingface_hub import hf_hub_download


OUTPUT_DIR = Path(__file__).parent.parent / "Tests" / "GLiNER2SwiftTests" / "Fixtures"


def main():
    print("Downloading model weights...")
    weights_path = hf_hub_download("fastino/gliner2-base-v1", "model.safetensors")

    print(f"Loading weights from: {weights_path}")

    l1_sums = {}
    encoder_weights = {}
    other_weights = {}

    def finite(value):
        """JSON has no NaN/Infinity literals, and Foundation's JSONSerialization
        rejects Python's non-standard spelling outright -- which silently disables
        the Swift-side gate. Single-element tensors give std() == NaN, so map every
        non-finite statistic to null."""
        value = float(value)
        return value if math.isfinite(value) else None

    with safe_open(weights_path, framework="pt") as f:
        for key in sorted(f.keys()):
            tensor = f.get_tensor(key)
            info = {
                "shape": list(tensor.shape),
                "l1_sum": finite(tensor.abs().sum().item()),
                "mean": finite(tensor.mean().item()),
                "std": finite(tensor.std().item()),
                "min": finite(tensor.min().item()),
                "max": finite(tensor.max().item()),
            }
            l1_sums[key] = info

            if key.startswith("encoder."):
                encoder_weights[key] = info
            else:
                other_weights[key] = info

    # Save full reference
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "weight_reference.json"
    with open(output_path, "w") as f:
        json.dump(l1_sums, f, indent=2)
    print(f"\nSaved full reference to: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("ENCODER WEIGHTS SUMMARY")
    print("=" * 70)

    # Group by component
    components = {}
    for key, info in encoder_weights.items():
        # Extract component path
        parts = key.split(".")
        if len(parts) >= 4:
            component = ".".join(parts[:4])
        else:
            component = key
        if component not in components:
            components[component] = []
        components[component].append((key, info))

    for component in sorted(components.keys()):
        print(f"\n{component}:")
        for key, info in components[component]:
            short_key = key[len(component)+1:] if key.startswith(component + ".") else key
            print(f"  {short_key}: shape={info['shape']}, L1={info['l1_sum']:.4f}")

    # Critical weights to check
    print("\n" + "=" * 70)
    print("CRITICAL WEIGHTS FOR PARITY")
    print("=" * 70)

    critical = [
        "encoder.embeddings.word_embeddings.weight",
        "encoder.embeddings.LayerNorm.weight",
        "encoder.embeddings.LayerNorm.bias",
        "encoder.encoder.rel_embeddings.weight",
        "encoder.encoder.LayerNorm.weight",
        "encoder.encoder.LayerNorm.bias",
        "encoder.encoder.layer.0.attention.self.query_proj.weight",
        "encoder.encoder.layer.0.attention.self.key_proj.weight",
        "encoder.encoder.layer.0.attention.self.value_proj.weight",
        "encoder.encoder.layer.0.attention.output.dense.weight",
        "encoder.encoder.layer.0.attention.output.LayerNorm.weight",
        "encoder.encoder.layer.0.intermediate.dense.weight",
        "encoder.encoder.layer.0.output.dense.weight",
        "encoder.encoder.layer.0.output.LayerNorm.weight",
    ]

    for key in critical:
        if key in l1_sums:
            info = l1_sums[key]
            print(f"{key}")
            print(f"  shape: {info['shape']}")
            print(f"  L1: {info['l1_sum']:.6f}")
            print(f"  mean: {info['mean']:.6f}, std: {info['std']:.6f}")
        else:
            print(f"{key} NOT FOUND!")

    # Print total weight count
    print(f"\n" + "=" * 70)
    print(f"Total encoder weights: {len(encoder_weights)}")
    print(f"Total other weights: {len(other_weights)}")
    print(f"Total weights: {len(l1_sums)}")


if __name__ == "__main__":
    main()
