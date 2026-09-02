#!/usr/bin/env python3
"""
Generate end-to-end inference fixtures using LOCAL raw PyTorch weights.

These fixtures test that Swift can load raw HuggingFace model.safetensors
directly (via Extractor.sanitize) and produce identical results to Python.

Usage:
    cd GLiNER2Swift/scripts
    python generate_raw_weights_fixtures.py

Output:
    ../Tests/GLiNER2SwiftTests/Fixtures/raw_weights/*.json
"""

import os
import sys
import json

# Add parent directory to path for gliner2 imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from gliner2 import GLiNER2


def save_json(name: str, data, output_dir: str):
    """Save data as JSON."""
    path = os.path.join(output_dir, f"{name}.json")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {name}.json")


def generate_entity_fixtures(model: GLiNER2, output_dir: str):
    """Generate entity extraction fixtures."""
    print("\n=== Entity Extraction ===")

    cases = [
        {
            "name": "entity_basic",
            "text": "Tim Cook is CEO of Apple.",
            "entity_types": ["person", "company"],
        },
        {
            "name": "entity_multi",
            "text": "John and Jane work at Google in Mountain View.",
            "entity_types": ["person", "organization", "location"],
        },
        {
            "name": "entity_no_match",
            "text": "The weather is nice today.",
            "entity_types": ["person", "company"],
        },
    ]

    for case in cases:
        print(f"\n  {case['name']}: \"{case['text']}\"")

        result = model.extract_entities(
            case["text"],
            case["entity_types"],
            threshold=0.5,
            include_confidence=True,
            include_spans=True
        )

        print(f"    Result: {json.dumps(result, indent=2)}")

        save_json(f"{case['name']}_metadata", {
            "text": case["text"],
            "entity_types": case["entity_types"],
            "threshold": 0.5,
        }, output_dir)

        save_json(f"{case['name']}_result", result, output_dir)


def generate_classification_fixtures(model: GLiNER2, output_dir: str):
    """Generate classification fixtures."""
    print("\n=== Classification ===")

    cases = [
        {
            "name": "classify_sentiment_positive",
            "text": "Great product! I love it.",
            "task": "sentiment",
            "labels": ["positive", "negative", "neutral"],
            "multi_label": False,
        },
        {
            "name": "classify_sentiment_negative",
            "text": "Terrible service. Very disappointed.",
            "task": "sentiment",
            "labels": ["positive", "negative", "neutral"],
            "multi_label": False,
        },
    ]

    for case in cases:
        print(f"\n  {case['name']}: \"{case['text']}\"")

        tasks = {case["task"]: {
            "labels": case["labels"],
            "multi_label": case["multi_label"],
        }}

        result = model.classify_text(
            case["text"],
            tasks,
            threshold=0.5,
            include_confidence=True,
        )

        print(f"    Result: {json.dumps(result, indent=2)}")

        save_json(f"{case['name']}_metadata", {
            "text": case["text"],
            "task": case["task"],
            "labels": case["labels"],
            "multi_label": case["multi_label"],
            "threshold": 0.5,
        }, output_dir)

        save_json(f"{case['name']}_result", result, output_dir)


def generate_structure_fixtures(model: GLiNER2, output_dir: str):
    """Generate structure extraction fixtures."""
    print("\n=== Structure Extraction ===")

    cases = [
        {
            "name": "struct_person",
            "text": "John Smith is 35 years old and lives in New York.",
            "structure_name": "person_info",
            "fields": ["name", "age", "location"],
        },
        {
            "name": "struct_product",
            "text": "iPhone 15 Pro costs $999 and is made by Apple.",
            "structure_name": "product",
            "fields": ["name", "price", "manufacturer"],
        },
    ]

    for case in cases:
        print(f"\n  {case['name']}: \"{case['text']}\"")

        structures = {case["structure_name"]: case["fields"]}

        result = model.extract_json(
            case["text"],
            structures,
            threshold=0.5,
            include_confidence=True,
            include_spans=True,
        )

        print(f"    Result: {json.dumps(result, indent=2)}")

        save_json(f"{case['name']}_metadata", {
            "text": case["text"],
            "structure_name": case["structure_name"],
            "fields": case["fields"],
            "threshold": 0.5,
        }, output_dir)

        save_json(f"{case['name']}_result", result, output_dir)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '../Tests/GLiNER2SwiftTests/Fixtures/raw_weights')
    os.makedirs(output_dir, exist_ok=True)

    # Use the HuggingFace hub model (has raw PyTorch keys)
    model_id = "fastino/gliner2-base-v1"

    print(f"Model: {model_id}")
    print(f"Output dir: {output_dir}")

    # Verify raw weights
    from huggingface_hub import hf_hub_download
    raw_path = hf_hub_download(model_id, 'model.safetensors')
    from safetensors import safe_open
    with safe_open(raw_path, framework='pt', device='cpu') as f:
        raw_keys = [k for k in f.keys() if k.startswith('span_rep.')]
        print(f"Raw weights at: {raw_path}")
        print(f"Confirmed raw PyTorch format ({len(raw_keys)} span_rep.* keys)")

    print("\nLoading model from HuggingFace hub...")
    model = GLiNER2.from_pretrained(model_id)
    model.eval()
    print("Model loaded.")

    generate_entity_fixtures(model, output_dir)
    generate_classification_fixtures(model, output_dir)
    generate_structure_fixtures(model, output_dir)

    # Save summary with model path for test reference
    save_json("test_config", {
        "model_path": "gliner2-base-v1",
        "weights_format": "raw_pytorch",
        "description": "Fixtures generated from raw PyTorch model.safetensors",
    }, output_dir)

    print("\n=== Done! ===")
    print(f"Fixtures saved to: {output_dir}")


if __name__ == "__main__":
    main()
