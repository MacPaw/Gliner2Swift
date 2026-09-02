#!/usr/bin/env python3
"""
Generate LoRA parity test fixtures for GLiNER2Swift.

This script:
1. Loads the base GLiNER2 model
2. Loads a LoRA adapter via load_adapter()
3. Runs test scenarios (entity extraction, classification)
4. Saves results as JSON + encoder outputs as .npy
5. Also loads the pre-merged model and verifies identical results

Usage:
    cd /path/to/GLiNER2
    uv run python GLiNER2Swift/scripts/generate_lora_fixtures.py
"""

import json
import os
import sys
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gliner2 import GLiNER2

# Paths
BASE_MODEL_PATH = "gliner2-base-v1"
ADAPTER_PATH = "gliner2-base-v1/final"
FIXTURES_DIR = "GLiNER2Swift/Tests/GLiNER2SwiftTests/Fixtures/lora"

os.makedirs(FIXTURES_DIR, exist_ok=True)


def save_json(data, filename):
    path = os.path.join(FIXTURES_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved {path}")


def save_npy(arr, filename):
    path = os.path.join(FIXTURES_DIR, filename)
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    np.save(path, arr)
    print(f"  Saved {path} shape={arr.shape} dtype={arr.dtype}")


def main():
    print("=" * 70)
    print("Generating LoRA Parity Fixtures")
    print("=" * 70)

    # ===== Step 1: Load base model + adapter =====
    # GLiNER2 inherits from Extractor (it IS the model, no .model attribute)
    print("\n[1/5] Loading base model...")
    model = GLiNER2.from_pretrained(BASE_MODEL_PATH)
    model.eval()

    print("\n[2/5] Loading LoRA adapter...")
    model.load_adapter(ADAPTER_PATH)
    model.eval()

    # ===== Step 2: Entity extraction test =====
    print("\n[3/5] Running entity extraction test...")
    entity_text = "Tim Cook is CEO of Apple."
    entity_types = ["person", "company"]
    entity_result = model.extract_entities(
        entity_text, entity_types,
        include_confidence=True, include_spans=True
    )
    print(f"  Entity result: {json.dumps(entity_result, indent=2, default=str)}")

    save_json({
        "text": entity_text,
        "entity_types": entity_types,
        "result": entity_result
    }, "lora_entity_result.json")

    # ===== Step 3: Classification test =====
    print("\n[4/5] Running classification test...")
    cls_text = "I love this product! It's amazing."
    cls_result = model.classify_text(
        cls_text, {"sentiment": ["positive", "negative"]},
        include_confidence=True
    )
    print(f"  Classification result: {json.dumps(cls_result, indent=2, default=str)}")

    save_json({
        "text": cls_text,
        "task": "sentiment",
        "labels": ["positive", "negative"],
        "result": cls_result
    }, "lora_classify_result.json")

    # ===== Step 4: Capture encoder output for numerical parity =====
    print("\n[5/5] Capturing encoder output for numerical parity...")

    # Use the entity extraction text for encoder output comparison
    processor = model.processor
    normalized_text = entity_text  # Already ends with '.'

    schema_dict = {
        "json_structures": [],
        "classifications": [],
        "entities": {"person": "", "company": ""},
        "relations": [],
        "json_descriptions": {},
        "entity_descriptions": {},
    }
    record = processor.transform_and_format(normalized_text, schema_dict)
    batch = processor.collate_fn_inference([(normalized_text, record)])

    # Save input tensors
    input_ids = batch.input_ids
    attention_mask = batch.attention_mask
    save_npy(input_ids, "lora_input_ids.npy")
    save_npy(attention_mask, "lora_attention_mask.npy")

    # Run encoder
    with torch.no_grad():
        encoder_output = model.encoder(
            input_ids, attention_mask=attention_mask
        )
        hidden_states = encoder_output.last_hidden_state
    save_npy(hidden_states, "lora_encoder_output.npy")

    # ===== Step 5: Verify pre-merged model gives same results =====
    print("\n[VERIFY] Loading pre-merged model for comparison...")
    merged_model = GLiNER2.from_pretrained(ADAPTER_PATH)
    merged_model.eval()

    merged_entity_result = merged_model.extract_entities(
        entity_text, entity_types,
        include_confidence=True, include_spans=True
    )
    merged_cls_result = merged_model.classify_text(
        cls_text, {"sentiment": ["positive", "negative"]},
        include_confidence=True
    )

    # Compare entity results
    print(f"\n  Pre-merged entity result: {json.dumps(merged_entity_result, indent=2, default=str)}")
    print(f"\n  Pre-merged cls result: {json.dumps(merged_cls_result, indent=2, default=str)}")

    # Also capture pre-merged encoder output for comparison
    with torch.no_grad():
        merged_encoder_output = merged_model.encoder(
            input_ids, attention_mask=attention_mask
        )
        merged_hidden = merged_encoder_output.last_hidden_state

    diff = (hidden_states - merged_hidden).abs()
    print(f"\n  Encoder output diff: mean={diff.mean().item():.8f}, max={diff.max().item():.8f}")

    # Check entity match
    entity_match = True
    for etype in entity_types:
        adapter_entities = entity_result.get("entities", {}).get(etype, [])
        merged_entities = merged_entity_result.get("entities", {}).get(etype, [])
        if len(adapter_entities) != len(merged_entities):
            entity_match = False
            print(f"  MISMATCH: {etype} count differs: {len(adapter_entities)} vs {len(merged_entities)}")
        else:
            for a, m in zip(adapter_entities, merged_entities):
                if a.get("text") != m.get("text"):
                    entity_match = False
                    print(f"  MISMATCH: {etype} text: '{a.get('text')}' vs '{m.get('text')}'")

    if entity_match:
        print("  Entity results MATCH between adapter and pre-merged model")

    # Save metadata
    save_json({
        "base_model_path": BASE_MODEL_PATH,
        "adapter_path": ADAPTER_PATH,
        "adapter_config": {
            "lora_r": 16,
            "lora_alpha": 32.0,
            "lora_dropout": 0.05,
            "target_modules": ["classifier", "count_embed", "count_pred", "encoder", "span_rep"]
        },
        "entity_test": {
            "text": entity_text,
            "entity_types": entity_types,
        },
        "classify_test": {
            "text": cls_text,
            "task": "sentiment",
            "labels": ["positive", "negative"],
        },
        "encoder_diff_mean": float(diff.mean().item()),
        "encoder_diff_max": float(diff.max().item()),
    }, "lora_metadata.json")

    print("\n" + "=" * 70)
    print("Fixture generation complete!")
    print(f"Fixtures saved to: {FIXTURES_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
