#!/usr/bin/env python3
"""
Generate end-to-end inference test fixtures for Swift/MLX parity testing.

This script creates test fixtures for:
1. Entity extraction parity
2. Classification parity
3. Structure extraction parity

Usage:
    cd GLiNER2Swift/scripts
    python generate_inference_fixtures.py

Output:
    ../Tests/GLiNER2SwiftTests/Fixtures/inference/*.npy
    ../Tests/GLiNER2SwiftTests/Fixtures/inference/*.json
"""

import os
import re
import sys
import json
import platform
import numpy as np
import torch

# Add parent directory to path for gliner2 imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from gliner2 import GLiNER2
from gliner2.inference.engine import RegexValidator
from transformers import AutoTokenizer


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_fixture(name: str, data: np.ndarray, output_dir: str):
    """Save a numpy array as a fixture."""
    path = os.path.join(output_dir, f"{name}.npy")
    np.save(path, data)
    print(f"  Saved: {name}.npy {data.shape} {data.dtype}")


def save_json(name: str, data: dict, output_dir: str):
    """Save metadata as JSON."""
    path = os.path.join(output_dir, f"{name}.json")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {name}.json")


def generate_entity_extraction_fixtures(model: GLiNER2, output_dir: str):
    """Generate entity extraction fixtures."""
    print("\n=== Entity Extraction Fixtures ===")

    test_cases = [
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

    for case in test_cases:
        print(f"\n  Test: {case['name']}")
        name = case["name"]
        text = case["text"]
        entity_types = case["entity_types"]

        # Build schema
        schema = model.create_schema().entities(entity_types)

        # Get intermediate values by processing manually
        model.eval()
        model.processor.change_mode(is_training=False)

        # Normalize text
        if text and not text.endswith(('.', '!', '?')):
            text_normalized = text + "."
        else:
            text_normalized = text or "."

        # Process text
        schema_dict = schema.build()
        dataset = [(text_normalized, schema_dict)]

        from gliner2.training.trainer import ExtractorCollator
        collator = ExtractorCollator(model.processor, is_training=False)
        batch = collator(dataset)

        # Move to device
        device = next(model.parameters()).device
        batch = batch.to(device)

        # Get encoder output
        with torch.no_grad():
            encoder_output = model.encoder(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask
            ).last_hidden_state

        # Save fixtures
        save_fixture(f"{name}_input_ids", batch.input_ids.cpu().numpy().astype(np.int32), output_dir)
        save_fixture(f"{name}_attention_mask", batch.attention_mask.cpu().numpy().astype(np.int32), output_dir)
        save_fixture(f"{name}_encoder_output", encoder_output.cpu().numpy().astype(np.float32), output_dir)

        # Get extraction result
        result = model.extract_entities(
            case["text"],
            entity_types,
            threshold=0.5,
            include_confidence=True,
            include_spans=True
        )

        # Save result and metadata
        save_json(f"{name}_result", result, output_dir)
        save_json(f"{name}_metadata", {
            "text": case["text"],
            "text_normalized": text_normalized,
            "entity_types": entity_types,
            "schema": schema_dict,
        }, output_dir)


def generate_classification_fixtures(model: GLiNER2, output_dir: str):
    """Generate classification fixtures."""
    print("\n=== Classification Fixtures ===")

    test_cases = [
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
        {
            "name": "classify_topic_multi",
            "text": "Apple released a new iPhone with improved AI capabilities.",
            "task": "topics",
            "labels": ["technology", "business", "sports", "entertainment"],
            "multi_label": True,
        },
    ]

    for case in test_cases:
        print(f"\n  Test: {case['name']}")
        name = case["name"]
        text = case["text"]

        # Build schema
        schema = model.create_schema().classification(
            task=case["task"],
            labels=case["labels"],
            multi_label=case["multi_label"]
        )

        # Get intermediate values
        model.eval()
        model.processor.change_mode(is_training=False)

        # Normalize text
        if text and not text.endswith(('.', '!', '?')):
            text_normalized = text + "."
        else:
            text_normalized = text or "."

        # Process text
        schema_dict = schema.build()
        dataset = [(text_normalized, schema_dict)]

        from gliner2.training.trainer import ExtractorCollator
        collator = ExtractorCollator(model.processor, is_training=False)
        batch = collator(dataset)

        # Move to device
        device = next(model.parameters()).device
        batch = batch.to(device)

        # Get encoder output
        with torch.no_grad():
            encoder_output = model.encoder(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask
            ).last_hidden_state

            # Extract schema embeddings for classification
            all_token_embs, all_schema_embs = model.processor.extract_embeddings_from_batch(
                encoder_output,
                batch.input_ids,
                batch
            )

            # Get classification embeddings (skip [P] token)
            if all_schema_embs[0] and len(all_schema_embs[0]) > 0:
                schema_embs = all_schema_embs[0][0]  # First sample, first schema
                if len(schema_embs) > 1:
                    cls_embeds = torch.stack(schema_embs[1:])  # Skip [P]
                    # Run classifier
                    logits = model.classifier(cls_embeds).squeeze(-1)
                    save_fixture(f"{name}_cls_logits", logits.cpu().numpy().astype(np.float32), output_dir)

        # Save fixtures
        save_fixture(f"{name}_input_ids", batch.input_ids.cpu().numpy().astype(np.int32), output_dir)
        save_fixture(f"{name}_attention_mask", batch.attention_mask.cpu().numpy().astype(np.int32), output_dir)
        save_fixture(f"{name}_encoder_output", encoder_output.cpu().numpy().astype(np.float32), output_dir)

        # Get classification result
        tasks = {case["task"]: {"labels": case["labels"], "multi_label": case["multi_label"]}}
        result = model.classify_text(
            case["text"],
            tasks,
            threshold=0.5,
            include_confidence=True
        )

        # Save result and metadata
        save_json(f"{name}_result", result, output_dir)
        save_json(f"{name}_metadata", {
            "text": case["text"],
            "text_normalized": text_normalized,
            "task": case["task"],
            "labels": case["labels"],
            "multi_label": case["multi_label"],
            "schema": schema_dict,
        }, output_dir)


def generate_structure_extraction_fixtures(model: GLiNER2, output_dir: str):
    """Generate structure extraction fixtures."""
    print("\n=== Structure Extraction Fixtures ===")

    test_cases = [
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

    for case in test_cases:
        print(f"\n  Test: {case['name']}")
        name = case["name"]
        text = case["text"]

        # Build schema using the fluent API
        # StructureBuilder auto-finishes when accessing Schema methods via __getattr__
        builder = model.create_schema().structure(case["structure_name"])
        for field in case["fields"]:
            builder = builder.field(field)
        # Access .build() triggers __getattr__ which auto-finishes and returns Schema.build
        schema_dict = builder.build()

        # Get intermediate values
        model.eval()
        model.processor.change_mode(is_training=False)

        # Normalize text
        if text and not text.endswith(('.', '!', '?')):
            text_normalized = text + "."
        else:
            text_normalized = text or "."
        dataset = [(text_normalized, schema_dict)]

        from gliner2.training.trainer import ExtractorCollator
        collator = ExtractorCollator(model.processor, is_training=False)
        batch = collator(dataset)

        # Move to device
        device = next(model.parameters()).device
        batch = batch.to(device)

        # Get encoder output
        with torch.no_grad():
            encoder_output = model.encoder(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask
            ).last_hidden_state

        # Save fixtures
        save_fixture(f"{name}_input_ids", batch.input_ids.cpu().numpy().astype(np.int32), output_dir)
        save_fixture(f"{name}_attention_mask", batch.attention_mask.cpu().numpy().astype(np.int32), output_dir)
        save_fixture(f"{name}_encoder_output", encoder_output.cpu().numpy().astype(np.float32), output_dir)

        # Get extraction result
        structures = {case["structure_name"]: case["fields"]}
        result = model.extract_json(
            case["text"],
            structures,
            threshold=0.5,
            include_confidence=True,
            include_spans=True
        )

        # Save result and metadata
        save_json(f"{name}_result", result, output_dir)
        save_json(f"{name}_metadata", {
            "text": case["text"],
            "text_normalized": text_normalized,
            "structure_name": case["structure_name"],
            "fields": case["fields"],
            "schema": schema_dict,
        }, output_dir)


def generate_tokenizer_fixtures(output_dir: str):
    """Generate tokenizer parity fixtures.

    These fixtures verify that Swift's UnigramTokenizer produces the same
    token IDs as Python's HuggingFace tokenizer.
    """
    print("\n=== Tokenizer Parity Fixtures ===")

    # Load the tokenizer from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained("fastino/gliner2-base-v1")

    test_cases = [
        "Hello, world!",
        "Tim Cook is CEO of Apple.",
        "The quick brown fox.",
        "John and Jane work at Google in Mountain View.",
        "Great product! I love it.",
        "iPhone 15 Pro costs $999 and is made by Apple.",
        "John Smith is 35 years old.",
        "This is a test with punctuation: commas, periods. And questions?",
        "[P] person [E] name",  # Special tokens test
        "  leading and trailing spaces  ",  # Whitespace handling
    ]

    fixtures = []

    for text in test_cases:
        # Encode without special tokens (just the text tokens)
        ids = tokenizer.encode(text, add_special_tokens=False)
        tokens = tokenizer.tokenize(text)

        fixtures.append({
            "text": text,
            "token_ids": ids,
            "tokens": tokens,
        })

        print(f"  '{text[:40]}...' -> {len(ids)} tokens")

    # Save fixtures
    save_json("tokenizer_parity", {"test_cases": fixtures}, output_dir)

    # Also save expected special token IDs
    special_tokens = {
        "pad_token_id": tokenizer.pad_token_id,
        "cls_token_id": tokenizer.cls_token_id,
        "sep_token_id": tokenizer.sep_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "vocab_size": tokenizer.vocab_size,
    }
    save_json("tokenizer_special_tokens", special_tokens, output_dir)


def generate_validator_filter_fixtures(output_dir: str):
    """Generate model-independent RegexValidator filter parity fixtures.

    Each case runs a list of synthetic ExtractedSpan-shaped inputs through the
    Python RegexValidator filter (post-threshold AND-logic) and records the
    result. The Swift test reconstructs the same validator objects locally
    (validators are never serialized into the schema dict on either side —
    engine.py stores them out-of-band in Schema._field_metadata).
    """
    print("\n=== RegexValidator Filter Parity Fixtures ===")

    # Reusable synthetic spans. Offsets are illustrative — they pass through
    # the filter unchanged, we only care about the text-based filtering.
    spans_emails = [
        {"text": "alice@example.com", "confidence": 0.95, "charStart": 0,  "charEnd": 17},
        {"text": "bob.dylan@music.io", "confidence": 0.88, "charStart": 20, "charEnd": 38},
        {"text": "not an email",       "confidence": 0.70, "charStart": 40, "charEnd": 52},
        {"text": "also@invalid",       "confidence": 0.65, "charStart": 55, "charEnd": 67},
        {"text": "x@y.z",              "confidence": 0.55, "charStart": 70, "charEnd": 75},
    ]
    spans_phones = [
        {"text": "Call (555) 123-4567 today",     "confidence": 0.90, "charStart": 0,  "charEnd": 25},
        {"text": "phone 867-5309 here",           "confidence": 0.82, "charStart": 30, "charEnd": 49},
        {"text": "(800) 555-1212",                "confidence": 0.75, "charStart": 52, "charEnd": 66},
        {"text": "no number in sight",            "confidence": 0.60, "charStart": 70, "charEnd": 88},
    ]
    spans_identifiers = [
        {"text": "user123",  "confidence": 0.90, "charStart": 0,  "charEnd": 7},
        {"text": "test_abc", "confidence": 0.85, "charStart": 10, "charEnd": 18},
        {"text": "demo",     "confidence": 0.80, "charStart": 20, "charEnd": 24},
        {"text": "ab",       "confidence": 0.75, "charStart": 26, "charEnd": 28},
        {"text": "sample42", "confidence": 0.70, "charStart": 30, "charEnd": 38},
        {"text": "ALICE",    "confidence": 0.65, "charStart": 40, "charEnd": 45},
    ]

    test_cases = [
        {
            "name": "email_full_match",
            "spans": spans_emails,
            "validators": [{"pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$", "mode": "full", "exclude": False}],
        },
        {
            "name": "phone_partial_match",
            "spans": spans_phones,
            "validators": [{"pattern": r"\(\d{3}\)\s\d{3}-\d{4}", "mode": "partial", "exclude": False}],
        },
        {
            "name": "exclude_test_demo_sample",
            "spans": spans_identifiers,
            "validators": [{"pattern": r"^(test|demo|sample)", "mode": "partial", "exclude": True}],
        },
        {
            "name": "and_alphanumeric_and_length",
            "spans": spans_identifiers,
            "validators": [
                {"pattern": r"^[a-zA-Z0-9_]+$", "mode": "full",  "exclude": False},
                {"pattern": r"^.{3,20}$",        "mode": "full",  "exclude": False},
            ],
        },
        {
            "name": "rejects_all_list_dtype",
            "spans": spans_emails,
            "validators": [{"pattern": r"^\d{20}$", "mode": "full", "exclude": False}],
        },
        {
            "name": "rejects_all_str_dtype",
            "spans": spans_emails[:1],  # single-span case for dtype=str parity
            "validators": [{"pattern": r"^\d{20}$", "mode": "full", "exclude": False}],
        },
        {
            "name": "empty_validators_noop",
            "spans": spans_identifiers,
            "validators": [],
        },
    ]

    fixture = {
        "python_version": platform.python_version(),
        "re_flags_default": "re.IGNORECASE",
        "cases": [],
    }

    for case in test_cases:
        validators = [
            RegexValidator(pattern=v["pattern"], mode=v["mode"], exclude=v["exclude"])
            for v in case["validators"]
        ]
        if validators:
            filtered = [
                s for s in case["spans"]
                if all(v.validate(s["text"]) for v in validators)
            ]
        else:
            filtered = list(case["spans"])

        fixture["cases"].append({
            "name": case["name"],
            "validators": case["validators"],
            "spans": case["spans"],
            "filtered": filtered,
        })
        print(f"  {case['name']}: {len(case['spans'])} → {len(filtered)} spans")

    save_json("validator_filter_cases", fixture, output_dir)


def generate_regex_engine_matrix(output_dir: str):
    """Generate a pattern × text boolean matrix for Python re vs NSRegularExpression parity.

    Swift runs the same (pattern, text, mode) triples through NSRegularExpression
    and asserts identical boolean outcomes. Catches subtle engine divergences
    ($, \\b, \\w, \\d, \\s, case folding) early.

    The patterns are the intersection of what both Python re and ICU
    NSRegularExpression support — no variable-width lookbehind, no possessive
    quantifiers, no atomic groups (those are rejected at Swift init-time).
    """
    print("\n=== Regex Engine Parity Matrix ===")

    patterns = [
        # (pattern, mode)
        (r"^\d+$",                          "full"),
        (r"^[a-zA-Z0-9_]+$",                "full"),
        (r"^.{3,20}$",                      "full"),
        (r"[a-z]+",                         "partial"),
        (r"\bword\b",                       "partial"),
        (r"\d{3}-\d{4}",                    "partial"),
        (r"^[\w\.-]+@[\w\.-]+\.\w+$",       "full"),
        (r"\s+",                            "partial"),
        (r"(cat|dog|bird)",                 "partial"),
        (r"^(test|demo)",                   "partial"),
        (r"[A-Z][a-z]+",                    "partial"),
        (r"^\(\d{3}\)\s\d{3}-\d{4}$",       "full"),
        (r"\d{4}",                          "partial"),
        (r"^[^\d]+$",                       "full"),
        (r"apple",                          "partial"),
    ]

    texts = [
        "12345", "abc", "abc123", "hello world", "the word is here",
        "call 555-0199 now", "alice@example.com", "tab\there",
        "cat in hat", "testify", "demonstrate", "Sample Text",
        "(555) 867-5309", "year 2024", "only-letters", "APPLE pie",
        "", "a", "  spaces  ", "no digits here",
    ]

    results = []
    for pattern, mode in patterns:
        compiled = re.compile(pattern, re.IGNORECASE)
        row = []
        for text in texts:
            if mode == "full":
                matched = compiled.fullmatch(text) is not None
            else:
                matched = compiled.search(text) is not None
            row.append(matched)
        results.append(row)

    fixture = {
        "python_version": platform.python_version(),
        "patterns": [{"pattern": p, "mode": m} for p, m in patterns],
        "texts": texts,
        "results": results,
    }

    save_json("regex_engine_matrix", fixture, output_dir)
    print(f"  {len(patterns)} patterns × {len(texts)} texts")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '../Tests/GLiNER2SwiftTests/Fixtures/inference')
    ensure_dir(output_dir)

    print(f"Generating inference fixtures in: {output_dir}")

    # Generate tokenizer fixtures first (doesn't require model)
    generate_tokenizer_fixtures(output_dir)

    # Generate model-independent fixtures (no model load needed)
    generate_validator_filter_fixtures(output_dir)
    generate_regex_engine_matrix(output_dir)

    print("\nLoading GLiNER2 model...")

    # Load model
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    model.eval()

    print(f"Model loaded. Device: {next(model.parameters()).device}")

    # Generate fixtures
    generate_entity_extraction_fixtures(model, output_dir)
    generate_classification_fixtures(model, output_dir)
    generate_structure_extraction_fixtures(model, output_dir)

    print("\n=== Done! ===")
    print(f"Fixtures saved to: {output_dir}")
    print("\nTo run Swift tests:")
    print("  cd GLiNER2Swift && swift test --filter InferenceParityTests")
    print("  cd GLiNER2Swift && swift test --filter testTokenizerParityWithPython")


if __name__ == "__main__":
    main()
