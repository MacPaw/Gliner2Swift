#!/usr/bin/env python3
"""
Generate parity test results for Swift vs Python comparison.

This script runs GLiNER2 on predefined test cases and saves the results
as JSON fixtures. The Swift tests will load these fixtures and compare
against their own inference results.

Usage:
    cd GLiNER2Swift/scripts
    python generate_parity_test_results.py

Output:
    ../Tests/GLiNER2SwiftTests/Fixtures/parity/*.json
"""

import os
import sys
import json

# Add parent directory to path for gliner2 imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from gliner2 import GLiNER2


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(path, data):
    """Save data as JSON with pretty printing."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {os.path.basename(path)}")


def generate_entity_extraction_fixtures(model, output_dir):
    """Generate entity extraction test fixtures."""
    print("\n=== Entity Extraction Fixtures ===")

    test_cases = [
        {
            "name": "entity_multiple_people",
            "text": "John, Mary, and Bob went to the store.",
            "entity_types": ["person"],
            "purpose": "Multiple same-type entities"
        },
        {
            "name": "entity_multi_type",
            "text": "Microsoft is based in Seattle, Washington.",
            "entity_types": ["organization", "location"],
            "purpose": "Multiple entity types"
        },
        {
            "name": "entity_no_match",
            "text": "The weather is nice today.",
            "entity_types": ["person"],
            "purpose": "No matches case"
        },
        {
            "name": "entity_tech_companies",
            "text": "Apple and Google announced a partnership.",
            "entity_types": ["organization"],
            "purpose": "Tech companies"
        },
        {
            "name": "entity_with_titles",
            "text": "Dr. Smith works at General Hospital.",
            "entity_types": ["person", "organization"],
            "purpose": "Entities with titles"
        },
    ]

    for tc in test_cases:
        print(f"\n  Processing: {tc['name']}")
        print(f"    Text: \"{tc['text']}\"")
        print(f"    Types: {tc['entity_types']}")

        # Run inference
        result = model.extract_entities(
            tc["text"],
            tc["entity_types"],
            threshold=0.5,
            include_confidence=True,
            include_spans=True
        )

        # Save metadata
        metadata = {
            "text": tc["text"],
            "entity_types": tc["entity_types"],
            "purpose": tc["purpose"],
            "threshold": 0.5
        }

        # Save files
        save_json(os.path.join(output_dir, f"{tc['name']}_metadata.json"), metadata)
        save_json(os.path.join(output_dir, f"{tc['name']}_result.json"), result)

        # Print result summary
        total_entities = sum(len(v) for v in result.get("entities", {}).values())
        print(f"    Found: {total_entities} entities")
        for et, entities in result.get("entities", {}).items():
            if entities:
                texts = [e.get("text", e) if isinstance(e, dict) else e for e in entities]
                print(f"      {et}: {texts}")


def generate_classification_fixtures(model, output_dir):
    """Generate classification test fixtures."""
    print("\n=== Classification Fixtures ===")

    test_cases = [
        {
            "name": "classify_sentiment_negative",
            "text": "This product is terrible. I hate it.",
            "task": "sentiment",
            "labels": ["positive", "negative", "neutral"],
            "purpose": "Negative sentiment"
        },
        {
            "name": "classify_topic_sports",
            "text": "The Lakers won the championship last night.",
            "task": "topic",
            "labels": ["sports", "technology", "politics"],
            "purpose": "Sports topic"
        },
        {
            "name": "classify_sentiment_neutral",
            "text": "Just got my package delivered.",
            "task": "sentiment",
            "labels": ["positive", "negative", "neutral"],
            "purpose": "Neutral sentiment"
        },
        {
            "name": "classify_topic_tech",
            "text": "New AI model breaks records!",
            "task": "topic",
            "labels": ["sports", "technology", "politics"],
            "purpose": "Technology topic"
        },
    ]

    for tc in test_cases:
        print(f"\n  Processing: {tc['name']}")
        print(f"    Text: \"{tc['text']}\"")
        print(f"    Task: {tc['task']}, Labels: {tc['labels']}")

        # Run inference - classify_text expects a tasks dict
        tasks = {tc["task"]: tc["labels"]}
        result = model.classify_text(
            tc["text"],
            tasks,
            threshold=0.5,
            include_confidence=True
        )

        # Save metadata
        metadata = {
            "text": tc["text"],
            "task": tc["task"],
            "labels": tc["labels"],
            "purpose": tc["purpose"],
            "threshold": 0.5
        }

        # Save files
        save_json(os.path.join(output_dir, f"{tc['name']}_metadata.json"), metadata)
        save_json(os.path.join(output_dir, f"{tc['name']}_result.json"), result)

        # Print result summary
        task_result = result.get(tc["task"], {})
        if isinstance(task_result, dict):
            label = task_result.get("label", "N/A")
            conf = task_result.get("confidence", "N/A")
            print(f"    Predicted: {label} (confidence: {conf})")
        else:
            print(f"    Predicted: {task_result}")


def generate_structure_extraction_fixtures(model, output_dir):
    """Generate structure extraction test fixtures."""
    print("\n=== Structure Extraction Fixtures ===")

    test_cases = [
        {
            "name": "struct_product",
            "text": "iPhone 15 costs $999 and is made by Apple.",
            "structure_name": "product",
            "fields": ["name", "price", "manufacturer"],
            "purpose": "Product extraction"
        },
        {
            "name": "struct_event",
            "text": "The conference is March 15 in San Francisco.",
            "structure_name": "event",
            "fields": ["name", "date", "location"],
            "purpose": "Event extraction"
        },
        {
            "name": "struct_contact",
            "text": "Contact john@email.com or call 555-1234.",
            "structure_name": "contact",
            "fields": ["email", "phone"],
            "purpose": "Contact info extraction"
        },
    ]

    for tc in test_cases:
        print(f"\n  Processing: {tc['name']}")
        print(f"    Text: \"{tc['text']}\"")
        print(f"    Structure: {tc['structure_name']}")

        # Build structures dict
        structures = {tc["structure_name"]: tc["fields"]}

        # Run inference using extract_json convenience method
        result = model.extract_json(
            tc["text"],
            structures,
            threshold=0.5,
            include_confidence=True,
            include_spans=True
        )

        # Save metadata
        metadata = {
            "text": tc["text"],
            "structure_name": tc["structure_name"],
            "fields": tc["fields"],
            "purpose": tc["purpose"],
            "threshold": 0.5
        }

        # Save files
        save_json(os.path.join(output_dir, f"{tc['name']}_metadata.json"), metadata)
        save_json(os.path.join(output_dir, f"{tc['name']}_result.json"), result)

        # Print result summary
        struct_results = result.get(tc["structure_name"], [])
        if isinstance(struct_results, list):
            print(f"    Found: {len(struct_results)} instances")
            for i, instance in enumerate(struct_results[:2]):  # Show first 2
                print(f"      Instance {i}: {instance}")
        else:
            print(f"    Result: {struct_results}")


def generate_summary(output_dir):
    """Generate a summary JSON of all test cases."""
    print("\n=== Generating Summary ===")

    summary = {
        "description": "GLiNER2 Python parity test results",
        "model": "fastino/gliner2-base-v1",
        "threshold": 0.5,
        "test_categories": {
            "entity_extraction": [
                "entity_multiple_people",
                "entity_multi_type",
                "entity_no_match",
                "entity_tech_companies",
                "entity_with_titles"
            ],
            "classification": [
                "classify_sentiment_negative",
                "classify_topic_sports",
                "classify_sentiment_neutral",
                "classify_topic_tech"
            ],
            "structure_extraction": [
                "struct_product",
                "struct_event",
                "struct_contact"
            ]
        }
    }

    save_json(os.path.join(output_dir, "parity_test_summary.json"), summary)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '../Tests/GLiNER2SwiftTests/Fixtures/parity')
    ensure_dir(output_dir)

    print(f"Output directory: {output_dir}")

    # Load model
    print("\n=== Loading GLiNER2 Model ===")
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    print("  Model loaded successfully")

    # Generate fixtures for each category
    generate_entity_extraction_fixtures(model, output_dir)
    generate_classification_fixtures(model, output_dir)
    generate_structure_extraction_fixtures(model, output_dir)
    generate_summary(output_dir)

    print("\n=== Done! ===")
    print(f"Parity fixtures saved to: {output_dir}")
    print("\nTo run Swift parity tests:")
    print("  cd /Users/tmwstw/Documents/mnemos/GLiNER2/GLiNER2Swift")
    print("  swift test --filter RealWeightsTests")


if __name__ == "__main__":
    main()
