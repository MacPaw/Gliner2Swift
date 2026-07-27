#!/usr/bin/env python3
"""Generate tokenizer fixtures for Swift parity testing.

This script generates reference tokenization results from the Python
HuggingFace tokenizer for comparison with the Swift implementation.

Usage:
    python generate_tokenizer_fixtures.py

Output:
    ../Tests/GLiNER2SwiftTests/Fixtures/tokenizer_fixtures.json
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path for gliner2 import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoTokenizer


def main():
    print("Loading tokenizer from fastino/gliner2-base-v1...")
    tokenizer = AutoTokenizer.from_pretrained("fastino/gliner2-base-v1")

    # Test cases with categories
    test_cases = [
        # Basic text
        ("Hello, world!", "basic_punctuation"),
        ("Tim Cook is CEO of Apple.", "basic_sentence"),
        ("Hello", "single_word"),

        # Edge cases
        ("", "empty_string"),
        ("   ", "whitespace_only"),
        ("   spaces   ", "whitespace_padding"),

        # Punctuation handling (should stay attached)
        ("don't", "apostrophe"),
        ("U.S.A.", "abbreviation"),
        ("test@email.com", "email"),
        ("https://example.com", "url"),
        ("Hello, world! How are you?", "multiple_punctuation"),
        ("It's a test.", "contraction_sentence"),

        # Unicode
        ("Café résumé naïve", "accents"),
        ("日本語テスト", "japanese"),
        ("Привет мир", "cyrillic"),
        ("Cześć świat", "polish"),
        ("中文测试", "chinese"),

        # GLiNER2 special tokens
        ("[P]", "p_token_only"),
        ("[E]", "e_token_only"),
        ("[P] entities ( [E] person [E] company )", "schema_tokens"),
        ("[SEP_STRUCT]", "sep_struct"),
        ("[SEP_TEXT]", "sep_text"),
        ("[CLS] Hello [SEP]", "cls_sep_tokens"),
        ("[P] entities ( [E] person ) [SEP_TEXT] Tim Cook is CEO", "full_schema_example"),

        # Numbers
        ("12345", "numbers"),
        ("$1,234.56", "currency"),
        ("2024-01-28", "date"),
        ("3.14159", "decimal"),
        ("1,000,000", "large_number"),

        # Long text
        ("The quick brown fox jumps over the lazy dog.", "pangram"),
        ("The quick brown fox jumps over the lazy dog. " * 10, "long_text"),

        # Mixed content
        ("John earned $50,000 in 2024.", "mixed_content"),
        ("The meeting is at 3:30 PM.", "time"),
        ("Email me at test@example.com or call 555-1234.", "contact_info"),

        # Entity extraction realistic examples
        ("Apple Inc. was founded by Steve Jobs.", "company_person"),
        ("New York City is in the United States.", "locations"),
        ("The Eiffel Tower is 330 meters tall.", "measurement"),
    ]

    results = {}

    print("\nGenerating tokenizer fixtures:")
    print("-" * 60)

    for text, name in test_cases:
        # Tokenize without special tokens (add_special_tokens=False)
        ids_no_special = tokenizer.encode(text, add_special_tokens=False)
        tokens_no_special = tokenizer.tokenize(text)

        # Tokenize with special tokens
        ids_with_special = tokenizer.encode(text, add_special_tokens=True)

        results[name] = {
            "text": text,
            "token_ids": ids_no_special,
            "token_ids_with_special": ids_with_special,
            "tokens": tokens_no_special,
            "num_tokens": len(ids_no_special),
        }

        # Print summary
        ids_preview = str(ids_no_special[:8])
        if len(ids_no_special) > 8:
            ids_preview = ids_preview[:-1] + ", ...]"
        print(f"  {name}: {ids_preview} ({len(ids_no_special)} tokens)")

    # Add special token ID reference
    results["_special_token_ids"] = {
        "pad_token_id": tokenizer.pad_token_id,
        "cls_token_id": tokenizer.cls_token_id,
        "sep_token_id": tokenizer.sep_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "mask_token_id": tokenizer.mask_token_id,
        "vocab_size": tokenizer.vocab_size,
    }

    # Check for GLiNER2 special tokens
    gliner2_special = {}
    for token_name in ["[P]", "[E]", "[C]", "[R]", "[L]", "[SEP_STRUCT]", "[SEP_TEXT]",
                       "[EXAMPLE]", "[OUTPUT]", "[DESCRIPTION]"]:
        token_id = tokenizer.convert_tokens_to_ids(token_name)
        if token_id != tokenizer.unk_token_id:
            gliner2_special[token_name] = token_id

    results["_gliner2_special_tokens"] = gliner2_special

    print("-" * 60)
    print("\nSpecial token IDs:")
    print(f"  PAD: {tokenizer.pad_token_id}")
    print(f"  CLS: {tokenizer.cls_token_id}")
    print(f"  SEP: {tokenizer.sep_token_id}")
    print(f"  UNK: {tokenizer.unk_token_id}")
    print(f"  MASK: {tokenizer.mask_token_id}")
    print(f"  Vocab size: {tokenizer.vocab_size}")

    print("\nGLiNER2 special tokens:")
    for token, id in gliner2_special.items():
        print(f"  {token}: {id}")

    # Output path
    output_dir = Path(__file__).parent.parent / "Tests" / "GLiNER2SwiftTests" / "Fixtures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tokenizer_fixtures.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved fixtures to: {output_path}")
    print(f"Total test cases: {len(test_cases)}")


if __name__ == "__main__":
    main()
