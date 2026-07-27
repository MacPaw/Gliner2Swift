#!/usr/bin/env python3
"""Generate component-level fixtures for Swift parity testing.

This script generates intermediate values from the Python GLiNER2 model
at each component level for comparison with the Swift implementation.

Usage:
    python generate_component_fixtures.py

Output:
    ../Tests/GLiNER2SwiftTests/Fixtures/*.npz
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gliner2 import GLiNER2


OUTPUT_DIR = Path(__file__).parent.parent / "Tests" / "GLiNER2SwiftTests" / "Fixtures"


def save_fixture(name: str, data: dict):
    """Save fixture as .npz file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np_data = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            np_data[k] = v.detach().cpu().numpy()
        elif isinstance(v, np.ndarray):
            np_data[k] = v
        else:
            np_data[k] = np.array(v)
    np.savez(OUTPUT_DIR / f"{name}.npz", **np_data)
    print(f"  Saved: {name}.npz")


def generate_embedding_fixtures(model):
    """Test 1-2: Word embeddings and LayerNorm."""
    print("\n1. Generating embedding fixtures...")

    # Fixed input for reproducibility (3 tokens)
    input_ids = torch.tensor([[287, 128003, 6967]])

    with torch.no_grad():
        # Get word embeddings (before LayerNorm)
        word_emb = model.encoder.embeddings.word_embeddings(input_ids)

        # Get after LayerNorm
        emb_ln = model.encoder.embeddings.LayerNorm(word_emb)

        # Get final embeddings output (with dropout disabled)
        model.encoder.embeddings.eval()
        full_emb = model.encoder.embeddings(input_ids)

    save_fixture("embeddings", {
        "input_ids": input_ids,
        "word_embeddings": word_emb,
        "after_layernorm": emb_ln,
        "full_embeddings": full_emb,
    })

    print(f"    word_emb L1: {word_emb.abs().sum():.4f}")
    print(f"    after_ln L1: {emb_ln.abs().sum():.4f}")
    print(f"    full_emb L1: {full_emb.abs().sum():.4f}")


def generate_position_bucket_fixtures():
    """Test 3: Position bucket computation."""
    print("\n2. Generating position bucket fixtures...")

    # Import the function from DeBERTa
    from transformers.models.deberta_v2.modeling_deberta_v2 import make_log_bucket_position

    for seq_len in [5, 10, 18]:
        buckets = make_log_bucket_position(
            relative_pos=torch.arange(-seq_len+1, seq_len).unsqueeze(0),
            bucket_size=256,
            max_position=512
        )

        # Build full position matrix
        pos_matrix = torch.zeros(seq_len, seq_len, dtype=torch.int32)
        for i in range(seq_len):
            for j in range(seq_len):
                rel_pos = i - j
                # Index into buckets: offset by (seq_len - 1)
                pos_matrix[i, j] = buckets[0, rel_pos + seq_len - 1]

        save_fixture(f"position_buckets_seq{seq_len}", {
            "seq_len": np.array(seq_len),
            "position_matrix": pos_matrix,
        })

        print(f"    seq_len={seq_len}: buckets range [{pos_matrix.min()}, {pos_matrix.max()}]")


def generate_attention_fixtures(model):
    """Test 4-7: Attention components."""
    print("\n3. Generating attention fixtures...")

    # Small sequence for debugging
    seq_len = 5
    batch_size = 1

    # Get embeddings first
    input_ids = torch.tensor([[287, 128003, 6967, 5365, 260]])  # 5 tokens
    model.encoder.embeddings.eval()

    with torch.no_grad():
        hidden = model.encoder.embeddings(input_ids)

        # Get layer 0 attention
        layer = model.encoder.encoder.layer[0]
        attention = layer.attention.self

        # Get Q, K, V projections
        query = attention.query_proj(hidden)
        key = attention.key_proj(hidden)
        value = attention.value_proj(hidden)

        # Get relative embeddings (normalized)
        rel_emb_raw = model.encoder.encoder.rel_embeddings.weight
        rel_emb = model.encoder.encoder.LayerNorm(rel_emb_raw)

        # c2c: content-to-content attention
        # Reshape for multi-head: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        num_heads = attention.num_attention_heads
        head_dim = hidden.size(-1) // num_heads

        q_heads = query.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        k_heads = key.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        v_heads = value.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

        # c2c scores
        c2c = torch.matmul(q_heads, k_heads.transpose(-1, -2))

    save_fixture("attention_layer0", {
        "input_ids": input_ids,
        "input_hidden": hidden,
        "query": query,
        "key": key,
        "value": value,
        "query_heads": q_heads,
        "key_heads": k_heads,
        "value_heads": v_heads,
        "c2c_scores": c2c,
        "rel_embeddings": rel_emb,
        "rel_embeddings_raw": rel_emb_raw,
    })

    print(f"    hidden L1: {hidden.abs().sum():.4f}")
    print(f"    query L1: {query.abs().sum():.4f}")
    print(f"    c2c L1: {c2c.abs().sum():.4f}")


def generate_encoder_layer_fixtures(model):
    """Test 8: Single encoder layer."""
    print("\n4. Generating encoder layer fixtures...")

    # Use the standard test input
    input_ids = torch.tensor([[287, 128003, 6967, 287, 128005, 483, 128005, 604, 1263, 1263, 128002, 41718, 3712, 269, 101312, 265, 6038, 323]])
    seq_len = input_ids.shape[1]

    # Create attention mask (all ones = all valid)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)

    model.encoder.eval()

    with torch.no_grad():
        # Get embeddings
        emb = model.encoder.embeddings(input_ids)

        # Get normalized relative embeddings
        rel_emb = model.encoder.encoder.LayerNorm(model.encoder.encoder.rel_embeddings.weight)

        # Run each layer and capture intermediate outputs
        layer_outputs = [emb]

        hidden = emb
        for i, layer in enumerate(model.encoder.encoder.layer):
            hidden = layer(hidden, attention_mask=attention_mask, rel_embeddings=rel_emb)[0]
            layer_outputs.append(hidden)

    # Save individual layer outputs
    save_fixture("encoder_layers", {
        "input_ids": input_ids,
        "embeddings": emb,
        "rel_embeddings": rel_emb,
        "layer0_output": layer_outputs[1],
        "layer1_output": layer_outputs[2],
        "layer11_output": layer_outputs[12],
        "final_output": layer_outputs[-1],
    })

    print(f"    embeddings L1: {emb.abs().sum():.4f}")
    print(f"    layer0 L1: {layer_outputs[1].abs().sum():.4f}")
    print(f"    layer1 L1: {layer_outputs[2].abs().sum():.4f}")
    print(f"    final L1: {layer_outputs[-1].abs().sum():.4f}")


def generate_full_encoder_fixtures(model):
    """Test 9: Full 12-layer encoder."""
    print("\n5. Generating full encoder fixtures...")

    input_ids = torch.tensor([[287, 128003, 6967, 287, 128005, 483, 128005, 604, 1263, 1263, 128002, 41718, 3712, 269, 101312, 265, 6038, 323]])

    model.encoder.eval()

    with torch.no_grad():
        output = model.encoder(input_ids)
        hidden = output.last_hidden_state

    save_fixture("full_encoder", {
        "input_ids": input_ids,
        "hidden_states": hidden,
    })

    # Per-position L1 sums
    print("    Per-position L1:")
    for pos in [0, 1, 10, 11, 17]:
        print(f"      Position {pos}: {hidden[0, pos].abs().sum():.4f}")

    # First 10 values at position 0
    print(f"    Position 0 first 10: {hidden[0, 0, :10].tolist()}")


def generate_full_model_fixtures(model):
    """Test 10+: Full model through SpanMarker."""
    print("\n6. Generating full model fixtures...")

    # Use model's extract method to get a realistic input
    text = "Tim Cook is CEO of Apple."
    schema = {"entities": {"person": [], "company": []}}

    # Get model output
    model.eval()

    with torch.no_grad():
        # Run preprocessing
        from gliner2.processor import SchemaTransformer
        processor = SchemaTransformer(model.tokenizer)
        batch = processor.collate_fn([processor.transform(text, schema)])
        batch = batch.to(model.device)

        # Run encoder
        encoder_output = model.encoder(batch.input_ids, attention_mask=batch.attention_mask)
        hidden = encoder_output.last_hidden_state

        # Run span marker
        span_output = model.model.span_marker(hidden)

    save_fixture("full_model", {
        "input_ids": batch.input_ids,
        "attention_mask": batch.attention_mask,
        "encoder_output": hidden,
        "span_marker_output": span_output,
    })

    print(f"    encoder L1: {hidden.abs().sum():.4f}")
    print(f"    span_marker L1: {span_output.abs().sum():.4f}")


def main():
    print("Loading GLiNER2 model...")
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    model.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Generate all fixtures
    generate_embedding_fixtures(model)
    generate_position_bucket_fixtures()
    generate_attention_fixtures(model)
    generate_encoder_layer_fixtures(model)
    generate_full_encoder_fixtures(model)

    try:
        generate_full_model_fixtures(model)
    except Exception as e:
        print(f"  Warning: Could not generate full model fixtures: {e}")

    print("\nDone! All fixtures saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
