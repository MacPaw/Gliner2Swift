#!/usr/bin/env python3
"""Generate layer 0 fixtures for Swift parity testing.

This generates intermediate values from Python's first encoder layer
so we can verify Swift matches exactly.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gliner2 import GLiNER2
from safetensors.numpy import save_file

OUTPUT_DIR = Path(__file__).parent.parent / "Tests" / "GLiNER2SwiftTests" / "Fixtures"


def main():
    print("Loading GLiNER2 model...")
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    model.eval()

    encoder = model.encoder

    print(f"\n=== GENERATING LAYER 0 FIXTURES ===")

    # Use the same input as diagnostic test
    input_ids = torch.tensor([[287, 128003, 6967, 287, 128005, 483, 128005, 604, 1263, 1263, 128002, 41718, 3712, 269, 101312, 265, 6038, 323]])

    with torch.no_grad():
        # Get embeddings
        embeddings = encoder.embeddings(input_ids)
        print(f"Embeddings L1: {embeddings.abs().sum().item():.4f}")

        # Get normalized rel_embeddings
        rel_emb = encoder.encoder.rel_embeddings.weight
        rel_emb_normalized = encoder.encoder.LayerNorm(rel_emb)
        print(f"Normalized rel_embeddings L1: {rel_emb_normalized.abs().sum().item():.4f}")

        # Run layer 0
        layer0 = encoder.encoder.layer[0]

        # Hook to capture intermediate values
        attention_output = None
        attention_probs = None

        def attention_hook(module, input, output):
            nonlocal attention_output
            attention_output = output[0]  # output is (hidden_states, attention_probs) or just hidden_states

        layer0.attention.register_forward_hook(attention_hook)

        # Create attention mask (all ones = no masking)
        attention_mask = torch.ones_like(input_ids)
        # Expand to [batch, 1, 1, seq] for DeBERTa
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2).float()

        # Run layer 0
        layer0_output = layer0(embeddings, extended_mask, rel_embeddings=rel_emb_normalized)[0]

        print(f"\nLayer 0 attention output L1: {attention_output.abs().sum().item():.4f}")
        print(f"Layer 0 final output L1: {layer0_output.abs().sum().item():.4f}")

        # Also get attention self output (before output dense)
        attention = layer0.attention
        attention_self = attention.self

        # Manually compute to get intermediate values
        query = attention_self.query_proj(embeddings)
        key = attention_self.key_proj(embeddings)
        value = attention_self.value_proj(embeddings)

        print(f"\nLayer 0 query L1: {query.abs().sum().item():.4f}")
        print(f"Layer 0 key L1: {key.abs().sum().item():.4f}")
        print(f"Layer 0 value L1: {value.abs().sum().item():.4f}")

        # Full encoder output
        encoder_output = encoder(input_ids).last_hidden_state
        print(f"\nFull encoder output L1: {encoder_output.abs().sum().item():.4f}")

        # Per-position L1
        print(f"\nPer-position L1 (encoder output):")
        for pos in [0, 1, 11]:
            l1 = encoder_output[0, pos].abs().sum().item()
            print(f"  Position {pos}: {l1:.4f}")

    # Save fixtures
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tensors = {
        "input_ids": input_ids.numpy().astype(np.int32),
        "embeddings": embeddings.detach().numpy(),
        "rel_embeddings_normalized": rel_emb_normalized.detach().numpy(),
        "layer0_attention_output": attention_output.detach().numpy(),
        "layer0_output": layer0_output.detach().numpy(),
        "encoder_output": encoder_output.detach().numpy(),
        "query": query.detach().numpy(),
        "key": key.detach().numpy(),
        "value": value.detach().numpy(),
    }

    output_path = OUTPUT_DIR / "layer0_debug.safetensors"
    save_file(tensors, str(output_path))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
