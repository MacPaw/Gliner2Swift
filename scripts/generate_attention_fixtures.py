#!/usr/bin/env python3
"""Generate attention computation fixtures for Swift parity testing.

This generates intermediate values from Python's attention computation
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
    layer0 = encoder.encoder.layer[0]
    attention = layer0.attention.self

    print(f"\n=== GENERATING ATTENTION FIXTURES ===")

    # Use small sequence for debugging
    input_ids = torch.tensor([[287, 128003, 6967, 287, 128005]])  # 5 tokens
    seq_len = 5

    with torch.no_grad():
        # Get embeddings
        embeddings = encoder.embeddings(input_ids)
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Embeddings L1: {embeddings.abs().sum().item():.4f}")

        # Get Q, K projections
        query = attention.query_proj(embeddings)
        key = attention.key_proj(embeddings)

        print(f"Query L1: {query.abs().sum().item():.4f}")
        print(f"Key L1: {key.abs().sum().item():.4f}")

        # Reshape for heads
        num_heads = 12
        head_dim = 64
        q = query.view(1, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        k = key.view(1, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

        # Get rel_embeddings
        rel_emb = encoder.encoder.rel_embeddings.weight

        # pos_key
        pos_key = attention.key_proj(rel_emb)
        pos_key_heads = pos_key.view(-1, num_heads, head_dim).permute(1, 0, 2)

        # c2c
        c2c = torch.matmul(q, k.transpose(-1, -2))
        print(f"c2c L1: {c2c.abs().sum().item():.4f}")

        # c2p (full)
        c2p_full = torch.einsum('bhsd,hpd->bhsp', q, pos_key_heads)
        print(f"c2p_full L1: {c2p_full.abs().sum().item():.4f}")

        # c2p indices
        q_ids = torch.arange(seq_len)
        k_ids = torch.arange(seq_len)
        rel_pos = q_ids.unsqueeze(1) - k_ids.unsqueeze(0)
        att_span = 128
        c2p_pos = torch.clamp(rel_pos + att_span, 0, 255)
        c2p_pos_exp = c2p_pos.unsqueeze(0).unsqueeze(0).expand(1, num_heads, -1, -1)

        # c2p gathered
        c2p = torch.gather(c2p_full, dim=-1, index=c2p_pos_exp.long())
        print(f"c2p L1: {c2p.abs().sum().item():.4f}")

        # pos_query
        pos_query = attention.query_proj(rel_emb)
        pos_query_heads = pos_query.view(-1, num_heads, head_dim).permute(1, 0, 2)

        # p2c (full)
        p2c_full = torch.einsum('bhsd,hpd->bhsp', k, pos_query_heads)
        print(f"p2c_full L1: {p2c_full.abs().sum().item():.4f}")

        # p2c gathered (uses c2p_pos, then transpose)
        p2c_gathered = torch.gather(p2c_full, dim=-1, index=c2p_pos_exp.long())
        p2c = p2c_gathered.permute(0, 1, 3, 2)
        print(f"p2c L1: {p2c.abs().sum().item():.4f}")

    # Save fixtures
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tensors = {
        "input_ids": input_ids.numpy().astype(np.int32),
        "embeddings": embeddings.numpy(),
        "query": query.numpy(),
        "key": key.numpy(),
        "q_heads": q.numpy(),
        "k_heads": k.numpy(),
        "c2c": c2c.numpy(),
        "c2p_full": c2p_full.numpy(),
        "c2p_pos": c2p_pos.numpy().astype(np.int32),
        "c2p": c2p.numpy(),
        "p2c_full": p2c_full.numpy(),
        "p2c": p2c.numpy(),
        "rel_pos": rel_pos.numpy().astype(np.int32),
    }

    output_path = OUTPUT_DIR / "attention_layer0.safetensors"
    save_file(tensors, str(output_path))
    print(f"\nSaved to: {output_path}")

    # Print values for reference
    print("\n=== REFERENCE VALUES ===")
    print(f"c2p[0,0] (head 0):")
    print(c2p[0, 0].numpy())

    print(f"\np2c[0,0] (head 0):")
    print(p2c[0, 0].numpy())

    print(f"\nc2c[0,0] (head 0):")
    print(c2c[0, 0].numpy())


if __name__ == "__main__":
    main()
