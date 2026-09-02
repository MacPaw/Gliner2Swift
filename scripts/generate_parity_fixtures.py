#!/usr/bin/env python3
"""
Generate test fixtures for Swift/MLX parity testing.

This script creates numpy arrays that can be loaded in Swift tests
to verify numerical parity between Python and Swift implementations.

Usage:
    cd GLiNER2Swift/scripts
    python generate_parity_fixtures.py

Output:
    ../Tests/GLiNER2SwiftTests/Fixtures/*.npy
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path for gliner2 imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from gliner2.layers import CountLSTMv2, DownscaledTransformer, create_mlp

# Get DeBERTa position bucketing
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    make_log_bucket_position,
    build_relative_position
)


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


def generate_position_bucket_fixtures(output_dir: str):
    """Generate position bucketing fixtures."""
    print("\n=== Position Bucketing Fixtures ===")

    # CRITICAL: DeBERTa uses i-j direction (q_ids[:, None] - k_ids[None, :])
    # This means: rel_pos[i, j] = i - j
    # unsqueeze(1) creates column vector, unsqueeze(0) creates row vector
    # So: arange.unsqueeze(1) - arange.unsqueeze(0) gives i - j

    # Test case 1: Small sequence (5x5)
    seq_len = 5
    # FIXED: Changed from j-i (wrong) to i-j (correct)
    rel_pos = torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0)
    buckets = make_log_bucket_position(rel_pos, bucket_size=256, max_position=512)
    save_fixture("position_buckets_5x5", buckets.numpy().astype(np.int32), output_dir)

    # Test case 2: Medium sequence (32x32)
    seq_len = 32
    rel_pos = torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0)
    buckets = make_log_bucket_position(rel_pos, bucket_size=256, max_position=512)
    save_fixture("position_buckets_32x32", buckets.numpy().astype(np.int32), output_dir)

    # Test case 3: Edge case - single position
    seq_len = 1
    rel_pos = torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0)
    buckets = make_log_bucket_position(rel_pos, bucket_size=256, max_position=512)
    save_fixture("position_buckets_1x1", buckets.numpy().astype(np.int32), output_dir)

    # Save the relative positions too for debugging
    seq_len = 5
    rel_pos = torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0)
    save_fixture("relative_positions_5x5", rel_pos.numpy().astype(np.int32), output_dir)


def generate_gru_fixtures(output_dir: str):
    """Generate GRU forward pass fixtures."""
    print("\n=== GRU Fixtures ===")

    torch.manual_seed(42)

    hidden_size = 768
    seq_len = 3
    batch_size = 2

    # Create GRU
    gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)

    # Input: [seq_len, batch, hidden]
    input_tensor = torch.randn(seq_len, batch_size, hidden_size)
    h0 = torch.randn(1, batch_size, hidden_size)

    # Forward pass
    output, hn = gru(input_tensor, h0)

    # Save inputs
    save_fixture("gru_input", input_tensor.detach().numpy().astype(np.float32), output_dir)
    save_fixture("gru_h0", h0.detach().numpy().astype(np.float32), output_dir)

    # Save weights
    save_fixture("gru_weight_ih", gru.weight_ih_l0.detach().numpy().astype(np.float32), output_dir)
    save_fixture("gru_weight_hh", gru.weight_hh_l0.detach().numpy().astype(np.float32), output_dir)
    save_fixture("gru_bias_ih", gru.bias_ih_l0.detach().numpy().astype(np.float32), output_dir)
    save_fixture("gru_bias_hh", gru.bias_hh_l0.detach().numpy().astype(np.float32), output_dir)

    # Save outputs
    save_fixture("gru_output", output.detach().numpy().astype(np.float32), output_dir)
    save_fixture("gru_hn", hn.detach().numpy().astype(np.float32), output_dir)

    # Save metadata
    save_json("gru_metadata", {
        "hidden_size": hidden_size,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "input_shape": list(input_tensor.shape),
        "output_shape": list(output.shape),
    }, output_dir)


def generate_downscaled_transformer_fixtures(output_dir: str):
    """Generate DownscaledTransformer fixtures."""
    print("\n=== DownscaledTransformer Fixtures ===")

    torch.manual_seed(42)

    input_size = 768
    hidden_size = 128
    L = 3  # count
    M = 4  # fields

    # Create model
    model = DownscaledTransformer(
        input_size=input_size,
        hidden_size=hidden_size,
        num_heads=4,
        num_layers=2,
        dropout=0.0  # Disable dropout for deterministic testing
    )
    model.eval()

    # Input: [L, M, input_size]
    input_tensor = torch.randn(L, M, input_size)

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)

    # Save input/output
    save_fixture("dst_input", input_tensor.numpy().astype(np.float32), output_dir)
    save_fixture("dst_output", output.numpy().astype(np.float32), output_dir)

    # Save intermediate values for debugging
    with torch.no_grad():
        projected = model.in_projector(input_tensor)
        save_fixture("dst_after_in_proj", projected.numpy().astype(np.float32), output_dir)

        transformed = model.transformer(projected)
        save_fixture("dst_after_transformer", transformed.numpy().astype(np.float32), output_dir)

        concatenated = torch.cat([transformed, input_tensor], dim=-1)
        save_fixture("dst_after_concat", concatenated.numpy().astype(np.float32), output_dir)

    # Save weights
    save_fixture("dst_in_projector_weight", model.in_projector.weight.detach().numpy().astype(np.float32), output_dir)
    save_fixture("dst_in_projector_bias", model.in_projector.bias.detach().numpy().astype(np.float32), output_dir)

    # Transformer layer weights
    for i, layer in enumerate(model.transformer.layers):
        prefix = f"dst_transformer_layer_{i}"
        # Self-attention (combined in_proj for Q, K, V)
        save_fixture(f"{prefix}_self_attn_in_proj_weight", layer.self_attn.in_proj_weight.detach().numpy().astype(np.float32), output_dir)
        save_fixture(f"{prefix}_self_attn_in_proj_bias", layer.self_attn.in_proj_bias.detach().numpy().astype(np.float32), output_dir)
        save_fixture(f"{prefix}_self_attn_out_proj_weight", layer.self_attn.out_proj.weight.detach().numpy().astype(np.float32), output_dir)
        save_fixture(f"{prefix}_self_attn_out_proj_bias", layer.self_attn.out_proj.bias.detach().numpy().astype(np.float32), output_dir)
        # FFN
        save_fixture(f"{prefix}_linear1_weight", layer.linear1.weight.detach().numpy().astype(np.float32), output_dir)
        save_fixture(f"{prefix}_linear1_bias", layer.linear1.bias.detach().numpy().astype(np.float32), output_dir)
        save_fixture(f"{prefix}_linear2_weight", layer.linear2.weight.detach().numpy().astype(np.float32), output_dir)
        save_fixture(f"{prefix}_linear2_bias", layer.linear2.bias.detach().numpy().astype(np.float32), output_dir)
        # Layer norms
        save_fixture(f"{prefix}_norm1_weight", layer.norm1.weight.detach().numpy().astype(np.float32), output_dir)
        save_fixture(f"{prefix}_norm1_bias", layer.norm1.bias.detach().numpy().astype(np.float32), output_dir)
        save_fixture(f"{prefix}_norm2_weight", layer.norm2.weight.detach().numpy().astype(np.float32), output_dir)
        save_fixture(f"{prefix}_norm2_bias", layer.norm2.bias.detach().numpy().astype(np.float32), output_dir)

    # Out projector (3 linear layers at indices 0, 2, 4)
    save_fixture("dst_out_projector_0_weight", model.out_projector[0].weight.detach().numpy().astype(np.float32), output_dir)
    save_fixture("dst_out_projector_0_bias", model.out_projector[0].bias.detach().numpy().astype(np.float32), output_dir)
    save_fixture("dst_out_projector_2_weight", model.out_projector[2].weight.detach().numpy().astype(np.float32), output_dir)
    save_fixture("dst_out_projector_2_bias", model.out_projector[2].bias.detach().numpy().astype(np.float32), output_dir)
    save_fixture("dst_out_projector_4_weight", model.out_projector[4].weight.detach().numpy().astype(np.float32), output_dir)
    save_fixture("dst_out_projector_4_bias", model.out_projector[4].bias.detach().numpy().astype(np.float32), output_dir)

    save_json("dst_metadata", {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "L": L,
        "M": M,
        "input_shape": list(input_tensor.shape),
        "output_shape": list(output.shape),
    }, output_dir)


def generate_count_lstm_v2_fixtures(output_dir: str):
    """Generate CountLSTMv2 fixtures."""
    print("\n=== CountLSTMv2 Fixtures ===")

    torch.manual_seed(42)

    hidden_size = 768
    max_count = 20
    M = 4  # number of fields
    gold_count = 3

    # Create model
    model = CountLSTMv2(hidden_size=hidden_size, max_count=max_count)
    model.eval()

    # Disable dropout in transformer for deterministic testing
    model.transformer.transformer.layers[0].dropout.p = 0
    model.transformer.transformer.layers[1].dropout.p = 0

    # Input: pc_emb [M, hidden_size]
    pc_emb = torch.randn(M, hidden_size)

    # Forward pass
    with torch.no_grad():
        output = model(pc_emb, gold_count)

    # Save input/output
    save_fixture("clv2_pc_emb", pc_emb.numpy().astype(np.float32), output_dir)
    save_fixture("clv2_output", output.numpy().astype(np.float32), output_dir)

    # Save intermediate values
    with torch.no_grad():
        count_idx = torch.arange(gold_count)
        pos_seq = model.pos_embedding(count_idx)
        save_fixture("clv2_pos_seq", pos_seq.numpy().astype(np.float32), output_dir)

        pos_seq_expanded = pos_seq.unsqueeze(1).expand(-1, M, -1)
        save_fixture("clv2_pos_seq_expanded", pos_seq_expanded.numpy().astype(np.float32), output_dir)

        h0 = pc_emb.unsqueeze(0)
        gru_output, _ = model.gru(pos_seq_expanded, h0)
        save_fixture("clv2_gru_output", gru_output.numpy().astype(np.float32), output_dir)

        # Key: CountLSTMv2 uses ADDITION, not concatenation
        pc_broadcast = pc_emb.unsqueeze(0).expand_as(gru_output)
        added = gru_output + pc_broadcast
        save_fixture("clv2_after_addition", added.numpy().astype(np.float32), output_dir)

    # Save key weights
    save_fixture("clv2_pos_embedding", model.pos_embedding.weight.detach().numpy().astype(np.float32), output_dir)
    save_fixture("clv2_gru_weight_ih", model.gru.weight_ih_l0.detach().numpy().astype(np.float32), output_dir)
    save_fixture("clv2_gru_weight_hh", model.gru.weight_hh_l0.detach().numpy().astype(np.float32), output_dir)
    save_fixture("clv2_gru_bias_ih", model.gru.bias_ih_l0.detach().numpy().astype(np.float32), output_dir)
    save_fixture("clv2_gru_bias_hh", model.gru.bias_hh_l0.detach().numpy().astype(np.float32), output_dir)

    # Save DownscaledTransformer weights (inside CountLSTMv2)
    dst = model.transformer
    save_fixture("clv2_dst_in_projector_weight", dst.in_projector.weight.detach().numpy().astype(np.float32), output_dir)
    save_fixture("clv2_dst_in_projector_bias", dst.in_projector.bias.detach().numpy().astype(np.float32), output_dir)

    # Transformer layer weights
    for i, layer in enumerate(dst.transformer.layers):
        prefix = f"clv2_dst_transformer_layer_{i}"
        # Self-attention
        save_fixture(f"{prefix}_self_attn_in_proj_weight", layer.self_attn.in_proj_weight.detach().numpy().astype(np.float32), output_dir)
        save_fixture(f"{prefix}_self_attn_in_proj_bias", layer.self_attn.in_proj_bias.detach().numpy().astype(np.float32), output_dir)
        save_fixture(f"{prefix}_self_attn_out_proj_weight", layer.self_attn.out_proj.weight.detach().numpy().astype(np.float32), output_dir)
        save_fixture(f"{prefix}_self_attn_out_proj_bias", layer.self_attn.out_proj.bias.detach().numpy().astype(np.float32), output_dir)
        # FFN
        save_fixture(f"{prefix}_linear1_weight", layer.linear1.weight.detach().numpy().astype(np.float32), output_dir)
        save_fixture(f"{prefix}_linear1_bias", layer.linear1.bias.detach().numpy().astype(np.float32), output_dir)
        save_fixture(f"{prefix}_linear2_weight", layer.linear2.weight.detach().numpy().astype(np.float32), output_dir)
        save_fixture(f"{prefix}_linear2_bias", layer.linear2.bias.detach().numpy().astype(np.float32), output_dir)
        # Layer norms
        save_fixture(f"{prefix}_norm1_weight", layer.norm1.weight.detach().numpy().astype(np.float32), output_dir)
        save_fixture(f"{prefix}_norm1_bias", layer.norm1.bias.detach().numpy().astype(np.float32), output_dir)
        save_fixture(f"{prefix}_norm2_weight", layer.norm2.weight.detach().numpy().astype(np.float32), output_dir)
        save_fixture(f"{prefix}_norm2_bias", layer.norm2.bias.detach().numpy().astype(np.float32), output_dir)

    # Out projector
    save_fixture("clv2_dst_out_projector_0_weight", dst.out_projector[0].weight.detach().numpy().astype(np.float32), output_dir)
    save_fixture("clv2_dst_out_projector_0_bias", dst.out_projector[0].bias.detach().numpy().astype(np.float32), output_dir)
    save_fixture("clv2_dst_out_projector_2_weight", dst.out_projector[2].weight.detach().numpy().astype(np.float32), output_dir)
    save_fixture("clv2_dst_out_projector_2_bias", dst.out_projector[2].bias.detach().numpy().astype(np.float32), output_dir)
    save_fixture("clv2_dst_out_projector_4_weight", dst.out_projector[4].weight.detach().numpy().astype(np.float32), output_dir)
    save_fixture("clv2_dst_out_projector_4_bias", dst.out_projector[4].bias.detach().numpy().astype(np.float32), output_dir)

    save_json("clv2_metadata", {
        "hidden_size": hidden_size,
        "max_count": max_count,
        "M": M,
        "gold_count": gold_count,
        "output_shape": list(output.shape),
    }, output_dir)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '../Tests/GLiNER2SwiftTests/Fixtures')
    ensure_dir(output_dir)

    print(f"Generating fixtures in: {output_dir}")

    generate_position_bucket_fixtures(output_dir)
    generate_gru_fixtures(output_dir)
    generate_downscaled_transformer_fixtures(output_dir)
    generate_count_lstm_v2_fixtures(output_dir)

    print("\n=== Done! ===")
    print(f"Fixtures saved to: {output_dir}")
    print("\nTo use in Xcode:")
    print("1. Add the Fixtures folder to your test target")
    print("2. Load fixtures with: Bundle.module.url(forResource: 'name', withExtension: 'npy')")


if __name__ == "__main__":
    main()