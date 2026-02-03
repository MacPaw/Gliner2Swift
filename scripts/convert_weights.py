#!/usr/bin/env python3
"""
Convert GLiNER2 PyTorch SafeTensors weights to MLX-compatible format.

This script converts weights from the PyTorch model to a format that can be
loaded by the Swift/MLX implementation. It handles:
- DeBERTa encoder weights (with disentangled attention - NO separate pos_key/query_proj)
- SpanMarkerV0 span representation weights
- CountLSTMv2 weights (GRU + DownscaledTransformer)
- Classifier and count prediction MLPs

CRITICAL DeBERTa Notes:
- share_att_key=true: query_proj and key_proj are used for BOTH content and position
- NO pos_key_proj or pos_query_proj weights exist in the model
- rel_embeddings are shared across all encoder layers

Usage:
    # Default: single combined file (recommended)
    python convert_weights.py --model fastino/gliner2-base-v1 --output ../weights

    # Legacy: split files (for backwards compatibility)
    python convert_weights.py --model fastino/gliner2-base-v1 --output ../weights --split-files

Output (default - single file):
    - model.safetensors: All weights (encoder + GLiNER2 model) with Swift-compatible names
    - weight_mapping.json: Mapping from PyTorch names to Swift names

Output (--split-files):
    - gliner2_weights.safetensors: GLiNER2-specific weights (span_rep, classifier, etc.)
    - encoder_weights.safetensors: DeBERTa encoder weights (~400MB)
    - weight_mapping.json: Mapping from PyTorch names to Swift names
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def get_pytorch_weights(model_path: str) -> Dict[str, torch.Tensor]:
    """Load weights from HuggingFace model."""
    from huggingface_hub import hf_hub_download

    if os.path.isdir(model_path):
        weights_path = os.path.join(model_path, "model.safetensors")
    else:
        weights_path = hf_hub_download(model_path, "model.safetensors")

    weights = {}
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    return weights


def map_weight_name(pytorch_name: str) -> str:
    """
    Map PyTorch weight name to Swift/MLX naming convention.

    PyTorch uses snake_case, Swift uses camelCase.
    Also handles structural differences in module naming.
    """
    # Weight name mappings: PyTorch -> Swift
    mappings = {
        # Span representation
        "span_rep.span_rep_layer.project_start": "spanRep.spanRepLayer.projectStart",
        "span_rep.span_rep_layer.project_end": "spanRep.spanRepLayer.projectEnd",
        "span_rep.span_rep_layer.out_project": "spanRep.spanRepLayer.outProject",

        # Classifier MLP
        "classifier.0": "classifier.layers.0",
        "classifier.2": "classifier.layers.1",  # After ReLU

        # Count prediction MLP
        "count_pred.0": "countPred.layers.0",
        "count_pred.2": "countPred.layers.1",

        # CountLSTMv2 components
        "count_embed.pos_embedding": "countEmbed.posEmbedding",
        "count_embed.gru.weight_ih_l0": "countEmbed.gru.weightIH",
        "count_embed.gru.weight_hh_l0": "countEmbed.gru.weightHH",
        "count_embed.gru.bias_ih_l0": "countEmbed.gru.biasIH",
        "count_embed.gru.bias_hh_l0": "countEmbed.gru.biasHH",

        # DownscaledTransformer
        "count_embed.transformer.in_projector": "countEmbed.transformer.inProjector",
        "count_embed.transformer.out_projector": "countEmbed.transformer.outProjector",
        "count_embed.transformer.transformer.layers": "countEmbed.transformer.transformerLayers",
    }

    swift_name = pytorch_name

    # Apply mappings
    for pt_prefix, swift_prefix in mappings.items():
        if pytorch_name.startswith(pt_prefix):
            swift_name = pytorch_name.replace(pt_prefix, swift_prefix, 1)
            break

    return swift_name


def split_encoder_weights(
    weights: Dict[str, torch.Tensor]
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Split weights into encoder and non-encoder components.

    Returns:
        (encoder_weights, model_weights)
    """
    encoder_weights = {}
    model_weights = {}

    for name, tensor in weights.items():
        if name.startswith("encoder."):
            # Keep 'encoder.' prefix for consistent loading in Swift
            # Our Swift loader expects: encoder.embeddings.*, encoder.encoder.layer.*, etc.
            encoder_weights[name] = tensor
        else:
            model_weights[name] = tensor

    return encoder_weights, model_weights


def process_deberta_encoder_weights(
    encoder_weights: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Process DeBERTa encoder weights for Swift/MLX compatibility.

    DeBERTa v2/v3 weight structure:
    - encoder.embeddings.word_embeddings.weight: [vocab_size, hidden_size]
    - encoder.embeddings.LayerNorm.weight/bias: [hidden_size]
    - encoder.encoder.layer.{i}.attention.self.query_proj.weight/bias: [hidden, hidden]
    - encoder.encoder.layer.{i}.attention.self.key_proj.weight/bias: [hidden, hidden]
    - encoder.encoder.layer.{i}.attention.self.value_proj.weight/bias: [hidden, hidden]
    - encoder.encoder.layer.{i}.attention.output.dense.weight/bias: [hidden, hidden]
    - encoder.encoder.layer.{i}.attention.output.LayerNorm.weight/bias: [hidden]
    - encoder.encoder.layer.{i}.intermediate.dense.weight/bias: [intermediate, hidden]
    - encoder.encoder.layer.{i}.output.dense.weight/bias: [hidden, intermediate]
    - encoder.encoder.layer.{i}.output.LayerNorm.weight/bias: [hidden]
    - encoder.encoder.rel_embeddings.weight: [max_position, hidden_size]
    - encoder.encoder.LayerNorm.weight/bias: [hidden_size]

    CRITICAL: share_att_key=true means NO pos_key_proj or pos_query_proj exist.
    The query_proj and key_proj are reused for both content and position attention.
    """
    processed = {}

    for name, tensor in encoder_weights.items():
        # Weights are already in correct format for DeBERTa
        # Just pass through with validation
        processed[name] = tensor

    # Validate expected structure
    expected_prefixes = [
        "encoder.embeddings.word_embeddings",
        "encoder.embeddings.LayerNorm",
        "encoder.encoder.layer",
        "encoder.encoder.rel_embeddings",
        "encoder.encoder.LayerNorm",
    ]

    found_prefixes = set()
    for name in processed.keys():
        for prefix in expected_prefixes:
            if name.startswith(prefix):
                found_prefixes.add(prefix)
                break

    missing = set(expected_prefixes) - found_prefixes
    if missing:
        print(f"WARNING: Missing expected encoder weight prefixes: {missing}")

    # Verify NO pos_key_proj or pos_query_proj (would indicate wrong model config)
    for name in processed.keys():
        if "pos_key_proj" in name or "pos_query_proj" in name:
            print(f"WARNING: Found unexpected position projection weight: {name}")
            print("         This suggests share_att_key=false, which is not expected for gliner2-base-v1")

    return processed


def convert_transformer_layer_weights(
    weights: Dict[str, torch.Tensor],
    layer_prefix: str
) -> Dict[str, torch.Tensor]:
    """
    Convert PyTorch TransformerEncoderLayer weights to MLX format.

    PyTorch stores Q, K, V in a combined in_proj_weight tensor.
    We need to split them for MLX MultiHeadAttention.
    """
    converted = {}

    # Self-attention in_proj (combined Q, K, V)
    in_proj_key = f"{layer_prefix}.self_attn.in_proj_weight"
    if in_proj_key in weights:
        in_proj = weights[in_proj_key]
        hidden_size = in_proj.shape[0] // 3

        # Split into Q, K, V
        converted[f"{layer_prefix}.self_attn.q_proj.weight"] = in_proj[:hidden_size]
        converted[f"{layer_prefix}.self_attn.k_proj.weight"] = in_proj[hidden_size:2*hidden_size]
        converted[f"{layer_prefix}.self_attn.v_proj.weight"] = in_proj[2*hidden_size:]

    in_proj_bias_key = f"{layer_prefix}.self_attn.in_proj_bias"
    if in_proj_bias_key in weights:
        in_proj_bias = weights[in_proj_bias_key]
        hidden_size = in_proj_bias.shape[0] // 3

        converted[f"{layer_prefix}.self_attn.q_proj.bias"] = in_proj_bias[:hidden_size]
        converted[f"{layer_prefix}.self_attn.k_proj.bias"] = in_proj_bias[hidden_size:2*hidden_size]
        converted[f"{layer_prefix}.self_attn.v_proj.bias"] = in_proj_bias[2*hidden_size:]

    # Output projection (pass through)
    out_proj_key = f"{layer_prefix}.self_attn.out_proj.weight"
    if out_proj_key in weights:
        converted[f"{layer_prefix}.self_attn.out_proj.weight"] = weights[out_proj_key]
    out_proj_bias_key = f"{layer_prefix}.self_attn.out_proj.bias"
    if out_proj_bias_key in weights:
        converted[f"{layer_prefix}.self_attn.out_proj.bias"] = weights[out_proj_bias_key]

    # FFN layers (pass through with name mapping)
    for suffix in [".linear1.weight", ".linear1.bias", ".linear2.weight", ".linear2.bias",
                   ".norm1.weight", ".norm1.bias", ".norm2.weight", ".norm2.bias"]:
        key = layer_prefix + suffix
        if key in weights:
            converted[key] = weights[key]

    return converted


def process_gliner2_weights(
    model_weights: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Process and convert GLiNER2-specific weights.

    This handles:
    - Transformer layer splitting (Q, K, V from in_proj)
    - Name mapping to Swift conventions
    - Data type conversion if needed
    """
    converted = {}

    # Process DownscaledTransformer layers
    for layer_idx in [0, 1]:
        layer_prefix = f"count_embed.transformer.transformer.layers.{layer_idx}"
        layer_weights = convert_transformer_layer_weights(model_weights, layer_prefix)

        # Apply name mapping to converted layer weights
        for name, tensor in layer_weights.items():
            swift_name = map_weight_name(name)
            converted[swift_name] = tensor

        # Remove original combined weights
        for key in list(model_weights.keys()):
            if key.startswith(layer_prefix):
                del model_weights[key]

    # Process remaining weights with name mapping
    for name, tensor in model_weights.items():
        swift_name = map_weight_name(name)
        converted[swift_name] = tensor

    return converted


def save_weight_mapping(
    pytorch_names: list[str],
    output_dir: Path
) -> None:
    """Save weight name mapping for debugging."""
    mapping = {}
    for name in pytorch_names:
        mapping[name] = map_weight_name(name)

    with open(output_dir / "weight_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Convert GLiNER2 weights to MLX format")
    parser.add_argument(
        "--model",
        type=str,
        default="fastino/gliner2-base-v1",
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../weights",
        help="Output directory for converted weights"
    )
    parser.add_argument(
        "--split-files",
        action="store_true",
        help="Output separate encoder and model weight files (legacy mode)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Output data type"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading weights from: {args.model}")
    weights = get_pytorch_weights(args.model)
    print(f"Loaded {len(weights)} weight tensors")

    # Split encoder and model weights
    encoder_weights, model_weights = split_encoder_weights(weights)
    print(f"Encoder weights: {len(encoder_weights)}")
    print(f"Model weights: {len(model_weights)}")

    # Save original name mapping for debugging
    save_weight_mapping(list(weights.keys()), output_dir)
    print(f"Saved weight mapping to {output_dir / 'weight_mapping.json'}")

    # Convert GLiNER2 model weights (with Swift name mapping)
    converted_model_weights = process_gliner2_weights(model_weights.copy())

    # Process and validate encoder weights (keeps encoder. prefix)
    processed_encoder = process_deberta_encoder_weights(encoder_weights)

    if args.split_files:
        # Legacy mode: output separate files
        # Convert to target dtype
        if args.dtype == "float16":
            converted_model_weights = {k: v.half() for k, v in converted_model_weights.items()}
        elif args.dtype == "bfloat16":
            converted_model_weights = {k: v.bfloat16() for k, v in converted_model_weights.items()}

        # Save converted model weights
        model_output = output_dir / "gliner2_weights.safetensors"
        save_file(converted_model_weights, str(model_output))
        print(f"Saved model weights to {model_output}")
        print(f"  - {len(converted_model_weights)} tensors")
        total_params = sum(t.numel() for t in converted_model_weights.values())
        print(f"  - {total_params:,} parameters")

        # Convert encoder weights dtype
        if args.dtype == "float16":
            processed_encoder = {k: v.half() for k, v in processed_encoder.items()}
        elif args.dtype == "bfloat16":
            processed_encoder = {k: v.bfloat16() for k, v in processed_encoder.items()}

        encoder_output = output_dir / "encoder_weights.safetensors"
        save_file(processed_encoder, str(encoder_output))
        print(f"Saved encoder weights to {encoder_output}")
        print(f"  - {len(processed_encoder)} tensors")
        encoder_params = sum(t.numel() for t in processed_encoder.values())
        print(f"  - {encoder_params:,} parameters")

        # Print summary
        print("\n" + "=" * 60)
        print("Conversion complete! (split files mode)")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print("\nFiles created:")
        print(f"  - gliner2_weights.safetensors")
        print(f"  - encoder_weights.safetensors")
        print(f"  - weight_mapping.json")
    else:
        # Default mode: single combined file
        # Combine all weights into single dict
        all_weights = {}

        # Add encoder weights (keeps encoder. prefix)
        all_weights.update(processed_encoder)

        # Add model weights (with Swift name mapping)
        all_weights.update(converted_model_weights)

        # Convert to target dtype
        if args.dtype == "float16":
            all_weights = {k: v.half() for k, v in all_weights.items()}
        elif args.dtype == "bfloat16":
            all_weights = {k: v.bfloat16() for k, v in all_weights.items()}

        # Save single combined file
        combined_output = output_dir / "model.safetensors"
        save_file(all_weights, str(combined_output))
        print(f"Saved combined weights to {combined_output}")
        print(f"  - {len(all_weights)} tensors")
        total_params = sum(t.numel() for t in all_weights.values())
        print(f"  - {total_params:,} parameters")

        # Print summary
        print("\n" + "=" * 60)
        print("Conversion complete! (single file mode)")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print("\nFiles created:")
        print(f"  - model.safetensors")
        print(f"  - weight_mapping.json")

    # Print encoder structure info
    print("\nDeBERTa encoder structure:")
    embedding_keys = [k for k in processed_encoder if "embeddings" in k]
    layer_keys = [k for k in processed_encoder if "layer.0." in k]
    print(f"  - Embedding weights: {len(embedding_keys)}")
    print(f"  - Weights per layer: {len(layer_keys)}")
    if "encoder.encoder.rel_embeddings.weight" in processed_encoder:
        rel_shape = list(processed_encoder["encoder.encoder.rel_embeddings.weight"].shape)
        print(f"  - Rel embeddings shape: {rel_shape}")

    print("\nKey weight shapes (GLiNER2 model):")
    shape_info = [
        ("GRU weight_ih", converted_model_weights.get("countEmbed.gru.weightIH")),
        ("GRU weight_hh", converted_model_weights.get("countEmbed.gru.weightHH")),
        ("Pos embedding", converted_model_weights.get("countEmbed.posEmbedding.weight")),
    ]
    for name, tensor in shape_info:
        if tensor is not None:
            print(f"  - {name}: {list(tensor.shape)}")

    print("\nKey weight shapes (DeBERTa encoder):")
    encoder_shape_info = [
        ("Word embeddings", processed_encoder.get("encoder.embeddings.word_embeddings.weight")),
        ("Rel embeddings", processed_encoder.get("encoder.encoder.rel_embeddings.weight")),
        ("Query proj L0", processed_encoder.get("encoder.encoder.layer.0.attention.self.query_proj.weight")),
        ("Key proj L0", processed_encoder.get("encoder.encoder.layer.0.attention.self.key_proj.weight")),
    ]
    for name, tensor in encoder_shape_info:
        if tensor is not None:
            print(f"  - {name}: {list(tensor.shape)}")


if __name__ == "__main__":
    main()
