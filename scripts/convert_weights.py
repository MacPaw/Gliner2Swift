#!/usr/bin/env python3
# Copyright 2026 MacPaw Way Ltd.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
"""
Convert a GLiNER2 PyTorch checkpoint to the MLX model directory the Swift
package loads, at a chosen precision, and optionally push it to the Hub.

The output directory is self-contained and loads directly through
`GLiNER2.fromPretrained(<dir>)`:

    model.safetensors   converted weights (fp16/fp32/bf16, or int8-quantized)
    config.json         extractor config (+ a `quantization` block when quantized)
    tokenizer.json      copied from the source (the only tokenizer file Swift reads)
    tokenizer_config.json, spm.model, ...   copied through for completeness

What it does to the weights:
  - DeBERTa encoder (disentangled attention, share_att_key=true — NO pos_key/query_proj)
  - SpanMarkerV0 span representation, CountLSTMv2 (GRU + DownscaledTransformer),
    classifier / count-prediction MLPs
  - PyTorch snake_case -> Swift camelCase key remapping
  - DownscaledTransformer combined in_proj -> split q/k/v

Quantization (`--quantize int8`):
  Reproduces exactly what `Extractor.quantize(.int8)` does at load time in Swift:
  affine group quantization of the ENCODER's Linear weights (and, by default, the
  word-embedding table), group size 64, leaving everything else at fp16. The packed
  layout (`.weight` uint32 + `.scales` + `.biases`) and the `quantization` config block
  are MLX's standard `QuantizedLinear`/`QuantizedEmbedding` format.

Usage:
    # fp16 directory (what the shipped model uses)
    python convert_weights.py --model fastino/gliner2-base-v1 \
        --output ../out/gliner2_mlx_fp16 --dtype fp16

    # int8-quantized directory
    python convert_weights.py --model fastino/gliner2-base-v1 \
        --output ../out/gliner2_mlx_int8 --quantize int8

    # convert and push
    python convert_weights.py --model fastino/gliner2-base-v1 \
        --output ../out/gliner2_mlx_int8 --quantize int8 \
        --push-to-hub your-org/gliner2_mlx_int8 --private
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
import torch
from safetensors import safe_open

# ---------------------------------------------------------------------------
# Weight loading + PyTorch -> Swift name mapping (proven mapping, unchanged)
# ---------------------------------------------------------------------------

# Tokenizer / config files copied verbatim into the output directory so the
# result loads without touching the source. `tokenizer.json` is the only one the
# Swift tokenizer actually reads; the rest are carried for completeness / Python use.
SIDE_CAR_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "spm.model",
    "special_tokens_map.json",
    "added_tokens.json",
]


def resolve_source_file(model: str, filename: str) -> Optional[str]:
    """Return a local path to `filename` for a local dir or Hub id, or None."""
    if os.path.isdir(model):
        path = os.path.join(model, filename)
        return path if os.path.exists(path) else None
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(model, filename)
    except Exception:
        return None


def get_pytorch_weights(model_path: str) -> Dict[str, torch.Tensor]:
    """Load PyTorch weights from a local dir or a Hub id."""
    weights_path = resolve_source_file(model_path, "model.safetensors")
    if weights_path is None:
        raise FileNotFoundError(f"model.safetensors not found for '{model_path}'")

    weights = {}
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


def map_weight_name(pytorch_name: str) -> str:
    """Map a PyTorch snake_case key to the Swift camelCase key."""
    mappings = {
        "span_rep.span_rep_layer.project_start": "spanRep.spanRepLayer.projectStart",
        "span_rep.span_rep_layer.project_end": "spanRep.spanRepLayer.projectEnd",
        "span_rep.span_rep_layer.out_project": "spanRep.spanRepLayer.outProject",
        "classifier.0": "classifier.layers.0",
        "classifier.2": "classifier.layers.1",
        "count_pred.0": "countPred.layers.0",
        "count_pred.2": "countPred.layers.1",
        "count_embed.pos_embedding": "countEmbed.posEmbedding",
        "count_embed.gru.weight_ih_l0": "countEmbed.gru.weightIH",
        "count_embed.gru.weight_hh_l0": "countEmbed.gru.weightHH",
        "count_embed.gru.bias_ih_l0": "countEmbed.gru.biasIH",
        "count_embed.gru.bias_hh_l0": "countEmbed.gru.biasHH",
        "count_embed.transformer.in_projector": "countEmbed.transformer.inProjector",
        "count_embed.transformer.out_projector": "countEmbed.transformer.outProjector",
        "count_embed.transformer.transformer.layers": "countEmbed.transformer.transformerLayers",
    }
    for pt_prefix, swift_prefix in mappings.items():
        if pytorch_name.startswith(pt_prefix):
            return pytorch_name.replace(pt_prefix, swift_prefix, 1)
    return pytorch_name


def convert_transformer_layer_weights(
    weights: Dict[str, torch.Tensor], layer_prefix: str
) -> Dict[str, torch.Tensor]:
    """Split the DownscaledTransformer's combined in_proj into q/k/v; pass FFN through."""
    converted = {}
    in_proj_key = f"{layer_prefix}.self_attn.in_proj_weight"
    if in_proj_key in weights:
        in_proj = weights[in_proj_key]
        h = in_proj.shape[0] // 3
        converted[f"{layer_prefix}.self_attn.q_proj.weight"] = in_proj[:h]
        converted[f"{layer_prefix}.self_attn.k_proj.weight"] = in_proj[h : 2 * h]
        converted[f"{layer_prefix}.self_attn.v_proj.weight"] = in_proj[2 * h :]
    in_proj_bias_key = f"{layer_prefix}.self_attn.in_proj_bias"
    if in_proj_bias_key in weights:
        b = weights[in_proj_bias_key]
        h = b.shape[0] // 3
        converted[f"{layer_prefix}.self_attn.q_proj.bias"] = b[:h]
        converted[f"{layer_prefix}.self_attn.k_proj.bias"] = b[h : 2 * h]
        converted[f"{layer_prefix}.self_attn.v_proj.bias"] = b[2 * h :]
    for suffix in (
        ".self_attn.out_proj.weight",
        ".self_attn.out_proj.bias",
        ".linear1.weight",
        ".linear1.bias",
        ".linear2.weight",
        ".linear2.bias",
        ".norm1.weight",
        ".norm1.bias",
        ".norm2.weight",
        ".norm2.bias",
    ):
        key = layer_prefix + suffix
        if key in weights:
            converted[key] = weights[key]
    return converted


def convert_to_swift_keys(weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Full PyTorch checkpoint -> Swift-keyed tensor dict (encoder kept prefixed)."""
    encoder = {k: v for k, v in weights.items() if k.startswith("encoder.")}
    model = {k: v for k, v in weights.items() if not k.startswith("encoder.")}

    for name in encoder:
        if "pos_key_proj" in name or "pos_query_proj" in name:
            print(f"WARNING: unexpected position projection weight {name} "
                  "(share_att_key should be true for gliner2-base-v1)")

    out: Dict[str, torch.Tensor] = dict(encoder)  # encoder passes through unchanged

    # DownscaledTransformer layers need in_proj splitting before name mapping.
    for layer_idx in (0, 1):
        prefix = f"count_embed.transformer.transformer.layers.{layer_idx}"
        for name, tensor in convert_transformer_layer_weights(model, prefix).items():
            out[map_weight_name(name)] = tensor
        for key in [k for k in model if k.startswith(prefix)]:
            del model[key]

    for name, tensor in model.items():
        out[map_weight_name(name)] = tensor
    return out


# ---------------------------------------------------------------------------
# Precision + quantization (MLX)
# ---------------------------------------------------------------------------

DTYPES = {
    "float32": mx.float32, "fp32": mx.float32,
    "float16": mx.float16, "fp16": mx.float16,
    "bfloat16": mx.bfloat16, "bf16": mx.bfloat16,
}


def to_mx_float32(t: torch.Tensor) -> mx.array:
    """torch tensor -> mx.array. Floats go via float32 numpy; ints preserved."""
    if t.is_floating_point():
        return mx.array(t.detach().to(torch.float32).cpu().numpy())
    return mx.array(t.detach().cpu().numpy())


def should_quantize(key: str, arr: mx.array, group_size: int, include_embeddings: bool) -> bool:
    """Exactly the set `Extractor.quantize(.int8)` touches: encoder Linear weights
    (2-D `.weight` under `encoder.`, excluding the 1-D LayerNorms and the non-module
    `rel_embeddings`) plus, optionally, the word-embedding table."""
    if not key.startswith("encoder.") or not key.endswith(".weight"):
        return False
    if arr.ndim != 2 or "rel_embeddings" in key:
        return False
    if "word_embeddings" in key and not include_embeddings:
        return False
    return arr.shape[-1] % group_size == 0


def build_output_weights(
    swift_weights: Dict[str, torch.Tensor],
    dtype: str,
    quantize: bool,
    bits: int,
    group_size: int,
    include_embeddings: bool,
):
    """Return (arrays_for_safetensors, num_quantized). When quantizing, non-quantized
    floats are fp16 (a quantized checkpoint is inherently fp16 + packed int)."""
    residual = mx.float16 if quantize else DTYPES[dtype]
    out: Dict[str, mx.array] = {}
    num_quantized = 0

    for key, tensor in swift_weights.items():
        arr = to_mx_float32(tensor)
        is_float = tensor.is_floating_point()

        if quantize and should_quantize(key, arr, group_size, include_embeddings):
            wq, scales, biases = mx.quantize(arr.astype(mx.float16),
                                             group_size=group_size, bits=bits)
            base = key[: -len(".weight")]
            out[key] = wq
            out[base + ".scales"] = scales
            out[base + ".biases"] = biases
            num_quantized += 1
        else:
            out[key] = arr.astype(residual) if is_float else arr

    mx.eval(list(out.values()))
    return out, num_quantized


# ---------------------------------------------------------------------------
# Directory assembly + Hub push
# ---------------------------------------------------------------------------

def write_config(model: str, output_dir: Path, quantize: bool, bits: int, group_size: int) -> None:
    """Copy the source config.json, adding a `quantization` block when quantized."""
    src = resolve_source_file(model, "config.json")
    config = json.load(open(src)) if src else {
        "model_type": "extractor",
        "counting_layer": "count_lstm_v2",
        "max_width": 8,
        "model_name": "microsoft/deberta-v3-base",
        "token_pooling": "first",
    }
    if quantize:
        config["quantization"] = {"group_size": group_size, "bits": bits}
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def copy_side_cars(model: str, output_dir: Path) -> None:
    """Copy tokenizer files (config.json is handled by write_config)."""
    for name in SIDE_CAR_FILES:
        if name == "config.json":
            continue
        src = resolve_source_file(model, name)
        if src:
            shutil.copyfile(src, output_dir / name)
            print(f"  copied {name}")


def push_to_hub(output_dir: Path, repo_id: str, private: bool, message: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=str(output_dir), repo_id=repo_id, commit_message=message)
    print(f"Pushed {output_dir} -> https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a GLiNER2 PyTorch checkpoint to an MLX model directory."
    )
    parser.add_argument("--model", default="fastino/gliner2-base-v1",
                        help="Source: HuggingFace id or local directory")
    parser.add_argument("--output", required=True, help="Output model directory")
    parser.add_argument("--dtype", default="fp16",
                        choices=sorted(DTYPES.keys()),
                        help="Precision for the non-quantized weights (default: fp16)")
    parser.add_argument("--quantize", choices=["int8"], default=None,
                        help="Also quantize the encoder to int8 (affine, group 64)")
    parser.add_argument("--q-bits", type=int, default=8, help="Quantization bits")
    parser.add_argument("--q-group-size", type=int, default=64, help="Quantization group size")
    parser.add_argument("--no-quantize-embeddings", action="store_true",
                        help="Leave the word-embedding table at fp16 when quantizing "
                             "(the largest tensor; excluding it forfeits most of the memory win)")
    parser.add_argument("--push-to-hub", metavar="REPO_ID", default=None,
                        help="Upload the output directory to this HuggingFace repo")
    parser.add_argument("--private", action="store_true", help="Create the Hub repo private")
    parser.add_argument("--commit-message", default="Add converted GLiNER2 MLX weights")
    args = parser.parse_args()

    quantize = args.quantize == "int8"
    include_embeddings = quantize and not args.no_quantize_embeddings
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading weights from: {args.model}")
    weights = get_pytorch_weights(args.model)
    print(f"  {len(weights)} source tensors")

    swift_weights = convert_to_swift_keys(weights)
    print(f"  {len(swift_weights)} Swift-keyed tensors")

    print(f"Building {'int8-quantized ' if quantize else ''}weights (residual dtype "
          f"{'fp16' if quantize else args.dtype})...")
    out_arrays, num_quantized = build_output_weights(
        swift_weights, args.dtype, quantize, args.q_bits, args.q_group_size, include_embeddings
    )
    if quantize:
        print(f"  quantized {num_quantized} tensors to {args.q_bits}-bit "
              f"(group {args.q_group_size}, embeddings {'in' if include_embeddings else 'ex'}cluded)")

    weights_path = output_dir / "model.safetensors"
    mx.save_safetensors(str(weights_path), out_arrays,
                        metadata={"format": "mlx", "converter": "convert_weights.py"})
    size_mb = weights_path.stat().st_size / (1024 * 1024)
    print(f"Saved {weights_path} ({size_mb:.0f} MB, {len(out_arrays)} tensors)")

    write_config(args.model, output_dir, quantize, args.q_bits, args.q_group_size)
    copy_side_cars(args.model, output_dir)

    print("\n" + "=" * 60)
    print("Conversion complete.")
    print(f"  {output_dir}")
    print('  Load in Swift:  GLiNER2.fromPretrained("<dir>")')
    print("=" * 60)

    if args.push_to_hub:
        push_to_hub(output_dir, args.push_to_hub, args.private, args.commit_message)


if __name__ == "__main__":
    main()
