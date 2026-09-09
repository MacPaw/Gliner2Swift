#!/usr/bin/env python3
"""Verify GLiNER2 weights: reference statistics for Swift loading, or a converted directory.

Mode 1 (legacy, no arguments): generate L1 sums / shapes / stats for every tensor of
``fastino/gliner2-base-v1`` so the Swift weight-loading test can check itself.

    python verify_all_weights.py
    -> ../Tests/GLiNER2SwiftTests/Fixtures/weight_reference.json

Mode 2 (``--dir``): verify a directory produced by ``convert_weights.py`` (v1 or
GLiNER 2.5 "boundary"): tensor counts (334 for a 2.5 fp16 directory; for int8 the
same 334 logical tensors plus ``.scales``/``.biases`` per packed encoder Linear),
``weight_mapping.json`` coverage, no leftover snake_case head keys, all values finite,
and -- with ``--source`` -- shape + L1 agreement with the original PyTorch checkpoint
(int8 tensors are dequantised with MLX first).

    python verify_all_weights.py --dir ../out/gliner25_small_mlx_fp16 \
        --source fastino/gliner2.5-small-v1 --expect-count 334

Exit status is non-zero when any check fails.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from huggingface_hub import hf_hub_download


OUTPUT_DIR = Path(__file__).parent.parent / "Tests" / "GLiNER2SwiftTests" / "Fixtures"

# Relative L1 tolerance |converted - source|_1 / |source|_1 per tensor.
FP16_TOLERANCE = 2e-3     # fp16 rounding of fp32 weights is ~5e-4 on average
BF16_TOLERANCE = 1.5e-2
INT8_TOLERANCE = 5e-2     # affine group-64 8-bit; typical ~5e-3


def finite(value):
    """JSON has no NaN/Infinity literals, and Foundation's JSONSerialization
    rejects Python's non-standard spelling outright -- which silently disables
    the Swift-side gate. Single-element tensors give std() == NaN, so map every
    non-finite statistic to null."""
    value = float(value)
    return value if math.isfinite(value) else None


# ---------------------------------------------------------------------------
# Mode 1: reference statistics for the v1 base checkpoint
# ---------------------------------------------------------------------------

def generate_reference():
    print("Downloading model weights...")
    weights_path = hf_hub_download("fastino/gliner2-base-v1", "model.safetensors")

    print(f"Loading weights from: {weights_path}")

    l1_sums = {}
    encoder_weights = {}
    other_weights = {}

    with safe_open(weights_path, framework="pt") as f:
        for key in sorted(f.keys()):
            tensor = f.get_tensor(key)
            info = {
                "shape": list(tensor.shape),
                "l1_sum": finite(tensor.abs().sum().item()),
                "mean": finite(tensor.mean().item()),
                "std": finite(tensor.std().item()),
                "min": finite(tensor.min().item()),
                "max": finite(tensor.max().item()),
            }
            l1_sums[key] = info

            if key.startswith("encoder."):
                encoder_weights[key] = info
            else:
                other_weights[key] = info

    # Save full reference
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "weight_reference.json"
    with open(output_path, "w") as f:
        json.dump(l1_sums, f, indent=2)
    print(f"\nSaved full reference to: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("ENCODER WEIGHTS SUMMARY")
    print("=" * 70)

    # Group by component
    components = {}
    for key, info in encoder_weights.items():
        # Extract component path
        parts = key.split(".")
        if len(parts) >= 4:
            component = ".".join(parts[:4])
        else:
            component = key
        if component not in components:
            components[component] = []
        components[component].append((key, info))

    for component in sorted(components.keys()):
        print(f"\n{component}:")
        for key, info in components[component]:
            short_key = key[len(component)+1:] if key.startswith(component + ".") else key
            print(f"  {short_key}: shape={info['shape']}, L1={info['l1_sum']:.4f}")

    # Critical weights to check
    print("\n" + "=" * 70)
    print("CRITICAL WEIGHTS FOR PARITY")
    print("=" * 70)

    critical = [
        "encoder.embeddings.word_embeddings.weight",
        "encoder.embeddings.LayerNorm.weight",
        "encoder.embeddings.LayerNorm.bias",
        "encoder.encoder.rel_embeddings.weight",
        "encoder.encoder.LayerNorm.weight",
        "encoder.encoder.LayerNorm.bias",
        "encoder.encoder.layer.0.attention.self.query_proj.weight",
        "encoder.encoder.layer.0.attention.self.key_proj.weight",
        "encoder.encoder.layer.0.attention.self.value_proj.weight",
        "encoder.encoder.layer.0.attention.output.dense.weight",
        "encoder.encoder.layer.0.attention.output.LayerNorm.weight",
        "encoder.encoder.layer.0.intermediate.dense.weight",
        "encoder.encoder.layer.0.output.dense.weight",
        "encoder.encoder.layer.0.output.LayerNorm.weight",
    ]

    for key in critical:
        if key in l1_sums:
            info = l1_sums[key]
            print(f"{key}")
            print(f"  shape: {info['shape']}")
            print(f"  L1: {info['l1_sum']:.6f}")
            print(f"  mean: {info['mean']:.6f}, std: {info['std']:.6f}")
        else:
            print(f"{key} NOT FOUND!")

    # Print total weight count
    print(f"\n" + "=" * 70)
    print(f"Total encoder weights: {len(encoder_weights)}")
    print(f"Total other weights: {len(other_weights)}")
    print(f"Total weights: {len(l1_sums)}")


# ---------------------------------------------------------------------------
# Mode 2: verify a converted MLX directory
# ---------------------------------------------------------------------------

def _source_weights_path(source: str) -> Path:
    if Path(source).is_dir():
        return Path(source) / "model.safetensors"
    return Path(hf_hub_download(source, "model.safetensors"))


SNAKE_SEGMENT = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)+$")


def verify_directory(dir_path: Path, source: str, expect_count: int) -> int:
    import mlx.core as mx

    failures = []

    def fail(message):
        failures.append(message)
        print(f"  FAIL  {message}")

    weights_path = dir_path / "model.safetensors"
    config = json.load(open(dir_path / "config.json"))
    architecture = str(config.get("architecture", "span")).lower()
    quantization = config.get("quantization")
    mapping_path = dir_path / "weight_mapping.json"
    mapping = json.load(open(mapping_path))["mapping"] if mapping_path.exists() else None

    print(f"Verifying {dir_path}")
    print(f"  architecture={architecture} quantization={quantization} "
          f"mapping={'yes' if mapping else 'no'}")
    for required in ("config.json", "tokenizer.json", "model.safetensors"):
        if not (dir_path / required).exists():
            fail(f"missing {required}")
    if architecture == "boundary" and not (dir_path / "encoder_config" / "config.json").exists():
        fail("missing encoder_config/config.json (boundary loader reads it)")

    arrays = mx.load(str(weights_path))
    keys = sorted(arrays)
    scales = {k[: -len(".scales")] for k in keys if k.endswith(".scales")}
    biases = {k[: -len(".biases")] for k in keys if k.endswith(".biases")}
    if scales != biases:
        fail("mismatched .scales / .biases sets")
    logical = [k for k in keys if not (k.endswith(".scales") or k.endswith(".biases"))]
    encoder = [k for k in logical if k.startswith("encoder.")]
    head = [k for k in logical if not k.startswith("encoder.")]
    by_prefix = {}
    for key in head:
        by_prefix[key.split(".")[0]] = by_prefix.get(key.split(".")[0], 0) + 1
    print(f"  tensors: {len(keys)} on disk = {len(logical)} logical "
          f"({len(encoder)} encoder + {len(head)} head) + {2 * len(scales)} quant sidecars")
    print(f"  head prefixes: {by_prefix}")
    if quantization and not scales:
        fail("config declares quantization but no .scales tensors found")
    if scales and not quantization:
        fail(".scales tensors present but config has no quantization block")
    if expect_count is not None and len(logical) != expect_count:
        fail(f"expected {expect_count} logical tensors, found {len(logical)}")

    # No snake_case left outside the encoder.
    for key in head:
        segments = key.split(".")
        bad = [s for s in segments if SNAKE_SEGMENT.match(s)]
        if bad:
            fail(f"snake_case segment {bad} in head key {key}")
    # Mapping coverage.
    if mapping:
        mapped = {v for v in mapping.values() if "(" not in v}
        missing = sorted(mapped - set(logical))
        extra = sorted(set(logical) - mapped)
        if missing:
            fail(f"{len(missing)} mapped keys absent from model.safetensors, e.g. {missing[:3]}")
        if extra and architecture == "boundary":
            fail(f"{len(extra)} tensors not in weight_mapping.json, e.g. {extra[:3]}")

    # Finite values + dtype census.
    dtypes = {}
    for key in keys:
        arr = arrays[key]
        dtypes[str(arr.dtype)] = dtypes.get(str(arr.dtype), 0) + 1
        if arr.dtype in (mx.float16, mx.bfloat16, mx.float32):
            if not bool(mx.all(mx.isfinite(arr)).item()):
                fail(f"non-finite values in {key}")
    print(f"  dtypes: {dtypes}")

    # Source comparison.
    if source:
        inverse = {v: k for k, v in mapping.items()} if mapping else {k: k for k in logical}
        src_path = _source_weights_path(source)
        worst = {}
        group_size = (quantization or {}).get("group_size", 64)
        bits = (quantization or {}).get("bits", 8)
        with safe_open(str(src_path), framework="pt") as src:
            src_keys = set(src.keys())
            unmapped = sorted(src_keys - set(inverse.values()))
            if unmapped:
                fail(f"{len(unmapped)} source tensors have no converted counterpart, e.g. {unmapped[:3]}")
            for key in logical:
                pt_key = inverse.get(key)
                if pt_key is None or pt_key not in src_keys:
                    fail(f"{key}: no source tensor ({pt_key})")
                    continue
                reference = src.get_tensor(pt_key).float().numpy()
                arr = arrays[key]
                base = key[: -len(".weight")] if key.endswith(".weight") else None
                if base is not None and base in scales:
                    arr = mx.dequantize(arr, arrays[base + ".scales"], arrays[base + ".biases"],
                                        group_size=group_size, bits=bits)
                    tolerance, kind = INT8_TOLERANCE, "int8"
                elif arr.dtype == mx.bfloat16:
                    tolerance, kind = BF16_TOLERANCE, "bf16"
                elif arr.dtype == mx.float16:
                    tolerance, kind = FP16_TOLERANCE, "fp16"
                else:
                    tolerance, kind = 1e-6, str(arr.dtype)
                converted = np.array(arr.astype(mx.float32))
                if converted.shape != reference.shape:
                    fail(f"{key}: shape {converted.shape} != source {reference.shape}")
                    continue
                if not np.issubdtype(reference.dtype, np.floating):
                    continue
                denom = np.abs(reference).sum()
                rel = float(np.abs(converted - reference).sum() / denom) if denom > 0 else 0.0
                if rel > worst.get(kind, (0.0, ""))[0]:
                    worst[kind] = (rel, key)
                if rel > tolerance:
                    fail(f"{key}: relative L1 diff {rel:.3e} > {tolerance:.1e} ({kind})")
        for kind, (rel, key) in sorted(worst.items()):
            print(f"  worst {kind:5s} relative L1 diff {rel:.3e} ({key})")

    status = "OK" if not failures else f"{len(failures)} FAILURE(S)"
    print(f"  => {status}: {dir_path.name}: {len(logical)} logical tensors "
          f"({len(keys)} on disk), {len(scales)} quantized")
    return 0 if not failures else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dir", default=None, help="Converted MLX model directory to verify")
    parser.add_argument("--source", default=None,
                        help="Original checkpoint (HF id or dir) to compare shapes/L1 against")
    parser.add_argument("--expect-count", type=int, default=None,
                        help="Expected number of logical tensors (334 for a GLiNER 2.5 checkpoint)")
    args = parser.parse_args()
    if args.dir is None:
        generate_reference()
        return
    sys.exit(verify_directory(Path(args.dir), args.source, args.expect_count))


if __name__ == "__main__":
    main()
