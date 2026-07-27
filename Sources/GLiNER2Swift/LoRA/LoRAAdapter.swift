// LoRAAdapter.swift
// LoRA weight merging for inference
//
// Matches Python: gliner2/training/lora.py:LoRALayer.merge_weights()
// Formula: W_merged = W_base + (lora_B @ lora_A) * (alpha / r)

import MLX

/// Merge LoRA adapter weights into base weights dictionary (in Python key space).
///
/// The merge happens before sanitization so that all existing weight loading
/// code continues to work unchanged.
///
/// - Parameters:
///   - baseWeights: Mutable base weight dictionary (Python keys, before sanitization)
///   - adapterWeights: Adapter weight dictionary from adapter_weights.safetensors
///   - config: LoRA adapter configuration (provides scaling = alpha / r)
/// - Returns: Number of weight matrices that were merged
@discardableResult
public func mergeLoRAWeights(
    into baseWeights: inout [String: MLXArray],
    adapterWeights: [String: MLXArray],
    config: LoRAAdapterConfig
) -> Int {
    let scaling = config.scaling  // alpha / r

    // Group adapter weights into (path -> (loraA, loraB)) pairs
    // Keys look like: "encoder.encoder.layer.0.attention.self.query_proj.lora_A"
    var pairs: [String: (a: MLXArray?, b: MLXArray?)] = [:]
    for (key, value) in adapterWeights {
        if key.hasSuffix(".lora_A") {
            let base = String(key.dropLast(7))  // strip ".lora_A"
            pairs[base, default: (nil, nil)].a = value
        } else if key.hasSuffix(".lora_B") {
            let base = String(key.dropLast(7))  // strip ".lora_B"
            pairs[base, default: (nil, nil)].b = value
        }
    }

    // Merge each (A, B) pair into the corresponding base weight
    var merged = 0
    for (path, pair) in pairs {
        guard let loraA = pair.a, let loraB = pair.b else { continue }

        // Adapter keys are always emitted in Python (snake_case) space, but the base
        // checkpoint may already be converted to camelCase (anything produced by
        // convert_weights.py, including the shipped fp16 model). Try the raw key first,
        // then the converted one. Without this fallback only `encoder.*` pairs match —
        // those keys are identical in both spaces — and every span_rep / count_embed /
        // count_pred / classifier delta is silently dropped, partially applying the adapter.
        let candidates = [path + ".weight", Extractor.mapRawKey(path) + ".weight"]
        guard let weightKey = candidates.first(where: { baseWeights[$0] != nil }),
              let baseWeight = baseWeights[weightKey] else { continue }

        // W_merged = W_base + (B @ A) * scaling
        let delta = matmul(loraB, loraA) * MLXArray(scaling)
        baseWeights[weightKey] = baseWeight + delta
        merged += 1
    }
    return merged
}
