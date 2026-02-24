// Copyright 2026 MacPaw Way Ltd.
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
//
// ModuleUpdates.swift
// Centralized weight update helpers for MLX modules

import MLX
import MLXNN

// MARK: - Module Update Helpers

/// Update a Linear module's weights
public func updateLinearWeights(_ module: Linear, weight: MLXArray?, bias: MLXArray?, debugKey: String? = nil) {
    var params: [String: MLXArray] = [:]
    if let w = weight {
        params["weight"] = w
    } else if let key = debugKey {
        print("⚠️ WARNING: weight is nil for '\(key)', keeping random initialization!")
    }
    if let b = bias {
        params["bias"] = b
    } else if module.bias != nil, let key = debugKey {
        print("⚠️ WARNING: bias is nil for '\(key)', keeping random initialization!")
    }
    if !params.isEmpty {
        module.update(parameters: ModuleParameters.unflattened(params))
    } else if let key = debugKey {
        print("❌ ERROR: No weights loaded for '\(key)'!")
    }
}

/// Update a LayerNorm module's weights
public func updateLayerNormWeights(_ module: LayerNorm, weight: MLXArray?, bias: MLXArray?) {
    var params: [String: MLXArray] = [:]
    if let w = weight { params["weight"] = w }
    if let b = bias { params["bias"] = b }
    if !params.isEmpty {
        module.update(parameters: ModuleParameters.unflattened(params))
    }
}

/// Update an Embedding module's weights
public func updateEmbeddingWeights(_ module: Embedding, weight: MLXArray?) {
    if let w = weight {
        module.update(parameters: ModuleParameters.unflattened(["weight": w]))
    }
}

// MARK: - Weight Dictionary Helpers

/// Load Linear weights from a weight dictionary
public func loadLinearFromDict(_ module: Linear, weights: [String: MLXArray], prefix: String) {
    let p = prefix.isEmpty ? "" : "\(prefix)."
    updateLinearWeights(
        module,
        weight: weights["\(p)weight"],
        bias: weights["\(p)bias"]
    )
}

/// Load LayerNorm weights from a weight dictionary
public func loadLayerNormFromDict(_ module: LayerNorm, weights: [String: MLXArray], prefix: String) {
    let p = prefix.isEmpty ? "" : "\(prefix)."
    updateLayerNormWeights(
        module,
        weight: weights["\(p)weight"],
        bias: weights["\(p)bias"]
    )
}

/// Load Embedding weights from a weight dictionary
public func loadEmbeddingFromDict(_ module: Embedding, weights: [String: MLXArray], prefix: String) {
    let p = prefix.isEmpty ? "" : "\(prefix)."
    updateEmbeddingWeights(module, weight: weights["\(p)weight"])
}
