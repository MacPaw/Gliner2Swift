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
// DownscaledTransformer.swift
// Downscaled transformer for efficient count embedding
//
// Matches Python: gliner2/layers.py:DownscaledTransformer
//
// Architecture:
//   original_x = x                      # Save original (L, M, 768)
//   x = in_projector(x)                 # (L, M, 128)
//   x = transformer(x)                  # (L, M, 128)
//   x = cat([x, original_x], -1)        # (L, M, 896)  ← CONCATENATION
//   x = out_projector(x)                # (L, M, 768)

import Foundation
import MLX
import MLXNN

/// Downscaled transformer that projects to a smaller dimension, applies transformer layers,
/// then projects back to original dimension using concatenation with skip connection.
///
/// Weight structure from Python model:
/// ```
/// count_embed.transformer.in_projector: [128, 768]
/// count_embed.transformer.transformer.layers.{0,1}:
///   - self_attn.in_proj: [384, 128]  # 4 heads (384 = 3*128 for Q,K,V)
///   - self_attn.out_proj: [128, 128]
///   - linear1: [256, 128], linear2: [128, 256]  # FFN
///   - norm1, norm2: [128]
/// count_embed.transformer.out_projector:  # MLP: 896→768→768→768
///   - .0: [768, 896]
///   - .2: [768, 768]
///   - .4: [768, 768]
/// ```
public class DownscaledTransformer: Module {
    public let inputSize: Int
    public let hiddenSize: Int
    public let numHeads: Int
    public let numLayers: Int

    /// Input projection: inputSize → hiddenSize
    public let inProjector: Linear

    /// Transformer encoder layers
    public let transformerLayers: [TransformerEncoderLayer]

    /// Output MLP: (hiddenSize + inputSize) → inputSize → inputSize → inputSize
    public let outProjector: Sequential

    /// Initialize DownscaledTransformer
    ///
    /// - Parameters:
    ///   - inputSize: Original input dimension (e.g., 768)
    ///   - hiddenSize: Downscaled dimension (e.g., 128)
    ///   - numHeads: Number of attention heads (e.g., 4)
    ///   - numLayers: Number of transformer layers (e.g., 2)
    ///   - dropout: Dropout probability (default: 0.1)
    public init(
        inputSize: Int,
        hiddenSize: Int = 128,
        numHeads: Int = 4,
        numLayers: Int = 2,
        dropout: Float = 0.1
    ) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numHeads = numHeads
        self.numLayers = numLayers

        // Input projection: 768 → 128
        self.inProjector = Linear(inputSize, hiddenSize)

        // Transformer encoder layers
        var layers: [TransformerEncoderLayer] = []
        for _ in 0..<numLayers {
            layers.append(TransformerEncoderLayer(
                dModel: hiddenSize,
                nHead: numHeads,
                dimFeedforward: hiddenSize * 2,  // 256
                dropout: dropout
            ))
        }
        self.transformerLayers = layers

        // Output MLP: 896 → 768 → 768 → 768
        // Note: No dropout, ReLU activation
        self.outProjector = createMLP(
            inputDim: hiddenSize + inputSize,  // 896
            intermediateDims: [inputSize, inputSize],  // [768, 768]
            outputDim: inputSize,  // 768
            dropout: 0.0,
            activation: .relu,
            addLayerNorm: false
        )
    }

    /// Forward pass
    ///
    /// - Parameter x: Input tensor [L, M, inputSize]
    /// - Returns: Output tensor [L, M, inputSize]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Save original for skip connection
        let originalX = x

        // Project down: (L, M, 768) → (L, M, 128)
        var h = inProjector(x)

        // Apply transformer layers
        for layer in transformerLayers {
            h = layer(h)
        }

        // Concatenate with original: (L, M, 128) + (L, M, 768) → (L, M, 896)
        let concatenated = MLX.concatenated([h, originalX], axis: -1)

        // Project back: (L, M, 896) → (L, M, 768)
        return outProjector(concatenated)
    }
}

// MARK: - Custom Multi-Head Attention

/// Custom multi-head attention with explicit weight access for loading.
///
/// This implementation mirrors PyTorch's nn.MultiheadAttention but with
/// separate projections (not combined in_proj) for easier weight loading.
public class CustomMultiHeadAttention: Module {
    public let dims: Int
    public let numHeads: Int
    public let headDim: Int

    public let queryProj: Linear
    public let keyProj: Linear
    public let valueProj: Linear
    public let outProj: Linear

    public init(dims: Int, numHeads: Int) {
        self.dims = dims
        self.numHeads = numHeads
        self.headDim = dims / numHeads

        self.queryProj = Linear(dims, dims)
        self.keyProj = Linear(dims, dims)
        self.valueProj = Linear(dims, dims)
        self.outProj = Linear(dims, dims)
    }

    public func callAsFunction(queries: MLXArray, keys: MLXArray, values: MLXArray) -> MLXArray {
        let B = queries.dim(0)
        let L = queries.dim(1)

        // Project
        var q = queryProj(queries)  // [B, L, D]
        var k = keyProj(keys)
        var v = valueProj(values)

        // Reshape for multi-head: [B, L, D] -> [B, L, H, D/H] -> [B, H, L, D/H]
        q = q.reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)
        k = k.reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)
        v = v.reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)

        // Attention.
        //
        // The divisor is a Swift scalar, NOT an `MLXArray`. A strongly typed float32 array
        // here promotes the whole product to float32 — and since this attention feeds
        // countEmbed, that promotion used to spread downstream all the way through the
        // span-score einsum and sigmoid, so an fp16 checkpoint still computed its scores in
        // fp32. Swift scalars adopt the array's dtype instead.
        let scores = MLX.matmul(q, k.transposed(0, 1, 3, 2)) / Foundation.sqrt(Float(headDim))
        let attnWeights = MLX.softmax(scores, axis: -1)

        // Apply attention
        var output = MLX.matmul(attnWeights, v)  // [B, H, L, D/H]

        // Reshape back: [B, H, L, D/H] -> [B, L, H, D/H] -> [B, L, D]
        output = output.transposed(0, 2, 1, 3).reshaped([B, L, dims])

        // Output projection
        return outProj(output)
    }
}

// MARK: - Transformer Encoder Layer

/// Single transformer encoder layer with post-norm architecture.
///
/// Architecture:
///   x = norm1(x + self_attn(x))
///   x = norm2(x + ffn(x))
public class TransformerEncoderLayer: Module {
    public let dModel: Int
    public let nHead: Int

    /// Multi-head self-attention (custom for weight loading)
    public let selfAttn: CustomMultiHeadAttention

    /// Feed-forward network
    public let linear1: Linear
    public let linear2: Linear

    /// Layer norms
    public let norm1: LayerNorm
    public let norm2: LayerNorm

    /// Dropout
    public let dropout: Dropout

    public init(
        dModel: Int,
        nHead: Int,
        dimFeedforward: Int,
        dropout: Float = 0.1
    ) {
        self.dModel = dModel
        self.nHead = nHead

        self.selfAttn = CustomMultiHeadAttention(dims: dModel, numHeads: nHead)
        self.linear1 = Linear(dModel, dimFeedforward)
        self.linear2 = Linear(dimFeedforward, dModel)
        self.norm1 = LayerNorm(dimensions: dModel)
        self.norm2 = LayerNorm(dimensions: dModel)
        self.dropout = Dropout(p: dropout)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Self-attention with residual
        let attnOut = selfAttn(queries: x, keys: x, values: x)
        var h = norm1(x + dropout(attnOut))

        // FFN with residual
        var ffnOut = linear1(h)
        ffnOut = MLX.maximum(ffnOut, MLXArray(0))  // ReLU
        ffnOut = dropout(ffnOut)
        ffnOut = linear2(ffnOut)

        h = norm2(h + dropout(ffnOut))

        return h
    }
}

// MARK: - Weight Loading

extension DownscaledTransformer {
    /// Load weights from a dictionary (SafeTensors format)
    ///
    /// Converted weight structure (camelCase from convert_weights.py):
    /// - {prefix}.inProjector.weight, {prefix}.inProjector.bias
    /// - {prefix}.transformerLayers.{i}.self_attn.*
    /// - {prefix}.transformerLayers.{i}.linear1.*, linear2.*, norm1.*, norm2.*
    /// - {prefix}.outProjector.{0,2,4}.*
    public func loadWeights(_ weights: [String: MLXArray], prefix: String) {
        // Load inProjector (camelCase)
        if let w = weights["\(prefix).inProjector.weight"],
           let b = weights["\(prefix).inProjector.bias"] {
            update(module: inProjector, weight: w, bias: b)
        }

        // Load transformer layers (camelCase: transformerLayers)
        for (i, layer) in transformerLayers.enumerated() {
            let layerPrefix = "\(prefix).transformerLayers.\(i)"

            // Self-attention
            // PyTorch stores Q, K, V in a combined in_proj_weight [3*dim, dim]
            // We need to handle both combined format and split format
            if let inProj = weights["\(layerPrefix).self_attn.in_proj_weight"],
               let inProjBias = weights["\(layerPrefix).self_attn.in_proj_bias"] {
                // Split into Q, K, V
                let dim = hiddenSize
                let qWeight = inProj[0..<dim]
                let kWeight = inProj[dim..<(2*dim)]
                let vWeight = inProj[(2*dim)..<(3*dim)]

                let qBias = inProjBias[0..<dim]
                let kBias = inProjBias[dim..<(2*dim)]
                let vBias = inProjBias[(2*dim)..<(3*dim)]

                // Use update helpers for attention projections
                updateLinearWeights(layer.selfAttn.queryProj, weight: qWeight, bias: qBias)
                updateLinearWeights(layer.selfAttn.keyProj, weight: kWeight, bias: kBias)
                updateLinearWeights(layer.selfAttn.valueProj, weight: vWeight, bias: vBias)
            }
            // Handle pre-split format (from convert_weights.py)
            else if let qWeight = weights["\(layerPrefix).self_attn.q_proj.weight"],
                    let kWeight = weights["\(layerPrefix).self_attn.k_proj.weight"],
                    let vWeight = weights["\(layerPrefix).self_attn.v_proj.weight"] {
                updateLinearWeights(layer.selfAttn.queryProj, weight: qWeight, bias: weights["\(layerPrefix).self_attn.q_proj.bias"])
                updateLinearWeights(layer.selfAttn.keyProj, weight: kWeight, bias: weights["\(layerPrefix).self_attn.k_proj.bias"])
                updateLinearWeights(layer.selfAttn.valueProj, weight: vWeight, bias: weights["\(layerPrefix).self_attn.v_proj.bias"])
            }

            if let outProj = weights["\(layerPrefix).self_attn.out_proj.weight"] {
                updateLinearWeights(layer.selfAttn.outProj, weight: outProj, bias: weights["\(layerPrefix).self_attn.out_proj.bias"])
            }

            // FFN
            if let w1 = weights["\(layerPrefix).linear1.weight"],
               let b1 = weights["\(layerPrefix).linear1.bias"] {
                update(module: layer.linear1, weight: w1, bias: b1)
            }
            if let w2 = weights["\(layerPrefix).linear2.weight"],
               let b2 = weights["\(layerPrefix).linear2.bias"] {
                update(module: layer.linear2, weight: w2, bias: b2)
            }

            // Layer norms
            if let w = weights["\(layerPrefix).norm1.weight"],
               let b = weights["\(layerPrefix).norm1.bias"] {
                update(module: layer.norm1, weight: w, bias: b)
            }
            if let w = weights["\(layerPrefix).norm2.weight"],
               let b = weights["\(layerPrefix).norm2.bias"] {
                update(module: layer.norm2, weight: w, bias: b)
            }
        }

        // Load outProjector MLP (camelCase)
        // Structure: Sequential([Linear, ReLU, Linear, ReLU, Linear])
        // Python indices: 0, 2, 4 are Linear layers (ReLU at 1, 3)
        // Swift Sequential indices: 0=Linear, 1=ReLU, 2=Linear, 3=ReLU, 4=Linear

        // First Linear: 896 -> 768
        if let w0 = weights["\(prefix).outProjector.0.weight"],
           let b0 = weights["\(prefix).outProjector.0.bias"],
           let linear0 = outProjector.layers[0] as? Linear {
            update(module: linear0, weight: w0, bias: b0)
        }

        // Second Linear: 768 -> 768 (Python index 2, Swift index 2)
        if let w2 = weights["\(prefix).outProjector.2.weight"],
           let b2 = weights["\(prefix).outProjector.2.bias"],
           let linear2 = outProjector.layers[2] as? Linear {
            update(module: linear2, weight: w2, bias: b2)
        }

        // Third Linear: 768 -> 768 (Python index 4, Swift index 4)
        if let w4 = weights["\(prefix).outProjector.4.weight"],
           let b4 = weights["\(prefix).outProjector.4.bias"],
           let linear4 = outProjector.layers[4] as? Linear {
            update(module: linear4, weight: w4, bias: b4)
        }
    }

    private func update(module: Linear, weight: MLXArray, bias: MLXArray?) {
        // Use MLX's module parameter update mechanism
        updateLinearWeights(module, weight: weight, bias: bias)
    }

    private func update(module: LayerNorm, weight: MLXArray, bias: MLXArray?) {
        // Use MLX's module parameter update mechanism
        var params: [String: MLXArray] = [:]
        params["weight"] = weight
        if let bias = bias {
            params["bias"] = bias
        }
        module.update(parameters: ModuleParameters.unflattened(params))
    }
}
