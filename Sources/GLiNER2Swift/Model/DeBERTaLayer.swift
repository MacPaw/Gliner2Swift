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
// DeBERTaLayer.swift
// Single DeBERTa encoder layer
//
// Matches Python: transformers/models/deberta_v2/modeling_deberta_v2.py:DebertaLayer
//
// Architecture (POST-norm, not pre-norm):
//   attention_output = DisentangledSelfAttention(x, rel_embeddings)
//   attention_output = attention_output_dense(attention_output)
//   attention_output = dropout(attention_output)
//   x = LayerNorm(x + attention_output)
//
//   intermediate = intermediate_dense(x)
//   intermediate = gelu(intermediate)
//
//   output = output_dense(intermediate)
//   output = dropout(output)
//   x = LayerNorm(x + output)

import MLX
import MLXNN

/// Single DeBERTa encoder layer.
///
/// Contains:
/// - Disentangled self-attention
/// - Attention output projection + LayerNorm
/// - Feed-forward network (intermediate + output) + LayerNorm
///
/// CRITICAL: Uses POST-norm (add then norm), not PRE-norm!
/// CRITICAL: LayerNorm eps must be 1e-7!
public class DeBERTaLayer: Module {
    /// Hidden size
    public let hiddenSize: Int

    /// Intermediate (FFN) size
    public let intermediateSize: Int

    /// Disentangled self-attention
    public let attention: DisentangledSelfAttention

    /// Attention output dense
    @ModuleInfo public var attentionOutputDense: Linear

    /// Attention output LayerNorm
    public let attentionLayerNorm: LayerNorm

    /// Intermediate (FFN first) dense
    @ModuleInfo public var intermediateDense: Linear

    /// Output (FFN second) dense
    @ModuleInfo public var outputDense: Linear

    /// Output LayerNorm
    public let outputLayerNorm: LayerNorm

    /// Dropout
    public let dropout: Dropout

    /// Initialize DeBERTa layer
    ///
    /// - Parameters:
    ///   - hiddenSize: Hidden dimension (768)
    ///   - intermediateSize: FFN intermediate dimension (3072)
    ///   - numHeads: Number of attention heads (12)
    ///   - positionBuckets: Number of position buckets (256)
    ///   - maxPosition: Maximum relative position (512)
    ///   - dropoutProb: Dropout probability (0.1)
    ///   - layerNormEps: LayerNorm epsilon (1e-7)
    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        numHeads: Int,
        positionBuckets: Int = 256,
        maxPosition: Int = 512,
        dropoutProb: Float = 0.1,
        layerNormEps: Float = 1e-7
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize

        // Disentangled attention
        self.attention = DisentangledSelfAttention(
            hiddenSize: hiddenSize,
            numHeads: numHeads,
            positionBuckets: positionBuckets,
            maxPosition: maxPosition,
            dropoutProb: dropoutProb
        )

        // Attention output projection
        self.attentionOutputDense = Linear(hiddenSize, hiddenSize)

        // Attention LayerNorm (CRITICAL: eps=1e-7)
        self.attentionLayerNorm = LayerNorm(dimensions: hiddenSize, eps: layerNormEps)

        // FFN layers
        self.intermediateDense = Linear(hiddenSize, intermediateSize)
        self.outputDense = Linear(intermediateSize, hiddenSize)

        // Output LayerNorm (CRITICAL: eps=1e-7)
        self.outputLayerNorm = LayerNorm(dimensions: hiddenSize, eps: layerNormEps)

        // Dropout
        self.dropout = Dropout(p: dropoutProb)
    }

    /// Forward pass
    ///
    /// - Parameters:
    ///   - hiddenStates: Input [batch, seq, hidden]
    ///   - relEmbeddings: Relative position embeddings [num_buckets, hidden]
    ///   - attentionMask: Optional attention mask
    /// - Returns: Output [batch, seq, hidden]
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        relEmbeddings: MLXArray,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        // Self-attention
        var attentionOutput = attention(
            hiddenStates,
            relEmbeddings: relEmbeddings,
            attentionMask: attentionMask
        )

        // Attention output projection + residual + norm
        attentionOutput = attentionOutputDense(attentionOutput)
        attentionOutput = dropout(attentionOutput)
        let hiddenStatesAfterAttention = attentionLayerNorm(hiddenStates + attentionOutput)

        // FFN: intermediate
        var intermediate = intermediateDense(hiddenStatesAfterAttention)
        intermediate = gelu(intermediate)

        // FFN: output
        var output = outputDense(intermediate)
        output = dropout(output)

        // Residual + LayerNorm
        output = outputLayerNorm(hiddenStatesAfterAttention + output)

        return output
    }
}

// MARK: - GELU Activation

/// GELU activation function (Gaussian Error Linear Units)
/// Uses MLX's built-in GELU which computes the exact formula:
/// gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
///
/// CRITICAL: DeBERTa uses exact GELU (with erf), NOT the tanh approximation!
/// The tanh approximation can introduce cumulative numerical differences.
private func gelu(_ x: MLXArray) -> MLXArray {
    // Use MLX's built-in exact GELU
    return MLXNN.gelu(x)
}

// MARK: - Weight Loading

extension DeBERTaLayer {
    /// Load weights from a dictionary
    ///
    /// Expected structure (DeBERTa v2 naming):
    /// - {prefix}.attention.self.query_proj.weight/bias
    /// - {prefix}.attention.self.key_proj.weight/bias
    /// - {prefix}.attention.self.value_proj.weight/bias
    /// - {prefix}.attention.output.dense.weight/bias
    /// - {prefix}.attention.output.LayerNorm.weight/bias
    /// - {prefix}.intermediate.dense.weight/bias
    /// - {prefix}.output.dense.weight/bias
    /// - {prefix}.output.LayerNorm.weight/bias
    public func loadWeights(_ weights: [String: MLXArray], prefix: String) {
        let p = prefix.isEmpty ? "" : "\(prefix)."

        // Helper to update a Linear module
        func updateLinear(_ module: Linear, weightKey: String, biasKey: String) {
            var params: [String: MLXArray] = [:]
            if let w = weights[weightKey] { params["weight"] = w }
            if let b = weights[biasKey] { params["bias"] = b }
            if !params.isEmpty {
                module.update(parameters: ModuleParameters.unflattened(params))
            }
        }

        // Helper to update a LayerNorm module
        func updateLayerNorm(_ module: LayerNorm, weightKey: String, biasKey: String) {
            var params: [String: MLXArray] = [:]
            if let w = weights[weightKey] { params["weight"] = w }
            if let b = weights[biasKey] { params["bias"] = b }
            if !params.isEmpty {
                module.update(parameters: ModuleParameters.unflattened(params))
            }
        }

        // Load attention weights
        attention.loadWeights(weights, prefix: "\(p)attention.self")

        // Attention output dense
        updateLinear(attentionOutputDense,
                     weightKey: "\(p)attention.output.dense.weight",
                     biasKey: "\(p)attention.output.dense.bias")

        // Attention output LayerNorm
        updateLayerNorm(attentionLayerNorm,
                        weightKey: "\(p)attention.output.LayerNorm.weight",
                        biasKey: "\(p)attention.output.LayerNorm.bias")

        // Intermediate dense
        updateLinear(intermediateDense,
                     weightKey: "\(p)intermediate.dense.weight",
                     biasKey: "\(p)intermediate.dense.bias")

        // Output dense
        updateLinear(outputDense,
                     weightKey: "\(p)output.dense.weight",
                     biasKey: "\(p)output.dense.bias")

        // Output LayerNorm
        updateLayerNorm(outputLayerNorm,
                        weightKey: "\(p)output.LayerNorm.weight",
                        biasKey: "\(p)output.LayerNorm.bias")
    }
}
