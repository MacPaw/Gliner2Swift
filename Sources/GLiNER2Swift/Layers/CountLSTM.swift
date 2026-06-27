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
// CountLSTM.swift
// Count-aware embedding modules for structure extraction
//
// Matches Python: gliner2/layers.py:CountLSTM, CountLSTMv2

import MLX
import MLXNN

/// Count-aware label projection. Two backbone variants exist (`count_lstm`
/// projector vs `count_lstm_v2` transformer) selected by the model config; both
/// turn label embeddings into count-aware structure projections for span scoring.
public protocol CountEmbedding: Module {
    func callAsFunction(_ pcEmb: MLXArray, goldCountVal: Int) -> MLXArray
    func loadWeights(_ weights: [String: MLXArray], prefix: String)
}

// MARK: - CountLSTM (Basic Version)

/// Basic CountLSTM using GRU with MLP projector.
///
/// Architecture:
///   pos_seq = pos_embedding(count_indices)  # (L, D)
///   pos_seq = pos_seq.unsqueeze(1).expand(L, M, D)
///   h0 = pc_emb.unsqueeze(0)  # (1, M, D)
///   output, _ = gru(pos_seq, h0)  # (L, M, D)
///   return projector(cat([output, pc_emb.expand], dim=-1))  # (L, M, D)
public class CountLSTM: Module, CountEmbedding {
    public let hiddenSize: Int
    public let maxCount: Int

    /// Positional embedding for count indices
    public let posEmbedding: Embedding

    /// GRU layer
    public let gru: GRU

    /// Output projector MLP: 2*hidden → 4*hidden → hidden
    public let projector: Sequential

    public init(hiddenSize: Int, maxCount: Int = 20) {
        self.hiddenSize = hiddenSize
        self.maxCount = maxCount

        self.posEmbedding = Embedding(embeddingCount: maxCount, dimensions: hiddenSize)
        self.gru = GRU(inputSize: hiddenSize, hiddenSize: hiddenSize)
        self.projector = createMLP(
            inputDim: hiddenSize * 2,
            intermediateDims: [hiddenSize * 4],
            outputDim: hiddenSize,
            dropout: 0.0,
            activation: .relu,
            addLayerNorm: false
        )
    }

    /// Forward pass
    ///
    /// - Parameters:
    ///   - pcEmb: Field embeddings [M, hiddenSize]
    ///   - goldCountVal: Number of count steps
    /// - Returns: Count-aware structure embeddings [goldCountVal, M, hiddenSize]
    public func callAsFunction(_ pcEmb: MLXArray, goldCountVal: Int) -> MLXArray {
        let M = pcEmb.dim(0)
        let D = pcEmb.dim(1)

        // Cap count value
        let count = min(goldCountVal, maxCount)
        guard count > 0 else {
            return MLXArray.zeros([0, M, D])
        }

        // Get positional embeddings: (count,) -> (count, D)
        let countIndices = MLXArray(Array(0..<count))
        var posSeq = posEmbedding(countIndices)

        // Expand over batch dimension: (count, D) -> (count, M, D)
        posSeq = posSeq.expandedDimensions(axis: 1)
        posSeq = MLX.broadcast(posSeq, to: [count, M, D])

        // Initialize GRU hidden state: (M, D) -> (1, M, D)
        let h0 = pcEmb.expandedDimensions(axis: 0)

        // Run GRU: (count, M, D), (1, M, D) -> (count, M, D)
        let (output, _) = gru(posSeq, h0: h0)

        // Expand pcEmb and concatenate: (count, M, 2*D)
        let pcBroadcast = MLX.broadcast(h0, to: [count, M, D])
        let concatenated = MLX.concatenated([output, pcBroadcast], axis: -1)

        // Project: (count, M, 2*D) -> (count, M, D)
        return projector(concatenated)
    }
}

// MARK: - CountLSTMv2 (with DownscaledTransformer)

/// CountLSTMv2 using GRU with DownscaledTransformer.
///
/// CRITICAL DIFFERENCE from CountLSTM:
/// - Uses ADDITION instead of concatenation!
/// - Uses DownscaledTransformer instead of MLP projector
///
/// Architecture:
///   pos_seq = pos_embedding(count_indices)
///   pos_seq = pos_seq.unsqueeze(1).expand(L, M, D)
///   h0 = pc_emb.unsqueeze(0)
///   output, _ = gru(pos_seq, h0)           # (L, M, D)
///   pc_broadcast = pc_emb.unsqueeze(0).expand_as(output)
///   return transformer(output + pc_broadcast)  # ← ADDITION here!
public class CountLSTMv2: Module, CountEmbedding {
    public let hiddenSize: Int
    public let maxCount: Int

    /// Positional embedding for count indices
    public let posEmbedding: Embedding

    /// GRU layer
    public let gru: GRU

    /// DownscaledTransformer for output processing
    public let transformer: DownscaledTransformer

    public init(hiddenSize: Int, maxCount: Int = 20) {
        self.hiddenSize = hiddenSize
        self.maxCount = maxCount

        self.posEmbedding = Embedding(embeddingCount: maxCount, dimensions: hiddenSize)
        self.gru = GRU(inputSize: hiddenSize, hiddenSize: hiddenSize)
        self.transformer = DownscaledTransformer(
            inputSize: hiddenSize,
            hiddenSize: 128,
            numHeads: 4,
            numLayers: 2,
            dropout: 0.1
        )
    }

    /// Forward pass
    ///
    /// - Parameters:
    ///   - pcEmb: Field embeddings [M, hiddenSize]
    ///   - goldCountVal: Number of count steps
    /// - Returns: Count-aware structure embeddings [goldCountVal, M, hiddenSize]
    public func callAsFunction(_ pcEmb: MLXArray, goldCountVal: Int) -> MLXArray {
        let M = pcEmb.dim(0)
        let D = pcEmb.dim(1)

        // Cap count value
        let count = min(goldCountVal, maxCount)
        guard count > 0 else {
            return MLXArray.zeros([0, M, D])
        }

        // Get positional embeddings: (count,) -> (count, D)
        let countIndices = MLXArray(Array(0..<count))
        var posSeq = posEmbedding(countIndices)

        // Expand over batch dimension: (count, D) -> (count, M, D)
        posSeq = posSeq.expandedDimensions(axis: 1)
        posSeq = MLX.broadcast(posSeq, to: [count, M, D])

        // Initialize GRU hidden state: (M, D) -> (1, M, D)
        let h0 = pcEmb.expandedDimensions(axis: 0)

        // Run GRU: (count, M, D), (1, M, D) -> (count, M, D)
        let (output, _) = gru(posSeq, h0: h0)

        // CRITICAL: Use ADDITION, not concatenation!
        // Expand pcEmb and add: (count, M, D)
        let pcBroadcast = MLX.broadcast(h0, to: [count, M, D])
        let combined = output + pcBroadcast

        // Apply transformer: (count, M, D) -> (count, M, D)
        return transformer(combined)
    }
}

// MARK: - Weight Loading

extension CountLSTM {
    /// Load weights from a dictionary (SafeTensors format)
    /// Note: CountLSTM (basic) is not used by gliner2-base-v1, which uses CountLSTMv2
    public func loadWeights(_ weights: [String: MLXArray], prefix: String) {
        // Load positional embedding (camelCase)
        loadEmbeddingFromDict(posEmbedding, weights: weights, prefix: "\(prefix).posEmbedding")

        // Load GRU weights
        gru.loadWeights(weights, prefix: "\(prefix).gru")

        // Load projector MLP
        // Projector structure: Linear(2*D, 4*D) -> ReLU -> Linear(4*D, D)
        if let linear0 = projector.layers[0] as? Linear {
            updateLinearWeights(
                linear0,
                weight: weights["\(prefix).projector.0.weight"],
                bias: weights["\(prefix).projector.0.bias"]
            )
        }
        if let linear2 = projector.layers[2] as? Linear {
            updateLinearWeights(
                linear2,
                weight: weights["\(prefix).projector.2.weight"],
                bias: weights["\(prefix).projector.2.bias"]
            )
        }
    }
}

extension CountLSTMv2 {
    /// Load weights from a dictionary (SafeTensors format)
    ///
    /// Converted weight structure (camelCase):
    /// - countEmbed.posEmbedding.weight: [20, 768]
    /// - countEmbed.gru.weightIH: [2304, 768]
    /// - countEmbed.gru.weightHH: [2304, 768]
    /// - countEmbed.gru.biasIH: [2304]
    /// - countEmbed.gru.biasHH: [2304]
    /// - countEmbed.transformer.*
    public func loadWeights(_ weights: [String: MLXArray], prefix: String) {
        // Load positional embedding (camelCase: posEmbedding)
        loadEmbeddingFromDict(posEmbedding, weights: weights, prefix: "\(prefix).posEmbedding")

        // Load GRU weights (GRU.loadWeights handles camelCase)
        gru.loadWeights(weights, prefix: "\(prefix).gru")

        // Load transformer weights
        transformer.loadWeights(weights, prefix: "\(prefix).transformer")
    }
}
