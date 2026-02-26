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
// DeBERTaEmbeddings.swift
// Word embeddings for DeBERTa encoder
//
// Matches Python: transformers/models/deberta_v2/modeling_deberta_v2.py:DebertaV2Embeddings
//
// DeBERTa v3 uses word embeddings only (no token_type_embeddings).
// Position information is handled separately via relative position embeddings
// in the disentangled attention mechanism.

import MLX
import MLXNN

/// DeBERTa v2/v3 Embeddings layer.
///
/// Architecture:
///   word_embeddings(input_ids)  -> [batch, seq, hidden]
///   LayerNorm(eps=1e-7)         -> [batch, seq, hidden]
///   Dropout                     -> [batch, seq, hidden]
///
/// Note: DeBERTa uses relative position embeddings in attention layers,
/// not absolute position embeddings in the embedding layer.
public class DeBERTaEmbeddings: Module {
    /// Word embedding table
    public let wordEmbeddings: Embedding

    /// Layer normalization (CRITICAL: eps=1e-7 for DeBERTa)
    public let layerNorm: LayerNorm

    /// Dropout for regularization
    public let dropout: Dropout

    /// Hidden size
    public let hiddenSize: Int

    /// Vocabulary size
    public let vocabSize: Int

    /// Initialize DeBERTa embeddings
    ///
    /// - Parameters:
    ///   - vocabSize: Size of vocabulary (128011 for gliner2-base-v1)
    ///   - hiddenSize: Hidden dimension (768 for base model)
    ///   - dropoutProb: Dropout probability (default: 0.1)
    ///   - layerNormEps: LayerNorm epsilon (default: 1e-7 for DeBERTa)
    public init(
        vocabSize: Int,
        hiddenSize: Int,
        dropoutProb: Float = 0.1,
        layerNormEps: Float = 1e-7
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize

        self.wordEmbeddings = Embedding(embeddingCount: vocabSize, dimensions: hiddenSize)
        self.layerNorm = LayerNorm(dimensions: hiddenSize, eps: layerNormEps)
        self.dropout = Dropout(p: dropoutProb)
    }

    /// Forward pass
    ///
    /// - Parameter inputIds: Token IDs [batch, seq_len]
    /// - Returns: Embeddings [batch, seq_len, hidden_size]
    public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        // Look up word embeddings
        var embeddings = wordEmbeddings(inputIds)

        // Apply layer normalization
        embeddings = layerNorm(embeddings)

        // Apply dropout
        embeddings = dropout(embeddings)

        return embeddings
    }
}

// MARK: - Weight Loading

extension DeBERTaEmbeddings {
    /// Load weights from a dictionary
    ///
    /// Expected keys (with prefix):
    /// - {prefix}.word_embeddings.weight: [vocab_size, hidden_size]
    /// - {prefix}.LayerNorm.weight: [hidden_size]
    /// - {prefix}.LayerNorm.bias: [hidden_size]
    ///
    /// - Parameters:
    ///   - weights: Weight dictionary
    ///   - prefix: Key prefix (e.g., "embeddings")
    public func loadWeights(_ weights: [String: MLXArray], prefix: String) {
        let p = prefix.isEmpty ? "" : "\(prefix)."

        // Load word embeddings
        loadEmbeddingFromDict(wordEmbeddings, weights: weights, prefix: "\(p)word_embeddings")

        // Load LayerNorm
        loadLayerNormFromDict(layerNorm, weights: weights, prefix: "\(p)LayerNorm")
    }
}
