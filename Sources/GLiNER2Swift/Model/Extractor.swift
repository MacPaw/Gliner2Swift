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
// Extractor.swift
// GLiNER2 Extractor model
//
// Matches Python: gliner2/model.py:Extractor

import Foundation
import MLX
import MLXNN

/// GLiNER2 Extractor Model.
///
/// This model accepts PreprocessedBatch for efficient training and inference.
/// Architecture:
/// - Encoder: DeBERTa-v3-base (microsoft/deberta-v3-base) with disentangled attention
/// - Span representation: SpanMarkerV0
/// - Classifier: MLP for classification tasks
/// - Count prediction: MLP for count prediction
/// - Count embedding: CountLSTMv2 for structure extraction
public class Extractor: Module {
    /// Model configuration
    public let config: ExtractorConfig

    /// Maximum span width
    public let maxWidth: Int

    /// Hidden size from encoder
    public let hiddenSize: Int

    /// DeBERTa encoder with disentangled attention
    public let encoder: DeBERTaEncoder

    /// Span representation layer
    public let spanRep: SpanRepLayer

    /// Classifier for classification tasks: hidden → 2*hidden → 1
    public let classifier: Sequential

    /// Count prediction layer: hidden → 2*hidden → 20
    public let countPred: Sequential

    /// Count embedding module (CountLSTMv2 for gliner2-base-v1)
    public let countEmbed: CountLSTMv2

    /// Initialize Extractor from configuration
    ///
    /// - Parameter config: Model configuration
    public init(config: ExtractorConfig) {
        self.config = config
        self.maxWidth = config.maxWidth
        self.hiddenSize = config.hiddenSize

        // Initialize DeBERTa encoder
        let debertaConfig = DeBERTaConfig(
            vocabSize: config.vocabSize,
            hiddenSize: config.hiddenSize,
            numHiddenLayers: 12,
            numAttentionHeads: 12,
            intermediateSize: 3072,
            positionBuckets: 256,
            maxPositionEmbeddings: 512,
            hiddenDropoutProb: 0.1,
            attentionDropoutProb: 0.1,
            layerNormEps: 1e-7,
            posAttType: ["c2p", "p2c"]
        )
        self.encoder = DeBERTaEncoder(config: debertaConfig)

        // Initialize span representation layer
        self.spanRep = SpanRepLayer(
            hiddenSize: hiddenSize,
            maxWidth: maxWidth,
            spanMode: .markerV0,
            dropout: config.spanDropout
        )

        // Initialize classifier: hidden → 2*hidden → 1
        self.classifier = createMLP(
            inputDim: hiddenSize,
            intermediateDims: [hiddenSize * 2],
            outputDim: 1,
            dropout: 0.0,
            activation: .relu,
            addLayerNorm: false
        )

        // Initialize count prediction: hidden → 2*hidden → 20
        self.countPred = createMLP(
            inputDim: hiddenSize,
            intermediateDims: [hiddenSize * 2],
            outputDim: config.maxCount,
            dropout: 0.0,
            activation: .relu,
            addLayerNorm: false
        )

        // Initialize count embedding
        self.countEmbed = CountLSTMv2(
            hiddenSize: hiddenSize,
            maxCount: config.maxCount
        )
    }

    // MARK: - Encoder Forward

    /// Run encoder on input IDs
    ///
    /// - Parameters:
    ///   - inputIds: Token IDs [batch, seq_len]
    ///   - attentionMask: Optional attention mask [batch, seq_len]
    /// - Returns: Encoder output with hidden states
    public func encode(
        _ inputIds: MLXArray,
        attentionMask: MLXArray? = nil
    ) -> DeBERTaEncoderOutput {
        encoder(inputIds, attentionMask: attentionMask)
    }

    // MARK: - Span Representation

    /// Compute span representations for token embeddings.
    ///
    /// - Parameters:
    ///   - tokenEmbeddings: Token embeddings [textLen, hidden]
    ///   - debug: If true, print intermediate values for debugging
    /// - Returns: Dictionary with span_rep, spans_idx, and span_mask
    public func computeSpanRep(_ tokenEmbeddings: MLXArray, debug: Bool = false) -> SpanInfo {
        let textLength = tokenEmbeddings.dim(0)

        // Build span indices: (start, end) for each position and width
        var spansIdx: [(Int, Int)] = []
        for i in 0..<textLength {
            for j in 0..<maxWidth {
                if i + j < textLength {
                    spansIdx.append((i, i + j))
                } else {
                    spansIdx.append((-1, -1))  // Invalid span
                }
            }
        }

        // Convert to MLXArray [1, numSpans, 2]
        let flatSpans = spansIdx.flatMap { [$0.0, $0.1] }
        var spanIdxArray = MLXArray(flatSpans.map { Int32($0) })
        spanIdxArray = spanIdxArray.reshaped([1, spansIdx.count, 2])

        // Create span mask: true for invalid spans
        let startInvalid = MLX.equal(spanIdxArray[0..., 0..., 0], MLXArray(Int32(-1)))
        let endInvalid = MLX.equal(spanIdxArray[0..., 0..., 1], MLXArray(Int32(-1)))
        let spanMask = MLX.logicalOr(startInvalid, endInvalid)

        // Replace invalid indices with (0, 0) for safe indexing
        let safeSpans = MLX.where(
            spanMask.expandedDimensions(axis: -1),
            MLXArray.zeros([1, spansIdx.count, 2]),
            spanIdxArray
        )

        // Compute span representations
        let tokenEmbsBatched = tokenEmbeddings.expandedDimensions(axis: 0)  // [1, textLen, hidden]
        var spanRepResult = spanRep(tokenEmbsBatched, spanIdx: safeSpans, debug: debug)  // [1, textLen, maxWidth, hidden]
        spanRepResult = spanRepResult.squeezed(axis: 0)  // [textLen, maxWidth, hidden]

        // Reshape to [numSpans, hidden]
        let numSpans = textLength * maxWidth
        spanRepResult = spanRepResult.reshaped([numSpans, hiddenSize])

        return SpanInfo(
            spanRep: spanRepResult,
            spansIdx: spanIdxArray,
            spanMask: spanMask
        )
    }

    // MARK: - Inference Helpers

    /// Extract scores for a schema
    ///
    /// - Parameters:
    ///   - spanInfo: Span representation info
    ///   - schemaEmb: Schema embeddings [numFields+1, hidden]
    ///   - predCount: Predicted count value
    /// - Returns: Span scores [predCount, numFields, numSpans, maxWidth]
    public func computeSpanScores(
        spanInfo: SpanInfo,
        schemaEmb: MLXArray,
        predCount: Int
    ) -> MLXArray {
        // Get field embeddings (skip [P] token)
        let fieldEmbs = schemaEmb[1...]

        // Get count-aware structure projections
        let structProj = countEmbed(fieldEmbs, goldCountVal: predCount)  // [count, fields, hidden]

        // Compute scores via einsum: 'lkd,bpd->bplk'
        // spanRep: [L*maxWidth, D] -> reshape to [L, maxWidth, D]
        // structProj: [count, fields, D]
        let L = spanInfo.spansIdx.dim(1) / maxWidth
        let spanRepReshaped = spanInfo.spanRep.reshaped([L, maxWidth, hiddenSize])

        // Einsum: scores[b,p,l,k] = sum_d(spanRep[l,k,d] * structProj[b,p,d])
        // We need [count, fields, L, maxWidth]
        var scores = MLX.einsum("lkd,cpd->cplk", spanRepReshaped, structProj)

        // Apply sigmoid
        scores = MLX.sigmoid(scores)

        return scores
    }
}

// MARK: - Span Info

/// Container for span representation information
public struct SpanInfo {
    /// Span representations [numSpans, hidden]
    public let spanRep: MLXArray

    /// Span indices [1, numSpans, 2]
    public let spansIdx: MLXArray

    /// Span mask [1, numSpans] - true for invalid spans
    public let spanMask: MLXArray
}

// MARK: - Weight Loading

extension Extractor {

    // MARK: - Format Detection

    /// Detect whether weights are in raw PyTorch format (snake_case keys)
    /// vs pre-converted format (camelCase keys from convert_weights.py).
    static func isRawPyTorchFormat(_ weights: [String: MLXArray]) -> Bool {
        weights.keys.contains(where: { $0.hasPrefix("span_rep.") })
    }

    // MARK: - Key Sanitization

    /// Key mappings from raw PyTorch snake_case to converted camelCase format.
    /// Ordered longest-prefix-first to avoid false prefix matches.
    private static let keyMappings: [(String, String)] = [
        // Span representation
        ("span_rep.span_rep_layer.project_start", "spanRep.spanRepLayer.projectStart"),
        ("span_rep.span_rep_layer.project_end", "spanRep.spanRepLayer.projectEnd"),
        ("span_rep.span_rep_layer.out_project", "spanRep.spanRepLayer.outProject"),
        // DownscaledTransformer (before shorter count_embed prefixes)
        ("count_embed.transformer.transformer.layers", "countEmbed.transformer.transformerLayers"),
        ("count_embed.transformer.in_projector", "countEmbed.transformer.inProjector"),
        ("count_embed.transformer.out_projector", "countEmbed.transformer.outProjector"),
        // GRU keys (full terminal keys, no suffix)
        ("count_embed.gru.weight_ih_l0", "countEmbed.gru.weightIH"),
        ("count_embed.gru.weight_hh_l0", "countEmbed.gru.weightHH"),
        ("count_embed.gru.bias_ih_l0", "countEmbed.gru.biasIH"),
        ("count_embed.gru.bias_hh_l0", "countEmbed.gru.biasHH"),
        // Count embedding
        ("count_embed.pos_embedding", "countEmbed.posEmbedding"),
        // Classifier MLP (index 2→1, skipping ReLU)
        ("classifier.2", "classifier.layers.1"),
        ("classifier.0", "classifier.layers.0"),
        // Count prediction MLP (same index shift)
        ("count_pred.2", "countPred.layers.1"),
        ("count_pred.0", "countPred.layers.0"),
    ]

    /// Remap raw PyTorch weight keys to the converted format expected by loadWeights methods.
    ///
    /// If the weights are already in converted format (detected by absence of `span_rep.` keys),
    /// they are returned unchanged. Encoder weights (`encoder.*`) pass through unmodified.
    ///
    /// This eliminates the need for the Python `convert_weights.py` script.
    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        guard isRawPyTorchFormat(weights) else { return weights }

        var result: [String: MLXArray] = [:]
        for (key, value) in weights {
            result[mapRawKey(key)] = value
        }
        return result
    }

    /// Map a single raw PyTorch key into the converted key space.
    ///
    /// Encoder keys are identical in both spaces and pass through unchanged. Works for any
    /// suffix, so it also maps LoRA keys (`.lora_A` / `.lora_B`), not just `.weight`/`.bias`.
    static func mapRawKey(_ key: String) -> String {
        // Encoder weights pass through unchanged
        if key.hasPrefix("encoder.") { return key }
        // Apply first matching prefix replacement
        for (prefix, replacement) in keyMappings where key.hasPrefix(prefix) {
            return replacement + key.dropFirst(prefix.count)
        }
        return key
    }

    // MARK: - Weight Loading

    /// Load all weights from a single combined SafeTensors file.
    ///
    /// Auto-detects whether the file contains raw PyTorch keys or pre-converted keys.
    /// Both formats are supported transparently.
    ///
    /// - Parameter url: URL to model.safetensors (combined weights file)
    public func loadWeights(from url: URL) throws {
        let rawWeights = try loadArrays(url: url)
        let weights = Extractor.sanitize(weights: rawWeights)

        // Load encoder weights (keys starting with "encoder.")
        encoder.loadWeights(weights, prefix: "encoder")

        // Load model-specific weights (spanRep, classifier, countPred, countEmbed)
        loadModelWeights(weights)
    }

    /// Load all weights from separate SafeTensors files (legacy method)
    ///
    /// - Parameters:
    ///   - modelWeightsUrl: URL to gliner2_weights.safetensors
    ///   - encoderWeightsUrl: URL to encoder_weights.safetensors
    public func loadWeights(modelWeightsUrl: URL, encoderWeightsUrl: URL) throws {
        // Load model weights
        let rawModelWeights = try loadArrays(url: modelWeightsUrl)
        let modelWeights = Extractor.sanitize(weights: rawModelWeights)
        loadModelWeights(modelWeights)

        // Load encoder weights
        let encoderWeights = try loadArrays(url: encoderWeightsUrl)
        encoder.loadWeights(encoderWeights, prefix: "encoder")
    }

    /// Load GLiNER2 model weights (excluding encoder)
    ///
    /// - Parameter weights: Dictionary mapping parameter names to arrays
    public func loadModelWeights(_ weights: [String: MLXArray]) {
        // Load span representation weights
        // Converted weights use camelCase: spanRep.spanRepLayer.*
        spanRep.loadWeights(weights, prefix: "spanRep")

        // Load classifier weights
        // Converted weights use: classifier.layers.{0,1}.weight/bias
        loadMLPWeights(classifier, weights: weights, prefix: "classifier.layers")

        // Load count prediction weights
        // Converted weights use: countPred.layers.{0,1}.weight/bias
        loadMLPWeights(countPred, weights: weights, prefix: "countPred.layers")

        // Load count embedding weights
        // Converted weights use: countEmbed.*
        countEmbed.loadWeights(weights, prefix: "countEmbed")
    }

    /// Load encoder weights separately
    ///
    /// - Parameter weights: Dictionary with encoder weights
    public func loadEncoderWeights(_ weights: [String: MLXArray]) {
        encoder.loadWeights(weights, prefix: "encoder")
    }

    /// Load base weights with LoRA adapter merged in.
    ///
    /// Loads base weights, applies LoRA deltas (W += B@A * alpha/r),
    /// then sanitizes keys and loads merged weights into the model.
    ///
    /// - Parameters:
    ///   - baseWeightsUrl: URL to model.safetensors (base model)
    ///   - adapterPath: Directory containing adapter_config.json + adapter_weights.safetensors
    public func loadWeightsWithLoRA(baseWeightsUrl: URL, adapterPath: URL) throws {
        // 1. Load raw base weights (Python keys, before sanitization)
        var rawWeights = try loadArrays(url: baseWeightsUrl)

        // 2. Load adapter config and weights
        let config = try LoRAAdapterConfig.load(from: adapterPath)
        let adapterUrl = adapterPath.appendingPathComponent("adapter_weights.safetensors")
        let adapterWeights = try loadArrays(url: adapterUrl)

        // 3. Merge LoRA deltas into base weights. `mergeLoRAWeights` accepts either key
        //    space for the base checkpoint (raw snake_case or converted camelCase).
        mergeLoRAWeights(into: &rawWeights, adapterWeights: adapterWeights, config: config)

        // 4. Sanitize and load (reuses existing code path)
        let weights = Extractor.sanitize(weights: rawWeights)
        encoder.loadWeights(weights, prefix: "encoder")
        loadModelWeights(weights)
    }

    private func loadMLPWeights(_ mlp: Sequential, weights: [String: MLXArray], prefix: String) {
        // MLP structure: Linear → ReLU → Linear
        // Swift Sequential layers array: [0]=Linear, [1]=ReLU, [2]=Linear
        //
        // Weight mapping (convert_weights.py remaps PyTorch indices):
        //   classifier.0.weight → classifier.layers.0.weight (first Linear)
        //   classifier.2.weight → classifier.layers.1.weight (second Linear)
        //
        // So with prefix "classifier.layers", we look for .0 and .1

        if let linear0 = mlp.layers[0] as? Linear {
            updateLinearWeights(
                linear0,
                weight: weights["\(prefix).0.weight"],
                bias: weights["\(prefix).0.bias"]
            )
        }

        if let linear2 = mlp.layers[2] as? Linear {
            // Note: converted weights use .1 for second linear, not .2
            updateLinearWeights(
                linear2,
                weight: weights["\(prefix).1.weight"],
                bias: weights["\(prefix).1.bias"]
            )
        }
    }
}
