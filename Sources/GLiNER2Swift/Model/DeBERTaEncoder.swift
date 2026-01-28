// DeBERTaEncoder.swift
// Complete DeBERTa v2/v3 encoder
//
// Matches Python: transformers/models/deberta_v2/modeling_deberta_v2.py:DebertaV2Model
//
// Architecture:
//   embeddings = DeBERTaEmbeddings(input_ids)
//   for layer in encoder_layers:
//       hidden_states = layer(hidden_states, rel_embeddings)
//   hidden_states = encoder_layer_norm(hidden_states)
//
// CRITICAL: rel_embeddings is SHARED across ALL layers!

import MLX
import MLXNN

/// DeBERTa v2/v3 Encoder configuration
public struct DeBERTaConfig {
    /// Vocabulary size (128011 for gliner2-base-v1)
    public var vocabSize: Int

    /// Hidden size (768 for base)
    public var hiddenSize: Int

    /// Number of encoder layers (12 for base)
    public var numHiddenLayers: Int

    /// Number of attention heads (12 for base)
    public var numAttentionHeads: Int

    /// FFN intermediate size (3072 for base)
    public var intermediateSize: Int

    /// Number of position buckets (256)
    public var positionBuckets: Int

    /// Maximum relative position (512)
    public var maxPositionEmbeddings: Int

    /// Hidden dropout probability
    public var hiddenDropoutProb: Float

    /// Attention dropout probability
    public var attentionDropoutProb: Float

    /// LayerNorm epsilon (CRITICAL: 1e-7 for DeBERTa)
    public var layerNormEps: Float

    /// Position attention types
    public var posAttType: Set<String>

    /// Default configuration for gliner2-base-v1
    public static var gliner2Base: DeBERTaConfig {
        DeBERTaConfig(
            vocabSize: 128011,
            hiddenSize: 768,
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
    }

    public init(
        vocabSize: Int = 128011,
        hiddenSize: Int = 768,
        numHiddenLayers: Int = 12,
        numAttentionHeads: Int = 12,
        intermediateSize: Int = 3072,
        positionBuckets: Int = 256,
        maxPositionEmbeddings: Int = 512,
        hiddenDropoutProb: Float = 0.1,
        attentionDropoutProb: Float = 0.1,
        layerNormEps: Float = 1e-7,
        posAttType: Set<String> = ["c2p", "p2c"]
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.intermediateSize = intermediateSize
        self.positionBuckets = positionBuckets
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.hiddenDropoutProb = hiddenDropoutProb
        self.attentionDropoutProb = attentionDropoutProb
        self.layerNormEps = layerNormEps
        self.posAttType = posAttType
    }
}

// MARK: - DeBERTa Encoder

/// Complete DeBERTa v2/v3 encoder model.
///
/// Architecture:
/// - Word embeddings + LayerNorm + Dropout
/// - N encoder layers with disentangled attention
/// - Shared relative position embeddings across all layers
/// - Final encoder LayerNorm
public class DeBERTaEncoder: Module {
    /// Model configuration
    public let config: DeBERTaConfig

    /// Word embeddings layer
    public let embeddings: DeBERTaEmbeddings

    /// Encoder layers
    public let layers: [DeBERTaLayer]

    /// Relative position embeddings (SHARED across all layers)
    /// Shape: [position_buckets, hidden_size] = [512, 768]
    public var relEmbeddings: MLXArray

    /// LayerNorm for relative embeddings (NOT final output!)
    /// In DeBERTa v2/v3, this normalizes rel_embeddings, not hidden states
    public let relEmbeddingsLayerNorm: LayerNorm

    /// Initialize DeBERTa encoder
    ///
    /// - Parameter config: Model configuration
    public init(config: DeBERTaConfig = .gliner2Base) {
        self.config = config

        // Embeddings layer
        self.embeddings = DeBERTaEmbeddings(
            vocabSize: config.vocabSize,
            hiddenSize: config.hiddenSize,
            dropoutProb: config.hiddenDropoutProb,
            layerNormEps: config.layerNormEps
        )

        // Encoder layers
        var encoderLayers: [DeBERTaLayer] = []
        for _ in 0..<config.numHiddenLayers {
            encoderLayers.append(DeBERTaLayer(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize,
                numHeads: config.numAttentionHeads,
                positionBuckets: config.positionBuckets,
                maxPosition: config.maxPositionEmbeddings,
                dropoutProb: config.attentionDropoutProb,
                layerNormEps: config.layerNormEps
            ))
        }
        self.layers = encoderLayers

        // Relative position embeddings (initialized, will be loaded from weights)
        // Shape: [max_position_embeddings, hidden_size] = [512, 768]
        self.relEmbeddings = MLXRandom.normal([config.maxPositionEmbeddings, config.hiddenSize])

        // LayerNorm for relative embeddings (NOT for final output!)
        // This normalizes rel_embeddings before they're used in attention
        self.relEmbeddingsLayerNorm = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    /// Forward pass
    ///
    /// - Parameters:
    ///   - inputIds: Token IDs [batch, seq_len]
    ///   - attentionMask: Optional attention mask [batch, seq_len]
    /// - Returns: Encoder output with hidden states and optional attention weights
    public func callAsFunction(
        _ inputIds: MLXArray,
        attentionMask: MLXArray? = nil
    ) -> DeBERTaEncoderOutput {
        // Get embeddings
        var hiddenStates = embeddings(inputIds)

        // Prepare attention mask if provided
        // Convert [batch, seq] -> [batch, 1, 1, seq] with proper values
        var expandedMask: MLXArray? = nil
        if let mask = attentionMask {
            expandedMask = prepareAttentionMask(mask, dtype: hiddenStates.dtype)
        }

        // Apply LayerNorm to relative embeddings (DeBERTa v2/v3 behavior)
        // This is done once before passing to all layers
        let normalizedRelEmbeddings = relEmbeddingsLayerNorm(relEmbeddings)

        // Apply encoder layers
        var allHiddenStates: [MLXArray] = [hiddenStates]

        for layer in layers {
            hiddenStates = layer(
                hiddenStates,
                relEmbeddings: normalizedRelEmbeddings,
                attentionMask: expandedMask
            )
            allHiddenStates.append(hiddenStates)
        }

        // NOTE: DeBERTa does NOT apply final LayerNorm to hidden states!
        // The LayerNorm is only used for relative embeddings

        return DeBERTaEncoderOutput(
            lastHiddenState: hiddenStates,
            hiddenStates: allHiddenStates
        )
    }

    /// Prepare attention mask for DeBERTa
    ///
    /// Converts [batch, seq_len] mask (1 = valid, 0 = masked)
    /// to [batch, 1, 1, seq_len] with large negative values for masked positions
    private func prepareAttentionMask(_ mask: MLXArray, dtype: DType) -> MLXArray {
        // Expand dimensions: [batch, seq] -> [batch, 1, 1, seq]
        let expanded = mask.expandedDimensions(axes: [1, 2])

        // Convert: 1 -> 0.0, 0 -> -10000.0 (large negative for softmax)
        // inverted_mask = (1.0 - mask) * -10000.0
        let invertedMask = (1.0 - expanded.asType(dtype)) * -10000.0

        return invertedMask
    }
}

// MARK: - Encoder Output

/// Output from DeBERTa encoder
public struct DeBERTaEncoderOutput {
    /// Final layer hidden states [batch, seq, hidden]
    public let lastHiddenState: MLXArray

    /// Hidden states from all layers (including embeddings)
    public let hiddenStates: [MLXArray]

    /// Get output at specific layer (0 = embeddings, 1-12 = layers)
    public func hiddenState(at layer: Int) -> MLXArray? {
        guard layer >= 0 && layer < hiddenStates.count else { return nil }
        return hiddenStates[layer]
    }
}

// MARK: - Weight Loading

extension DeBERTaEncoder {
    /// Load weights from a dictionary
    ///
    /// Expected structure (DeBERTa v2 naming with "encoder." prefix):
    /// - encoder.embeddings.word_embeddings.weight
    /// - encoder.embeddings.LayerNorm.weight/bias
    /// - encoder.encoder.layer.{i}.attention.self.query_proj.weight/bias
    /// - encoder.encoder.layer.{i}.attention.self.key_proj.weight/bias
    /// - encoder.encoder.layer.{i}.attention.self.value_proj.weight/bias
    /// - encoder.encoder.layer.{i}.attention.output.dense.weight/bias
    /// - encoder.encoder.layer.{i}.attention.output.LayerNorm.weight/bias
    /// - encoder.encoder.layer.{i}.intermediate.dense.weight/bias
    /// - encoder.encoder.layer.{i}.output.dense.weight/bias
    /// - encoder.encoder.layer.{i}.output.LayerNorm.weight/bias
    /// - encoder.encoder.rel_embeddings.weight
    /// - encoder.encoder.LayerNorm.weight/bias (for rel_embeddings normalization)
    ///
    /// - Parameters:
    ///   - weights: Weight dictionary
    ///   - prefix: Key prefix (typically "encoder" or empty)
    public func loadWeights(_ weights: [String: MLXArray], prefix: String = "") {
        let p = prefix.isEmpty ? "" : "\(prefix)."

        // Load embeddings
        embeddings.loadWeights(weights, prefix: "\(p)embeddings")

        // Load encoder layers
        for (i, layer) in layers.enumerated() {
            layer.loadWeights(weights, prefix: "\(p)encoder.layer.\(i)")
        }

        // Load shared relative position embeddings
        if let relW = weights["\(p)encoder.rel_embeddings.weight"] {
            relEmbeddings = relW
        }

        // Load LayerNorm for relative embeddings (NOT final output LayerNorm!)
        loadLayerNormFromDict(relEmbeddingsLayerNorm, weights: weights, prefix: "\(p)encoder.LayerNorm")
    }

    /// Load from converted GLiNER2 weights
    ///
    /// GLiNER2 stores encoder weights with "encoder." prefix in the model weights.
    /// This method handles that structure.
    public func loadGLiNER2Weights(_ weights: [String: MLXArray]) {
        // GLiNER2 weights have structure: encoder.embeddings.*, encoder.encoder.layer.*
        loadWeights(weights, prefix: "encoder")
    }
}

// MARK: - Factory

extension DeBERTaEncoder {
    /// Create encoder configured for GLiNER2 base model
    public static func forGLiNER2Base() -> DeBERTaEncoder {
        DeBERTaEncoder(config: .gliner2Base)
    }

    /// Create encoder from configuration dictionary (e.g., from config.json)
    public static func fromConfigDict(_ dict: [String: Any]) throws -> DeBERTaEncoder {
        var config = DeBERTaConfig()

        if let vocabSize = dict["vocab_size"] as? Int {
            config.vocabSize = vocabSize
        }
        if let hiddenSize = dict["hidden_size"] as? Int {
            config.hiddenSize = hiddenSize
        }
        if let numLayers = dict["num_hidden_layers"] as? Int {
            config.numHiddenLayers = numLayers
        }
        if let numHeads = dict["num_attention_heads"] as? Int {
            config.numAttentionHeads = numHeads
        }
        if let intermediateSize = dict["intermediate_size"] as? Int {
            config.intermediateSize = intermediateSize
        }
        if let positionBuckets = dict["position_buckets"] as? Int {
            config.positionBuckets = positionBuckets
        }
        if let maxPosition = dict["max_position_embeddings"] as? Int {
            config.maxPositionEmbeddings = maxPosition
        }
        if let dropoutProb = dict["hidden_dropout_prob"] as? Double {
            config.hiddenDropoutProb = Float(dropoutProb)
        }
        if let attentionDropout = dict["attention_probs_dropout_prob"] as? Double {
            config.attentionDropoutProb = Float(attentionDropout)
        }
        if let eps = dict["layer_norm_eps"] as? Double {
            config.layerNormEps = Float(eps)
        }
        if let posAttTypeArray = dict["pos_att_type"] as? [String] {
            config.posAttType = Set(posAttTypeArray)
        }

        return DeBERTaEncoder(config: config)
    }
}
