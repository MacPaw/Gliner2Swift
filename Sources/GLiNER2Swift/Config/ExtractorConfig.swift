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
// ExtractorConfig.swift
// Configuration for the GLiNER2 Extractor model
//
// Matches Python: gliner2/model.py:ExtractorConfig

import Foundation

/// Configuration for the GLiNER2 Extractor model.
///
/// This struct holds all model hyperparameters and architecture settings.
/// Default values match the `fastino/gliner2-base-v1` model configuration.
public struct ExtractorConfig: Codable, Sendable {
    /// The base encoder model name (e.g., "microsoft/deberta-v3-base")
    public let modelName: String

    /// Maximum span width for entity extraction (default: 8)
    public let maxWidth: Int

    /// Type of counting layer: "count_lstm", "count_lstm_v2", or "count_lstm_moe"
    public let countingLayer: CountingLayerType

    /// Token pooling strategy: "first", "mean", or "max"
    public let tokenPooling: TokenPoolingType

    /// Hidden size from encoder (default: 768 for DeBERTa-base)
    public let hiddenSize: Int

    /// Vocabulary size including special tokens (default: 128011)
    public let vocabSize: Int

    /// Maximum count value for CountLSTM (default: 20)
    public let maxCount: Int

    /// Dropout rate for span representation (default: 0.1)
    public let spanDropout: Float

    /// Present when the checkpoint on disk is already int-quantized (an MLX `quantization`
    /// block in config.json). The loader uses it to build the `QuantizedLinear` structure
    /// before reading the packed weights.
    public let quantization: QuantizationConfig?

    // MARK: - Counting Layer Types

    /// Affine int-quantization parameters read from config.json's `quantization` block.
    public struct QuantizationConfig: Codable, Sendable, Equatable {
        public let groupSize: Int
        public let bits: Int
        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits
        }
    }

    public enum CountingLayerType: String, Codable, Sendable {
        case countLSTM = "count_lstm"
        case countLSTMv2 = "count_lstm_v2"
        case countLSTMMoE = "count_lstm_moe"
    }

    public enum TokenPoolingType: String, Codable, Sendable {
        case first = "first"
        case mean = "mean"
        case max = "max"
    }

    // MARK: - Initialization

    /// Initialize with default values for GLiNER2-base-v1
    public init(
        modelName: String = "microsoft/deberta-v3-base",
        maxWidth: Int = 8,
        countingLayer: CountingLayerType = .countLSTMv2,
        tokenPooling: TokenPoolingType = .first,
        hiddenSize: Int = 768,
        vocabSize: Int = 128011,
        maxCount: Int = 20,
        spanDropout: Float = 0.1,
        quantization: QuantizationConfig? = nil
    ) {
        self.modelName = modelName
        self.maxWidth = maxWidth
        self.countingLayer = countingLayer
        self.tokenPooling = tokenPooling
        self.hiddenSize = hiddenSize
        self.vocabSize = vocabSize
        self.maxCount = maxCount
        self.spanDropout = spanDropout
        self.quantization = quantization
    }

    // MARK: - Loading from HuggingFace config.json

    /// Load configuration from a HuggingFace config.json file
    public static func load(from url: URL) throws -> ExtractorConfig {
        let data = try Data(contentsOf: url)
        return try load(from: data)
    }

    /// Load configuration from JSON data
    public static func load(from data: Data) throws -> ExtractorConfig {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase

        // First decode the raw JSON to handle optional fields
        let rawConfig = try JSONDecoder().decode(RawConfig.self, from: data)

        return ExtractorConfig(
            modelName: rawConfig.modelName ?? "microsoft/deberta-v3-base",
            maxWidth: rawConfig.maxWidth ?? 8,
            countingLayer: CountingLayerType(rawValue: rawConfig.countingLayer ?? "count_lstm_v2") ?? .countLSTMv2,
            tokenPooling: TokenPoolingType(rawValue: rawConfig.tokenPooling ?? "first") ?? .first,
            hiddenSize: rawConfig.hiddenSize ?? 768,
            vocabSize: rawConfig.vocabSize ?? 128011,
            maxCount: 20,
            spanDropout: 0.1,
            quantization: rawConfig.quantization
        )
    }

    /// Raw config structure for flexible JSON parsing
    private struct RawConfig: Codable {
        let modelName: String?
        let maxWidth: Int?
        let countingLayer: String?
        let tokenPooling: String?
        let hiddenSize: Int?
        let vocabSize: Int?
        let quantization: QuantizationConfig?

        enum CodingKeys: String, CodingKey {
            case modelName = "model_name"
            case maxWidth = "max_width"
            case countingLayer = "counting_layer"
            case tokenPooling = "token_pooling"
            case hiddenSize = "hidden_size"
            case vocabSize = "vocab_size"
            case quantization
        }
    }
}

// MARK: - Derived Properties

extension ExtractorConfig {
    /// DownscaledTransformer hidden size (for CountLSTMv2)
    public var downscaledHiddenSize: Int { 128 }

    /// DownscaledTransformer number of heads
    public var downscaledNumHeads: Int { 4 }

    /// DownscaledTransformer number of layers
    public var downscaledNumLayers: Int { 2 }

    /// DownscaledTransformer feedforward dimension
    public var downscaledFeedforwardDim: Int { 256 }

    /// Classifier MLP intermediate dimension
    public var classifierIntermediateDim: Int { hiddenSize * 2 }

    /// Count prediction MLP intermediate dimension
    public var countPredIntermediateDim: Int { hiddenSize * 2 }

    /// SpanMarkerV0 projection expansion factor
    public var spanProjectionExpansion: Int { 4 }
}

// MARK: - Debug Description

extension ExtractorConfig: CustomStringConvertible {
    public var description: String {
        """
        ExtractorConfig:
          Encoder model      : \(modelName)
          Counting layer     : \(countingLayer.rawValue)
          Token pooling      : \(tokenPooling.rawValue)
          Hidden size        : \(hiddenSize)
          Max width          : \(maxWidth)
          Vocab size         : \(vocabSize)
        """
    }
}
