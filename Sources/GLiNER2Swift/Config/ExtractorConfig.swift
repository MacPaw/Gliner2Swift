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
// Matches Python: gliner2/configuration.py:ExtractorConfig (+ BoundaryHeadSettings)

import Foundation

/// Which decoder architecture a checkpoint uses (`config.json["architecture"]`).
public enum ModelArchitecture: String, Codable, Sendable {
    /// GLiNER2 v1: span enumeration (SpanMarker + CountLSTM). The default when absent.
    case span
    /// GLiNER2.5: boundary scoring with a shared document-level candidate pool.
    case boundary
}

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

    /// Decoder architecture. `.span` for every v1 checkpoint, `.boundary` for GLiNER2.5.
    public let architecture: ModelArchitecture

    /// The `boundary_head` block of a GLiNER2.5 config. `nil` on span checkpoints.
    public let boundaryHead: BoundaryHeadSettings?

    /// `max_len` from config.json (GLiNER2.5 declares 4096). Informational: the Python
    /// runtime does not truncate on it either.
    public let maxLen: Int?

    // Encoder shape. Read from `encoder_config/config.json` when the model directory has
    // one (every fastino checkpoint does); the defaults are deberta-v3-base.

    /// Number of encoder layers (12 for base).
    public let numHiddenLayers: Int

    /// Number of attention heads (12 for base, 6 for xsmall, 16 for large).
    public let numAttentionHeads: Int

    /// FFN intermediate size (3072 for base).
    public let intermediateSize: Int

    /// Relative-position buckets (256).
    public let positionBuckets: Int

    /// Relative-position embedding rows (512).
    public let maxPositionEmbeddings: Int

    /// Encoder LayerNorm epsilon (1e-7 for DeBERTa-v3).
    public let layerNormEps: Float

    /// Whether the encoder shape came from `encoder_config/config.json` rather than the
    /// deberta-v3-base defaults. Surfaced so a non-base `modelName` without the file can
    /// be warned about instead of silently mis-shaped.
    public let encoderConfigLoaded: Bool

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
        quantization: QuantizationConfig? = nil,
        architecture: ModelArchitecture = .span,
        boundaryHead: BoundaryHeadSettings? = nil,
        maxLen: Int? = nil,
        numHiddenLayers: Int = 12,
        numAttentionHeads: Int = 12,
        intermediateSize: Int = 3072,
        positionBuckets: Int = 256,
        maxPositionEmbeddings: Int = 512,
        layerNormEps: Float = 1e-7,
        encoderConfigLoaded: Bool = false
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
        self.architecture = architecture
        self.boundaryHead = boundaryHead
        self.maxLen = maxLen
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.intermediateSize = intermediateSize
        self.positionBuckets = positionBuckets
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.layerNormEps = layerNormEps
        self.encoderConfigLoaded = encoderConfigLoaded
    }

    // MARK: - Loading from HuggingFace config.json

    /// Load configuration from a HuggingFace config.json file
    public static func load(from url: URL) throws -> ExtractorConfig {
        let data = try Data(contentsOf: url)
        return try load(from: data)
    }

    /// Load the configuration of a model directory: `config.json`, plus the encoder shape
    /// from `encoder_config/config.json` when present.
    ///
    /// The top-level config of a fastino checkpoint carries no encoder dimensions at all,
    /// so without the sibling file every model is built as deberta-v3-base. That was right
    /// for `gliner2-base-v1` only; `gliner2.5-small-v1` (384 wide, 6 heads),
    /// `gliner2-large-v1` (1024 wide, 24 layers) and the mdeberta models (250k vocab)
    /// need the real numbers.
    public static func load(fromDirectory directory: URL) throws -> ExtractorConfig {
        let base = try load(from: directory.appendingPathComponent("config.json"))
        let encoderUrl = directory.appendingPathComponent("encoder_config/config.json")
        guard FileManager.default.fileExists(atPath: encoderUrl.path) else {
            if base.modelName != "microsoft/deberta-v3-base" {
                print("⚠️ GLiNER2: \(directory.lastPathComponent) has no encoder_config/config.json; "
                      + "building the encoder with deberta-v3-base dimensions for \(base.modelName)")
            }
            return base
        }
        let data = try Data(contentsOf: encoderUrl)
        guard let dict = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw GLiNER2Error.invalidConfig("encoder_config/config.json is not a JSON object")
        }
        return base.applyingEncoderConfig(dict)
    }

    /// Return a copy with the encoder dimensions taken from a DeBERTa-v2 `config.json`
    /// dictionary (the `encoder_config/config.json` of a fastino checkpoint).
    public func applyingEncoderConfig(_ dict: [String: Any]) -> ExtractorConfig {
        func int(_ key: String, _ fallback: Int) -> Int { (dict[key] as? Int) ?? fallback }
        let eps = (dict["layer_norm_eps"] as? Double).map { Float($0) } ?? layerNormEps
        return ExtractorConfig(
            modelName: modelName,
            maxWidth: maxWidth,
            countingLayer: countingLayer,
            tokenPooling: tokenPooling,
            hiddenSize: int("hidden_size", hiddenSize),
            vocabSize: int("vocab_size", vocabSize),
            maxCount: maxCount,
            spanDropout: spanDropout,
            quantization: quantization,
            architecture: architecture,
            boundaryHead: boundaryHead,
            maxLen: maxLen,
            numHiddenLayers: int("num_hidden_layers", numHiddenLayers),
            numAttentionHeads: int("num_attention_heads", numAttentionHeads),
            intermediateSize: int("intermediate_size", intermediateSize),
            positionBuckets: int("position_buckets", positionBuckets),
            maxPositionEmbeddings: int("max_position_embeddings", maxPositionEmbeddings),
            layerNormEps: eps,
            encoderConfigLoaded: true
        )
    }

    /// Load configuration from JSON data
    public static func load(from data: Data) throws -> ExtractorConfig {
        // First decode the raw JSON to handle optional fields
        let rawConfig = try JSONDecoder().decode(RawConfig.self, from: data)

        let architecture = ModelArchitecture(rawValue: rawConfig.architecture ?? "span") ?? .span
        var boundaryHead: BoundaryHeadSettings? = nil
        if architecture == .boundary {
            let settings = rawConfig.boundaryHead ?? BoundaryHeadSettings()
            try settings.validateSupported()
            boundaryHead = settings
        }

        return ExtractorConfig(
            modelName: rawConfig.modelName ?? "microsoft/deberta-v3-base",
            maxWidth: rawConfig.maxWidth ?? 8,
            countingLayer: CountingLayerType(rawValue: rawConfig.countingLayer ?? "count_lstm_v2") ?? .countLSTMv2,
            tokenPooling: TokenPoolingType(rawValue: rawConfig.tokenPooling ?? "first") ?? .first,
            hiddenSize: rawConfig.hiddenSize ?? 768,
            vocabSize: rawConfig.vocabSize ?? 128011,
            maxCount: 20,
            spanDropout: 0.1,
            quantization: rawConfig.quantization,
            architecture: architecture,
            boundaryHead: boundaryHead,
            maxLen: rawConfig.maxLen
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
        let architecture: String?
        let boundaryHead: BoundaryHeadSettings?
        let maxLen: Int?

        enum CodingKeys: String, CodingKey {
            case modelName = "model_name"
            case maxWidth = "max_width"
            case countingLayer = "counting_layer"
            case tokenPooling = "token_pooling"
            case hiddenSize = "hidden_size"
            case vocabSize = "vocab_size"
            case quantization
            case architecture
            case boundaryHead = "boundary_head"
            case maxLen = "max_len"
        }
    }
}

// MARK: - Boundary head settings

/// The inference-relevant fields of a GLiNER2.5 `boundary_head` config block.
///
/// Defaults are the values every shipped 2.5 checkpoint uses, so a missing field still
/// builds the shipped model. Fields that only matter for training are not represented.
/// Reference: gliner2/configuration.py:BoundaryHeadSettings.
public struct BoundaryHeadSettings: Codable, Sendable, Equatable {
    public var boundaryDim: Int = 128
    public var pairDim: Int = 128
    public var contentDim: Int = 64
    public var boundaryAttentionLayers: Int = 2
    public var boundaryAttentionHeads: Int = 4
    public var boundaryAttentionWindow: Int = 128
    public var boundaryRefinementLayers: Int = 1
    public var boundaryFfnMultiplier: Float = 2.0
    public var enableSpanContent: Bool = true
    public var contentSoftMaxPool: Bool = false
    public var enableRotaryEndpoints: Bool = true
    public var rotaryBase: Float = 10000.0
    public var useInsideEvidence: Bool = true
    public var queryConditionedInsideWeight: Bool = true
    public var endpointDifferenceFeatures: Bool = true
    public var rerankerEndpointCompat: Bool = true
    public var multiheadPairCompatHeads: Int = 8
    public var candidatePool: String = "shared"
    public var poolSize: Int = 192
    public var poolBoundaryTopK: Int = 32
    public var minPoolPerQuery: Int = 8
    public var candidateAttentionLayers: Int = 0
    public var queryAttentionLayers: Int = 0
    public var enableAbstention: Bool = true
    public var abstentionThreshold: Float = 0.5
    public var enableCountHead: Bool = true
    public var adaptiveThreshold: Bool = false
    public var overlapPolicy: String = "flat"
    public var pairTemperature: Float = 1.0
    public var classificationTemperature: Float = 1.0
    public var relationTemperature: Float = 1.0
    public var recordTemperature: Float = 1.0
    public var enableRecords: Bool = true
    public var enableRelations: Bool = true
    public var startTopK: Int = 24
    public var endTopK: Int = 24
    public var endsPerStart: Int = 12
    public var startsPerEnd: Int = 12
    public var candidateBudget: Int = 192
    public var endBlockSize: Int = 256
    public var bidirectionalProposals: Bool = true
    public var boundaryTopKAlpha: Float = 0.08
    public var boundaryTopKMax: Int = 128
    public var boundaryTopKBucket: Int = 8
    public var recordDim: Int = 128
    public var recordInstanceQueries: Int = 32
    public var relationHeadsPerType: Int = 32
    public var relationTailsPerType: Int = 32
    public var relationPairCap: Int = 64
    public var relationArgumentProposalThreshold: Float = 0.2
    public var directionalRelationStates: Bool = true
    public var relationBiaffineContent: Bool = true

    public init() {}

    enum CodingKeys: String, CodingKey {
        case boundaryDim = "boundary_dim"
        case pairDim = "pair_dim"
        case contentDim = "content_dim"
        case boundaryAttentionLayers = "boundary_attention_layers"
        case boundaryAttentionHeads = "boundary_attention_heads"
        case boundaryAttentionWindow = "boundary_attention_window"
        case boundaryRefinementLayers = "boundary_refinement_layers"
        case boundaryFfnMultiplier = "boundary_ffn_multiplier"
        case enableSpanContent = "enable_span_content"
        case contentSoftMaxPool = "content_soft_max_pool"
        case enableRotaryEndpoints = "enable_rotary_endpoints"
        case rotaryBase = "rotary_base"
        case useInsideEvidence = "use_inside_evidence"
        case queryConditionedInsideWeight = "query_conditioned_inside_weight"
        case endpointDifferenceFeatures = "endpoint_difference_features"
        case rerankerEndpointCompat = "reranker_endpoint_compat"
        case multiheadPairCompatHeads = "multihead_pair_compat_heads"
        case candidatePool = "candidate_pool"
        case poolSize = "pool_size"
        case poolBoundaryTopK = "pool_boundary_top_k"
        case minPoolPerQuery = "min_pool_per_query"
        case candidateAttentionLayers = "candidate_attention_layers"
        case queryAttentionLayers = "query_attention_layers"
        case enableAbstention = "enable_abstention"
        case abstentionThreshold = "abstention_threshold"
        case enableCountHead = "enable_count_head"
        case adaptiveThreshold = "adaptive_threshold"
        case overlapPolicy = "overlap_policy"
        case pairTemperature = "pair_temperature"
        case classificationTemperature = "classification_temperature"
        case relationTemperature = "relation_temperature"
        case recordTemperature = "record_temperature"
        case enableRecords = "enable_records"
        case enableRelations = "enable_relations"
        case startTopK = "start_top_k"
        case endTopK = "end_top_k"
        case endsPerStart = "ends_per_start"
        case startsPerEnd = "starts_per_end"
        case candidateBudget = "candidate_budget"
        case endBlockSize = "end_block_size"
        case bidirectionalProposals = "bidirectional_proposals"
        case boundaryTopKAlpha = "boundary_top_k_alpha"
        case boundaryTopKMax = "boundary_top_k_max"
        case boundaryTopKBucket = "boundary_top_k_bucket"
        case recordDim = "record_dim"
        case recordInstanceQueries = "record_instance_queries"
        case relationHeadsPerType = "relation_heads_per_type"
        case relationTailsPerType = "relation_tails_per_type"
        case relationPairCap = "relation_pair_cap"
        case relationArgumentProposalThreshold = "relation_argument_proposal_threshold"
        case directionalRelationStates = "directional_relation_states"
        case relationBiaffineContent = "relation_biaffine_content"
    }

    /// Every field is optional in the JSON; a missing one keeps its shipped default.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func int(_ key: CodingKeys, _ d: Int) -> Int { (try? c.decodeIfPresent(Int.self, forKey: key)) ?? nil ?? d }
        func flt(_ key: CodingKeys, _ d: Float) -> Float { (try? c.decodeIfPresent(Float.self, forKey: key)) ?? nil ?? d }
        func bool(_ key: CodingKeys, _ d: Bool) -> Bool { (try? c.decodeIfPresent(Bool.self, forKey: key)) ?? nil ?? d }
        func str(_ key: CodingKeys, _ d: String) -> String { (try? c.decodeIfPresent(String.self, forKey: key)) ?? nil ?? d }

        boundaryDim = int(.boundaryDim, boundaryDim)
        pairDim = int(.pairDim, pairDim)
        contentDim = int(.contentDim, contentDim)
        boundaryAttentionLayers = int(.boundaryAttentionLayers, boundaryAttentionLayers)
        boundaryAttentionHeads = int(.boundaryAttentionHeads, boundaryAttentionHeads)
        boundaryAttentionWindow = int(.boundaryAttentionWindow, boundaryAttentionWindow)
        boundaryRefinementLayers = int(.boundaryRefinementLayers, boundaryRefinementLayers)
        boundaryFfnMultiplier = flt(.boundaryFfnMultiplier, boundaryFfnMultiplier)
        enableSpanContent = bool(.enableSpanContent, enableSpanContent)
        contentSoftMaxPool = bool(.contentSoftMaxPool, contentSoftMaxPool)
        enableRotaryEndpoints = bool(.enableRotaryEndpoints, enableRotaryEndpoints)
        rotaryBase = flt(.rotaryBase, rotaryBase)
        useInsideEvidence = bool(.useInsideEvidence, useInsideEvidence)
        queryConditionedInsideWeight = bool(.queryConditionedInsideWeight, queryConditionedInsideWeight)
        endpointDifferenceFeatures = bool(.endpointDifferenceFeatures, endpointDifferenceFeatures)
        rerankerEndpointCompat = bool(.rerankerEndpointCompat, rerankerEndpointCompat)
        multiheadPairCompatHeads = int(.multiheadPairCompatHeads, multiheadPairCompatHeads)
        candidatePool = str(.candidatePool, candidatePool)
        poolSize = int(.poolSize, poolSize)
        poolBoundaryTopK = int(.poolBoundaryTopK, poolBoundaryTopK)
        minPoolPerQuery = int(.minPoolPerQuery, minPoolPerQuery)
        candidateAttentionLayers = int(.candidateAttentionLayers, candidateAttentionLayers)
        queryAttentionLayers = int(.queryAttentionLayers, queryAttentionLayers)
        enableAbstention = bool(.enableAbstention, enableAbstention)
        abstentionThreshold = flt(.abstentionThreshold, abstentionThreshold)
        enableCountHead = bool(.enableCountHead, enableCountHead)
        adaptiveThreshold = bool(.adaptiveThreshold, adaptiveThreshold)
        overlapPolicy = str(.overlapPolicy, overlapPolicy)
        pairTemperature = flt(.pairTemperature, pairTemperature)
        classificationTemperature = flt(.classificationTemperature, classificationTemperature)
        relationTemperature = flt(.relationTemperature, relationTemperature)
        recordTemperature = flt(.recordTemperature, recordTemperature)
        enableRecords = bool(.enableRecords, enableRecords)
        enableRelations = bool(.enableRelations, enableRelations)
        startTopK = int(.startTopK, startTopK)
        endTopK = int(.endTopK, endTopK)
        endsPerStart = int(.endsPerStart, endsPerStart)
        startsPerEnd = int(.startsPerEnd, startsPerEnd)
        candidateBudget = int(.candidateBudget, candidateBudget)
        endBlockSize = int(.endBlockSize, endBlockSize)
        bidirectionalProposals = bool(.bidirectionalProposals, bidirectionalProposals)
        boundaryTopKAlpha = flt(.boundaryTopKAlpha, boundaryTopKAlpha)
        boundaryTopKMax = int(.boundaryTopKMax, boundaryTopKMax)
        boundaryTopKBucket = int(.boundaryTopKBucket, boundaryTopKBucket)
        recordDim = int(.recordDim, recordDim)
        recordInstanceQueries = int(.recordInstanceQueries, recordInstanceQueries)
        relationHeadsPerType = int(.relationHeadsPerType, relationHeadsPerType)
        relationTailsPerType = int(.relationTailsPerType, relationTailsPerType)
        relationPairCap = int(.relationPairCap, relationPairCap)
        relationArgumentProposalThreshold = flt(.relationArgumentProposalThreshold, relationArgumentProposalThreshold)
        directionalRelationStates = bool(.directionalRelationStates, directionalRelationStates)
        relationBiaffineContent = bool(.relationBiaffineContent, relationBiaffineContent)
    }

    /// The Swift port hardcodes the shipped structure in a few places. Refuse anything
    /// else up front rather than silently producing wrong output on a future checkpoint.
    public func validateSupported() throws {
        var problems: [String] = []
        if candidatePool != "shared" { problems.append("candidate_pool=\(candidatePool) (only \"shared\")") }
        if candidateAttentionLayers != 0 { problems.append("candidate_attention_layers=\(candidateAttentionLayers) (only 0)") }
        if queryAttentionLayers != 0 { problems.append("query_attention_layers=\(queryAttentionLayers) (only 0)") }
        if contentSoftMaxPool { problems.append("content_soft_max_pool=true (only false)") }
        if !enableSpanContent { problems.append("enable_span_content=false (only true)") }
        if !enableRotaryEndpoints { problems.append("enable_rotary_endpoints=false (only true)") }
        if !useInsideEvidence { problems.append("use_inside_evidence=false (only true)") }
        guard problems.isEmpty else {
            throw GLiNER2Error.invalidConfig(
                "unsupported boundary_head settings: " + problems.joined(separator: ", "))
        }
    }

    /// The canonical overlap policy name for this checkpoint (`flat` → `disallow`).
    public var canonicalOverlapPolicy: OverlapPolicy {
        OverlapPolicy(rawName: overlapPolicy) ?? .disallow
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

    /// The encoder configuration this model builds.
    public var debertaConfig: DeBERTaConfig {
        DeBERTaConfig(
            vocabSize: vocabSize,
            hiddenSize: hiddenSize,
            numHiddenLayers: numHiddenLayers,
            numAttentionHeads: numAttentionHeads,
            intermediateSize: intermediateSize,
            positionBuckets: positionBuckets,
            maxPositionEmbeddings: maxPositionEmbeddings,
            hiddenDropoutProb: 0.1,
            attentionDropoutProb: 0.1,
            layerNormEps: layerNormEps,
            posAttType: ["c2p", "p2c"]
        )
    }
}

// MARK: - Debug Description

extension ExtractorConfig: CustomStringConvertible {
    public var description: String {
        """
        ExtractorConfig:
          Architecture       : \(architecture.rawValue)
          Encoder model      : \(modelName)
          Encoder shape      : \(numHiddenLayers) layers x \(hiddenSize) (\(numAttentionHeads) heads, ffn \(intermediateSize))\(encoderConfigLoaded ? "" : " [defaults]")
          Counting layer     : \(countingLayer.rawValue)
          Token pooling      : \(tokenPooling.rawValue)
          Hidden size        : \(hiddenSize)
          Max width          : \(maxWidth)
          Vocab size         : \(vocabSize)
        """
    }
}
