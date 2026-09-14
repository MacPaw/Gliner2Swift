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
// GLiNER2 Extractor models
//
// Matches Python: gliner2/models/base.py:BaseExtractorModel (shared encoder + classifier),
// gliner2/models/span/model.py:SpanExtractorModel (SpanExtractor below) and
// gliner2/models/boundary/model.py:BoundaryExtractorModel (BoundaryExtractor, in
// Model/Boundary/BoundaryExtractor.swift).

import Foundation
import MLX
import MLXNN

/// What both GLiNER2 architectures share: the DeBERTa encoder, the classification MLP,
/// weight loading, quantization and the optional compiled encoder.
///
/// `GLiNER2.model` is typed as this class. The decoder-specific heads live in the two
/// subclasses, `SpanExtractor` (v1 checkpoints) and `BoundaryExtractor` (GLiNER2.5);
/// `Extractor.make(config:)` picks the one the checkpoint's `architecture` asks for.
public class Extractor: Module {
    /// Model configuration
    public let config: ExtractorConfig

    /// Hidden size from encoder
    public let hiddenSize: Int

    /// DeBERTa encoder with disentangled attention
    public let encoder: DeBERTaEncoder

    /// Classifier for classification tasks: hidden → 2*hidden → 1
    public let classifier: Sequential

    /// Compiled encoder forward, when enabled.
    ///
    /// Deliberately not an `MLXArray`-bearing stored property that `Module` reflection
    /// would see; a closure is `.other` to reflection. It captures the traced graph, so it
    /// must be discarded whenever the weights change.
    private var compiledEncode: (@Sendable (MLXArray, MLXArray) -> MLXArray)?

    /// Whether the caller asked for a compiled encoder, so it can be rebuilt after a load.
    private var wantsCompiledEncoder = false

    /// Initialize the shared encoder + classifier from configuration.
    ///
    /// Prefer `Extractor.make(config:)`, which returns the subclass for the checkpoint's
    /// architecture; this initializer alone builds a model without a decoder head.
    public init(config: ExtractorConfig) {
        self.config = config
        self.hiddenSize = config.hiddenSize

        // The encoder shape comes from encoder_config/config.json when the model
        // directory has one (see ExtractorConfig.load(fromDirectory:)).
        self.encoder = DeBERTaEncoder(config: config.debertaConfig)

        // Initialize classifier: hidden → 2*hidden → 1
        self.classifier = createMLP(
            inputDim: config.hiddenSize,
            intermediateDims: [config.hiddenSize * 2],
            outputDim: 1,
            dropout: 0.0,
            activation: .relu,
            addLayerNorm: false
        )
        super.init()
    }

    /// Build the extractor subclass the configuration's `architecture` calls for.
    public static func make(config: ExtractorConfig) -> Extractor {
        switch config.architecture {
        case .span: return SpanExtractor(config: config)
        case .boundary: return BoundaryExtractor(config: config)
        }
    }

    /// The decoder architecture this model implements.
    public var architecture: ModelArchitecture { config.architecture }

    // MARK: - Encoder Forward

    /// Run encoder on input IDs
    ///
    /// - Parameters:
    ///   - inputIds: Token IDs [batch, seq_len]
    ///   - attentionMask: Optional attention mask [batch, seq_len]
    ///   - outputHiddenStates: Whether to also return every intermediate layer's output
    /// - Returns: Encoder output with hidden states
    public func encode(
        _ inputIds: MLXArray,
        attentionMask: MLXArray? = nil,
        outputHiddenStates: Bool = false
    ) -> DeBERTaEncoderOutput {
        // The compiled path returns only the last hidden state, so anything asking for the
        // per-layer outputs falls back to the interpreted forward.
        if let mask = attentionMask, !outputHiddenStates, let compiled = compiledEncode {
            encoder.primeCaches(seqLen: inputIds.dim(1))
            return DeBERTaEncoderOutput(
                lastHiddenState: compiled(inputIds, mask),
                hiddenStates: []
            )
        }
        return encoder(inputIds, attentionMask: attentionMask, outputHiddenStates: outputHiddenStates)
    }

    // MARK: - Compiled Encoder

    /// Build (or discard) a compiled encoder forward.
    ///
    /// `MLX.compile` keeps a per-shape cache, so this only pays off when batches arrive at
    /// a small set of sequence lengths — see `SchemaTransformer.sequenceLengthBucket`.
    /// Not `shapeless`: the encoder reads dimensions into Swift constants, which shapeless
    /// tracing would silently bake in.
    public func setCompiledEncoderEnabled(_ enabled: Bool) {
        wantsCompiledEncoder = enabled
        rebuildCompiledEncoder()
    }

    /// Rebuild or drop the compiled closure. Every weight-loading path must call this: the
    /// traced graph captures the memoized relative-position projections as constants, so a
    /// closure built against the previous weights would keep serving them.
    func rebuildCompiledEncoder() {
        guard wantsCompiledEncoder else {
            compiledEncode = nil
            return
        }
        let encoder = self.encoder
        compiledEncode = compile(inputs: [encoder], outputs: [encoder]) {
            (ids: MLXArray, mask: MLXArray) in
            encoder(ids, attentionMask: mask).lastHiddenState
        }
    }

    var isCompiledEncoderEnabled: Bool { compiledEncode != nil }

    // MARK: - Architecture hooks (overridden by the subclasses)

    /// Load the non-encoder weights. The base loads the classifier; subclasses add their
    /// heads and call `super`.
    ///
    /// - Parameter weights: Dictionary mapping parameter names to arrays (converted keys)
    public func loadModelWeights(_ weights: [String: MLXArray]) {
        // Converted weights use: classifier.layers.{0,1}.weight/bias
        loadMLPWeights(classifier, weights: weights, prefix: "classifier.layers")
    }

    /// Keys a checkpoint must carry for this model to run on trained values, checked by
    /// `loadWeights(..., strict: true)`. `exact` keys must be present verbatim; for every
    /// `family` prefix at least one key must start with it.
    ///
    /// The base covers the encoder (per configured layer count) and the classifier.
    func requiredWeightKeys() -> (exact: [String], families: [String]) {
        var exact = [
            "encoder.embeddings.word_embeddings.weight",
            "encoder.embeddings.LayerNorm.weight",
            "encoder.encoder.rel_embeddings.weight",
            "encoder.encoder.LayerNorm.weight",
        ]
        for i in 0..<config.numHiddenLayers {
            exact.append("encoder.encoder.layer.\(i).attention.self.query_proj.weight")
            exact.append("encoder.encoder.layer.\(i).output.dense.weight")
        }
        return (exact, ["classifier.layers."])
    }

    /// Throw if the checkpoint is missing any critical key for this architecture.
    func verifyRequiredWeights(_ weights: [String: MLXArray]) throws {
        let required = requiredWeightKeys()
        try Extractor.verify(weights, exact: required.exact, families: required.families)
    }

    static func verify(_ weights: [String: MLXArray], exact: [String], families: [String]) throws {
        let present = Set(weights.keys)
        var missing = exact.filter { !present.contains($0) }
        missing += families
            .filter { family in !present.contains { $0.hasPrefix(family) } }
            .map { $0 + "*" }
        guard missing.isEmpty else {
            throw GLiNER2Error.weightLoadingFailed(
                "strict load: \(missing.count) required weight(s) missing "
                + "(would keep random init): \(missing.prefix(8).joined(separator: ", "))"
                + (missing.count > 8 ? " …" : ""))
        }
    }

    // MARK: - Shared helpers

    func loadMLPWeights(_ mlp: Sequential, weights: [String: MLXArray], prefix: String) {
        // MLP structure: Linear → activation → Linear
        // Swift Sequential layers array: [0]=Linear, [1]=activation, [2]=Linear
        //
        // Converted keys number the Linear layers, not the Sequential slots
        // (convert_weights.py remaps PyTorch indices):
        //   classifier.0.weight → classifier.layers.0.weight (first Linear)
        //   classifier.2.weight → classifier.layers.1.weight (second Linear; .3 on 2.5)
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
            updateLinearWeights(
                linear2,
                weight: weights["\(prefix).1.weight"],
                bias: weights["\(prefix).1.bias"]
            )
        }
    }
}

// MARK: - Span extractor (GLiNER2 v1)

/// GLiNER2 v1 Extractor Model.
///
/// Architecture:
/// - Encoder: DeBERTa-v3 with disentangled attention
/// - Span representation: SpanMarkerV0
/// - Classifier: MLP for classification tasks
/// - Count prediction: MLP for count prediction
/// - Count embedding: CountLSTMv2 for structure extraction
public final class SpanExtractor: Extractor {
    /// Maximum span width
    public let maxWidth: Int

    /// Span representation layer
    public let spanRep: SpanRepLayer

    /// Count prediction layer: hidden → 2*hidden → 20
    public let countPred: Sequential

    /// Count embedding module (CountLSTMv2 for gliner2-base-v1)
    public let countEmbed: CountLSTMv2

    /// Initialize Extractor from configuration
    ///
    /// - Parameter config: Model configuration
    public override init(config: ExtractorConfig) {
        self.maxWidth = config.maxWidth
        let hiddenSize = config.hiddenSize

        // Initialize span representation layer
        self.spanRep = SpanRepLayer(
            hiddenSize: hiddenSize,
            maxWidth: config.maxWidth,
            spanMode: .markerV0,
            dropout: config.spanDropout
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
        super.init(config: config)
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
        let numSpans = textLength * maxWidth

        // Build the (start, end) index pair and the validity flag for every span in one
        // CPU pass. The previous version wrote (-1, -1) for out-of-range spans and then
        // recovered the same information on the GPU with two `equal`s, a `logicalOr` and a
        // `where` against a float32 `zeros` — which also promoted the whole index tensor to
        // float32 and made the gather float-indexed. Invalid entries get (0, 0) here, which
        // is exactly what that `where` produced, so the gathered values are unchanged.
        var flatSpans = [Int32](repeating: 0, count: numSpans * 2)
        var invalid = [Bool](repeating: false, count: numSpans)

        var span = 0
        for i in 0..<textLength {
            for j in 0..<maxWidth {
                let end = i + j
                if end < textLength {
                    flatSpans[span * 2] = Int32(i)
                    flatSpans[span * 2 + 1] = Int32(end)
                } else {
                    invalid[span] = true   // indices stay (0, 0)
                }
                span += 1
            }
        }

        let spanIdxArray = MLXArray(flatSpans).reshaped([1, numSpans, 2])
        let spanMask = MLXArray(invalid).reshaped([1, numSpans])

        // Compute span representations
        let tokenEmbsBatched = tokenEmbeddings.expandedDimensions(axis: 0)  // [1, textLen, hidden]
        var spanRepResult = spanRep(tokenEmbsBatched, spanIdx: spanIdxArray, debug: debug)  // [1, textLen, maxWidth, hidden]
        spanRepResult = spanRepResult.squeezed(axis: 0)  // [textLen, maxWidth, hidden]

        // Reshape to [numSpans, hidden]
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

    // MARK: - Weights

    /// Load GLiNER2 model weights (excluding encoder)
    ///
    /// - Parameter weights: Dictionary mapping parameter names to arrays
    public override func loadModelWeights(_ weights: [String: MLXArray]) {
        super.loadModelWeights(weights)

        // Load span representation weights
        // Converted weights use camelCase: spanRep.spanRepLayer.*
        spanRep.loadWeights(weights, prefix: "spanRep")

        // Load count prediction weights
        // Converted weights use: countPred.layers.{0,1}.weight/bias
        loadMLPWeights(countPred, weights: weights, prefix: "countPred.layers")

        // Load count embedding weights
        // Converted weights use: countEmbed.*
        countEmbed.loadWeights(weights, prefix: "countEmbed")
    }

    override func requiredWeightKeys() -> (exact: [String], families: [String]) {
        var required = super.requiredWeightKeys()
        required.exact += [
            "countEmbed.gru.weightIH",
            "countEmbed.gru.weightHH",
            "countEmbed.posEmbedding.weight",
        ]
        required.families += [
            "spanRep.spanRepLayer.projectStart.",
            "spanRep.spanRepLayer.projectEnd.",
            "spanRep.spanRepLayer.outProject.",
            "countPred.layers.",
            "countEmbed.transformer.",
        ]
        return required
    }
}

// MARK: - Span Info

/// Container for span representation information
public struct SpanInfo {
    /// Span representations [numSpans, hidden]
    public let spanRep: MLXArray

    /// Span indices [1, numSpans, 2], int32 and gather-safe: spans that run past the end of
    /// the text are collapsed to (0, 0) and flagged in `spanMask` rather than held as (-1, -1).
    public let spansIdx: MLXArray

    /// Span mask [1, numSpans] - true for invalid spans
    public let spanMask: MLXArray
}

// MARK: - Weight Loading

/// What to do with a checkpoint's floating-point weights at load time.
public enum DTypePolicy: Sendable {
    /// Keep whatever the checkpoint stores: an fp16 file loads as fp16, fp32 as fp32.
    case auto
    case float16
    case bfloat16
    case float32

    var target: DType? {
        switch self {
        case .auto: return nil
        case .float16: return .float16
        case .bfloat16: return .bfloat16
        case .float32: return .float32
        }
    }

    /// Cast every floating-point tensor, leaving integer tensors alone.
    ///
    /// There is no global compute dtype in MLX — activations take their dtype from the
    /// weights that produce them — so this is the only place the choice is made.
    func apply(to weights: [String: MLXArray]) -> [String: MLXArray] {
        guard let target else { return weights }
        return weights.mapValues { $0.dtype.isFloatingPoint ? $0.asType(target) : $0 }
    }
}

/// Optional 8-bit quantization of the encoder, applied after weights are loaded.
///
/// **This trades accuracy for memory. It is off by default and should stay off unless the
/// memory matters more than exactness on your data.**
///
/// Measured on the canonical fp16 snapshot (M3 Pro): active memory 415 -> 253 MB, peak
/// 521 -> 355 MB, and a few per cent off latency. The embedding table is `[128011, 768]`,
/// the single largest tensor in the model, so `includeEmbeddings: false` gives up most of
/// the memory win (415 -> 341 MB) for no accuracy benefit that has been measured.
///
/// The cost, on the 58-case prediction-parity corpus: **56 cases match Python instead of
/// 58** — two borderline decisions move (an entity within a hair of the threshold, and a
/// confidence just outside the ±0.02 fp16 band). Both are borderline rather than wholesale
/// errors, but they are real, and they are why this is not the default. Re-run
/// `PredictionParityTests` with `GLINER2_QUANTIZE=int8` against your own corpus first.
///
/// This can be applied two ways, which are numerically identical (`OnDiskQuantizationTests`
/// pins that): at load, by passing `.int8` to `fromPretrained` on an fp16 model; or ahead
/// of time, by loading a directory that `convert_weights.py --quantize int8` already packed
/// — `fromPretrained` detects the `quantization` config block and loads it quantized. The
/// pre-quantized directory is ~234 MB vs the fp16 398 MB and skips the fp16→int8 transient
/// at load, so it is the better choice for a shipped int8-only app.
///
/// On a GLiNER2.5 model only the encoder is quantized; the boundary head (about 5 % of the
/// parameters) stays at the checkpoint's float dtype.
public enum QuantizationPolicy: Sendable, Equatable {
    case none

    /// 8-bit, group size 64. `includeEmbeddings: false` restricts it to the encoder's
    /// `Linear` layers, leaving the token embedding table at full precision.
    case int8(includeEmbeddings: Bool)

    /// 8-bit over the encoder's linear layers and the embedding table.
    public static let int8 = QuantizationPolicy.int8(includeEmbeddings: true)
}

/// How much GPU memory MLX may hold in its allocator cache.
///
/// This is a **process-global** MLX setting, not per-model, which is why the default here
/// changes nothing: a library should not silently reconfigure the host application's
/// allocator. Pass a policy explicitly when the process is the model.
public enum GPUCachePolicy: Sendable {
    /// Leave MLX's global cache limit exactly as it is. The default.
    case unchanged

    /// Unbounded on macOS, 256 MB on iOS, where the jetsam limit makes an unbounded
    /// allocator cache the difference between running and being killed.
    case platformDefault

    case limit(bytes: Int)

    func apply() {
        switch self {
        case .unchanged:
            return
        case .platformDefault:
            #if os(iOS) || os(tvOS) || os(visionOS) || os(watchOS)
            MLX.Memory.cacheLimit = 256 * 1024 * 1024
            #endif
        case .limit(let bytes):
            MLX.Memory.cacheLimit = bytes
        }
    }
}

extension Extractor {

    /// Whether `quantize(_:)` has been applied.
    ///
    /// Loading a LoRA adapter afterwards would write full-precision matrices over
    /// `QuantizedLinear`'s packed weights, so that combination is rejected rather than
    /// silently corrupting the model.
    public var isQuantized: Bool {
        encoder.leafModules().flattened().contains { $0.1 is QuantizedLinear }
    }

    /// Apply a quantization policy to the encoder. Call after weights are loaded.
    public func quantize(_ policy: QuantizationPolicy) {
        guard case .int8(let includeEmbeddings) = policy else { return }
        applyEncoderQuantization(bits: 8, groupSize: 64, includeEmbeddings: includeEmbeddings)
        rebuildCompiledEncoder()
    }

    /// Replace the encoder's `Linear` (and optionally `Embedding`) layers with quantized
    /// equivalents of the given width. Used both by the runtime `quantize(.int8)` path and,
    /// at load time, to build the structure a pre-quantized checkpoint's packed weights fill.
    func applyEncoderQuantization(bits: Int, groupSize: Int, includeEmbeddings: Bool) {
        MLXNN.quantize(model: encoder, groupSize: groupSize, bits: bits) { _, module in
            module is Linear || (includeEmbeddings && module is Embedding)
        }
        // The relative-position projections are derived from the query/key weights that
        // were just replaced.
        encoder.resetCaches()
    }

    // MARK: - Format Detection

    /// Detect whether weights are in raw PyTorch format (snake_case keys)
    /// vs pre-converted format (camelCase keys from convert_weights.py).
    static func isRawPyTorchFormat(_ weights: [String: MLXArray]) -> Bool {
        weights.keys.contains(where: { $0.hasPrefix("span_rep.") || $0.hasPrefix("boundary_head.") })
    }

    /// Whether a raw checkpoint is a GLiNER2.5 (boundary) one.
    static func isRawBoundaryFormat(_ weights: [String: MLXArray]) -> Bool {
        weights.keys.contains(where: { $0.hasPrefix("boundary_head.") })
    }

    // MARK: - Key Sanitization

    /// Key mappings from raw PyTorch snake_case to converted camelCase format (v1 span
    /// checkpoints). Ordered longest-prefix-first to avoid false prefix matches.
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
    /// If the weights are already in converted format (detected by absence of `span_rep.`
    /// / `boundary_head.` keys), they are returned unchanged. Encoder weights (`encoder.*`)
    /// pass through unmodified.
    ///
    /// This eliminates the need for the Python `convert_weights.py` script.
    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        guard isRawPyTorchFormat(weights) else { return weights }
        let boundary = isRawBoundaryFormat(weights)

        var result: [String: MLXArray] = [:]
        for (key, value) in weights {
            result[boundary ? mapRawBoundaryKey(key) : mapRawKey(key)] = value
        }
        return result
    }

    /// Map a single raw PyTorch key (v1 span checkpoint) into the converted key space.
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

    /// Map a single raw PyTorch key of a GLiNER2.5 checkpoint into the converted key space.
    ///
    /// One rule, the same `scripts/convert_weights.py` applies: every path segment goes
    /// `snake_case → camelCase`, integer segments are kept, and the two-Linear MLPs
    /// (`classifier`, `film_output`, `mlp`) number their Linear layers 0/1 under `.layers.`
    /// exactly as the v1 `classifier` conversion does (PyTorch index 0 → `layers.0`, the
    /// second Linear at index 2 or 3 → `layers.1`).
    static func mapRawBoundaryKey(_ key: String) -> String {
        if key.hasPrefix("encoder.") { return key }
        var segments = key.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
        for i in segments.indices where i > 0 && Int(segments[i]) != nil {
            let owner = segments[i - 1]
            if owner == "classifier" || owner == "film_output" || owner == "mlp" {
                segments[i] = "layers." + (segments[i] == "0" ? "0" : "1")
            }
        }
        return segments.map(camelCased).joined(separator: ".")
    }

    /// `shared_pool_scorer` → `sharedPoolScorer`; segments without underscores are kept.
    static func camelCased(_ segment: String) -> String {
        guard segment.contains("_") else { return segment }
        let parts = segment.split(separator: "_", omittingEmptySubsequences: false).map(String.init)
        guard let head = parts.first else { return segment }
        return head + parts.dropFirst().map { $0.prefix(1).uppercased() + $0.dropFirst() }.joined()
    }

    // MARK: - Strict-load key sets (v1 span defaults, kept for callers and tests)

    /// Critical weight keys a v1 span checkpoint must carry (see `requiredWeightKeys()`).
    static func requiredCriticalKeys() -> [String] {
        SpanExtractor(config: ExtractorConfig()).requiredWeightKeys().exact
    }

    /// Throw if a v1 span checkpoint is missing any critical key.
    static func verifyRequired(_ weights: [String: MLXArray]) throws {
        try SpanExtractor(config: ExtractorConfig()).verifyRequiredWeights(weights)
    }

    // MARK: - Weight Loading

    /// Load all weights from a single combined SafeTensors file.
    ///
    /// Auto-detects whether the file contains raw PyTorch keys or pre-converted keys.
    /// Both formats are supported transparently.
    ///
    /// - Parameters:
    ///   - url: URL to model.safetensors (combined weights file)
    ///   - dtype: Floating-point policy to apply to the loaded weights
    ///   - strict: Throw when a critical key is missing instead of keeping random init
    public func loadWeights(from url: URL, dtype: DTypePolicy = .auto, strict: Bool = false) throws {
        let rawWeights = try loadArrays(url: url)
        let sanitized = Extractor.sanitize(weights: rawWeights)
        if strict { try verifyRequiredWeights(sanitized) }

        // A pre-quantized checkpoint carries packed weights, detectable by `.scales` keys.
        // Build the quantized structure BEFORE loading so the packed tensors land in
        // QuantizedLinear/QuantizedEmbedding, and do not cast the packed uint32 / fp16
        // scales (dtype policy applies only to plain float checkpoints).
        let isQuantizedCheckpoint = sanitized.keys.contains { $0.hasSuffix(".scales") }
        if isQuantizedCheckpoint, let q = config.quantization, !isQuantized {
            let includeEmbeddings = sanitized["encoder.embeddings.word_embeddings.scales"] != nil
            applyEncoderQuantization(bits: q.bits, groupSize: q.groupSize,
                                     includeEmbeddings: includeEmbeddings)
        }
        let weights = isQuantizedCheckpoint ? sanitized : dtype.apply(to: sanitized)

        // Load encoder weights (keys starting with "encoder.")
        encoder.loadWeights(weights, prefix: "encoder")

        // Load model-specific weights (heads)
        loadModelWeights(weights)
        rebuildCompiledEncoder()
    }

    /// Load all weights from separate SafeTensors files (legacy method)
    ///
    /// - Parameters:
    ///   - modelWeightsUrl: URL to gliner2_weights.safetensors
    ///   - encoderWeightsUrl: URL to encoder_weights.safetensors
    public func loadWeights(
        modelWeightsUrl: URL,
        encoderWeightsUrl: URL,
        dtype: DTypePolicy = .auto
    ) throws {
        // Load model weights
        let rawModelWeights = try loadArrays(url: modelWeightsUrl)
        let modelWeights = dtype.apply(to: Extractor.sanitize(weights: rawModelWeights))
        loadModelWeights(modelWeights)

        // Load encoder weights
        let encoderWeights = try dtype.apply(to: loadArrays(url: encoderWeightsUrl))
        encoder.loadWeights(encoderWeights, prefix: "encoder")
        rebuildCompiledEncoder()
    }

    /// Load encoder weights separately
    ///
    /// - Parameter weights: Dictionary with encoder weights
    public func loadEncoderWeights(_ weights: [String: MLXArray]) {
        encoder.loadWeights(weights, prefix: "encoder")
        rebuildCompiledEncoder()
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
        rebuildCompiledEncoder()
    }
}
