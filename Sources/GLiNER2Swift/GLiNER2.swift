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
// GLiNER2.swift
// Main GLiNER2 inference class
//
// Matches Python: gliner2/inference/engine.py:GLiNER2

import MLX
import MLXNN
import Foundation

// MARK: - GLiNER2

/// GLiNER2 Information Extraction Model.
///
/// Provides efficient extraction for:
/// - Named Entity Recognition (NER)
/// - Text Classification
/// - Structured Data Extraction (JSON-like structures)
/// - Relation Extraction
public class GLiNER2 {
    /// Model configuration
    public let config: ExtractorConfig

    /// The extractor model
    public let model: Extractor

    /// Schema processor
    public let processor: SchemaTransformer

    /// URL of the loaded base weights (for adapter loading)
    private(set) var baseWeightsUrl: URL?

    /// Initialize GLiNER2
    ///
    /// - Parameters:
    ///   - config: Model configuration
    ///   - processor: Schema processor
    public init(config: ExtractorConfig, processor: SchemaTransformer) {
        self.config = config
        self.model = Extractor(config: config)
        self.processor = processor
    }

    // MARK: - Loading

    /// Load model from HuggingFace repository or local directory
    ///
    /// - Parameters:
    ///   - pathOrRepo: HuggingFace repo ID or local path
    ///   - progressHandler: Optional progress callback for Hub downloads
    /// - Returns: Initialized GLiNER2 model
    public static func fromPretrained(
        _ pathOrRepo: String,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> GLiNER2 {
        // Determine if local path or HuggingFace repo
        let isLocalPath = FileManager.default.fileExists(atPath: pathOrRepo)

        let baseUrl: URL
        if isLocalPath {
            baseUrl = URL(fileURLWithPath: pathOrRepo)
        } else {
            // Download from HuggingFace Hub (uses shared ~/.cache/huggingface/hub/ cache)
            baseUrl = try await downloadModelDirectory(
                repoId: pathOrRepo,
                progressHandler: progressHandler
            )
        }

        // Resolve file URLs from the model directory
        let configUrl = baseUrl.appendingPathComponent("config.json")

        var combinedWeightsUrl: URL?
        var splitModelWeightsUrl: URL?
        var splitEncoderWeightsUrl: URL?

        // Try single combined file first (preferred)
        let combinedPath = baseUrl.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: combinedPath.path) {
            combinedWeightsUrl = combinedPath
        } else {
            // Fall back to split files
            let modelWeightsPath = baseUrl.appendingPathComponent("gliner2_weights.safetensors")
            let encoderWeightsPath = baseUrl.appendingPathComponent("encoder_weights.safetensors")

            if FileManager.default.fileExists(atPath: modelWeightsPath.path) &&
               FileManager.default.fileExists(atPath: encoderWeightsPath.path) {
                splitModelWeightsUrl = modelWeightsPath
                splitEncoderWeightsUrl = encoderWeightsPath
            } else {
                throw GLiNER2Error.fileNotFound("model.safetensors or gliner2_weights.safetensors + encoder_weights.safetensors")
            }
        }

        // 1. Load configuration
        let config = try ExtractorConfig.load(from: configUrl)

        // 2. Initialize processor with tokenizer
        let poolingType: TokenPoolingType
        switch config.tokenPooling {
        case .first: poolingType = .first
        case .mean: poolingType = .mean
        case .max: poolingType = .max
        }

        let processor = try SchemaTransformer.createFromLocalDirectory(
            directoryUrl: baseUrl,
            tokenPooling: poolingType
        )

        // 3. Create model instance
        let gliner2 = GLiNER2(config: config, processor: processor)

        // 4. Load weights - try combined file first, fall back to split files
        if let combinedUrl = combinedWeightsUrl {
            try gliner2.model.loadWeights(from: combinedUrl)
            gliner2.baseWeightsUrl = combinedUrl
        } else if let modelUrl = splitModelWeightsUrl,
                  let encoderUrl = splitEncoderWeightsUrl {
            try gliner2.model.loadWeights(
                modelWeightsUrl: modelUrl,
                encoderWeightsUrl: encoderUrl
            )
            gliner2.baseWeightsUrl = modelUrl
        } else {
            throw GLiNER2Error.fileNotFound("model weights")
        }

        // 5. Set to evaluation mode (disables dropout)
        gliner2.model.train(false)
        gliner2.model.freeze()

        return gliner2
    }

    /// Load from local directory (synchronous convenience)
    public static func fromLocal(_ path: String) throws -> GLiNER2 {
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<GLiNER2, Error>!

        Task {
            do {
                let model = try await fromPretrained(path)
                result = .success(model)
            } catch {
                result = .failure(error)
            }
            semaphore.signal()
        }

        semaphore.wait()

        switch result! {
        case .success(let model):
            return model
        case .failure(let error):
            throw error
        }
    }

    // MARK: - LoRA Adapter

    /// Load a LoRA adapter onto this model.
    ///
    /// Re-loads base weights with LoRA deltas merged in.
    /// Produces identical results to Python's `model.load_adapter()` followed by `model.merge_lora()`.
    ///
    /// - Parameter adapterPath: Path to adapter directory containing adapter_config.json + adapter_weights.safetensors
    public func loadAdapter(from adapterPath: String) throws {
        guard let baseUrl = baseWeightsUrl else {
            throw GLiNER2Error.weightLoadingFailed("Base weights URL not available for adapter merging")
        }
        let adapterUrl = URL(fileURLWithPath: adapterPath)
        try model.loadWeightsWithLoRA(baseWeightsUrl: baseUrl, adapterPath: adapterUrl)
        model.train(false)
        model.freeze()
    }

    /// Unload the current LoRA adapter, restoring original base weights.
    ///
    /// Re-loads the base weights without any LoRA deltas applied.
    /// Matches Python's `model.unload_adapter()`.
    public func unloadAdapter() throws {
        guard let baseUrl = baseWeightsUrl else {
            throw GLiNER2Error.weightLoadingFailed("Base weights URL not available")
        }
        try model.loadWeights(from: baseUrl)
        model.train(false)
        model.freeze()
    }

    /// Load model from pretrained with optional LoRA adapter.
    ///
    /// - Parameters:
    ///   - pathOrRepo: HuggingFace repo ID or local path
    ///   - adapterPath: Optional path to LoRA adapter directory
    ///   - progressHandler: Optional progress callback for Hub downloads
    /// - Returns: Initialized GLiNER2 model with adapter merged
    public static func fromPretrained(
        _ pathOrRepo: String,
        adapterPath: String?,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> GLiNER2 {
        let model = try await fromPretrained(pathOrRepo, progressHandler: progressHandler)
        if let adapterPath = adapterPath {
            try model.loadAdapter(from: adapterPath)
        }
        return model
    }

    // MARK: - Schema Builder

    /// Create a new schema builder
    public func createSchema() -> Schema {
        Schema()
    }

    // MARK: - Main Extraction

    /// Extract from single text
    ///
    /// - Parameters:
    ///   - text: Input text
    ///   - schema: Extraction schema
    ///   - threshold: Confidence threshold (default: 0.5)
    ///   - includeConfidence: Include confidence scores
    ///   - includeSpans: Include character-level positions
    /// - Returns: Extraction results
    public func extract(
        text: String,
        schema: Schema,
        threshold: Float = 0.5,
        includeConfidence: Bool = false,
        includeSpans: Bool = false
    ) -> [String: Any] {
        let results = batchExtract(
            texts: [text],
            schema: schema,
            threshold: threshold,
            includeConfidence: includeConfidence,
            includeSpans: includeSpans
        )
        return results.first ?? [:]
    }

    /// Batch extract from multiple texts
    ///
    /// - Parameters:
    ///   - texts: Input texts
    ///   - schema: Extraction schema
    ///   - batchSize: Batch size for processing
    ///   - threshold: Confidence threshold
    ///   - includeConfidence: Include confidence scores
    ///   - includeSpans: Include character-level positions
    /// - Returns: List of extraction results
    public func batchExtract(
        texts: [String],
        schema: Schema,
        batchSize: Int = 8,
        threshold: Float = 0.5,
        includeConfidence: Bool = false,
        includeSpans: Bool = false
    ) -> [[String: Any]] {
        guard !texts.isEmpty else { return [] }

        // Build schema dictionary
        let internalSchemaDict = schema.build()

        // Transform all texts
        var records: [TransformedRecord] = []
        for text in texts {
            let normalizedText = normalizeText(text)
            let record = processor.transform(text: normalizedText, schema: internalSchemaDict)
            records.append(record)
        }

        // Process in batches
        var allResults: [[String: Any]] = []

        for batchStart in stride(from: 0, to: records.count, by: batchSize) {
            let batchEnd = min(batchStart + batchSize, records.count)
            let batchRecords = Array(records[batchStart..<batchEnd])
            let batch = processor.collateBatch(batchRecords)

            let batchResults = extractFromBatch(
                batch: batch,
                threshold: threshold,
                metadata: schema.metadata,
                includeConfidence: includeConfidence,
                includeSpans: includeSpans
            )

            allResults.append(contentsOf: batchResults)
        }

        return allResults
    }

    // MARK: - Convenience Methods

    /// Extract entities from text
    public func extractEntities(
        text: String,
        entityTypes: [String],
        threshold: Float = 0.5,
        includeConfidence: Bool = false,
        includeSpans: Bool = false
    ) -> [String: Any] {
        let schema = createSchema().entities(entityTypes)
        return extract(
            text: text,
            schema: schema,
            threshold: threshold,
            includeConfidence: includeConfidence,
            includeSpans: includeSpans
        )
    }

    /// Classify text
    public func classifyText(
        text: String,
        task: String,
        labels: [String],
        multiLabel: Bool = false,
        threshold: Float = 0.5,
        includeConfidence: Bool = false
    ) -> [String: Any] {
        let schema = createSchema().classification(task: task, labels: labels, multiLabel: multiLabel)
        return extract(
            text: text,
            schema: schema,
            threshold: threshold,
            includeConfidence: includeConfidence
        )
    }

    /// Extract relations from text
    public func extractRelations(
        text: String,
        relationTypes: [String],
        threshold: Float = 0.5,
        includeConfidence: Bool = false,
        includeSpans: Bool = false
    ) -> [String: Any] {
        let schema = createSchema().relations(relationTypes)
        return extract(
            text: text,
            schema: schema,
            threshold: threshold,
            includeConfidence: includeConfidence,
            includeSpans: includeSpans
        )
    }

    // MARK: - Private Helpers

    private func normalizeText(_ text: String) -> String {
        var normalized = text
        if !normalized.isEmpty && !normalized.hasSuffix(".") &&
           !normalized.hasSuffix("!") && !normalized.hasSuffix("?") {
            normalized += "."
        } else if normalized.isEmpty {
            normalized = "."
        }
        return normalized
    }

    private func extractFromBatch(
        batch: PreprocessedBatch,
        threshold: Float,
        metadata: SchemaMetadata,
        includeConfidence: Bool,
        includeSpans: Bool
    ) -> [[String: Any]] {
        guard !batch.isEmpty else { return [] }

        // 1. Run encoder on full batch
        let encoderOutput = model.encode(batch.inputIds, attentionMask: batch.attentionMask)
        let hiddenStates = encoderOutput.lastHiddenState  // [batch, seq_len, hidden]

        // Evaluate to ensure computation is complete
        MLX.eval(hiddenStates)

        var results: [[String: Any]] = []
        let decoder = SpanDecoder(maxWidth: config.maxWidth)

        // 2. Process each sample in the batch
        for sampleIdx in 0..<batch.count {
            let sampleHidden = hiddenStates[sampleIdx]  // [seq_len, hidden]
            let mappedIndices = batch.mappedIndices[sampleIdx]
            let taskTypes = batch.taskTypes[sampleIdx]
            let schemaTokensList = batch.schemaTokensList[sampleIdx]
            let originalSchemas = batch.originalSchemas[sampleIdx]

            // Find text tokens position
            let textStartIdx = batch.textStartIndices[sampleIdx]
            let seqLen = mappedIndices.count

            // Get text token embeddings (subword level)
            let subwordEmbeddings = sampleHidden[textStartIdx..<seqLen]  // [num_subwords, hidden]

            // Pool subword embeddings to word-level embeddings
            // This aggregates subwords belonging to the same whitespace-split word
            let pooledEmbeddings = poolTextEmbeddings(
                subwordEmbeddings: subwordEmbeddings,
                wordFirstIndices: batch.wordFirstIndices[sampleIdx],
                wordSubwordCounts: batch.wordSubwordCounts[sampleIdx],
                poolingType: processor.tokenPooling
            )
            // Pooled positions cover the classification prefix AND the real text words:
            // buildClassificationPrefix prepends its tokens to the text segment. Python
            // separates the two — `text_len = len(start_mapping)` counts only real words,
            // and the prefix occupies `scores[..., :-text_len]` — so the span decode path
            // must use the real text length, not the pooled length. They are equal (and
            // the prefix empty) for every schema without choice fields, which is why this
            // only matters once choices are in play.
            let pooledLen = pooledEmbeddings.dim(0)
            let textLen = min(batch.startMappings[sampleIdx].count, pooledLen)
            let prefixLen = pooledLen - textLen
            let allTextTokens = batch.textTokens[sampleIdx]
            let prefixTokens = prefixLen > 0
                ? Array(allTextTokens.prefix(prefixLen))
                : []

            // Positions of the marker tokens ([P], [C], [E], [R], [L]) that contribute
            // embeddings, grouped per schema. Recorded during tokenization (Phase 3.5), so
            // no per-position vocabulary lookup happens here any more.
            let markerPositions = batch.schemaMarkerPositions[sampleIdx]

            // Compute span info if we have any span tasks
            let hasSpanTask = taskTypes.contains { $0 != "classifications" }
            var spanInfo: SpanInfo? = nil
            if hasSpanTask && textLen > 0 {
                spanInfo = model.computeSpanRep(pooledEmbeddings)
            }

            // Build classification field map for structures with choices
            var clsFields: [String: [String]] = [:]
            if let structures = originalSchemas["json_structures"] as? [[String: Any]] {
                for structDict in structures {
                    for (parent, fields) in structDict {
                        if let fieldsDict = fields as? [String: Any] {
                            for (fname, fval) in fieldsDict {
                                if let fvalDict = fval as? [String: Any],
                                   let choices = fvalDict["choices"] as? [String] {
                                    clsFields["\(parent).\(fname)"] = choices
                                }
                            }
                        }
                    }
                }
            }

            // Build every schema's marker gather up front, and with it the count-prediction
            // argmax for the span tasks, then force the whole set with one barrier. Each
            // schema's countPred graph is left exactly as it was — stacking them into a
            // single matmul would change the accumulation order and could flip a near-tie
            // count — so this batches only the synchronization. `item()` on an already
            // evaluated array is a copy, so the decode loop below no longer blocks per
            // schema.
            let schemaCount = schemaTokensList.count
            var schemaEmbeddings = [MLXArray?](repeating: nil, count: schemaCount)
            var countArgmax = [MLXArray?](repeating: nil, count: schemaCount)
            var pendingCounts: [MLXArray] = []

            for schemaIdx in 0..<schemaCount {
                guard schemaTokensList[schemaIdx].count >= 4,
                      schemaIdx < markerPositions.count,
                      !markerPositions[schemaIdx].isEmpty,
                      schemaIdx < taskTypes.count else { continue }

                // Gather this schema's marker embeddings in one kernel, replacing a
                // per-marker row slice plus a stack.
                let embs = MLX.take(  // [numTokens, hidden]
                    sampleHidden,
                    MLXArray(markerPositions[schemaIdx].map { Int32($0) }),
                    axis: 0
                )
                schemaEmbeddings[schemaIdx] = embs

                if taskTypes[schemaIdx] != "classifications", spanInfo != nil {
                    let countLogits = model.countPred(embs[0].expandedDimensions(axis: 0))
                    let argmax = MLX.argMax(countLogits.squeezed(axis: 0))
                    countArgmax[schemaIdx] = argmax
                    pendingCounts.append(argmax)
                }
            }

            if !pendingCounts.isEmpty {
                MLX.eval(pendingCounts)
            }

            // Process each schema separately (like Python does)
            var sampleResult: [String: Any] = [:]

            for (schemaIdx, schemaTokens) in schemaTokensList.enumerated() {
                guard let embs = schemaEmbeddings[schemaIdx] else { continue }

                let taskType = taskTypes[schemaIdx]

                // Get schema name (token at index 2, before [DESCRIPTION] if present)
                let schemaName = schemaTokens[2].components(separatedBy: " [DESCRIPTION] ")[0]

                if taskType == "classifications" {
                    // Classification task - use classifier MLP
                    extractClassificationResult(
                        results: &sampleResult,
                        schemaName: schemaName,
                        schema: originalSchemas,
                        embs: embs,
                        schemaTokens: schemaTokens,
                        includeConfidence: includeConfidence
                    )
                } else {
                    // Span-based task (entities, json_structures, relations)
                    guard let info = spanInfo,
                          let predCountArray = countArgmax[schemaIdx] else { continue }

                    extractSpanResult(
                        results: &sampleResult,
                        schemaName: schemaName,
                        taskType: taskType,
                        embs: embs,
                        predCount: Int(predCountArray.item(Int32.self)),
                        spanInfo: info,
                        schemaTokens: schemaTokens,
                        textLen: textLen,
                        originalText: batch.originalTexts[sampleIdx],
                        startMappings: batch.startMappings[sampleIdx],
                        endMappings: batch.endMappings[sampleIdx],
                        threshold: threshold,
                        metadata: metadata,
                        clsFields: clsFields,
                        prefixTokens: prefixTokens,
                        decoder: decoder,
                        includeConfidence: includeConfidence,
                        includeSpans: includeSpans
                    )
                }
            }

            results.append(sampleResult)
        }

        return results
    }

    /// Pool subword embeddings to word-level embeddings.
    ///
    /// This aggregates subword embeddings belonging to the same whitespace-split word.
    /// Matches Python: processor.py:_aggregate() and extract_embeddings_from_batch()
    ///
    /// The word boundaries arrive precomputed from tokenization (Phase 3.5): each word owns
    /// a contiguous run of subwords, so `first ..< first + count` describes it completely.
    /// That turns what used to be one lazy row slice per subword plus a stack per word —
    /// hundreds of kernel launches on a long document — into a single gather.
    ///
    /// - Parameters:
    ///   - subwordEmbeddings: Subword token embeddings [num_subwords, hidden]
    ///   - wordFirstIndices: Index of each word's first subword
    ///   - wordSubwordCounts: Number of subwords in each word
    ///   - poolingType: How to aggregate subwords ("first", "mean", or "max")
    /// - Returns: Word-level embeddings [num_words, hidden]
    func poolTextEmbeddings(   // internal, not private: covered directly by GatherIndexTests
        subwordEmbeddings: MLXArray,
        wordFirstIndices: [Int],
        wordSubwordCounts: [Int],
        poolingType: TokenPoolingType
    ) -> MLXArray {
        guard subwordEmbeddings.dim(0) > 0, !wordFirstIndices.isEmpty else {
            return MLXArray.zeros([0, config.hiddenSize])
        }

        if poolingType == .first {
            return MLX.take(subwordEmbeddings, MLXArray(wordFirstIndices.map { Int32($0) }), axis: 0)
        }

        // Gather every word's subwords into a rectangular [words, maxSubwords, hidden]
        // block. Short words repeat their last subword: harmless for `max` (idempotent)
        // and masked back out for `mean`.
        let wordCount = wordFirstIndices.count
        let maxSubwords = wordSubwordCounts.max() ?? 1

        var gatherIndices = [Int32](repeating: 0, count: wordCount * maxSubwords)
        for w in 0..<wordCount {
            let first = wordFirstIndices[w]
            let last = wordSubwordCounts[w] - 1
            for k in 0..<maxSubwords {
                gatherIndices[w * maxSubwords + k] = Int32(first + min(k, last))
            }
        }

        let gathered = MLX.take(subwordEmbeddings, MLXArray(gatherIndices), axis: 0)
            .reshaped([wordCount, maxSubwords, config.hiddenSize])

        if poolingType == .max {
            return MLX.max(gathered, axis: 1)
        }

        var maskValues = [Float](repeating: 0, count: wordCount * maxSubwords)
        for w in 0..<wordCount {
            for k in 0..<wordSubwordCounts[w] {
                maskValues[w * maxSubwords + k] = 1
            }
        }
        let mask = MLXArray(maskValues)
            .reshaped([wordCount, maxSubwords, 1])
            .asType(gathered.dtype)
        let counts = MLXArray(wordSubwordCounts.map { Float($0) })
            .reshaped([wordCount, 1])
            .asType(gathered.dtype)

        return MLX.sum(gathered * mask, axis: 1) / counts
    }

    // MARK: - Classification Extraction

    /// Extract classification result using the classifier MLP.
    ///
    /// Matches Python: GLiNER2._extract_classification_result()
    private func extractClassificationResult(
        results: inout [String: Any],
        schemaName: String,
        schema: [String: Any],
        embs: MLXArray,
        schemaTokens: [String],
        includeConfidence: Bool
    ) {
        // Find the classification config for this schema
        guard let classifications = schema["classifications"] as? [[String: Any]] else { return }

        let clsConfig = classifications.first { config in
            guard let task = config["task"] as? String else { return false }
            return schemaTokens[2].hasPrefix(task)
        }

        guard let config = clsConfig,
              let labels = config["labels"] as? [String] else { return }

        let isMultiLabel = config["multi_label"] as? Bool ?? false
        let classThreshold = config["cls_threshold"] as? Float ?? 0.5
        let activation = config["class_act"] as? String ?? "auto"

        // Get label embeddings (skip [P] token at index 0)
        let clsEmbeds = embs[1...]  // [numLabels, hidden]

        // Run classifier MLP: [numLabels, hidden] -> [numLabels, 1] -> [numLabels]
        var logits = model.classifier(clsEmbeds)  // [numLabels, 1]
        logits = logits.squeezed(axis: -1)  // [numLabels]

        // Apply activation
        var probs: MLXArray
        if activation == "sigmoid" {
            probs = MLX.sigmoid(logits)
        } else if activation == "softmax" {
            probs = MLX.softmax(logits, axis: -1)
        } else {
            // "auto": sigmoid for multi-label, softmax for single-label
            probs = isMultiLabel ? MLX.sigmoid(logits) : MLX.softmax(logits, axis: -1)
        }

        // Single bulk copy to the CPU, then read every label in pure Swift
        // (replaces per-label `.item()` syncs and the redundant argmax evals).
        let probsArr = probs.asArray(Float32.self)

        // Extract results
        let numLabels = labels.count
        guard probsArr.count >= numLabels else { return }

        // argmax over the full vector, first-max on ties (matches MLX.argMax)
        func argmaxFull() -> Int {
            var bi = 0
            var bv = probsArr[0]
            for j in 1..<probsArr.count where probsArr[j] > bv { bv = probsArr[j]; bi = j }
            return bi
        }

        if isMultiLabel {
            // Multi-label: return all labels above threshold
            var chosen: [(String, Float)] = []
            for j in 0..<numLabels {
                let prob = probsArr[j]
                if prob >= classThreshold {
                    chosen.append((labels[j], prob))
                }
            }

            // If none above threshold, return the best one
            if chosen.isEmpty {
                let bestIdx = argmaxFull()
                if bestIdx < numLabels {
                    let bestProb = probsArr[bestIdx]
                    chosen = [(labels[bestIdx], bestProb)]
                }
            }

            if includeConfidence {
                results[schemaName] = chosen.map { ["label": $0.0, "confidence": $0.1] }
            } else {
                results[schemaName] = chosen.map { $0.0 }
            }
        } else {
            // Single-label: return the best label
            let bestIdx = argmaxFull()
            guard bestIdx < numLabels else { return }
            let bestProb = probsArr[bestIdx]

            if includeConfidence {
                results[schemaName] = ["label": labels[bestIdx], "confidence": bestProb]
            } else {
                results[schemaName] = labels[bestIdx]
            }
        }
    }

    // MARK: - Span-Based Extraction

    /// Extract span-based results (entities, structures, relations).
    ///
    /// Matches Python: GLiNER2._extract_span_result()
    private func extractSpanResult(
        results: inout [String: Any],
        schemaName: String,
        taskType: String,
        embs: MLXArray,
        predCount: Int,
        spanInfo: SpanInfo,
        schemaTokens: [String],
        textLen: Int,
        originalText: String,
        startMappings: [Int],
        endMappings: [Int],
        threshold: Float,
        metadata: SchemaMetadata,
        clsFields: [String: [String]],
        prefixTokens: [String],
        decoder: SpanDecoder,
        includeConfidence: Bool,
        includeSpans: Bool
    ) {
        // Get field names from schema tokens
        // Schema tokens: ["(", "[P]", "schemaName", "(", "[E]", "field1", "[E]", "field2", ..., ")", ")"]
        var fieldNames: [String] = []
        for j in 0..<(schemaTokens.count - 1) {
            let token = schemaTokens[j]
            if token == "[E]" || token == "[C]" || token == "[R]" {
                fieldNames.append(schemaTokens[j + 1])
            }
        }

        guard !fieldNames.isEmpty else {
            if taskType == "entities" {
                results[schemaName] = [:] as [String: Any]
            } else {
                results[schemaName] = [] as [[String: Any]]
            }
            return
        }

        // `predCount` was computed from this schema's [P] embedding by the caller, which
        // evaluates every schema's count argmax in one barrier (Phase 2.2).
        if predCount <= 0 {
            if taskType == "entities" {
                results[schemaName] = [:] as [String: Any]
            } else if taskType == "relations" {
                // Still register the relation under `relation_extraction` with an empty
                // list: Python lists every requested relation even when nothing matched.
                var grouped = results["relation_extraction"] as? [String: Any] ?? [:]
                grouped[schemaName] = [] as [Any]
                results["relation_extraction"] = grouped
            } else {
                // An empty structure formats to `{}` in Python, not `[]`.
                results[schemaName] = [String: Any]()
            }
            return
        }

        // Get field embeddings (skip [P] token)
        let fieldEmbs = embs[1...]  // [numFields, hidden]

        // Get count-aware structure projections
        let structProj = model.countEmbed(fieldEmbs, goldCountVal: predCount)  // [count, fields, hidden]

        // Compute span scores
        let L = spanInfo.spansIdx.dim(1) / config.maxWidth
        let spanRepReshaped = spanInfo.spanRep.reshaped([L, config.maxWidth, config.hiddenSize])

        // Einsum: scores[b,p,l,k] = sum_d(spanRep[l,k,d] * structProj[b,p,d])
        var scoreArray = MLX.einsum("lkd,cpd->cplk", spanRepReshaped, structProj)
        scoreArray = MLX.sigmoid(scoreArray)  // [count, fields, L, maxWidth]

        // One readback for the whole schema. `asArray` evaluates, so this also subsumes
        // the explicit eval that used to sit here (Phase 2.1).
        let spanScores = SpanScoreBuffer(scoreArray)

        // Extract based on task type
        if taskType == "entities" {
            extractEntities(
                results: &results,
                schemaName: schemaName,
                fieldNames: fieldNames,
                spanScores: spanScores,
                textLen: textLen,
                originalText: originalText,
                startMappings: startMappings,
                endMappings: endMappings,
                threshold: threshold,
                metadata: metadata,
                decoder: decoder,
                includeConfidence: includeConfidence,
                includeSpans: includeSpans
            )
        } else if taskType == "relations" {
            extractRelations(
                results: &results,
                schemaName: schemaName,
                fieldNames: fieldNames,
                spanScores: spanScores,
                predCount: predCount,
                textLen: textLen,
                originalText: originalText,
                startMappings: startMappings,
                endMappings: endMappings,
                threshold: threshold,
                metadata: metadata,
                decoder: decoder,
                includeConfidence: includeConfidence,
                includeSpans: includeSpans
            )
        } else {
            // json_structures
            extractStructures(
                results: &results,
                schemaName: schemaName,
                fieldNames: fieldNames,
                spanScores: spanScores,
                predCount: predCount,
                textLen: textLen,
                originalText: originalText,
                startMappings: startMappings,
                endMappings: endMappings,
                threshold: threshold,
                metadata: metadata,
                clsFields: clsFields,
                prefixTokens: prefixTokens,
                decoder: decoder,
                includeConfidence: includeConfidence,
                includeSpans: includeSpans
            )
        }
    }

    // MARK: - Entity Extraction

    private func extractEntities(
        results: inout [String: Any],
        schemaName: String,
        fieldNames: [String],
        spanScores: SpanScoreBuffer,
        textLen: Int,
        originalText: String,
        startMappings: [Int],
        endMappings: [Int],
        threshold: Float,
        metadata: SchemaMetadata,
        decoder: SpanDecoder,
        includeConfidence: Bool,
        includeSpans: Bool
    ) {
        // For entities, use scores[0, :, -textLen:] (first count slot, all fields, text portion)
        let startIdx = spanScores.rows - textLen

        var entityResults: [String: [Any]] = [:]

        for (fieldIdx, entityName) in fieldNames.enumerated() {
            guard fieldIdx < spanScores.fields else { continue }

            let fieldThreshold = metadata.entityMetadata[entityName]?.threshold ?? threshold

            let spans = decoder.findSpans(
                scores: spanScores,
                instance: 0,
                field: fieldIdx,
                rowOffset: startIdx,
                threshold: fieldThreshold,
                textLen: textLen,
                text: originalText,
                startMap: startMappings,
                endMap: endMappings
            )

            let formatted = decoder.formatSpans(
                spans,
                includeConfidence: includeConfidence,
                includeSpans: includeSpans
            )

            // Python's formatting pass keeps only the first span per distinct lowercased
            // surface text, even when spans are included (engine.py:_format_entity_dict).
            // A document repeating "Apple" ten times yields one entry, not ten.
            entityResults[entityName] = Self.dedupeByLowercasedText(formatted)
        }

        results[schemaName] = entityResults
    }

    /// Drop later values whose `text` (lowercased) was already seen, preserving order.
    ///
    /// Mirrors Python's `_format_entity_dict` / `_format_struct` de-duplication, which
    /// applies to both plain-string and span-dictionary output shapes.
    static func dedupeByLowercasedText(_ values: [Any]) -> [Any] {
        var unique: [Any] = []
        var seen: Set<String> = []
        unique.reserveCapacity(values.count)

        for value in values {
            let text: String?
            if let string = value as? String {
                text = string
            } else if let dict = value as? [String: Any] {
                text = dict["text"] as? String
            } else {
                text = nil
            }

            guard let key = text else {
                unique.append(value)   // shape we do not de-duplicate on
                continue
            }
            guard !key.isEmpty else { continue }   // Python drops falsy text
            let lowered = key.lowercased()
            if seen.insert(lowered).inserted {
                unique.append(value)
            }
        }
        return unique
    }

    // MARK: - Relation Extraction

    private func extractRelations(
        results: inout [String: Any],
        schemaName: String,
        fieldNames: [String],
        spanScores: SpanScoreBuffer,
        predCount: Int,
        textLen: Int,
        originalText: String,
        startMappings: [Int],
        endMappings: [Int],
        threshold: Float,
        metadata: SchemaMetadata,
        decoder: SpanDecoder,
        includeConfidence: Bool,
        includeSpans: Bool
    ) {
        let startIdx = spanScores.rows - textLen

        var instances: [Any] = []

        // Process each count instance
        for inst in 0..<predCount {
            var fieldData: [(String?, Float, Int, Int)?] = []

            for (fieldIdx, _) in fieldNames.enumerated() {
                guard fieldIdx < spanScores.fields else {
                    fieldData.append(nil)
                    continue
                }

                let spans = decoder.findSpans(
                    scores: spanScores,
                    instance: inst,
                    field: fieldIdx,
                    rowOffset: startIdx,
                    threshold: threshold,
                    textLen: textLen,
                    text: originalText,
                    startMap: startMappings,
                    endMap: endMappings
                )

                if let first = spans.first {
                    fieldData.append((first.text, first.confidence, first.charStart, first.charEnd))
                } else {
                    fieldData.append(nil)
                }
            }

            // Relations need at least head and tail
            if fieldData.count >= 2,
               let head = fieldData[0],
               let tail = fieldData[1],
               let headText = head.0,
               let tailText = tail.0 {

                if includeSpans && includeConfidence {
                    instances.append([
                        "head": ["text": headText, "confidence": head.1, "start": head.2, "end": head.3],
                        "tail": ["text": tailText, "confidence": tail.1, "start": tail.2, "end": tail.3]
                    ])
                } else if includeSpans {
                    instances.append([
                        "head": ["text": headText, "start": head.2, "end": head.3],
                        "tail": ["text": tailText, "start": tail.2, "end": tail.3]
                    ])
                } else if includeConfidence {
                    instances.append([
                        "head": ["text": headText, "confidence": head.1],
                        "tail": ["text": tailText, "confidence": tail.1]
                    ])
                } else {
                    instances.append((headText, tailText))
                }
            }
        }

        // Python groups every relation under a top-level `relation_extraction` key, and
        // lists each requested relation even when it matched nothing (engine.py:1066-1075).
        var grouped = results["relation_extraction"] as? [String: Any] ?? [:]
        grouped[schemaName] = instances
        results["relation_extraction"] = grouped
    }

    // MARK: - Structure Extraction

    private func extractStructures(
        results: inout [String: Any],
        schemaName: String,
        fieldNames: [String],
        spanScores: SpanScoreBuffer,
        predCount: Int,
        textLen: Int,
        originalText: String,
        startMappings: [Int],
        endMappings: [Int],
        threshold: Float,
        metadata: SchemaMetadata,
        clsFields: [String: [String]],
        prefixTokens: [String],
        decoder: SpanDecoder,
        includeConfidence: Bool,
        includeSpans: Bool
    ) {
        let startIdx = spanScores.rows - textLen

        var instances: [[String: Any]] = []

        for inst in 0..<predCount {
            var instance: [String: Any] = [:]

            for (fieldIdx, fieldName) in fieldNames.enumerated() {
                guard fieldIdx < spanScores.fields else { continue }

                let fieldKey = "\(schemaName).\(fieldName)"
                let fieldMeta = metadata.fieldMetadata[fieldKey]
                let fieldThreshold = fieldMeta?.threshold ?? threshold
                let dtype = fieldMeta?.dtype ?? "list"

                // Check if this is a choice field
                if let choices = clsFields[fieldKey] {
                    // Choice fields are scored against the classification-prefix region,
                    // which occupies the positions before the real text words. Matches
                    // Python: prefix_scores = span_scores[inst, fidx, :-text_len] and
                    // _find_choice_idx(choice, text_tokens[:-text_len]).
                    let result = decoder.decodeChoiceField(
                        scores: spanScores,
                        instance: inst,
                        field: fieldIdx,
                        prefixLength: startIdx,
                        choices: choices,
                        textTokens: prefixTokens,
                        threshold: fieldThreshold,
                        dtype: dtype,
                        includeConfidence: includeConfidence
                    )

                    // Python keeps the key with a null/empty value rather than dropping
                    // it (engine.py:660); the instance-level content gate below decides
                    // whether the whole instance survives.
                    instance[fieldName] = result ?? NSNull()
                } else {
                    // Regular span field: use text scores
                    var spans = decoder.findSpans(
                        scores: spanScores,
                        instance: inst,
                        field: fieldIdx,
                        rowOffset: startIdx,
                        threshold: fieldThreshold,
                        textLen: textLen,
                        text: originalText,
                        startMap: startMappings,
                        endMap: endMappings
                    )

                    // Drop spans failing any validator before formatting, as Python does
                    // (engine.py:668).
                    if let validators = fieldMeta?.validators, !validators.isEmpty {
                        spans = spans.filter { span in
                            validators.allSatisfy { $0.validate(span.text) }
                        }
                    }

                    if dtype == "list" {
                        instance[fieldName] = decoder.formatSpans(
                            spans,
                            includeConfidence: includeConfidence,
                            includeSpans: includeSpans
                        )
                    } else {
                        // dtype == "str": return first span only
                        if let first = spans.first {
                            if includeSpans && includeConfidence {
                                instance[fieldName] = [
                                    "text": first.text,
                                    "confidence": first.confidence,
                                    "start": first.charStart,
                                    "end": first.charEnd
                                ]
                            } else if includeSpans {
                                instance[fieldName] = [
                                    "text": first.text,
                                    "start": first.charStart,
                                    "end": first.charEnd
                                ]
                            } else if includeConfidence {
                                instance[fieldName] = ["text": first.text, "confidence": first.confidence]
                            } else {
                                instance[fieldName] = first.text
                            }
                        } else {
                            instance[fieldName] = nil
                        }
                    }
                }
            }

            // Only add if instance has any content. Matches Python's
            // `any(v is not None and v != [])` (engine.py:697) — a null choice value and
            // an empty span list both count as "no content".
            let hasContent = instance.values.contains { value in
                if value is NSNull { return false }
                if let arr = value as? [Any], arr.isEmpty { return false }
                if let str = value as? String, str.isEmpty { return false }
                return true
            }

            if hasContent {
                // De-duplicate repeated surface text within each list-valued field, as
                // Python's _format_struct does.
                for (fieldName, value) in instance {
                    if let list = value as? [Any] {
                        instance[fieldName] = Self.dedupeByLowercasedText(list)
                    }
                }
                instances.append(instance)
            }
        }

        // With no surviving instance Python leaves an empty struct dict, which formats to
        // `{}` — not the empty list Swift would otherwise emit.
        results[schemaName] = instances.isEmpty ? [String: Any]() : instances
    }
}

// MARK: - Schema Builder

/// Schema builder for extraction tasks.
public class Schema {
    /// Internal schema dictionary
    var internalSchemaDict: [String: Any] = [
        "json_structures": [],
        "classifications": [],
        "entities": [:],
        "relations": [],
        "json_descriptions": [:],
        "entity_descriptions": [:]
    ]

    /// Metadata for field configurations
    var metadata: SchemaMetadata = SchemaMetadata()

    public init() {}

    /// Add entity extraction task
    @discardableResult
    public func entities(
        _ entityTypes: [String],
        dtype: String = "list",
        threshold: Float? = nil
    ) -> Schema {
        var entitiesDict = internalSchemaDict["entities"] as? [String: Any] ?? [:]
        for entityType in entityTypes {
            entitiesDict[entityType] = ""
            metadata.entityMetadata[entityType] = EntityMetadata(dtype: dtype, threshold: threshold)
        }
        internalSchemaDict["entities"] = entitiesDict
        // Store entity order for parity with Python
        internalSchemaDict["_entity_order"] = entityTypes
        metadata.entityOrder = entityTypes
        return self
    }

    /// Add entity extraction task with descriptions
    ///
    /// - Parameter typesWithDescriptions: Dictionary mapping entity type names to their descriptions
    /// - Returns: Schema for fluent chaining
    ///
    /// Example:
    /// ```swift
    /// let schema = model.createSchema().entities([
    ///     "person": "A human being's name",
    ///     "company": "A business organization"
    /// ])
    /// ```
    @discardableResult
    public func entities(_ typesWithDescriptions: [String: String]) -> Schema {
        var entitiesDict = internalSchemaDict["entities"] as? [String: Any] ?? [:]
        // Note: Dictionary iteration order is not guaranteed, so sort for consistency
        let sortedKeys = typesWithDescriptions.keys.sorted()
        for entityType in sortedKeys {
            entitiesDict[entityType] = ""
        }
        internalSchemaDict["entities"] = entitiesDict
        internalSchemaDict["entity_descriptions"] = typesWithDescriptions
        // Store entity order for parity with Python (sorted since dict order is not preserved)
        internalSchemaDict["_entity_order"] = sortedKeys
        return self
    }

    /// Add classification task
    @discardableResult
    public func classification(
        task: String,
        labels: [String],
        multiLabel: Bool = false,
        threshold: Float = 0.5
    ) -> Schema {
        var classifications = internalSchemaDict["classifications"] as? [[String: Any]] ?? []
        classifications.append([
            "task": task,
            "labels": labels,
            "multi_label": multiLabel,
            "cls_threshold": threshold,
            "true_label": ["N/A"]
        ])
        internalSchemaDict["classifications"] = classifications
        return self
    }

    /// Add relation extraction task
    @discardableResult
    public func relations(_ relationTypes: [String], threshold: Float? = nil) -> Schema {
        var relations = internalSchemaDict["relations"] as? [[String: [String: Any]]] ?? []
        for relationType in relationTypes {
            relations.append([relationType: ["head": "", "tail": ""]])
            metadata.relationMetadata[relationType] = RelationMetadata(threshold: threshold)
        }
        internalSchemaDict["relations"] = relations
        metadata.relationOrder.append(contentsOf: relationTypes)
        return self
    }

    /// Add structure extraction task
    @discardableResult
    public func structure(_ name: String) -> StructureBuilder {
        StructureBuilder(schema: self, name: name)
    }

    /// Build the schema dictionary
    public func build() -> [String: Any] {
        internalSchemaDict
    }
}

/// Builder for structure schemas
public class StructureBuilder {
    private let schema: Schema
    private let name: String
    private var fields: [String: Any] = [:]
    private var fieldOrder: [String] = []  // Track insertion order
    private var descriptions: [String: String] = [:]

    init(schema: Schema, name: String) {
        self.schema = schema
        self.name = name
    }

    /// Add a field to the structure
    ///
    /// - Parameters:
    ///   - fieldName: Name of the field
    ///   - dtype: Data type ("list" or "str")
    ///   - choices: Optional list of choices for classification fields
    ///   - description: Optional description for the field (used in schema tokens)
    ///   - threshold: Optional confidence threshold for this field
    ///   - validators: Optional regex validators; spans failing any of them are dropped
    @discardableResult
    public func field(
        _ fieldName: String,
        dtype: String = "list",
        choices: [String]? = nil,
        description: String? = nil,
        threshold: Float? = nil,
        validators: [RegexValidator]? = nil
    ) -> StructureBuilder {
        if let choices = choices {
            fields[fieldName] = ["value": "", "choices": choices]
        } else {
            fields[fieldName] = ""
        }

        // Record the per-field configuration so decoding can honour it. Matches Python's
        // Schema._store_field_metadata; without this the decoder falls back to
        // dtype "list" and the call-level threshold for every field.
        schema.metadata.fieldMetadata["\(name).\(fieldName)"] = FieldMetadata(
            dtype: dtype,
            threshold: threshold,
            choices: choices,
            validators: validators
        )

        // Track insertion order (only add if new)
        if !fieldOrder.contains(fieldName) {
            fieldOrder.append(fieldName)
        }

        // Store description if provided
        if let description = description {
            descriptions[fieldName] = description
        }

        return self
    }

    /// Finish building and return to schema
    @discardableResult
    public func done() -> Schema {
        var structures = schema.internalSchemaDict["json_structures"] as? [[String: Any]] ?? []
        structures.append([name: fields])
        schema.internalSchemaDict["json_structures"] = structures

        // Store field order for this structure (critical for parity with Python)
        var fieldOrders = schema.internalSchemaDict["_field_orders"] as? [String: [String]] ?? [:]
        fieldOrders[name] = fieldOrder
        schema.internalSchemaDict["_field_orders"] = fieldOrders

        // Store descriptions if any were provided
        if !descriptions.isEmpty {
            var jsonDescriptions = schema.internalSchemaDict["json_descriptions"] as? [String: [String: String]] ?? [:]
            jsonDescriptions[name] = descriptions
            schema.internalSchemaDict["json_descriptions"] = jsonDescriptions
        }

        return schema
    }

    /// Chain to add entities
    @discardableResult
    public func entities(_ entityTypes: [String]) -> Schema {
        _ = done()
        return schema.entities(entityTypes)
    }

    /// Chain to add classification
    @discardableResult
    public func classification(task: String, labels: [String], multiLabel: Bool = false) -> Schema {
        _ = done()
        return schema.classification(task: task, labels: labels, multiLabel: multiLabel)
    }
}

/// Metadata for schema configurations
public struct SchemaMetadata {
    var fieldMetadata: [String: FieldMetadata] = [:]
    var entityMetadata: [String: EntityMetadata] = [:]
    var relationMetadata: [String: RelationMetadata] = [:]
    var fieldOrders: [String: [String]] = [:]
    var entityOrder: [String] = []
    var relationOrder: [String] = []
}

public struct FieldMetadata {
    var dtype: String = "list"
    var threshold: Float?
    var choices: [String]?
    var validators: [RegexValidator]?
}

public struct EntityMetadata {
    var dtype: String = "list"
    var threshold: Float?
}

public struct RelationMetadata {
    var threshold: Float?
}

// MARK: - Errors

/// Errors that can occur during GLiNER2 operations
public enum GLiNER2Error: Error, LocalizedError {
    case fileNotFound(String)
    case invalidConfig(String)
    case unsupportedFormat(String)
    case weightLoadingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let file):
            return "File not found: \(file)"
        case .invalidConfig(let reason):
            return "Invalid configuration: \(reason)"
        case .unsupportedFormat(let reason):
            return "Unsupported format: \(reason)"
        case .weightLoadingFailed(let reason):
            return "Failed to load weights: \(reason)"
        }
    }
}
