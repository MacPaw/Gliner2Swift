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
    /// - Parameter pathOrRepo: HuggingFace repo ID or local path
    /// - Returns: Initialized GLiNER2 model
    public static func fromPretrained(_ pathOrRepo: String) async throws -> GLiNER2 {
        // Determine if local path or HuggingFace repo
        let isLocalPath = FileManager.default.fileExists(atPath: pathOrRepo)

        let configUrl: URL
        let modelWeightsUrl: URL
        let encoderWeightsUrl: URL
        let tokenizerConfigUrl: URL

        if isLocalPath {
            // Local directory
            let baseUrl = URL(fileURLWithPath: pathOrRepo)
            configUrl = baseUrl.appendingPathComponent("config.json")
            modelWeightsUrl = baseUrl.appendingPathComponent("gliner2_weights.safetensors")
            encoderWeightsUrl = baseUrl.appendingPathComponent("encoder_weights.safetensors")
            tokenizerConfigUrl = baseUrl.appendingPathComponent("tokenizer.json")

            // Fall back to single model.safetensors if split weights don't exist
            if !FileManager.default.fileExists(atPath: modelWeightsUrl.path) {
                // Use combined weights file if separate don't exist
                throw GLiNER2Error.fileNotFound("gliner2_weights.safetensors")
            }
        } else {
            // Download from HuggingFace Hub
            let files = try await HuggingFaceLoader.downloadModel(repo: pathOrRepo)

            guard let config = files["config.json"] else {
                throw GLiNER2Error.fileNotFound("config.json")
            }
            configUrl = config

            // Try to get split weight files or fall back to combined
            if let modelW = files["gliner2_weights.safetensors"],
               let encoderW = files["encoder_weights.safetensors"] {
                modelWeightsUrl = modelW
                encoderWeightsUrl = encoderW
            } else if let combined = files["model.safetensors"] {
                // TODO: Need to handle combined weights
                throw GLiNER2Error.unsupportedFormat("Combined model.safetensors not yet supported. Use convert_weights.py to split.")
            } else {
                throw GLiNER2Error.fileNotFound("model weights")
            }

            guard let tokenizer = files["tokenizer.json"] else {
                throw GLiNER2Error.fileNotFound("tokenizer.json")
            }
            tokenizerConfigUrl = tokenizer
        }

        // 1. Load configuration
        let config = try ExtractorConfig.load(from: configUrl)

        // 2. Initialize processor with tokenizer
        // Convert ExtractorConfig.TokenPoolingType to SchemaTransformer.TokenPoolingType
        let poolingType: TokenPoolingType
        switch config.tokenPooling {
        case .first: poolingType = .first
        case .mean: poolingType = .mean
        case .max: poolingType = .max
        }

        let processor: SchemaTransformer
        if isLocalPath {
            // Load tokenizer from local directory (synchronous with custom tokenizer)
            processor = try SchemaTransformer.createFromLocalDirectory(
                directoryUrl: URL(fileURLWithPath: pathOrRepo),
                tokenPooling: poolingType
            )
        } else {
            // For HuggingFace repos, we need to download first then load locally
            let files = try await HuggingFaceLoader.downloadModel(repo: pathOrRepo)
            guard let tokenizerUrl = files["tokenizer.json"] else {
                throw GLiNER2Error.fileNotFound("tokenizer.json")
            }
            let tokenizerDir = tokenizerUrl.deletingLastPathComponent()
            processor = try SchemaTransformer.createFromLocalDirectory(
                directoryUrl: tokenizerDir,
                tokenPooling: poolingType
            )
        }

        // 3. Create model instance
        let gliner2 = GLiNER2(config: config, processor: processor)

        // 4. Load weights
        try gliner2.model.loadWeights(
            modelWeightsUrl: modelWeightsUrl,
            encoderWeightsUrl: encoderWeightsUrl
        )

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
            let textStartIdx = findTextStartIndex(mapping: mappedIndices)
            let seqLen = mappedIndices.count
            let textLen = seqLen - textStartIdx

            // Get text token embeddings
            let textEmbeddings = sampleHidden[textStartIdx..<seqLen]  // [text_len, hidden]

            // Extract schema embeddings per-schema (grouped by schema_idx in mapping)
            // Only special tokens ([P], [C], [E], [R], [L]) contribute embeddings
            // Get input IDs for this sample (needed for token string lookup)
            let sampleInputIds = batch.getInputIds(for: sampleIdx)

            let schemaEmbsList = extractSchemaEmbeddingsPerSchema(
                hiddenStates: sampleHidden,
                mappedIndices: mappedIndices,
                schemaTokensList: schemaTokensList,
                inputIds: sampleInputIds
            )

            // Compute span info if we have any span tasks
            let hasSpanTask = taskTypes.contains { $0 != "classifications" }
            var spanInfo: SpanInfo? = nil
            if hasSpanTask && textLen > 0 {
                spanInfo = model.computeSpanRep(textEmbeddings)
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

            // Process each schema separately (like Python does)
            var sampleResult: [String: Any] = [:]

            for (schemaIdx, schemaTokens) in schemaTokensList.enumerated() {
                guard schemaTokens.count >= 4 else { continue }
                guard schemaIdx < schemaEmbsList.count, !schemaEmbsList[schemaIdx].isEmpty else { continue }
                guard schemaIdx < taskTypes.count else { continue }

                let taskType = taskTypes[schemaIdx]

                // Get schema name (token at index 2, before [DESCRIPTION] if present)
                let schemaName = schemaTokens[2].components(separatedBy: " [DESCRIPTION] ")[0]

                // Stack schema embeddings for this schema
                let embs = MLX.stacked(schemaEmbsList[schemaIdx], axis: 0)  // [numTokens, hidden]

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
                    guard let info = spanInfo else { continue }

                    extractSpanResult(
                        results: &sampleResult,
                        schemaName: schemaName,
                        taskType: taskType,
                        embs: embs,
                        spanInfo: info,
                        schemaTokens: schemaTokens,
                        textLen: textLen,
                        originalText: batch.originalTexts[sampleIdx],
                        startMappings: batch.startMappings[sampleIdx],
                        endMappings: batch.endMappings[sampleIdx],
                        threshold: threshold,
                        metadata: metadata,
                        clsFields: clsFields,
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

    /// Find the index where text tokens start in the mapping
    private func findTextStartIndex(mapping: [MappedIndex]) -> Int {
        for (idx, map) in mapping.enumerated() {
            if map.segmentType == .text {
                return idx
            }
        }
        return mapping.count
    }

    /// Extract schema embeddings (only special tokens that contribute)
    private func extractSchemaEmbeddings(
        hiddenStates: MLXArray,
        mappedIndices: [MappedIndex]
    ) -> MLXArray {
        var schemaEmbList: [MLXArray] = []

        for (idx, mapping) in mappedIndices.enumerated() {
            if mapping.segmentType == .schema {
                // Only extract [P], [C], [E], [R], [L] token embeddings
                schemaEmbList.append(hiddenStates[idx].expandedDimensions(axis: 0))
            }
        }

        guard !schemaEmbList.isEmpty else {
            return MLXArray.zeros([0, config.hiddenSize])
        }

        return MLX.concatenated(schemaEmbList, axis: 0)
    }

    /// Special marker tokens that contribute embeddings
    private static let specialMarkerTokens: Set<String> = ["[P]", "[C]", "[E]", "[R]", "[L]"]

    /// Extract schema embeddings grouped by schema index.
    /// Returns a list of embedding arrays, one per schema.
    /// Only special tokens ([P], [C], [E], [R], [L]) contribute embeddings.
    ///
    /// This matches Python's approach in processor.py:1018-1025:
    /// ```python
    /// for j, tid in enumerate(ids):
    ///     seg_type, orig_idx, schema_idx = mappings[j]
    ///     emb = embs[j]
    ///     if seg_type == "schema":
    ///         tok = self.tokenizer.convert_ids_to_tokens(tid)
    ///         if tok in special_set:
    ///             schema_embs[schema_idx].append(emb)
    /// ```
    private func extractSchemaEmbeddingsPerSchema(
        hiddenStates: MLXArray,
        mappedIndices: [MappedIndex],
        schemaTokensList: [[String]],
        inputIds: [Int]
    ) -> [[MLXArray]] {
        let numSchemas = schemaTokensList.count
        var schemaEmbs: [[MLXArray]] = Array(repeating: [], count: numSchemas)

        // Match Python: iterate through all positions and check actual token string
        for (idx, mapping) in mappedIndices.enumerated() {
            guard mapping.segmentType == .schema else { continue }

            let schemaIdx = mapping.schemaIndex
            guard schemaIdx >= 0 && schemaIdx < numSchemas else { continue }
            guard idx < inputIds.count else { continue }

            // Get actual token string (matching Python's convert_ids_to_tokens)
            let tokenId = inputIds[idx]
            if let tokenStr = processor.tokenizer.idToToken(tokenId),
               Self.specialMarkerTokens.contains(tokenStr) {
                schemaEmbs[schemaIdx].append(hiddenStates[idx])
            }
        }

        return schemaEmbs
    }

    /// Get argmax of a 1D array
    private func argmax(_ arr: MLXArray) -> Int {
        MLX.eval(arr)
        let maxIdx = MLX.argMax(arr)
        MLX.eval(maxIdx)
        return Int(maxIdx.item(Int32.self))
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

        MLX.eval(probs)

        // Extract results
        let numLabels = labels.count
        guard probs.dim(0) >= numLabels else { return }

        if isMultiLabel {
            // Multi-label: return all labels above threshold
            var chosen: [(String, Float)] = []
            for j in 0..<numLabels {
                let prob = Float(probs[j].item(Float32.self))
                if prob >= classThreshold {
                    chosen.append((labels[j], prob))
                }
            }

            // If none above threshold, return the best one
            if chosen.isEmpty {
                let bestIdx = argmax(probs)
                if bestIdx < numLabels {
                    let bestProb = Float(probs[bestIdx].item(Float32.self))
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
            let bestIdx = argmax(probs)
            guard bestIdx < numLabels else { return }
            let bestProb = Float(probs[bestIdx].item(Float32.self))

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
        spanInfo: SpanInfo,
        schemaTokens: [String],
        textLen: Int,
        originalText: String,
        startMappings: [Int],
        endMappings: [Int],
        threshold: Float,
        metadata: SchemaMetadata,
        clsFields: [String: [String]],
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

        // Predict count using [P] token (first embedding)
        let countLogits = model.countPred(embs[0].expandedDimensions(axis: 0))
        let predCount = argmax(countLogits.squeezed(axis: 0))

        if predCount <= 0 {
            if taskType == "entities" {
                results[schemaName] = [:] as [String: Any]
            } else if taskType == "relations" {
                results[schemaName] = [] as [(String, String)]
            } else {
                results[schemaName] = [] as [[String: Any]]
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
        var spanScores = MLX.einsum("lkd,cpd->cplk", spanRepReshaped, structProj)
        spanScores = MLX.sigmoid(spanScores)  // [count, fields, L, maxWidth]
        MLX.eval(spanScores)

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
        spanScores: MLXArray,
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
        let totalLen = spanScores.dim(2)
        let startIdx = totalLen - textLen
        let scores = spanScores[0]  // [fields, L, maxWidth]

        var entityResults: [String: [Any]] = [:]

        for (fieldIdx, entityName) in fieldNames.enumerated() {
            guard fieldIdx < scores.dim(0) else { continue }

            // Get scores for this field's text spans
            let fieldScores = scores[fieldIdx, startIdx...]  // [textLen, maxWidth]

            let fieldThreshold = metadata.entityMetadata[entityName]?.threshold ?? threshold

            let spans = decoder.findSpans(
                scores: fieldScores,
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

            entityResults[entityName] = formatted
        }

        results[schemaName] = entityResults
    }

    // MARK: - Relation Extraction

    private func extractRelations(
        results: inout [String: Any],
        schemaName: String,
        fieldNames: [String],
        spanScores: MLXArray,
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
        let totalLen = spanScores.dim(2)
        let startIdx = totalLen - textLen

        var instances: [Any] = []

        // Process each count instance
        for inst in 0..<predCount {
            let instScores = spanScores[inst]  // [fields, L, maxWidth]

            var fieldData: [(String?, Float, Int, Int)?] = []

            for (fieldIdx, _) in fieldNames.enumerated() {
                guard fieldIdx < instScores.dim(0) else {
                    fieldData.append(nil)
                    continue
                }

                let fieldScores = instScores[fieldIdx, startIdx...]  // [textLen, maxWidth]

                let spans = decoder.findSpans(
                    scores: fieldScores,
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

        results[schemaName] = instances
    }

    // MARK: - Structure Extraction

    private func extractStructures(
        results: inout [String: Any],
        schemaName: String,
        fieldNames: [String],
        spanScores: MLXArray,
        predCount: Int,
        textLen: Int,
        originalText: String,
        startMappings: [Int],
        endMappings: [Int],
        threshold: Float,
        metadata: SchemaMetadata,
        clsFields: [String: [String]],
        decoder: SpanDecoder,
        includeConfidence: Bool,
        includeSpans: Bool
    ) {
        let totalLen = spanScores.dim(2)
        let startIdx = totalLen - textLen

        var instances: [[String: Any]] = []

        for inst in 0..<predCount {
            let instScores = spanScores[inst]  // [fields, L, maxWidth]

            var instance: [String: Any] = [:]

            for (fieldIdx, fieldName) in fieldNames.enumerated() {
                guard fieldIdx < instScores.dim(0) else { continue }

                let fieldKey = "\(schemaName).\(fieldName)"
                let fieldMeta = metadata.fieldMetadata[fieldKey]
                let fieldThreshold = fieldMeta?.threshold ?? threshold
                let dtype = fieldMeta?.dtype ?? "list"

                // Check if this is a choice field
                if let choices = clsFields[fieldKey] {
                    // Choice field: use prefix scores
                    let prefixScores = instScores[fieldIdx, 0..<startIdx]  // [prefixLen, maxWidth]

                    let result = decoder.decodeChoiceField(
                        prefixScores: prefixScores,
                        choices: choices,
                        textTokens: [],  // Not used with this implementation
                        threshold: fieldThreshold,
                        dtype: dtype
                    )

                    instance[fieldName] = result
                } else {
                    // Regular span field: use text scores
                    let fieldScores = instScores[fieldIdx, startIdx...]  // [textLen, maxWidth]

                    let spans = decoder.findSpans(
                        scores: fieldScores,
                        threshold: fieldThreshold,
                        textLen: textLen,
                        text: originalText,
                        startMap: startMappings,
                        endMap: endMappings
                    )

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

            // Only add if instance has any content
            let hasContent = instance.values.contains { value in
                if let arr = value as? [Any], arr.isEmpty { return false }
                if let str = value as? String, str.isEmpty { return false }
                return true
            }

            if hasContent {
                instances.append(instance)
            }
        }

        results[schemaName] = instances
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
    public func entities(_ entityTypes: [String]) -> Schema {
        var entitiesDict = internalSchemaDict["entities"] as? [String: Any] ?? [:]
        for entityType in entityTypes {
            entitiesDict[entityType] = ""
        }
        internalSchemaDict["entities"] = entitiesDict
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
        for entityType in typesWithDescriptions.keys {
            entitiesDict[entityType] = ""
        }
        internalSchemaDict["entities"] = entitiesDict
        internalSchemaDict["entity_descriptions"] = typesWithDescriptions
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
    public func relations(_ relationTypes: [String]) -> Schema {
        var relations = internalSchemaDict["relations"] as? [[String: [String: Any]]] ?? []
        for relationType in relationTypes {
            relations.append([relationType: ["head": "", "tail": ""]])
        }
        internalSchemaDict["relations"] = relations
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
    @discardableResult
    public func field(
        _ fieldName: String,
        dtype: String = "list",
        choices: [String]? = nil,
        description: String? = nil,
        threshold: Float? = nil
    ) -> StructureBuilder {
        if let choices = choices {
            fields[fieldName] = ["value": "", "choices": choices]
        } else {
            fields[fieldName] = ""
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
