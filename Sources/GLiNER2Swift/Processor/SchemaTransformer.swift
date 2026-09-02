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
// SchemaTransformer.swift
// Schema-based text transformer for GLiNER2
//
// Matches Python: gliner2/processor.py:SchemaTransformer

import Foundation
import MLX

// MARK: - Special Tokens

/// Special tokens used by GLiNER2
public enum SpecialTokens {
    public static let sepStruct = "[SEP_STRUCT]"
    public static let sepText = "[SEP_TEXT]"
    public static let pToken = "[P]"      // Parent/Prompt token
    public static let cToken = "[C]"      // Child token (JSON structures)
    public static let eToken = "[E]"      // Entity token
    public static let rToken = "[R]"      // Relation token
    public static let lToken = "[L]"      // Label token (classification)
    public static let exampleToken = "[EXAMPLE]"
    public static let outputToken = "[OUTPUT]"
    public static let descToken = "[DESCRIPTION]"

    public static let all: [String] = [
        sepStruct, sepText, pToken, cToken, eToken,
        rToken, lToken, exampleToken, outputToken, descToken
    ]

    /// Tokens that contribute embeddings for schemas
    public static let embeddingTokens: Set<String> = [
        pToken, cToken, eToken, rToken, lToken
    ]
}

// MARK: - Token Pooling

/// Token pooling strategy for aggregating subword embeddings
public enum TokenPoolingType: String, Sendable {
    case first
    case mean
    case max
}

// MARK: - Schema Transformer

/// Schema-based text transformer for GLiNER2.
///
/// Handles:
/// - Text tokenization and normalization
/// - Schema sequence building
/// - Classification prefix handling with [selection] wrapping
/// - Mapped indices as 3-tuples (segment_type, original_idx, schema_idx)
public class SchemaTransformer {
    /// The tokenizer (custom UnigramTokenizer for DeBERTa v3)
    public let tokenizer: UnigramTokenizer

    /// Whitespace tokenizer for text splitting
    public let wordSplitter: WhitespaceTokenSplitter

    /// Token pooling strategy
    public let tokenPooling: TokenPoolingType

    /// Whether in training mode
    public var isTraining: Bool = false

    /// Round each batch's padded length up to a fixed set of sizes.
    ///
    /// Off by default, because it makes the encoder process tokens nobody asked for. It
    /// exists for the compiled encoder, whose per-shape cache would otherwise recompile on
    /// nearly every call — real inputs rarely repeat an exact token count. The extra work
    /// is close to free at these sizes: a 12x longer sequence only doubles inference time,
    /// so per-call latency is dominated by fixed cost rather than sequence length.
    public var padsSequenceLengthToBuckets: Bool = false

    /// Granularity of the padded length when `padsSequenceLengthToBuckets` is on.
    ///
    /// Rounding up to a multiple of this rather than to a handful of coarse buckets keeps
    /// the wasted tokens bounded by `bucketGranularity - 1`. Coarse buckets (64/128/256)
    /// cost 15 % on batched workloads, where sequence length really does drive the cost.
    public static var bucketGranularity = 16

    static func sequenceLengthBucket(for length: Int) -> Int {
        guard length > 0 else { return 0 }
        let granularity = bucketGranularity
        return ((length + granularity - 1) / granularity) * granularity
    }

    /// Token ids that stand for a marker token contributing a schema embedding.
    ///
    /// Resolved through the tokenizer rather than assumed, and filtered back through
    /// `idToToken` so an id only counts when it really round-trips to the marker — the
    /// same test the decode path used to run at every position.
    private let markerTokenIds: Set<Int>

    /// Initialize from local directory containing tokenizer.json
    ///
    /// - Parameters:
    ///   - directoryUrl: URL to directory containing tokenizer.json
    ///   - tokenPooling: Token pooling strategy
    public static func createFromLocalDirectory(directoryUrl: URL, tokenPooling: TokenPoolingType = .first) throws -> SchemaTransformer {
        let tokenizerUrl = directoryUrl.appendingPathComponent("tokenizer.json")
        let tokenizer = try UnigramTokenizer(tokenizerJsonUrl: tokenizerUrl)
        return SchemaTransformer(tokenizer: tokenizer, tokenPooling: tokenPooling)
    }

    /// Initialize with existing tokenizer
    ///
    /// - Parameters:
    ///   - tokenizer: Pre-initialized tokenizer
    ///   - tokenPooling: Token pooling strategy
    public init(tokenizer: UnigramTokenizer, tokenPooling: TokenPoolingType = .first) {
        self.tokenizer = tokenizer
        self.wordSplitter = WhitespaceTokenSplitter()
        self.tokenPooling = tokenPooling

        var markerIds: Set<Int> = []
        for marker in SpecialTokens.embeddingTokens {
            guard let id = tokenizer.tokensToIds([marker]).first,
                  tokenizer.idToToken(id) == marker else { continue }
            markerIds.insert(id)
        }
        self.markerTokenIds = markerIds
    }

    // MARK: - Main Transformation

    /// Transform text and schema into a preprocessed record
    ///
    /// - Parameters:
    ///   - text: Input text
    ///   - schema: Schema dictionary
    ///   - maxLen: Optional cap on the number of whitespace-split text words; longer
    ///     inputs are truncated to the first `maxLen` words before schema/prefix encoding.
    /// - Returns: Transformed record ready for batching
    public func transform(text: String, schema: [String: Any], maxLen: Int? = nil) -> TransformedRecord {
        // Normalize text: ensure ends with punctuation
        var normalizedText = text
        if !normalizedText.isEmpty && !normalizedText.hasSuffix(".") &&
           !normalizedText.hasSuffix("!") && !normalizedText.hasSuffix("?") {
            normalizedText += "."
        } else if normalizedText.isEmpty {
            normalizedText = "."
        }

        // Build classification prefix
        var mutableSchema = schema
        let prefix = buildClassificationPrefix(schema: schema)

        // Wrap classification fields with [selection] prefix
        if !prefix.isEmpty {
            wrapClassificationFields(schema: &mutableSchema, prefix: prefix)
        }

        // Tokenize text into whitespace-split words (with char start/end maps).
        var textTokens = wordSplitter.tokenize(normalizedText, lower: true)

        // maxLen truncation (Python processor.py:408-410): keep the first `maxLen` words,
        // done here — after word splitting, before the prefix/schema is joined on. The kept
        // words' char start/end still index the original (normalized) string, so extracted
        // spans keep their real positions; only words beyond the cap are dropped.
        if let maxLen, maxLen >= 0, textTokens.count > maxLen {
            textTokens = Array(textTokens.prefix(maxLen))
        }

        let allTextTokens = prefix + textTokens.texts
        let prefixLen = prefix.count

        // Build schema outputs
        let schemaResults = buildSchemaOutputs(schema: mutableSchema, textTokens: allTextTokens, prefixLen: prefixLen)

        // Format input with mappings
        let schemaTokensList = schemaResults.map { $0.schemaTokens }
        let formatted = formatInputWithMapping(
            schemaTokensList: schemaTokensList,
            textTokens: allTextTokens
        )

        return TransformedRecord(
            inputIds: formatted.inputIds,
            mappedIndices: formatted.mappedIndices,
            schemaTokensList: schemaTokensList,
            textTokens: allTextTokens,
            structureLabels: schemaResults.map { $0.output as Any },
            taskTypes: schemaResults.map { $0.taskType },
            startTokenIdx: textTokens.starts,
            endTokenIdx: textTokens.ends,
            text: normalizedText,
            schema: schema,  // Original schema before modification
            textStartIndex: formatted.textStartIndex,
            wordFirstIndices: formatted.wordFirstIndices,
            wordSubwordCounts: formatted.wordSubwordCounts,
            schemaMarkerPositions: formatted.schemaMarkerPositions
        )
    }

    // MARK: - Classification Prefix

    /// Build classification prefix tokens for choice fields
    private func buildClassificationPrefix(schema: [String: Any]) -> [String] {
        var prefixTokens: [String] = []

        guard let jsonStructures = schema["json_structures"] as? [[String: [String: Any]]] else {
            return prefixTokens
        }

        for structure in jsonStructures {
            for (parent, fields) in structure {
                // Find classification fields (those with "value" and "choices")
                var classificationFields: [(String, Any)] = []

                for (fieldName, fieldValue) in fields {
                    if let fieldDict = fieldValue as? [String: Any],
                       fieldDict["value"] != nil,
                       fieldDict["choices"] != nil {
                        classificationFields.append((fieldName, fieldValue))
                    }
                }

                guard !classificationFields.isEmpty else { continue }

                // Build inner tokens
                var inner: [String] = []
                for (fieldName, fieldValue) in classificationFields {
                    guard let fieldDict = fieldValue as? [String: Any],
                          let choices = fieldDict["choices"] as? [String] else {
                        continue
                    }

                    var choiceTokens: [String] = []
                    for (i, choice) in choices.enumerated() {
                        if i > 0 {
                            choiceTokens.append("|")
                        }
                        choiceTokens.append(choice)
                    }

                    inner.append(fieldName)
                    inner.append("(")
                    inner.append(contentsOf: choiceTokens)
                    inner.append(")")
                    inner.append(",")
                }

                // Remove trailing comma
                if !inner.isEmpty {
                    inner.removeLast()
                    prefixTokens.append("(")
                    prefixTokens.append("\(parent):")
                    prefixTokens.append(contentsOf: inner)
                    prefixTokens.append(")")
                }
            }
        }

        return prefixTokens
    }

    /// Wrap classification field values with [selection] prefix
    private func wrapClassificationFields(schema: inout [String: Any], prefix: [String]) {
        guard var jsonStructures = schema["json_structures"] as? [[String: [String: Any]]] else {
            return
        }

        // Find classification field keys
        var classificationKeys: Set<String> = []
        for structure in jsonStructures {
            for (parent, fields) in structure {
                for (fieldName, fieldValue) in fields {
                    if let fieldDict = fieldValue as? [String: Any],
                       fieldDict["value"] != nil,
                       fieldDict["choices"] != nil {
                        classificationKeys.insert("\(parent).\(fieldName)")
                    }
                }
            }
        }

        // Wrap field values
        for i in 0..<jsonStructures.count {
            for (parent, var fields) in jsonStructures[i] {
                for fieldName in fields.keys {
                    let key = "\(parent).\(fieldName)"
                    guard classificationKeys.contains(key) else { continue }

                    if let fieldDict = fields[fieldName] as? [String: Any],
                       let value = fieldDict["value"] {
                        // Wrap with [selection] prefix
                        if let valueArray = value as? [String] {
                            fields[fieldName] = valueArray.map { "[selection]\($0)" }
                        } else if let valueString = value as? String {
                            fields[fieldName] = "[selection]\(valueString)"
                        }
                    }
                }
                jsonStructures[i][parent] = fields
            }
        }

        schema["json_structures"] = jsonStructures
    }

    // MARK: - Schema Building

    /// Result from building a single schema
    private struct SchemaResult {
        let taskType: String
        let schemaTokens: [String]
        let output: Any
    }

    /// Build outputs for all schemas
    private func buildSchemaOutputs(
        schema: [String: Any],
        textTokens: [String],
        prefixLen: Int
    ) -> [SchemaResult] {
        var results: [SchemaResult] = []

        // Process JSON structures
        if let jsonStructures = schema["json_structures"] as? [[String: [String: Any]]] {
            // Get field descriptions for all structures
            let jsonDescriptions = schema["json_descriptions"] as? [String: [String: String]] ?? [:]
            // Get preserved field orders (set by StructureBuilder)
            let fieldOrders = schema["_field_orders"] as? [String: [String]] ?? [:]

            for structure in jsonStructures {
                for (parent, fields) in structure {
                    // Use preserved field order if available, otherwise fall back to dictionary keys
                    let fieldNames: [String]
                    if let order = fieldOrders[parent], !order.isEmpty {
                        fieldNames = order
                    } else {
                        fieldNames = Array(fields.keys).sorted()  // Sort for consistency
                    }
                    guard !fieldNames.isEmpty else { continue }

                    // Get descriptions for this structure's fields
                    let fieldDescriptions = jsonDescriptions[parent]

                    let schemaTokens = buildSchemaTokens(
                        parent: parent,
                        fields: fieldNames,
                        childPrefix: SpecialTokens.cToken,
                        labelDescriptions: fieldDescriptions
                    )

                    // Build structure labels (simplified for now)
                    let output: [Any] = [1, []]  // [count, spans]

                    results.append(SchemaResult(
                        taskType: "json_structures",
                        schemaTokens: schemaTokens,
                        output: output
                    ))
                }
            }
        }

        // Process entities
        if let entities = schema["entities"] as? [String: Any] {
            // Use preserved entity order if available, otherwise sort keys for consistency
            let entityNames: [String]
            if let order = schema["_entity_order"] as? [String], !order.isEmpty {
                entityNames = order
            } else {
                entityNames = Array(entities.keys).sorted()
            }
            // Only process if there are entity names (don't early return - other schemas may follow)
            if !entityNames.isEmpty {
                // Get entity descriptions if available
                let entityDescriptions = schema["entity_descriptions"] as? [String: String]

                let schemaTokens = buildSchemaTokens(
                    parent: "entities",
                    fields: entityNames,
                    childPrefix: SpecialTokens.eToken,
                    labelDescriptions: entityDescriptions
                )

                let output: [Any] = [1, []]

                results.append(SchemaResult(
                    taskType: "entities",
                    schemaTokens: schemaTokens,
                    output: output
                ))
            }
        }

        // Process relations
        if let relations = schema["relations"] as? [[String: [String: Any]]] {
            // Get preserved relation field orders
            let relationOrders = schema["_relation_orders"] as? [String: [String]] ?? [:]

            for relation in relations {
                for (parent, fields) in relation {
                    // Use preserved field order if available, otherwise sort
                    let fieldNames: [String]
                    if let order = relationOrders[parent], !order.isEmpty {
                        fieldNames = order
                    } else {
                        fieldNames = Array(fields.keys).sorted()
                    }
                    guard !fieldNames.isEmpty else { continue }

                    let schemaTokens = buildSchemaTokens(
                        parent: parent,
                        fields: fieldNames,
                        childPrefix: SpecialTokens.rToken
                    )

                    let output: [Any] = [1, []]

                    results.append(SchemaResult(
                        taskType: "relations",
                        schemaTokens: schemaTokens,
                        output: output
                    ))
                }
            }
        }

        // Process classifications
        if let classifications = schema["classifications"] as? [[String: Any]] {
            for classification in classifications {
                guard let task = classification["task"] as? String,
                      let labels = classification["labels"] as? [String] else {
                    continue
                }

                // Classification extras (Phase 6.5): prompt, per-label descriptions, and
                // few-shot examples stored as [[input, output]] pairs.
                let prompt = classification["prompt"] as? String
                let labelDescriptions = classification["label_descriptions"] as? [String: String]
                let examples = (classification["examples"] as? [[String]])?.compactMap {
                    pair -> (input: String, output: String)? in
                    pair.count >= 2 ? (pair[0], pair[1]) : nil
                }

                let schemaTokens = buildSchemaTokens(
                    parent: task,
                    fields: labels,
                    childPrefix: SpecialTokens.lToken,
                    prompt: prompt,
                    labelDescriptions: labelDescriptions,
                    examples: examples
                )

                // Classification output: binary labels
                let trueLabels = classification["true_label"] as? [String] ?? []
                let binaryLabels = labels.map { trueLabels.contains($0) ? 1 : 0 }

                results.append(SchemaResult(
                    taskType: "classifications",
                    schemaTokens: schemaTokens,
                    output: binaryLabels
                ))
            }
        }

        return results
    }

    /// Build schema token sequence.
    ///
    /// The prompt string mirrors Python `_transform_schema` at inference (example_mode
    /// "both", no shuffling): optional `task: prompt`, then `[DESCRIPTION] label: desc` for
    /// each label that has one (in label order), then `[EXAMPLE] input [OUTPUT] output` for
    /// each few-shot example whose output is one of the labels.
    private func buildSchemaTokens(
        parent: String,
        fields: [String],
        childPrefix: String,
        prompt: String? = nil,
        labelDescriptions: [String: String]? = nil,
        examples: [(input: String, output: String)]? = nil
    ) -> [String] {
        var promptStr = parent
        if let prompt = prompt {
            promptStr = "\(parent): \(prompt)"
        }

        // Add descriptions if available (label order; only labels that have one).
        if let descriptions = labelDescriptions {
            for label in fields {
                if let desc = descriptions[label] {
                    promptStr += " \(SpecialTokens.descToken) \(label): \(desc)"
                }
            }
        }

        // Few-shot examples: kept in given order, only those whose output is a label.
        if let examples = examples {
            for example in examples where fields.contains(example.output) {
                promptStr += " \(SpecialTokens.exampleToken) \(example.input)"
                    + " \(SpecialTokens.outputToken) \(example.output)"
            }
        }

        // Build token sequence: ( [P] prompt ( [C/E/R/L] field1 [C/E/R/L] field2 ... ) )
        var tokens = ["(", SpecialTokens.pToken, promptStr, "("]
        for field in fields {
            tokens.append(childPrefix)
            tokens.append(field)
        }
        tokens.append(")")
        tokens.append(")")

        return tokens
    }

    // MARK: - Input Formatting

    /// Everything `formatInputWithMapping` derives in its single pass over the prompt.
    private struct FormattedInput {
        let inputIds: [Int]
        let mappedIndices: [MappedIndex]
        let textStartIndex: Int
        let wordFirstIndices: [Int]
        let wordSubwordCounts: [Int]
        let schemaMarkerPositions: [[Int]]
    }

    /// Format input and create token mappings
    ///
    /// Also records, in the same pass, the index arrays the decode path needs to gather
    /// word-level and schema-level embeddings without inspecting individual token ids
    /// again: where the text segment starts, which subword opens each word (and how many
    /// subwords it spans), and where each schema's marker tokens landed.
    ///
    /// - Parameters:
    ///   - schemaTokensList: List of schema token lists
    ///   - textTokens: Text tokens
    private func formatInputWithMapping(
        schemaTokensList: [[String]],
        textTokens: [String]
    ) -> FormattedInput {
        // Build combined tokens
        var combined: [String] = []
        for schemaTokens in schemaTokensList {
            combined.append(contentsOf: schemaTokens)
            combined.append(SpecialTokens.sepStruct)
        }
        // Remove last SEP_STRUCT
        if !combined.isEmpty {
            combined.removeLast()
        }
        combined.append(SpecialTokens.sepText)
        combined.append(contentsOf: textTokens)

        // Build token IDs and mappings directly
        var inputIds: [Int] = []
        var mappings: [MappedIndex] = []

        let numSchemas = schemaTokensList.count
        let textSchemaIdx = numSchemas
        var currentSchema = 0
        var foundSep = false

        var textStartIndex = -1
        var wordFirstIndices: [Int] = []
        var wordSubwordCounts: [Int] = []
        var schemaMarkerPositions: [[Int]] = Array(repeating: [], count: numSchemas)
        let markerIds = markerTokenIds

        for (origIdx, token) in combined.enumerated() {
            let segType: SegmentType
            let schemaIdx: Int

            if token == SpecialTokens.sepText {
                segType = .sep
                schemaIdx = textSchemaIdx
                foundSep = true
            } else if !foundSep {
                segType = .schema
                schemaIdx = currentSchema
                if token == SpecialTokens.sepStruct {
                    currentSchema += 1
                }
            } else {
                segType = .text
                schemaIdx = textSchemaIdx
            }

            // Tokenize token into subword IDs directly
            let subTokenIds = tokenize(token)
            let tokenStart = inputIds.count
            inputIds.append(contentsOf: subTokenIds)

            // Map each subword to original token
            for _ in subTokenIds {
                mappings.append(MappedIndex(segType, origIdx, schemaIdx))
            }

            switch segType {
            case .schema:
                if schemaIdx >= 0 && schemaIdx < numSchemas {
                    for (offset, id) in subTokenIds.enumerated() where markerIds.contains(id) {
                        schemaMarkerPositions[schemaIdx].append(tokenStart + offset)
                    }
                }
            case .text:
                if textStartIndex < 0 {
                    textStartIndex = tokenStart
                }
                // A word that produced no subword contributes no pooled position, which is
                // what the per-subword loop this replaces also did.
                if !subTokenIds.isEmpty {
                    wordFirstIndices.append(tokenStart - textStartIndex)
                    wordSubwordCounts.append(subTokenIds.count)
                }
            case .sep:
                break
            }
        }

        return FormattedInput(
            inputIds: inputIds,
            mappedIndices: mappings,
            textStartIndex: textStartIndex < 0 ? mappings.count : textStartIndex,
            wordFirstIndices: wordFirstIndices,
            wordSubwordCounts: wordSubwordCounts,
            schemaMarkerPositions: schemaMarkerPositions
        )
    }

    /// Tokenize a single token into subword IDs
    private func tokenize(_ token: String) -> [Int] {
        // Use the tokenizer to get subword IDs
        let encoded = tokenizer.encode(token)
        return encoded
    }

    /// Convert token strings to IDs
    private func convertTokensToIds(_ tokens: [String]) -> [Int] {
        return tokenizer.tokensToIds(tokens)
    }

    // MARK: - Batch Processing

    /// Collate records into a batch
    ///
    /// - Parameter records: Array of transformed records
    /// - Returns: Preprocessed batch
    public func collateBatch(_ records: [TransformedRecord]) -> PreprocessedBatch {
        guard !records.isEmpty else {
            return PreprocessedBatch.empty()
        }

        let longest = records.map { $0.inputIds.count }.max() ?? 0
        // Padded positions carry attention mask 0, and the decode path indexes by the
        // per-record mappings, so extra padding cannot reach the output.
        let maxLen = padsSequenceLengthToBuckets
            ? Self.sequenceLengthBucket(for: longest)
            : longest
        let batchSize = records.count

        // Pad input IDs and create attention masks
        var inputIdsList: [[Int32]] = []
        var attentionMaskList: [[Int32]] = []
        var originalLengths: [Int] = []

        for record in records {
            let seqLen = record.inputIds.count
            var ids = record.inputIds.map { Int32($0) }
            var mask = [Int32](repeating: 1, count: seqLen)

            // Pad to max length
            let padding = maxLen - seqLen
            if padding > 0 {
                ids.append(contentsOf: [Int32](repeating: 0, count: padding))
                mask.append(contentsOf: [Int32](repeating: 0, count: padding))
            }

            inputIdsList.append(ids)
            attentionMaskList.append(mask)
            originalLengths.append(seqLen)
        }

        // Convert to MLXArrays
        let inputIds = MLXArray(inputIdsList.flatMap { $0 }).reshaped([batchSize, maxLen])
        let attentionMask = MLXArray(attentionMaskList.flatMap { $0 }).reshaped([batchSize, maxLen])

        return PreprocessedBatch(
            inputIds: inputIds,
            attentionMask: attentionMask,
            mappedIndices: records.map { $0.mappedIndices },
            schemaCounts: records.map { $0.numSchemas },
            originalLengths: originalLengths,
            structureLabels: records.map { $0.structureLabels },
            taskTypes: records.map { $0.taskTypes },
            textTokens: records.map { $0.textTokens },
            schemaTokensList: records.map { $0.schemaTokensList },
            startMappings: records.map { $0.startTokenIdx },
            endMappings: records.map { $0.endTokenIdx },
            originalTexts: records.map { $0.text },
            originalSchemas: records.map { $0.schema },
            inputIdsCPU: records.map { $0.inputIds },
            textStartIndices: records.map { $0.textStartIndex },
            wordFirstIndices: records.map { $0.wordFirstIndices },
            wordSubwordCounts: records.map { $0.wordSubwordCounts },
            schemaMarkerPositions: records.map { $0.schemaMarkerPositions }
        )
    }
}
