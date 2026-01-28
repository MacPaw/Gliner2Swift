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
    }

    // MARK: - Main Transformation

    /// Transform text and schema into a preprocessed record
    ///
    /// - Parameters:
    ///   - text: Input text
    ///   - schema: Schema dictionary
    /// - Returns: Transformed record ready for batching
    public func transform(text: String, schema: [String: Any]) -> TransformedRecord {
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

        // Tokenize text
        let textTokens = wordSplitter.tokenize(normalizedText, lower: true)
        var allTextTokens = prefix + textTokens.texts
        let prefixLen = prefix.count

        // Build schema outputs
        let schemaResults = buildSchemaOutputs(schema: mutableSchema, textTokens: allTextTokens, prefixLen: prefixLen)

        // Format input with mappings
        let schemaTokensList = schemaResults.map { $0.schemaTokens }
        let (inputIds, mappedIndices) = formatInputWithMapping(
            schemaTokensList: schemaTokensList,
            textTokens: allTextTokens
        )

        return TransformedRecord(
            inputIds: inputIds,
            mappedIndices: mappedIndices,
            schemaTokensList: schemaTokensList,
            textTokens: allTextTokens,
            structureLabels: schemaResults.map { $0.output as Any },
            taskTypes: schemaResults.map { $0.taskType },
            startTokenIdx: textTokens.starts,
            endTokenIdx: textTokens.ends,
            text: normalizedText,
            schema: schema  // Original schema before modification
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
            for structure in jsonStructures {
                for (parent, fields) in structure {
                    let fieldNames = Array(fields.keys)
                    guard !fieldNames.isEmpty else { continue }

                    let schemaTokens = buildSchemaTokens(
                        parent: parent,
                        fields: fieldNames,
                        childPrefix: SpecialTokens.cToken
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
            let entityNames = Array(entities.keys)
            guard !entityNames.isEmpty else { return results }

            let schemaTokens = buildSchemaTokens(
                parent: "entities",
                fields: entityNames,
                childPrefix: SpecialTokens.eToken
            )

            let output: [Any] = [1, []]

            results.append(SchemaResult(
                taskType: "entities",
                schemaTokens: schemaTokens,
                output: output
            ))
        }

        // Process relations
        if let relations = schema["relations"] as? [[String: [String: Any]]] {
            for relation in relations {
                for (parent, fields) in relation {
                    let fieldNames = Array(fields.keys)
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

                let schemaTokens = buildSchemaTokens(
                    parent: task,
                    fields: labels,
                    childPrefix: SpecialTokens.lToken
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

    /// Build schema token sequence
    private func buildSchemaTokens(
        parent: String,
        fields: [String],
        childPrefix: String,
        prompt: String? = nil,
        labelDescriptions: [String: String]? = nil
    ) -> [String] {
        var promptStr = parent
        if let prompt = prompt {
            promptStr = "\(parent): \(prompt)"
        }

        // Add descriptions if available
        if let descriptions = labelDescriptions {
            for (label, desc) in descriptions {
                if fields.contains(label) {
                    promptStr += " \(SpecialTokens.descToken) \(label): \(desc)"
                }
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

    /// Format input and create token mappings
    ///
    /// - Parameters:
    ///   - schemaTokensList: List of schema token lists
    ///   - textTokens: Text tokens
    /// - Returns: (input_ids, mapped_indices)
    private func formatInputWithMapping(
        schemaTokensList: [[String]],
        textTokens: [String]
    ) -> ([Int], [MappedIndex]) {
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
            inputIds.append(contentsOf: subTokenIds)

            // Map each subword to original token
            for _ in subTokenIds {
                mappings.append(MappedIndex(segType, origIdx, schemaIdx))
            }
        }

        return (inputIds, mappings)
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

        let maxLen = records.map { $0.inputIds.count }.max() ?? 0
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
            originalSchemas: records.map { $0.schema }
        )
    }
}
