// PreprocessedBatch.swift
// Data structures for preprocessed batches
//
// Matches Python: gliner2/processor.py:TransformedRecord, PreprocessedBatch

import MLX
import Foundation

// MARK: - Segment Type

/// Segment type for token mapping
public enum SegmentType: String, Sendable {
    case schema
    case text
    case sep
}

// MARK: - Mapped Index

/// Token mapping information: (segment_type, original_index, schema_index)
public struct MappedIndex: Sendable, Equatable {
    public let segmentType: SegmentType
    public let originalIndex: Int
    public let schemaIndex: Int

    public init(_ segmentType: SegmentType, _ originalIndex: Int, _ schemaIndex: Int) {
        self.segmentType = segmentType
        self.originalIndex = originalIndex
        self.schemaIndex = schemaIndex
    }
}

// MARK: - Transformed Record

/// Single transformed record ready for batching.
public struct TransformedRecord: Sendable {
    /// Tokenized input IDs
    public let inputIds: [Int]

    /// Token mappings: (segment_type, original_idx, schema_idx)
    public let mappedIndices: [MappedIndex]

    /// Schema tokens for each schema
    public let schemaTokensList: [[String]]

    /// Text tokens (whitespace-split)
    public let textTokens: [String]

    /// Structure labels for each schema
    public let structureLabels: [Any]

    /// Task type for each schema
    public let taskTypes: [String]

    /// Start character positions for text tokens
    public let startTokenIdx: [Int]

    /// End character positions for text tokens
    public let endTokenIdx: [Int]

    /// Original text
    public let text: String

    /// Original schema dictionary
    public let schema: [String: Any]

    /// Number of schemas
    public var numSchemas: Int {
        schemaTokensList.count
    }
}

// MARK: - Preprocessed Batch

/// GPU-ready batch for training/inference.
public struct PreprocessedBatch: Sendable {
    /// Input token IDs [batch, max_seq_len]
    public let inputIds: MLXArray

    /// Attention mask [batch, max_seq_len]
    public let attentionMask: MLXArray

    /// Per-sample token mappings
    public let mappedIndices: [[MappedIndex]]

    /// Number of schemas per sample
    public let schemaCounts: [Int]

    /// Original sequence lengths
    public let originalLengths: [Int]

    /// Ground truth labels for each sample and schema
    public let structureLabels: [[Any]]

    /// Task types per schema per sample
    public let taskTypes: [[String]]

    /// Original text tokens per sample
    public let textTokens: [[String]]

    /// Schema tokens per sample
    public let schemaTokensList: [[[String]]]

    /// Char position start mappings per sample
    public let startMappings: [[Int]]

    /// Char position end mappings per sample
    public let endMappings: [[Int]]

    /// Original texts for result formatting
    public let originalTexts: [String]

    /// Original schemas for result formatting
    public let originalSchemas: [[String: Any]]

    /// Batch size
    public var count: Int {
        inputIds.dim(0)
    }

    /// Check if batch is empty
    public var isEmpty: Bool {
        count == 0
    }

    /// Get input IDs for a specific sample (unpadded)
    ///
    /// This is needed for token-by-token lookup during schema embedding extraction,
    /// matching Python's approach of using convert_ids_to_tokens(tid) to check
    /// if each token is a special marker token.
    public func getInputIds(for sampleIndex: Int) -> [Int] {
        guard sampleIndex < originalLengths.count else { return [] }
        let seqLen = originalLengths[sampleIndex]

        // Extract row from MLXArray and convert to [Int]
        let row = inputIds[sampleIndex, 0..<seqLen]
        MLX.eval(row)

        var ids: [Int] = []
        for i in 0..<seqLen {
            ids.append(Int(row[i].item(Int32.self)))
        }
        return ids
    }

    /// Move tensors to specific device/stream
    public func using(_ device: Device) -> PreprocessedBatch {
        return PreprocessedBatch(
            inputIds: inputIds,  // MLX handles device placement automatically
            attentionMask: attentionMask,
            mappedIndices: mappedIndices,
            schemaCounts: schemaCounts,
            originalLengths: originalLengths,
            structureLabels: structureLabels,
            taskTypes: taskTypes,
            textTokens: textTokens,
            schemaTokensList: schemaTokensList,
            startMappings: startMappings,
            endMappings: endMappings,
            originalTexts: originalTexts,
            originalSchemas: originalSchemas
        )
    }

    /// Create an empty batch
    public static func empty() -> PreprocessedBatch {
        PreprocessedBatch(
            inputIds: MLXArray.zeros([0, 0]),
            attentionMask: MLXArray.zeros([0, 0]),
            mappedIndices: [],
            schemaCounts: [],
            originalLengths: [],
            structureLabels: [],
            taskTypes: [],
            textTokens: [],
            schemaTokensList: [],
            startMappings: [],
            endMappings: [],
            originalTexts: [],
            originalSchemas: []
        )
    }
}

// MARK: - Structure Labels

/// Structure label for span-based tasks: (count, spans)
public struct StructureLabel {
    public let count: Int
    public let spans: [[[SpanPosition?]]]  // [instance][field][span]

    public init(count: Int, spans: [[[SpanPosition?]]]) {
        self.count = count
        self.spans = spans
    }
}

/// Span position (start, end) in token indices
public struct SpanPosition: Sendable, Equatable {
    public let start: Int
    public let end: Int

    public static let invalid = SpanPosition(start: -1, end: -1)

    public init(start: Int, end: Int) {
        self.start = start
        self.end = end
    }

    public var isValid: Bool {
        start >= 0 && end >= 0
    }
}

/// Classification label: array of binary values
public typealias ClassificationLabel = [Int]
