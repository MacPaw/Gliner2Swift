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
// BoundaryInputs.swift
// Routing tensors and query layout for the GLiNER2.5 boundary head, derived from a
// `PreprocessedBatch`.
//
// Matches Python: processor.py:_pad_batch (text_word_indices, query_marker_indices),
// processing/boundary_preprocessing.py:build_boundary_batch_metadata (QueryLayout) and
// BoundaryExtractor._query_thresholds.

import Foundation
import MLX

/// One extractive query: a `[E]` / `[C]` / `[R]` child marker of a non-classification
/// schema group. Query ids follow prompt order (json_structures, entities, relations;
/// fields in schema order).
public struct QuerySpec: Sendable, Equatable {
    public let queryId: Int
    /// Index of the schema group in the prompt.
    public let taskIndex: Int
    /// `"json_structures"`, `"entities"` or `"relations"`.
    public let taskType: String
    /// The group's name (`schemaTokens[2]` before any ` [DESCRIPTION] `).
    public let taskName: String
    /// Position of the field within its group.
    public let roleIndex: Int
    /// The field / entity label / relation role name.
    public let roleName: String
    /// `"taskName.roleName"` for structure fields, else `roleName`.
    public let fieldPath: String
}

/// Device tensors and host metadata the boundary path needs beyond `PreprocessedBatch`.
public struct BoundaryBatchInputs {
    /// `[B, Lpad]` int32 absolute sub-token index of each word's first piece.
    public let textWordIndices: MLXArray
    /// `[B, Lpad]` bool.
    public let textWordMask: MLXArray
    /// `[B, Qpad]` int32 absolute sub-token index of each query marker.
    public let queryMarkerIndices: MLXArray
    /// `[B, Qpad]` bool.
    public let queryMask: MLXArray
    /// Per-sample word count (classification-prefix words included).
    public let textLengths: [Int]
    /// Per-sample number of prefix words to subtract when mapping to user text.
    public let wordOffsets: [Int]
    /// Per-sample query layout.
    public let queryLayouts: [[QuerySpec]]
    /// Per-sample, per-query extraction thresholds `[B][Q]` (unpadded).
    public let queryThresholds: [[Float]]
    public let paddedWords: Int
    public let paddedQueries: Int

    /// Round `n` up to a multiple of `granularity` (at least `granularity`).
    static func bucket(_ n: Int, granularity: Int) -> Int {
        max(granularity, ((n + granularity - 1) / granularity) * granularity)
    }

    /// Build the routing tensors for a collated batch.
    ///
    /// - Parameters:
    ///   - batch: the collated batch
    ///   - metadata: schema metadata (per-entity / per-field thresholds)
    ///   - threshold: the call's default threshold
    ///   - wordGranularity: pad the word axis to a multiple of this (compiled graphs reuse
    ///     shapes; padded words are inert through the masks)
    ///   - queryGranularity: same for the query axis
    public static func build(
        from batch: PreprocessedBatch,
        metadata: SchemaMetadata,
        threshold: Float,
        wordGranularity: Int = 16,
        queryGranularity: Int = 8
    ) -> BoundaryBatchInputs {
        let B = batch.count
        var wordIndices: [[Int32]] = []
        var queryIndices: [[Int32]] = []
        var textLengths: [Int] = []
        var wordOffsets: [Int] = []
        var layouts: [[QuerySpec]] = []
        var thresholds: [[Float]] = []

        for i in 0..<B {
            let textStart = batch.textStartIndices[i]
            let words = batch.wordFirstIndices[i].map { Int32(textStart + $0) }
            wordIndices.append(words)
            textLengths.append(words.count)
            // Python: word_offsets = max(text_word_counts - len(start_mappings), 0)
            wordOffsets.append(max(words.count - batch.startMappings[i].count, 0))

            var layout: [QuerySpec] = []
            var markers: [Int32] = []
            var thr: [Float] = []
            let taskTypes = batch.taskTypes[i]
            let groups = batch.schemaTokensList[i]
            let markerPositions = batch.schemaMarkerPositions[i]
            for (g, tokens) in groups.enumerated() where g < taskTypes.count {
                let taskType = taskTypes[g]
                guard taskType != "classifications", g < markerPositions.count else { continue }
                let positions = markerPositions[g]
                guard positions.count >= 2, tokens.count >= 3 else { continue }
                let taskName = tokens[2].components(separatedBy: " [DESCRIPTION] ")[0]
                let fieldNames = Self.childNames(in: tokens)
                // positions[0] is [P]; the children follow in field order.
                for (role, position) in positions.dropFirst().enumerated() {
                    let roleName = role < fieldNames.count ? fieldNames[role] : ""
                    let fieldPath = taskType == "json_structures" ? "\(taskName).\(roleName)" : roleName
                    layout.append(QuerySpec(
                        queryId: layout.count, taskIndex: g, taskType: taskType, taskName: taskName,
                        roleIndex: role, roleName: roleName, fieldPath: fieldPath))
                    markers.append(Int32(position))
                    var configured: Float? = nil
                    if taskType == "entities" {
                        configured = metadata.entityMetadata[roleName]?.threshold
                    } else if taskType == "json_structures" {
                        configured = metadata.fieldMetadata[fieldPath]?.threshold
                    }
                    thr.append(configured ?? threshold)
                }
            }
            layouts.append(layout)
            queryIndices.append(markers)
            thresholds.append(thr)
        }

        let Lpad = bucket(textLengths.max() ?? 0, granularity: wordGranularity)
        let Qpad = bucket(queryIndices.map(\.count).max() ?? 0, granularity: queryGranularity)

        func pad(_ rows: [[Int32]], to width: Int) -> (MLXArray, MLXArray) {
            var values = [Int32](repeating: 0, count: B * width)
            var mask = [Bool](repeating: false, count: B * width)
            for (i, row) in rows.enumerated() {
                for (j, v) in row.enumerated() where j < width {
                    values[i * width + j] = v
                    mask[i * width + j] = true
                }
            }
            return (MLXArray(values).reshaped([B, width]), MLXArray(mask).reshaped([B, width]))
        }
        let (wi, wm) = pad(wordIndices, to: Lpad)
        let (qi, qm) = pad(queryIndices, to: Qpad)

        return BoundaryBatchInputs(
            textWordIndices: wi, textWordMask: wm,
            queryMarkerIndices: qi, queryMask: qm,
            textLengths: textLengths, wordOffsets: wordOffsets,
            queryLayouts: layouts, queryThresholds: thresholds,
            paddedWords: Lpad, paddedQueries: Qpad)
    }

    /// The field names of a schema group: the token after each child marker.
    /// Tokens: `( [P] name ( [E] f1 [E] f2 … ) )`.
    static func childNames(in tokens: [String]) -> [String] {
        var names: [String] = []
        var j = 4
        while j + 1 < tokens.count - 2 {
            let token = tokens[j]
            if token == SpecialTokens.eToken || token == SpecialTokens.cToken
                || token == SpecialTokens.rToken || token == SpecialTokens.lToken {
                names.append(tokens[j + 1])
                j += 2
            } else {
                j += 1
            }
        }
        return names
    }
}
