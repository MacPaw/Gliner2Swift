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
// GatherIndexTests.swift
// Guards the index arrays recorded during tokenization in Phase 3.5.
//
// The decode path no longer inspects token ids or segment types to find marker
// tokens and word boundaries — it gathers straight from these arrays. That makes
// them a silent single point of failure, so each one is checked here against the
// per-position scan it replaced.

import XCTest
import MLX
@testable import GLiNER2Swift

final class GatherIndexTests: XCTestCase {

    private static let markerTokens: Set<String> = ["[P]", "[C]", "[E]", "[R]", "[L]"]

    /// Schemas of every task type at once, so marker positions have to be split across
    /// four schema indices rather than trivially landing in one bucket.
    private func mixedSchema() -> [String: Any] {
        Schema()
            .structure("person")
            .field("name")
            .field("sentiment", choices: ["very positive", "positive", "negative"])
            .done()
            .entities(["person", "company"])
            .classification(task: "topic", labels: ["tech", "finance"])
            .relations(["works_at"])
            .build()
    }

    private func makeProcessor() throws -> SchemaTransformer {
        let tokenizerPath = try TestModel.requireTokenizerDirectory()
        return try SchemaTransformer.createFromLocalDirectory(directoryUrl: tokenizerPath)
    }

    // MARK: - Schema marker positions

    func testMarkerPositionsMatchPerPositionScan() throws {
        let processor = try makeProcessor()

        for text in ["Tim Cook is CEO of Apple in Cupertino.",
                     "Antidisestablishmentarianism flabbergasted the cryptozoologist."] {
            let record = processor.transform(text: text, schema: mixedSchema())

            // What the decode path used to do: walk every position, look the token id
            // back up in the vocabulary, keep it if it is a marker.
            var expected: [[Int]] = Array(repeating: [], count: record.schemaTokensList.count)
            for (idx, mapping) in record.mappedIndices.enumerated() {
                guard mapping.segmentType == .schema,
                      mapping.schemaIndex >= 0,
                      mapping.schemaIndex < expected.count,
                      idx < record.inputIds.count,
                      let token = processor.tokenizer.idToToken(record.inputIds[idx]),
                      Self.markerTokens.contains(token) else { continue }
                expected[mapping.schemaIndex].append(idx)
            }

            XCTAssertEqual(record.schemaMarkerPositions, expected,
                           "Recorded marker positions diverged from a per-position scan for: \(text)")
            XCTAssertFalse(expected.contains { $0.isEmpty },
                           "Test schema should put at least one marker in every schema")
        }
    }

    func testMarkerCountMatchesSchemaShape() throws {
        let processor = try makeProcessor()
        let record = processor.transform(text: "Tim Cook is CEO of Apple.", schema: mixedSchema())

        // One [P] plus one child marker per field, for every schema.
        for (schemaIdx, schemaTokens) in record.schemaTokensList.enumerated() {
            let markersInPrompt = schemaTokens.filter { Self.markerTokens.contains($0) }.count
            XCTAssertEqual(record.schemaMarkerPositions[schemaIdx].count, markersInPrompt,
                           "Schema \(schemaIdx) recorded a different marker count than its prompt has")
        }
    }

    // MARK: - Word pooling indices

    func testWordIndicesMatchOrigIdxRuns() throws {
        let processor = try makeProcessor()

        for text in ["John is 30 years old.",
                     "Antidisestablishmentarianism, unquestionably, bewildered Dr. Müller-Schmidt."] {
            let record = processor.transform(text: text, schema: mixedSchema())

            // What pooling used to do: group text-segment subwords into runs of equal
            // originalIndex, one pooled word per run.
            var expectedFirsts: [Int] = []
            var expectedCounts: [Int] = []
            var lastOrigIdx: Int? = nil
            for (idx, mapping) in record.mappedIndices.enumerated()
            where mapping.segmentType == .text {
                let relative = idx - record.textStartIndex
                if mapping.originalIndex == lastOrigIdx {
                    expectedCounts[expectedCounts.count - 1] += 1
                } else {
                    expectedFirsts.append(relative)
                    expectedCounts.append(1)
                }
                lastOrigIdx = mapping.originalIndex
            }

            XCTAssertEqual(record.wordFirstIndices, expectedFirsts,
                           "Word start indices diverged for: \(text)")
            XCTAssertEqual(record.wordSubwordCounts, expectedCounts,
                           "Word subword counts diverged for: \(text)")
        }
    }

    func testTextStartIndexIsFirstTextSubword() throws {
        let processor = try makeProcessor()
        let record = processor.transform(text: "John is 30 years old.", schema: mixedSchema())

        let expected = record.mappedIndices.firstIndex { $0.segmentType == .text }
            ?? record.mappedIndices.count
        XCTAssertEqual(record.textStartIndex, expected)
    }

    func testWordIndicesCoverTheWholeTextSegment() throws {
        let processor = try makeProcessor()
        let record = processor.transform(text: "Tim Cook is CEO of Apple.", schema: mixedSchema())

        // The runs must tile the text segment with no gap and no overlap, since pooling
        // reads each word as `first ..< first + count`.
        var cursor = 0
        for (first, count) in zip(record.wordFirstIndices, record.wordSubwordCounts) {
            XCTAssertEqual(first, cursor, "Word runs must be contiguous")
            XCTAssertGreaterThan(count, 0, "A pooled word must own at least one subword")
            cursor += count
        }
        XCTAssertEqual(cursor, record.mappedIndices.count - record.textStartIndex,
                       "Word runs must cover every text-segment subword")
    }

    // MARK: - Pooling arithmetic

    /// The shipped model pools with `.first`, so `.mean` and `.max` are exercised by no
    /// end-to-end test. Check them directly against a per-word reduction.
    func testMeanAndMaxPoolingMatchPerWordReduction() throws {
        let tokenizerPath = try TestModel.requireTokenizerDirectory()

        // Uneven word lengths, so the rectangular gather really has padding to handle.
        let firstIndices = [0, 2, 5, 6]
        let counts = [2, 3, 1, 4]
        let subwordCount = 10
        let hidden = 768   // must equal ExtractorConfig.hiddenSize: pooling reshapes by it

        var values: [Float] = []
        for s in 0..<subwordCount {
            for h in 0..<hidden {
                // Deliberately non-monotonic, so `max` cannot be satisfied by the last row.
                values.append(Float((s * 7 + h * 3) % 11) - 5)
            }
        }
        let subwords = MLXArray(values).reshaped([subwordCount, hidden])

        for pooling in [TokenPoolingType.mean, .max] {
            let model = GLiNER2(
                config: ExtractorConfig(),
                processor: try SchemaTransformer.createFromLocalDirectory(
                    directoryUrl: tokenizerPath, tokenPooling: pooling)
            )

            let pooled = model.poolTextEmbeddings(
                subwordEmbeddings: subwords,
                wordFirstIndices: firstIndices,
                wordSubwordCounts: counts,
                poolingType: pooling
            )
            XCTAssertEqual(pooled.shape, [firstIndices.count, hidden])
            let flat = pooled.asArray(Float.self)

            for (word, first) in firstIndices.enumerated() {
                for h in stride(from: 0, to: hidden, by: 97) {
                    let rows = (0..<counts[word]).map { values[(first + $0) * hidden + h] }
                    let expected = pooling == .mean
                        ? rows.reduce(0, +) / Float(rows.count)
                        : rows.max()!
                    XCTAssertEqual(flat[word * hidden + h], expected, accuracy: 1e-5,
                                   "\(pooling) pooling wrong at word \(word), dim \(h)")
                }
            }
        }
    }
}
