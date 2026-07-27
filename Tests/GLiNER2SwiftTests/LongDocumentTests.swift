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
// LongDocumentTests.swift
// Phase 6.2 — chunk splitting, span remapping onto the original document, and cross-chunk
// merge/dedup.

import XCTest
@testable import GLiNER2Swift

final class LongDocumentTests: XCTestCase {

    // MARK: - Chunker mechanics (pure, deterministic)

    func testChunkBoundariesAndOverlap() {
        let splitter = WhitespaceTokenSplitter()
        // 10 words, chunkSize 5, overlap 2 → step 3 → starts at word 0, 3, 6; the window at
        // 6 already covers through word 9, so the walk stops there (no redundant tail chunk).
        let text = "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9"
        let chunks = TextChunker.split(text, chunkSize: 5, overlap: 2, splitter: splitter)

        XCTAssertEqual(chunks.map(\.text), [
            "w0 w1 w2 w3 w4",   // words 0..4
            "w3 w4 w5 w6 w7",   // words 3..7
            "w6 w7 w8 w9",      // words 6..9 (reaches the end → stop)
        ])
        // Every chunk's text is the original sliced at its reported offset.
        let scalars = Array(text.unicodeScalars)
        for chunk in chunks {
            let slice = String(String.UnicodeScalarView(
                scalars[chunk.charOffset ..< chunk.charOffset + chunk.text.unicodeScalars.count]))
            XCTAssertEqual(slice, chunk.text)
        }
    }

    func testShortTextIsSingleChunk() {
        let splitter = WhitespaceTokenSplitter()
        let chunks = TextChunker.split("just three words", chunkSize: 384, overlap: 64, splitter: splitter)
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks.first?.charOffset, 0)
    }

    // MARK: - Merge (pure)

    func testMergeDedupsEntitiesAcrossChunks() {
        // Same "Apple" surfaced in two overlapping chunks → one entry.
        let a: [String: Any] = ["entities": ["org": [["text": "Apple", "start": 0, "end": 5]]]]
        let b: [String: Any] = ["entities": ["org": [["text": "apple", "start": 0, "end": 5],
                                                      ["text": "Google", "start": 10, "end": 16]]]]
        let merged = GLiNER2.mergeChunkResults([a, b])
        let org = (merged["entities"] as? [String: Any])?["org"] as? [Any] ?? []
        let texts = org.compactMap { ($0 as? [String: Any])?["text"] as? String }
        XCTAssertEqual(texts, ["Apple", "Google"])   // case-insensitive dedup kept first
    }

    func testMergeClassificationPicksHighestConfidence() {
        let a: [String: Any] = ["topic": ["label": "sport", "confidence": Float(0.6)]]
        let b: [String: Any] = ["topic": ["label": "business", "confidence": Float(0.9)]]
        let merged = GLiNER2.mergeChunkResults([a, b])
        let topic = merged["topic"] as? [String: Any]
        XCTAssertEqual(topic?["label"] as? String, "business")
    }

    func testOutputFlagsStripMetadata() {
        let withMeta: [String: Any] = ["entities": ["org": [["text": "Apple", "start": 0, "end": 5,
                                                             "confidence": Float(0.9)]]]]
        let bare = GLiNER2.applyOutputFlags(withMeta, includeConfidence: false, includeSpans: false)
        let org = (bare["entities"] as? [String: Any])?["org"] as? [Any] ?? []
        XCTAssertEqual(org.first as? String, "Apple")   // collapsed to bare string
    }

    // MARK: - End-to-end: spans remap onto the ORIGINAL text

    func testExtractLongRemapsSpansToOriginal() async throws {
        try TestModel.requireGPU()
        let model = try await GLiNER2.fromPretrained(try TestModel.requireFP16Model())

        // A distinctive location early and another far in the tail, past chunk 1.
        let head = "The conference opened in Reykjavik with a keynote. "
        let filler = String(repeating: "The session continued with more discussion. ", count: 25)
        let tail = "The closing remarks were delivered in Kathmandu."
        let text = head + filler + tail

        let result = model.extractLong(
            text: text, schema: model.createSchema().entities(["location"]),
            chunkSize: 40, overlap: 10, threshold: 0.3, includeSpans: true)

        let spans = (result["entities"] as? [String: Any])?["location"] as? [Any] ?? []
        XCTAssertFalse(spans.isEmpty, "expected locations across chunks")

        let scalars = Array(text.unicodeScalars)
        var found = Set<String>()
        for span in spans {
            guard let dict = span as? [String: Any],
                  let start = dict["start"] as? Int, let end = dict["end"] as? Int,
                  let entityText = dict["text"] as? String else { continue }
            XCTAssertTrue(start >= 0 && end <= scalars.count && start < end,
                          "span out of original bounds: \(start)..\(end) of \(scalars.count)")
            // The remapped span must slice back to exactly the entity text in the ORIGINAL.
            let sliced = String(String.UnicodeScalarView(scalars[start ..< end]))
            XCTAssertEqual(sliced, entityText, "remapped span does not match original text")
            found.insert(entityText)
        }
        // The tail location lives well past the first chunk; chunking must still reach it.
        XCTAssertTrue(found.contains { $0.contains("Kathmandu") },
                      "tail-of-document entity should be found via a later chunk; found \(found)")
    }
}
