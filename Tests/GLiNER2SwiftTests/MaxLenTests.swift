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
// MaxLenTests.swift
// Phase 6.1 — `maxLen` truncates the whitespace-split text words after splitting and
// before schema/prefix encoding, while the surviving words keep their original char
// offsets. Mirrors Python's tests/test_inference_max_len.py.

import XCTest
@testable import GLiNER2Swift

final class MaxLenTests: XCTestCase {

    private let words = "one two three four five six seven eight nine ten eleven twelve"

    // MARK: - Truncation contract (deterministic, tokenizer only)

    func testMaxLenTruncatesWordTokensAndKeepsCharMaps() throws {
        let dir = try TestModel.requireTokenizerDirectory()
        let processor = try SchemaTransformer.createFromLocalDirectory(directoryUrl: dir)
        let schema = Schema().entities(["thing"]).build()

        let full = processor.transform(text: words, schema: schema)
        let capped = processor.transform(text: words, schema: schema, maxLen: 5)

        // 13 word tokens in (12 number-words + the appended "." which the splitter emits
        // as its own token); first 5 kept.
        XCTAssertEqual(full.startTokenIdx.count, 13)
        XCTAssertEqual(capped.startTokenIdx.count, 5)

        // Surviving words keep the exact char offsets they had in the untruncated pass —
        // truncation drops tail words, it does not renumber the survivors.
        XCTAssertEqual(capped.startTokenIdx, Array(full.startTokenIdx.prefix(5)))
        XCTAssertEqual(capped.endTokenIdx, Array(full.endTokenIdx.prefix(5)))
        XCTAssertEqual(capped.textTokens, Array(full.textTokens.prefix(5)))
    }

    func testMaxLenLargerThanInputIsNoOp() throws {
        let dir = try TestModel.requireTokenizerDirectory()
        let processor = try SchemaTransformer.createFromLocalDirectory(directoryUrl: dir)
        let schema = Schema().entities(["thing"]).build()

        let full = processor.transform(text: words, schema: schema)
        let capped = processor.transform(text: words, schema: schema, maxLen: 1000)
        XCTAssertEqual(capped.startTokenIdx, full.startTokenIdx)
        XCTAssertEqual(capped.textTokens, full.textTokens)
    }

    // MARK: - End-to-end bound (deterministic: nothing extracted past the cut)

    func testMaxLenBoundsExtractedSpans() async throws {
        try TestModel.requireGPU()
        let model = try await GLiNER2.fromPretrained(try TestModel.requireFP16Model())

        // Distinct entities early and late; the late one lives past a small cap.
        let text = "Tim Cook leads Apple in Cupertino. "
            + String(repeating: "filler word here. ", count: 20)
            + "Satya Nadella runs Microsoft in Redmond."
        let schema = model.createSchema().entities(["person", "organization", "location"])
        let maxLen = 6

        // The char position where truncation cuts — end of the last kept word.
        let record = model.processor.transform(text: model.normalizeForTest(text),
                                               schema: schema.build(), maxLen: maxLen)
        let boundary = record.endTokenIdx.last ?? 0

        let capped = model.extract(text: text, schema: schema, threshold: 0.3,
                                   includeConfidence: false, includeSpans: true, maxLen: maxLen)
        guard let entities = capped["entities"] as? [String: [Any]] else {
            return XCTFail("no entities key")
        }
        var beyond: [String] = []
        for (_, spans) in entities {
            for span in spans {
                guard let dict = span as? [String: Any], let start = dict["start"] as? Int else { continue }
                if start >= boundary { beyond.append(dict["text"] as? String ?? "?") }
            }
        }
        XCTAssertTrue(beyond.isEmpty,
                      "maxLen \(maxLen) (char boundary \(boundary)) must not surface spans past the "
                      + "cut, but got: \(beyond)")

        // Sanity: the uncapped call reaches the tail entity that the cap excludes.
        let full = model.extract(text: text, schema: schema, threshold: 0.3, includeSpans: true)
        let fullOrgs = ((full["entities"] as? [String: [Any]])?["organization"] as? [Any] ?? [])
            .compactMap { ($0 as? [String: Any])?["text"] as? String }
        XCTAssertTrue(fullOrgs.contains { $0.contains("Microsoft") },
                      "uncapped run should find the tail organization; got \(fullOrgs)")
    }
}

extension GLiNER2 {
    /// Test hook: apply the same terminal-punctuation normalization the extract path does,
    /// so a boundary computed via `processor.transform` lines up with extracted spans.
    func normalizeForTest(_ text: String) -> String {
        if text.isEmpty { return "." }
        if text.hasSuffix(".") || text.hasSuffix("!") || text.hasSuffix("?") { return text }
        return text + "."
    }
}
