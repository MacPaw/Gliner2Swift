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
// ExtractJsonTests.swift
// Phase 6.4 — the `name::dtype::[a|b]::desc` field-spec parser, and that extractJson builds
// the same schema (and result) as the equivalent StructureBuilder calls.

import XCTest
@testable import GLiNER2Swift

final class ExtractJsonTests: XCTestCase {

    // MARK: - Parser (pure, deterministic)

    func testParseFieldSpecShapes() {
        // Bare name → single-value str field.
        XCTAssertEqual(GLiNER2.parseJSONFieldSpec("name"),
                       .init(name: "name", dtype: "str", choices: nil, description: nil))

        // name + dtype.
        XCTAssertEqual(GLiNER2.parseJSONFieldSpec("tags::list"),
                       .init(name: "tags", dtype: "list", choices: nil, description: nil))

        // Full form: name, dtype, choices, description.
        XCTAssertEqual(
            GLiNER2.parseJSONFieldSpec("sentiment::str::[positive|negative|neutral]::the tone"),
            .init(name: "sentiment", dtype: "str",
                  choices: ["positive", "negative", "neutral"], description: "the tone"))

        // Shape-matched, not position-strict: choices/description/dtype in any order.
        let reordered = GLiNER2.parseJSONFieldSpec("mood::[up|down]::a mood::list")
        XCTAssertEqual(reordered.name, "mood")
        XCTAssertEqual(reordered.dtype, "list")
        XCTAssertEqual(reordered.choices, ["up", "down"])
        XCTAssertEqual(reordered.description, "a mood")

        // Whitespace tolerated.
        XCTAssertEqual(GLiNER2.parseJSONFieldSpec("  city :: str "),
                       .init(name: "city", dtype: "str", choices: nil, description: nil))
    }

    // MARK: - Schema equivalence (extractJson == equivalent builder)

    func testExtractJsonMatchesEquivalentBuilder() async throws {
        try TestModel.requireGPU()
        let model = try await GLiNER2.fromPretrained(try TestModel.requireFP16Model())

        let text = "Tim Cook, 63, is the chief executive of Apple."
        let viaJson = model.extractJson(
            text: text, name: "person",
            fields: ["name::str", "age::str", "role::str::the job title"],
            threshold: 0.3, includeSpans: true)

        let builderSchema = model.createSchema()
            .structure("person")
            .field("name", dtype: "str")
            .field("age", dtype: "str")
            .field("role", dtype: "str", description: "the job title")
            .done()
        let viaBuilder = model.extract(text: text, schema: builderSchema,
                                       threshold: 0.3, includeSpans: true)

        XCTAssertEqual(key(viaJson), key(viaBuilder))
    }

    func testBatchExtractJsonMatchesSingle() async throws {
        try TestModel.requireGPU()
        let model = try await GLiNER2.fromPretrained(try TestModel.requireFP16Model())
        let texts = ["Tim Cook runs Apple.", "Satya Nadella runs Microsoft."]
        let fields = ["person::str", "company::str"]

        let batch = model.batchExtractJson(texts: texts, name: "record", fields: fields, threshold: 0.3)
        XCTAssertEqual(batch.count, texts.count)
        for (i, text) in texts.enumerated() {
            let single = model.extractJson(text: text, name: "record", fields: fields, threshold: 0.3)
            XCTAssertEqual(key(batch[i]), key(single))
        }
    }

    private func key(_ value: Any) -> String {
        switch value {
        case let d as [String: Any]:
            return "{" + d.keys.sorted().map { "\($0):\(key(d[$0]!))" }.joined(separator: ",") + "}"
        case let a as [Any]: return "[" + a.map { key($0) }.sorted().joined(separator: ",") + "]"
        case let s as String: return s
        case let n as Int: return String(n)
        case let f as Float: return String(format: "%.4f", f)
        case is NSNull: return "null"
        default: return "\(value)"
        }
    }
}
