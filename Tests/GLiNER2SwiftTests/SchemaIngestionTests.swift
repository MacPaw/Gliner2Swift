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
// SchemaIngestionTests.swift
// Phase 6.3 — toDict/fromDict/fromJSON round-trips (Python's build() is toDict; a raw dict
// or JSON is accepted directly) and per-text schema lists in batchExtract (engine.py:358).

import XCTest
@testable import GLiNER2Swift

final class SchemaIngestionTests: XCTestCase {

    // MARK: - Ingestion (pure / deterministic)

    func testFromDictFillsMissingTopLevelKeys() {
        // A minimal dict — only entities — is normalized so downstream indexing is safe.
        let schema = Schema.fromDict(["entities": ["person": "", "location": ""]])
        let dict = schema.build()
        XCTAssertNotNil(dict["json_structures"])
        XCTAssertNotNil(dict["classifications"])
        XCTAssertNotNil(dict["relations"])
        XCTAssertEqual((dict["entities"] as? [String: Any])?.keys.sorted(), ["location", "person"])
    }

    func testFromJSONParsesObject() throws {
        let json = #"{"entities": {"company": ""}}"#
        let schema = try Schema.fromJSON(json)
        XCTAssertEqual((schema.build()["entities"] as? [String: Any])?.keys.first, "company")
    }

    func testFromJSONRejectsNonObject() {
        XCTAssertThrowsError(try Schema.fromJSON("[1, 2, 3]"))
    }

    // MARK: - Round-trip end-to-end (builder → toDict → fromDict extracts identically)

    func testDictRoundTripExtractsIdentically() async throws {
        try TestModel.requireGPU()
        let model = try await GLiNER2.fromPretrained(try TestModel.requireFP16Model())
        let text = "Tim Cook leads Apple in Cupertino."

        let built = model.createSchema().entities(["person", "organization", "location"])
        let fromBuilt = model.extract(text: text, schema: built, threshold: 0.4, includeSpans: true)

        // Round-trip the built schema through a dict and re-ingest.
        let reingested = Schema.fromDict(built.toDict())
        let fromDict = model.extract(text: text, schema: reingested, threshold: 0.4, includeSpans: true)

        XCTAssertEqual(key(fromBuilt), key(fromDict))
    }

    // MARK: - Per-text schema lists

    func testPerTextSchemasApplyIndependently() async throws {
        try TestModel.requireGPU()
        let model = try await GLiNER2.fromPretrained(try TestModel.requireFP16Model())
        let texts = [
            "Tim Cook leads Apple in Cupertino.",
            "The match ended 2-1 in favour of the home side.",
        ]
        // Text 0 gets an entity schema, text 1 a classification schema.
        let schemas = [
            model.createSchema().entities(["person", "organization"]),
            model.createSchema().classification(task: "topic", labels: ["business", "sport"]),
        ]

        let batch = model.batchExtract(texts: texts, schemas: schemas, threshold: 0.4,
                                       includeSpans: true)
        XCTAssertEqual(batch.count, 2)

        // Each result matches extracting that text with its own schema.
        for (i, text) in texts.enumerated() {
            let single = model.extract(text: text, schema: schemas[i], threshold: 0.4,
                                       includeSpans: true)
            XCTAssertEqual(key(batch[i]), key(single), "per-text schema mismatch at \(i)")
        }
        // And the two carry different top-level keys (entities vs the classification task).
        XCTAssertNotNil(batch[0]["entities"])
        XCTAssertNil(batch[1]["entities"])
        XCTAssertNotNil(batch[1]["topic"])
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
