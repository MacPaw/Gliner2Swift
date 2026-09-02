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
// BatchWrapperTests.swift
// Phase 6.8 — the batch convenience wrappers are thin call-throughs; each must produce,
// per text, exactly what the single-text method produces for that text.

import XCTest
@testable import GLiNER2Swift

final class BatchWrapperTests: XCTestCase {

    private func makeModel() async throws -> GLiNER2 {
        try TestModel.requireGPU()
        return try await GLiNER2.fromPretrained(try TestModel.requireFP16Model())
    }

    private static let texts = [
        "Tim Cook leads Apple in Cupertino.",
        "Maria Gonzalez joined Siemens in Munich.",
    ]

    func testBatchExtractEntitiesMatchesSingle() async throws {
        let model = try await makeModel()
        let types = ["person", "organization", "location"]
        let batch = model.batchExtractEntities(texts: Self.texts, entityTypes: types,
                                               threshold: 0.4, includeSpans: true)
        XCTAssertEqual(batch.count, Self.texts.count)
        for (index, text) in Self.texts.enumerated() {
            let single = model.extractEntities(text: text, entityTypes: types,
                                               threshold: 0.4, includeSpans: true)
            XCTAssertEqual(Self.key(batch[index]), Self.key(single), "mismatch on: \(text)")
        }
    }

    func testBatchClassifyMatchesSingle() async throws {
        let model = try await makeModel()
        let batch = model.batchClassifyText(texts: Self.texts, task: "topic",
                                            labels: ["business", "sport"], includeConfidence: true)
        XCTAssertEqual(batch.count, Self.texts.count)
        for (index, text) in Self.texts.enumerated() {
            let single = model.classifyText(text: text, task: "topic",
                                            labels: ["business", "sport"], includeConfidence: true)
            XCTAssertEqual(Self.key(batch[index]), Self.key(single), "mismatch on: \(text)")
        }
    }

    func testBatchExtractRelationsMatchesSingle() async throws {
        let model = try await makeModel()
        let batch = model.batchExtractRelations(texts: Self.texts, relationTypes: ["works_at"],
                                                threshold: 0.3)
        XCTAssertEqual(batch.count, Self.texts.count)
        for (index, text) in Self.texts.enumerated() {
            let single = model.extractRelations(text: text, relationTypes: ["works_at"], threshold: 0.3)
            XCTAssertEqual(Self.key(batch[index]), Self.key(single), "mismatch on: \(text)")
        }
    }

    /// Stable, order-insensitive string form of a result dict for comparison.
    private static func key(_ value: Any) -> String {
        switch value {
        case let dict as [String: Any]:
            return "{" + dict.keys.sorted().map { "\($0):\(key(dict[$0]!))" }.joined(separator: ",") + "}"
        case let array as [Any]:
            return "[" + array.map { key($0) }.sorted().joined(separator: ",") + "]"
        case let s as String: return s
        case let n as Int: return String(n)
        case let f as Float: return String(format: "%.4f", f)
        default: return "\(value)"
        }
    }
}
