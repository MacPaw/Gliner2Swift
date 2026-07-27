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
// ClassificationExtrasTests.swift
// Phase 6.5 — classification prompt / label descriptions / few-shot examples serialize to
// exactly the schema tokens Python emits at inference (example_mode "both", no shuffle).
// Expected tokens captured from gliner2's `_transform_schema` on fastino/gliner2-base-v1.

import XCTest
@testable import GLiNER2Swift

final class ClassificationExtrasTests: XCTestCase {

    private let labels = ["positive", "negative", "neutral"]

    private func classificationTokens(_ schema: Schema) throws -> [String] {
        let dir = try TestModel.requireTokenizerDirectory()
        let processor = try SchemaTransformer.createFromLocalDirectory(directoryUrl: dir)
        let record = processor.transform(text: "The film.", schema: schema.build())
        // The classification block is the one carrying the label markers.
        return try XCTUnwrap(record.schemaTokensList.first { $0.contains("[L]") })
    }

    func testPromptSerialization() throws {
        let schema = Schema().classification(task: "sentiment", labels: labels,
                                             prompt: "how does the reviewer feel")
        XCTAssertEqual(try classificationTokens(schema), [
            "(", "[P]", "sentiment: how does the reviewer feel",
            "(", "[L]", "positive", "[L]", "negative", "[L]", "neutral", ")", ")",
        ])
    }

    func testDescriptionSerialization() throws {
        let schema = Schema().classification(
            task: "sentiment", labels: labels,
            labelDescriptions: ["positive": "good vibes", "negative": "bad vibes"])
        XCTAssertEqual(try classificationTokens(schema), [
            "(", "[P]",
            "sentiment [DESCRIPTION] positive: good vibes [DESCRIPTION] negative: bad vibes",
            "(", "[L]", "positive", "[L]", "negative", "[L]", "neutral", ")", ")",
        ])
    }

    func testExampleSerialization() throws {
        let schema = Schema().classification(
            task: "sentiment", labels: labels,
            examples: [("great film", "positive"), ("awful", "negative")])
        XCTAssertEqual(try classificationTokens(schema), [
            "(", "[P]",
            "sentiment [EXAMPLE] great film [OUTPUT] positive [EXAMPLE] awful [OUTPUT] negative",
            "(", "[L]", "positive", "[L]", "negative", "[L]", "neutral", ")", ")",
        ])
    }

    func testAllThreeSerialization() throws {
        let schema = Schema().classification(
            task: "sentiment", labels: labels, prompt: "tone",
            labelDescriptions: ["positive": "good", "neutral": "meh"],
            examples: [("loved it", "positive")])
        XCTAssertEqual(try classificationTokens(schema), [
            "(", "[P]",
            "sentiment: tone [DESCRIPTION] positive: good [DESCRIPTION] neutral: meh "
                + "[EXAMPLE] loved it [OUTPUT] positive",
            "(", "[L]", "positive", "[L]", "negative", "[L]", "neutral", ")", ")",
        ])
    }

    /// An example whose output is not one of the labels is dropped (Python `if out in fields`).
    func testExampleWithUnknownOutputIsDropped() throws {
        let schema = Schema().classification(
            task: "sentiment", labels: labels,
            examples: [("meh", "mixed"), ("nice", "positive")])
        let tokens = try classificationTokens(schema)
        XCTAssertEqual(tokens[2], "sentiment [EXAMPLE] nice [OUTPUT] positive")
    }

    // MARK: - Multi-task classifyText

    func testMultiTaskClassifyProducesAllTaskKeys() async throws {
        try TestModel.requireGPU()
        let model = try await GLiNER2.fromPretrained(try TestModel.requireFP16Model())
        let result = model.classifyText(text: "The team won the championship in overtime.", tasks: [
            .init(task: "topic", labels: ["sport", "business"]),
            .init(task: "tone", labels: ["exciting", "dull"]),
        ])
        XCTAssertNotNil(result["topic"])
        XCTAssertNotNil(result["tone"])
    }

    /// End-to-end parity against Python. Expected captured from
    /// `gliner2.classify_text` on fastino/gliner2-base-v1:
    ///   {"sentiment: reviewer sentiment": {"label": "positive", "confidence": 1.0}}
    /// The result KEY is `schema_tokens[2].split(" [DESCRIPTION] ")[0]` = "task: prompt"
    /// (engine.py:535), not the bare task name — this pins that.
    func testClassifyWithExtrasMatchesPython() async throws {
        try TestModel.requireGPU()
        let model = try await GLiNER2.fromPretrained(try TestModel.requireFP16Model())
        let result = model.classifyText(
            text: "Absolutely loved every minute of it, a true masterpiece.",
            task: "sentiment", labels: ["positive", "negative", "neutral"],
            includeConfidence: true,
            prompt: "reviewer sentiment",
            labelDescriptions: ["positive": "enjoyed it", "negative": "disliked it"],
            examples: [("a wonderful film", "positive")])

        XCTAssertNil(result["sentiment"], "Python keys by 'task: prompt', not the bare task")
        let entry = try XCTUnwrap(result["sentiment: reviewer sentiment"] as? [String: Any],
                                  "expected key 'sentiment: reviewer sentiment'")
        XCTAssertEqual(entry["label"] as? String, "positive")
        XCTAssertEqual(try XCTUnwrap(entry["confidence"] as? Float), 1.0, accuracy: 0.02)
    }
}
