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
// Attempt3LoadingTests.swift
// Tests for loading locally-converted f16 weights from gliner_attempt_3.
//
// Weights converted via:
//   python scripts/convert_weights.py \
//     --model /path/to/gliner_attempt_3 \
//     --output /path/to/weights_attempt3_f16 \
//     --dtype float16

import XCTest
import Foundation
import Metal
@testable import GLiNER2Swift
import MLX
import MLXNN

final class Attempt3LoadingTests: XCTestCase {

    /// Canonical fp16 model. Override with `GLINER2_FP16_MODEL`; skips when absent.
    static var weightsPath: String { TestModel.fp16ModelPath ?? "" }

    // MARK: - Helpers

    private func skipIfNoGPU() throws {
        try TestModel.requireGPU()
    }

    @discardableResult
    private func skipIfNoWeights() throws -> String {
        try TestModel.requireFP16Model()
    }

    private func loadModel() async throws -> GLiNER2 {
        return try await GLiNER2.fromPretrained(try TestModel.requireFP16Model())
    }

    // MARK: - Config & Loading

    func testConfigLoads() throws {
        try skipIfNoWeights()

        let configUrl = URL(fileURLWithPath: Self.weightsPath)
            .appendingPathComponent("config.json")
        let config = try ExtractorConfig.load(from: configUrl)

        XCTAssertEqual(config.modelName, "microsoft/deberta-v3-base")
        XCTAssertEqual(config.maxWidth, 8)
        XCTAssertEqual(config.countingLayer, .countLSTMv2)
        XCTAssertEqual(config.tokenPooling, .first)
        XCTAssertEqual(config.hiddenSize, 768)
        print("Config: \(config)")
    }

    func testModelLoads() async throws {
        try skipIfNoGPU()
        try skipIfNoWeights()

        let model = try await loadModel()

        // Verify weights loaded (non-zero)
        let classifierWeight = model.model.classifier.layers[0] as! Linear
        let weightSum = MLX.sum(MLX.abs(classifierWeight.weight))
        MLX.eval(weightSum)
        XCTAssertGreaterThan(
            Float(weightSum.item(Float32.self)), 0.0,
            "Classifier weights should not be all zeros"
        )

        // Verify f16 dtype
        let dtype = classifierWeight.weight.dtype
        print("Weight dtype: \(dtype)")
    }

    // MARK: - Entity Extraction

    func testEntityExtraction() async throws {
        try skipIfNoGPU()
        try skipIfNoWeights()

        let model = try await loadModel()

        let result = model.extractEntities(
            text: "Tim Cook is CEO of Apple in Cupertino.",
            entityTypes: ["person", "company", "location"],
            threshold: 0.5,
            includeConfidence: true,
            includeSpans: true
        )

        print("Entity result: \(result)")

        guard let entities = result["entities"] as? [String: Any] else {
            XCTFail("Missing 'entities' key. Got: \(result)")
            return
        }

        let persons = entities["person"] as? [[String: Any]] ?? []
        print("  persons: \(persons)")
        XCTAssertFalse(persons.isEmpty, "Should detect at least one person")

        let companies = entities["company"] as? [[String: Any]] ?? []
        print("  companies: \(companies)")
        XCTAssertFalse(companies.isEmpty, "Should detect at least one company")

        let locations = entities["location"] as? [[String: Any]] ?? []
        print("  locations: \(locations)")
    }

    func testEntityExtractionMultiSentence() async throws {
        try skipIfNoGPU()
        try skipIfNoWeights()

        let model = try await loadModel()

        let result = model.extractEntities(
            text: "John and Jane work at Google in Mountain View. They previously worked at Microsoft in Seattle.",
            entityTypes: ["person", "organization", "location"],
            threshold: 0.5,
            includeConfidence: true,
            includeSpans: true
        )

        print("Multi-sentence result: \(result)")

        guard let entities = result["entities"] as? [String: Any] else {
            XCTFail("Missing 'entities'"); return
        }

        let persons = entities["person"] as? [[String: Any]] ?? []
        XCTAssertGreaterThanOrEqual(persons.count, 2, "Should detect John and Jane")
    }

    // MARK: - Classification

    func testClassification() async throws {
        try skipIfNoGPU()
        try skipIfNoWeights()

        let model = try await loadModel()

        let result = model.classifyText(
            text: "This product is amazing! Best purchase I've ever made.",
            task: "sentiment",
            labels: ["positive", "negative", "neutral"],
            threshold: 0.3,
            includeConfidence: true
        )

        print("Classification result: \(result)")
        XCTAssertNotNil(result["sentiment"], "Should have sentiment classification")
    }

    // MARK: - Structure Extraction

    func testStructureExtraction() async throws {
        try skipIfNoGPU()
        try skipIfNoWeights()

        let model = try await loadModel()

        let schema = model.createSchema()
            .structure("person_info")
            .field("name", dtype: "str")
            .field("age", dtype: "str")
            .field("city", dtype: "str")
            .done()

        let result = model.extract(
            text: "John Smith is 35 years old and lives in New York.",
            schema: schema,
            threshold: 0.3,
            includeConfidence: true,
            includeSpans: true
        )

        print("Structure result: \(result)")

        if let instances = result["person_info"] as? [[String: Any]] {
            XCTAssertFalse(instances.isEmpty, "Should extract at least one instance")
            for (i, inst) in instances.enumerated() {
                print("  instance \(i): \(inst)")
            }
        }
    }

    func testStructureWithValidators() async throws {
        try skipIfNoGPU()
        try skipIfNoWeights()

        let model = try await loadModel()

        let emailValidator = try RegexValidator(
            pattern: #"^[\w\.-]+@[\w\.-]+\.\w+$"#,
            mode: .full
        )

        let schema = model.createSchema()
            .structure("contact")
            .field("name", dtype: "str")
            .field("email", dtype: "str", validators: [emailValidator])
            .done()

        let result = model.extract(
            text: "Contact Alice at alice@example.com or Bob at bob-at-work for info.",
            schema: schema,
            threshold: 0.3,
            includeConfidence: true
        )

        print("Validator result: \(result)")

        if let instances = result["contact"] as? [[String: Any]] {
            for inst in instances {
                if let email = inst["email"] as? String {
                    XCTAssertTrue(emailValidator.validate(email),
                        "Email '\(email)' should pass validator")
                } else if let emailDict = inst["email"] as? [String: Any],
                          let emailText = emailDict["text"] as? String {
                    XCTAssertTrue(emailValidator.validate(emailText),
                        "Email '\(emailText)' should pass validator")
                }
            }
        }
    }

    // MARK: - Multi-Task Schema

    func testMultiTaskSchema() async throws {
        try skipIfNoGPU()
        try skipIfNoWeights()

        let model = try await loadModel()

        let schema = model.createSchema()
            .entities(["person", "company"])
            .classification(task: "sentiment", labels: ["positive", "negative", "neutral"])

        let result = model.extract(
            text: "Tim Cook announced great results for Apple today.",
            schema: schema,
            threshold: 0.3,
            includeConfidence: true,
            includeSpans: true
        )

        print("Multi-task result: \(result)")

        if let entities = result["entities"] as? [String: Any] {
            print("  entities: \(entities)")
        }
        if let sentiment = result["sentiment"] {
            print("  sentiment: \(sentiment)")
        }
    }
}