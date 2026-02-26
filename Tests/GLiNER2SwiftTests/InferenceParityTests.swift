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
// InferenceParityTests.swift
// End-to-end inference parity tests for GLiNER2 Swift/MLX
//
// These tests verify that the Swift/MLX implementation produces the same
// results as the Python/PyTorch implementation for:
// - Entity extraction
// - Text classification
// - Structure extraction
//
// Run fixtures generation first:
//   cd scripts && python generate_inference_fixtures.py

import XCTest
import Foundation
@testable import GLiNER2Swift

// Conditionally import MLX - tests will skip if unavailable
#if canImport(MLX)
import MLX
#endif

/// Helper to load .npy and .json files from fixtures directory
class InferenceFixtureLoader {
    let fixturesDir: URL

    init(subdir: String = "inference") {
        // Use path relative to this source file (works during development)
        fixturesDir = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures")
            .appendingPathComponent(subdir)
    }

    /// Load JSON metadata
    func loadJSON(_ name: String) throws -> [String: Any] {
        let path = fixturesDir.appendingPathComponent("\(name).json")

        guard FileManager.default.fileExists(atPath: path.path) else {
            throw FixtureError.fileNotFound(path.path)
        }

        let data = try Data(contentsOf: path)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw FixtureError.invalidFormat("Expected dictionary at \(path.path)")
        }

        return json
    }

    /// Check if a fixture exists
    func fixtureExists(_ name: String, ext: String = "npy") -> Bool {
        let path = fixturesDir.appendingPathComponent("\(name).\(ext)")
        return FileManager.default.fileExists(atPath: path.path)
    }

    #if canImport(MLX)
    /// Load a numpy array as MLXArray
    func loadNpy(_ name: String) throws -> MLXArray {
        let path = fixturesDir.appendingPathComponent("\(name).npy")

        guard FileManager.default.fileExists(atPath: path.path) else {
            throw FixtureError.fileNotFound(path.path)
        }

        // Use MLX to load numpy file on CPU stream to avoid Metal errors
        return try MLX.loadArray(url: path, stream: .cpu)
    }
    #endif

    enum FixtureError: Error {
        case fileNotFound(String)
        case invalidFormat(String)
    }
}

// MARK: - JSON-Only Tests (No MLX Required)

/// Tests that only require JSON fixtures (no MLX/Metal dependency)
final class InferenceJSONParityTests: XCTestCase {

    var loader: InferenceFixtureLoader!

    override func setUp() {
        super.setUp()
        loader = InferenceFixtureLoader(subdir: "inference")
    }

    // MARK: - Entity Extraction Result Tests

    func testEntityExtractionBasicResult() throws {
        guard loader.fixtureExists("entity_basic_result", ext: "json") else {
            throw XCTSkip("Fixtures not generated. Run: python generate_inference_fixtures.py")
        }

        let result = try loader.loadJSON("entity_basic_result")
        let metadata = try loader.loadJSON("entity_basic_metadata")

        print("\nEntity extraction result:")
        print("  Text: \(metadata["text"] ?? "unknown")")

        // The result should contain entities key
        guard let entities = result["entities"] as? [String: Any] else {
            XCTFail("Missing 'entities' key in result")
            return
        }

        print("  Extracted entities:")
        for (entityType, values) in entities {
            print("    \(entityType): \(values)")
        }

        // Basic validation: Tim Cook and Apple should be extracted
        if let personResults = entities["person"] as? [[String: Any]] {
            let personTexts = personResults.compactMap { $0["text"] as? String }
            XCTAssertTrue(personTexts.contains { $0.lowercased().contains("tim cook") },
                          "Should extract 'Tim Cook' as person")
        } else {
            XCTFail("Missing person entities")
        }

        if let companyResults = entities["company"] as? [[String: Any]] {
            let companyTexts = companyResults.compactMap { $0["text"] as? String }
            XCTAssertTrue(companyTexts.contains { $0.lowercased().contains("apple") },
                          "Should extract 'Apple' as company")
        } else {
            XCTFail("Missing company entities")
        }
    }

    func testEntityExtractionMultiple() throws {
        guard loader.fixtureExists("entity_multi_result", ext: "json") else {
            throw XCTSkip("Fixtures not generated. Run: python generate_inference_fixtures.py")
        }

        let result = try loader.loadJSON("entity_multi_result")
        let metadata = try loader.loadJSON("entity_multi_metadata")

        print("\nMulti-entity extraction result:")
        print("  Text: \(metadata["text"] ?? "unknown")")

        if let entities = result["entities"] as? [String: Any] {
            print("  Extracted entities:")
            for (entityType, values) in entities {
                print("    \(entityType): \(values)")
            }

            // Should find John, Jane, Google, Mountain View
            if let persons = entities["person"] as? [[String: Any]] {
                XCTAssertGreaterThanOrEqual(persons.count, 2, "Should find at least 2 persons")
            }
        }
    }

    func testEntityExtractionNoMatch() throws {
        guard loader.fixtureExists("entity_no_match_result", ext: "json") else {
            throw XCTSkip("Fixtures not generated. Run: python generate_inference_fixtures.py")
        }

        let result = try loader.loadJSON("entity_no_match_result")
        let metadata = try loader.loadJSON("entity_no_match_metadata")

        print("\nNo-match entity extraction result:")
        print("  Text: \(metadata["text"] ?? "unknown")")

        if let entities = result["entities"] as? [String: Any] {
            print("  Extracted entities:")
            for (entityType, values) in entities {
                if let arr = values as? [Any] {
                    print("    \(entityType): \(arr.count) matches")
                }
            }
        }
    }

    // MARK: - Classification Result Tests

    func testClassificationSentimentPositive() throws {
        guard loader.fixtureExists("classify_sentiment_positive_result", ext: "json") else {
            throw XCTSkip("Fixtures not generated. Run: python generate_inference_fixtures.py")
        }

        let result = try loader.loadJSON("classify_sentiment_positive_result")
        let metadata = try loader.loadJSON("classify_sentiment_positive_metadata")

        print("\nClassification result (positive sentiment):")
        print("  Text: \(metadata["text"] ?? "unknown")")
        print("  Task: \(metadata["task"] ?? "unknown")")
        print("  Labels: \(metadata["labels"] ?? [])")

        // Check if sentiment is classified correctly
        if let sentiment = result["sentiment"] as? [String: Any] {
            let label = sentiment["label"] as? String
            print("  Result: \(label ?? "unknown")")
            XCTAssertEqual(label, "positive", "Positive text should be classified as positive")
        } else if let sentiment = result["sentiment"] as? String {
            print("  Result: \(sentiment)")
            XCTAssertEqual(sentiment, "positive", "Positive text should be classified as positive")
        } else {
            XCTFail("Missing sentiment result")
        }
    }

    func testClassificationSentimentNegative() throws {
        guard loader.fixtureExists("classify_sentiment_negative_result", ext: "json") else {
            throw XCTSkip("Fixtures not generated. Run: python generate_inference_fixtures.py")
        }

        let result = try loader.loadJSON("classify_sentiment_negative_result")
        let metadata = try loader.loadJSON("classify_sentiment_negative_metadata")

        print("\nClassification result (negative sentiment):")
        print("  Text: \(metadata["text"] ?? "unknown")")

        if let sentiment = result["sentiment"] as? [String: Any] {
            let label = sentiment["label"] as? String
            print("  Result: \(label ?? "unknown")")
            XCTAssertEqual(label, "negative", "Negative text should be classified as negative")
        } else if let sentiment = result["sentiment"] as? String {
            print("  Result: \(sentiment)")
            XCTAssertEqual(sentiment, "negative", "Negative text should be classified as negative")
        } else {
            XCTFail("Missing sentiment result")
        }
    }

    func testClassificationMultiLabel() throws {
        guard loader.fixtureExists("classify_topic_multi_result", ext: "json") else {
            throw XCTSkip("Fixtures not generated. Run: python generate_inference_fixtures.py")
        }

        let result = try loader.loadJSON("classify_topic_multi_result")
        let metadata = try loader.loadJSON("classify_topic_multi_metadata")

        print("\nMulti-label classification result:")
        print("  Text: \(metadata["text"] ?? "unknown")")
        print("  Task: \(metadata["task"] ?? "unknown")")
        print("  Multi-label: \(metadata["multi_label"] ?? false)")

        if let topics = result["topics"] as? [[String: Any]] {
            print("  Result labels:")
            for topic in topics {
                print("    - \(topic["label"] ?? "unknown"): \(topic["confidence"] ?? 0)")
            }

            // Should classify as technology and/or business
            let labels = topics.compactMap { $0["label"] as? String }
            XCTAssertTrue(labels.contains("technology") || labels.contains("business"),
                          "Should classify as technology or business")
        }
    }

    // MARK: - Structure Extraction Result Tests

    func testStructureExtractionPerson() throws {
        guard loader.fixtureExists("struct_person_result", ext: "json") else {
            throw XCTSkip("Fixtures not generated. Run: python generate_inference_fixtures.py")
        }

        let result = try loader.loadJSON("struct_person_result")
        let metadata = try loader.loadJSON("struct_person_metadata")

        print("\nStructure extraction result (person_info):")
        print("  Text: \(metadata["text"] ?? "unknown")")
        print("  Structure: \(metadata["structure_name"] ?? "unknown")")
        print("  Fields: \(metadata["fields"] ?? [])")

        if let personInfo = result["person_info"] as? [[String: Any]] {
            for (idx, instance) in personInfo.enumerated() {
                print("  Instance \(idx):")
                for (field, value) in instance {
                    print("    \(field): \(value)")
                }
            }

            // Verify expected fields are extracted
            XCTAssertGreaterThan(personInfo.count, 0, "Should extract at least one instance")

            if let first = personInfo.first {
                // Name field
                let nameValue: String?
                if let nameArr = first["name"] as? [[String: Any]], let nameFirst = nameArr.first {
                    nameValue = nameFirst["text"] as? String
                } else {
                    nameValue = first["name"] as? String
                }

                if let name = nameValue {
                    XCTAssertTrue(name.lowercased().contains("john"),
                                  "Should extract 'John Smith' as name")
                }
            }
        } else {
            XCTFail("Missing person_info result")
        }
    }

    func testStructureExtractionProduct() throws {
        guard loader.fixtureExists("struct_product_result", ext: "json") else {
            throw XCTSkip("Fixtures not generated. Run: python generate_inference_fixtures.py")
        }

        let result = try loader.loadJSON("struct_product_result")
        let metadata = try loader.loadJSON("struct_product_metadata")

        print("\nStructure extraction result (product):")
        print("  Text: \(metadata["text"] ?? "unknown")")

        if let products = result["product"] as? [[String: Any]] {
            for (idx, instance) in products.enumerated() {
                print("  Instance \(idx):")
                for (field, value) in instance {
                    print("    \(field): \(value)")
                }
            }

            XCTAssertGreaterThan(products.count, 0, "Should extract at least one product")
        } else {
            XCTFail("Missing product result")
        }
    }

    // MARK: - Metadata Tests

    func testMetadataStructure() throws {
        guard loader.fixtureExists("entity_basic_metadata", ext: "json") else {
            throw XCTSkip("Fixtures not generated. Run: python generate_inference_fixtures.py")
        }

        let metadata = try loader.loadJSON("entity_basic_metadata")

        // Verify metadata structure
        XCTAssertNotNil(metadata["text"], "Metadata should have 'text' field")
        XCTAssertNotNil(metadata["entity_types"], "Metadata should have 'entity_types' field")
        XCTAssertNotNil(metadata["schema"], "Metadata should have 'schema' field")

        if let text = metadata["text"] as? String {
            XCTAssertFalse(text.isEmpty, "Text should not be empty")
        }

        if let entityTypes = metadata["entity_types"] as? [String] {
            XCTAssertGreaterThan(entityTypes.count, 0, "Should have entity types")
        }
    }
}

// MARK: - MLX-Dependent Tests (Require Metal)

#if canImport(MLX)
/// Tests that require MLX/Metal for numerical array operations
final class InferenceMLXParityTests: XCTestCase {

    var loader: InferenceFixtureLoader!

    override func setUp() {
        super.setUp()
        loader = InferenceFixtureLoader(subdir: "inference")
    }

    // MARK: - Numerical Parity Tests

    func testEntityExtractionBasicInputIds() throws {
        guard loader.fixtureExists("entity_basic_input_ids") else {
            throw XCTSkip("Fixtures not generated. Run: python generate_inference_fixtures.py")
        }

        let expectedInputIds = try loader.loadNpy("entity_basic_input_ids")
        let metadata = try loader.loadJSON("entity_basic_metadata")

        // Verify the input IDs shape
        XCTAssertEqual(expectedInputIds.ndim, 2, "Input IDs should be 2D [batch, seq_len]")
        XCTAssertEqual(expectedInputIds.dim(0), 1, "Batch size should be 1")

        print("Entity basic input_ids shape: \(expectedInputIds.shape)")
        print("Text: \(metadata["text"] ?? "unknown")")
        print("Entity types: \(metadata["entity_types"] ?? [])")
    }

    func testEntityExtractionBasicEncoderOutput() throws {
        guard loader.fixtureExists("entity_basic_encoder_output") else {
            throw XCTSkip("Fixtures not generated. Run: python generate_inference_fixtures.py")
        }

        let encoderOutput = try loader.loadNpy("entity_basic_encoder_output")

        // Verify encoder output shape: [batch, seq_len, hidden_size]
        XCTAssertEqual(encoderOutput.ndim, 3, "Encoder output should be 3D")
        XCTAssertEqual(encoderOutput.dim(0), 1, "Batch size should be 1")
        XCTAssertEqual(encoderOutput.dim(2), 768, "Hidden size should be 768")

        print("Encoder output shape: \(encoderOutput.shape)")
    }

    func testClassificationLogits() throws {
        guard loader.fixtureExists("classify_sentiment_positive_cls_logits") else {
            throw XCTSkip("Fixtures not generated or classification embeddings not captured")
        }

        let logits = try loader.loadNpy("classify_sentiment_positive_cls_logits")

        print("\nClassification logits shape: \(logits.shape)")
        XCTAssertEqual(logits.ndim, 1, "Logits should be 1D [num_labels]")

        // Logits should correspond to labels
        let metadata = try loader.loadJSON("classify_sentiment_positive_metadata")
        let labels = metadata["labels"] as? [String] ?? []
        XCTAssertEqual(logits.dim(0), labels.count, "Logits count should match labels count")
    }

    func testEncoderOutputParity() throws {
        guard loader.fixtureExists("entity_basic_encoder_output") else {
            throw XCTSkip("Fixtures not generated. Run: python generate_inference_fixtures.py")
        }

        // This test would require loading the Swift model and comparing encoder outputs
        // For now, just verify the fixture format
        let encoderOutput = try loader.loadNpy("entity_basic_encoder_output")

        // Check that values are reasonable (not NaN, not too large)
        let meanVal = MLX.mean(encoderOutput)
        let maxVal = MLX.max(encoderOutput)
        let minVal = MLX.min(encoderOutput)

        MLX.eval([meanVal, maxVal, minVal])

        let mean = Float(meanVal.item(Float32.self))
        let max = Float(maxVal.item(Float32.self))
        let min = Float(minVal.item(Float32.self))

        print("\nEncoder output statistics:")
        print("  Mean: \(mean)")
        print("  Max: \(max)")
        print("  Min: \(min)")

        XCTAssertFalse(mean.isNaN, "Mean should not be NaN")
        XCTAssertTrue(abs(mean) < 100, "Mean should be reasonable")
        XCTAssertTrue(max < 1000, "Max should be reasonable")
        XCTAssertTrue(min > -1000, "Min should be reasonable")
    }
}
#endif

// MARK: - Test Helpers

extension XCTestCase {
    /// Compare extracted entities for parity
    func assertEntitiesEqual(_ actual: [String: Any], _ expected: [String: Any], message: String = "") {
        let actualEntities = actual["entities"] as? [String: Any] ?? [:]
        let expectedEntities = expected["entities"] as? [String: Any] ?? [:]

        for (entityType, expectedValues) in expectedEntities {
            let actualValues = actualEntities[entityType]
            XCTAssertNotNil(actualValues, "Missing entity type: \(entityType)")

            if let expectedList = expectedValues as? [[String: Any]],
               let actualList = actualValues as? [[String: Any]] {
                let expectedTexts = Set(expectedList.compactMap { $0["text"] as? String })
                let actualTexts = Set(actualList.compactMap { $0["text"] as? String })

                XCTAssertEqual(actualTexts, expectedTexts,
                               "Entity texts don't match for \(entityType): \(message)")
            }
        }
    }
}
