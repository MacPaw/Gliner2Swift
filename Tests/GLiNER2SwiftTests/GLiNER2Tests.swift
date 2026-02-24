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
// GLiNER2Tests.swift
// Unit tests for GLiNER2 Swift implementation

import XCTest
import MLX
@testable import GLiNER2Swift

final class GLiNER2Tests: XCTestCase {

    // MARK: - Configuration Tests

    func testExtractorConfigDefaults() throws {
        let config = ExtractorConfig()

        XCTAssertEqual(config.modelName, "microsoft/deberta-v3-base")
        XCTAssertEqual(config.maxWidth, 8)
        XCTAssertEqual(config.countingLayer, .countLSTMv2)
        XCTAssertEqual(config.tokenPooling, .first)
        XCTAssertEqual(config.hiddenSize, 768)
        XCTAssertEqual(config.maxCount, 20)
    }

    func testExtractorConfigDerivedProperties() throws {
        let config = ExtractorConfig()

        XCTAssertEqual(config.downscaledHiddenSize, 128)
        XCTAssertEqual(config.downscaledNumHeads, 4)
        XCTAssertEqual(config.downscaledNumLayers, 2)
        XCTAssertEqual(config.classifierIntermediateDim, 1536)  // 768 * 2
    }

    // MARK: - Tokenizer Tests

    func testWhitespaceTokenSplitter() throws {
        let splitter = WhitespaceTokenSplitter()

        // Basic tokenization
        let tokens = splitter.tokenize("Hello World!", lower: true)
        XCTAssertEqual(tokens.count, 3)
        XCTAssertEqual(tokens[0].text, "hello")
        XCTAssertEqual(tokens[1].text, "world")
        XCTAssertEqual(tokens[2].text, "!")

        // Character positions
        XCTAssertEqual(tokens[0].start, 0)
        XCTAssertEqual(tokens[0].end, 5)
        XCTAssertEqual(tokens[1].start, 6)
        XCTAssertEqual(tokens[1].end, 11)
    }

    func testWhitespaceTokenSplitterURLs() throws {
        let splitter = WhitespaceTokenSplitter()

        let tokens = splitter.tokenize("Visit https://example.com today", lower: false)
        XCTAssertEqual(tokens.count, 3)
        XCTAssertEqual(tokens[1].text, "https://example.com")
    }

    func testWhitespaceTokenSplitterEmails() throws {
        let splitter = WhitespaceTokenSplitter()

        let tokens = splitter.tokenize("Contact test@example.com for info", lower: true)
        XCTAssertEqual(tokens.count, 4)
        XCTAssertEqual(tokens[1].text, "test@example.com")
    }

    func testWhitespaceTokenSplitterHandles() throws {
        let splitter = WhitespaceTokenSplitter()

        let tokens = splitter.tokenize("Follow @user_name on Twitter", lower: true)
        XCTAssertTrue(tokens.contains { $0.text == "@user_name" })
    }

    // MARK: - Span Position Tests

    func testSpanPositionValidity() throws {
        let validSpan = SpanPosition(start: 0, end: 5)
        let invalidSpan = SpanPosition.invalid

        XCTAssertTrue(validSpan.isValid)
        XCTAssertFalse(invalidSpan.isValid)
    }

    // MARK: - Regex Validator Tests

    func testRegexValidatorFullMatch() throws {
        let validator = try RegexValidator(pattern: #"^\d+$"#, mode: .full)

        XCTAssertTrue(validator.validate("123"))
        XCTAssertFalse(validator.validate("abc"))
        XCTAssertFalse(validator.validate("123abc"))
    }

    func testRegexValidatorPartialMatch() throws {
        let validator = try RegexValidator(pattern: "test", mode: .partial)

        XCTAssertTrue(validator.validate("this is a test"))
        XCTAssertTrue(validator.validate("testing"))
        XCTAssertFalse(validator.validate("hello world"))
    }

    func testRegexValidatorExclude() throws {
        let validator = try RegexValidator(pattern: "test", mode: .partial, exclude: true)

        XCTAssertFalse(validator.validate("this is a test"))
        XCTAssertTrue(validator.validate("hello world"))
    }

    // MARK: - Schema Builder Tests

    func testSchemaBuilderEntities() throws {
        let schema = Schema()
            .entities(["person", "organization"])

        let built = schema.build()
        let entities = built["entities"] as? [String: Any]

        XCTAssertNotNil(entities)
        XCTAssertTrue(entities?.keys.contains("person") ?? false)
        XCTAssertTrue(entities?.keys.contains("organization") ?? false)
    }

    func testSchemaBuilderClassification() throws {
        let schema = Schema()
            .classification(task: "sentiment", labels: ["positive", "negative"], multiLabel: false)

        let built = schema.build()
        let classifications = built["classifications"] as? [[String: Any]]

        XCTAssertNotNil(classifications)
        XCTAssertEqual(classifications?.count, 1)
        XCTAssertEqual(classifications?[0]["task"] as? String, "sentiment")
    }

    func testSchemaBuilderChaining() throws {
        let schema = Schema()
            .entities(["person", "company"])
            .classification(task: "sentiment", labels: ["positive", "negative"])

        let built = schema.build()

        XCTAssertNotNil(built["entities"])
        XCTAssertNotNil(built["classifications"])
    }

    func testSchemaBuilderEntitiesWithDescriptions() throws {
        let schema = Schema()
            .entities([
                "person": "A human being's name",
                "company": "A business organization"
            ])

        let built = schema.build()
        let entities = built["entities"] as? [String: Any]
        let descriptions = built["entity_descriptions"] as? [String: String]

        XCTAssertNotNil(entities)
        XCTAssertTrue(entities?.keys.contains("person") ?? false)
        XCTAssertTrue(entities?.keys.contains("company") ?? false)

        XCTAssertNotNil(descriptions)
        XCTAssertEqual(descriptions?["person"], "A human being's name")
        XCTAssertEqual(descriptions?["company"], "A business organization")
    }

    func testSchemaBuilderEntitiesWithDescriptionsChaining() throws {
        let schema = Schema()
            .entities([
                "person": "A human being's name",
                "company": "A business organization"
            ])
            .classification(task: "sentiment", labels: ["positive", "negative"])

        let built = schema.build()

        XCTAssertNotNil(built["entities"])
        XCTAssertNotNil(built["entity_descriptions"])
        XCTAssertNotNil(built["classifications"])
    }
}

// MARK: - MLP Tests

final class MLPTests: XCTestCase {

    func testMLPCreation() throws {
        let mlp = createMLP(
            inputDim: 768,
            intermediateDims: [1536],
            outputDim: 1,
            dropout: 0.0,
            activation: .relu
        )

        // Test forward pass
        let input = MLXArray.ones([1, 768])
        let output = mlp(input)

        XCTAssertEqual(output.dim(0), 1)
        XCTAssertEqual(output.dim(1), 1)
    }

    func testMLPWithLayerNorm() throws {
        let mlp = createMLP(
            inputDim: 768,
            intermediateDims: [1536],
            outputDim: 768,
            dropout: 0.0,
            activation: .gelu,
            addLayerNorm: true
        )

        let input = MLXArray.ones([2, 768])
        let output = mlp(input)

        XCTAssertEqual(output.dim(0), 2)
        XCTAssertEqual(output.dim(1), 768)
    }
}

// MARK: - GRU Tests

final class GRUTests: XCTestCase {

    func testGRUForwardPass() throws {
        let gru = GRU(inputSize: 768, hiddenSize: 768)

        // Input: [seq_len, batch, input_size]
        let input = MLXArray.ones([5, 2, 768]) * 0.1
        let h0 = MLXArray.zeros([1, 2, 768])

        let (output, hN) = gru(input, h0: h0)

        XCTAssertEqual(output.dim(0), 5)  // seq_len
        XCTAssertEqual(output.dim(1), 2)  // batch
        XCTAssertEqual(output.dim(2), 768)  // hidden

        XCTAssertEqual(hN.dim(0), 1)
        XCTAssertEqual(hN.dim(1), 2)
        XCTAssertEqual(hN.dim(2), 768)
    }

    func testGRUWithoutInitialHidden() throws {
        let gru = GRU(inputSize: 768, hiddenSize: 768)

        let input = MLXArray.ones([3, 4, 768]) * 0.1
        let (output, hN) = gru(input, h0: nil)

        XCTAssertEqual(output.dim(0), 3)
        XCTAssertEqual(output.dim(1), 4)
        XCTAssertEqual(output.dim(2), 768)
    }
}

// MARK: - CountLSTMv2 Tests

final class CountLSTMv2Tests: XCTestCase {

    func testCountLSTMv2ForwardPass() throws {
        let countLSTM = CountLSTMv2(hiddenSize: 768, maxCount: 20)

        // Field embeddings: [M, hidden]
        let pcEmb = MLXArray.ones([4, 768]) * 0.1

        let output = countLSTM(pcEmb, goldCountVal: 3)

        // Output should be [count, M, hidden]
        XCTAssertEqual(output.dim(0), 3)  // count
        XCTAssertEqual(output.dim(1), 4)  // M (fields)
        XCTAssertEqual(output.dim(2), 768)  // hidden
    }

    func testCountLSTMv2ZeroCount() throws {
        let countLSTM = CountLSTMv2(hiddenSize: 768, maxCount: 20)
        let pcEmb = MLXArray.ones([4, 768]) * 0.1

        let output = countLSTM(pcEmb, goldCountVal: 0)

        XCTAssertEqual(output.dim(0), 0)
    }

    func testCountLSTMv2MaxCountCap() throws {
        let countLSTM = CountLSTMv2(hiddenSize: 768, maxCount: 20)
        let pcEmb = MLXArray.ones([4, 768]) * 0.1

        let output = countLSTM(pcEmb, goldCountVal: 100)

        // Should be capped at maxCount
        XCTAssertEqual(output.dim(0), 20)
    }
}

// MARK: - Span Decoder Tests

final class SpanDecoderTests: XCTestCase {

    func testSpanDecoderFormatSpansNoOverlap() throws {
        let decoder = SpanDecoder(maxWidth: 8)

        let spans = [
            ExtractedSpan(text: "Apple", confidence: 0.95, charStart: 0, charEnd: 5),
            ExtractedSpan(text: "Inc", confidence: 0.85, charStart: 6, charEnd: 9),
        ]

        let formatted = decoder.formatSpans(spans, includeConfidence: false, includeSpans: false)

        XCTAssertEqual(formatted.count, 2)
        XCTAssertEqual(formatted[0] as? String, "Apple")
        XCTAssertEqual(formatted[1] as? String, "Inc")
    }

    func testSpanDecoderFormatSpansWithOverlap() throws {
        let decoder = SpanDecoder(maxWidth: 8)

        // Overlapping spans - should keep higher confidence
        let spans = [
            ExtractedSpan(text: "Apple Inc", confidence: 0.9, charStart: 0, charEnd: 9),
            ExtractedSpan(text: "Apple", confidence: 0.95, charStart: 0, charEnd: 5),
        ]

        let formatted = decoder.formatSpans(spans, includeConfidence: false, includeSpans: false)

        // Should only keep "Apple" (higher confidence)
        XCTAssertEqual(formatted.count, 1)
        XCTAssertEqual(formatted[0] as? String, "Apple")
    }

    func testSpanDecoderFormatSpansWithConfidence() throws {
        let decoder = SpanDecoder(maxWidth: 8)

        let spans = [
            ExtractedSpan(text: "Apple", confidence: 0.95, charStart: 0, charEnd: 5),
        ]

        let formatted = decoder.formatSpans(spans, includeConfidence: true, includeSpans: false)

        XCTAssertEqual(formatted.count, 1)
        let first = formatted[0] as? [String: Any]
        XCTAssertEqual(first?["text"] as? String, "Apple")
        XCTAssertEqual(first?["confidence"] as? Float, 0.95)
    }

    func testSpanDecoderFormatSpansWithPositions() throws {
        let decoder = SpanDecoder(maxWidth: 8)

        let spans = [
            ExtractedSpan(text: "Apple", confidence: 0.95, charStart: 0, charEnd: 5),
        ]

        let formatted = decoder.formatSpans(spans, includeConfidence: false, includeSpans: true)

        XCTAssertEqual(formatted.count, 1)
        let first = formatted[0] as? [String: Any]
        XCTAssertEqual(first?["text"] as? String, "Apple")
        XCTAssertEqual(first?["start"] as? Int, 0)
        XCTAssertEqual(first?["end"] as? Int, 5)
    }
}

// MARK: - Structure Field Descriptions Tests

final class StructureDescriptionsTests: XCTestCase {

    func testStructureBuilderWithFieldDescriptions() throws {
        let schema = Schema()
            .structure("person")
            .field("name", description: "The person's full name")
            .field("age", description: "The person's age in years")
            .done()

        let built = schema.build()

        // Check structure was added
        let structures = built["json_structures"] as? [[String: Any]]
        XCTAssertNotNil(structures)
        XCTAssertEqual(structures?.count, 1)

        // Check descriptions were stored
        let jsonDescriptions = built["json_descriptions"] as? [String: [String: String]]
        XCTAssertNotNil(jsonDescriptions)
        XCTAssertEqual(jsonDescriptions?["person"]?["name"], "The person's full name")
        XCTAssertEqual(jsonDescriptions?["person"]?["age"], "The person's age in years")
    }

    func testStructureBuilderWithoutDescriptions() throws {
        let schema = Schema()
            .structure("person")
            .field("name")
            .field("age")
            .done()

        let built = schema.build()

        // Check structure was added
        let structures = built["json_structures"] as? [[String: Any]]
        XCTAssertNotNil(structures)

        // Check no descriptions were stored (empty or nil)
        let jsonDescriptions = built["json_descriptions"] as? [String: [String: String]]
        XCTAssertTrue(jsonDescriptions == nil || jsonDescriptions?.isEmpty == true)
    }

    func testStructureBuilderMixedDescriptions() throws {
        // Some fields have descriptions, some don't
        let schema = Schema()
            .structure("person")
            .field("name", description: "The person's full name")
            .field("age")  // No description
            .field("email", description: "Contact email address")
            .done()

        let built = schema.build()

        let jsonDescriptions = built["json_descriptions"] as? [String: [String: String]]
        XCTAssertNotNil(jsonDescriptions)
        XCTAssertEqual(jsonDescriptions?["person"]?["name"], "The person's full name")
        XCTAssertEqual(jsonDescriptions?["person"]?["email"], "Contact email address")
        XCTAssertNil(jsonDescriptions?["person"]?["age"])
    }

    func testStructureTokensIncludeDescriptions() throws {
        // Skip if tokenizer not available
        let tokenizerPath = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("gliner2-base-v1")
        guard FileManager.default.fileExists(atPath: tokenizerPath.path) else {
            throw XCTSkip("Tokenizer not available at \(tokenizerPath.path)")
        }

        let processor = try SchemaTransformer.createFromLocalDirectory(directoryUrl: tokenizerPath)

        // Build schema with field descriptions
        let schema = Schema()
            .structure("person")
            .field("name", description: "The person's full name")
            .field("age", description: "The person's age")
            .done()
        let schemaDict = schema.build()

        // Transform text
        let record = processor.transform(text: "John is 30 years old.", schema: schemaDict)

        // Verify schema tokens include descriptions
        let schemaTokensList = record.schemaTokensList
        XCTAssertEqual(schemaTokensList.count, 1, "Should have exactly 1 schema (structure)")

        let structureSchemaTokens = schemaTokensList[0]
        let joined = structureSchemaTokens.joined(separator: " ")

        // Check that [DESCRIPTION] token is present
        XCTAssertTrue(joined.contains("[DESCRIPTION]"), "Schema tokens should contain [DESCRIPTION] token")

        // Check that descriptions are included
        XCTAssertTrue(joined.contains("name: The person's full name") || joined.contains("age: The person's age"),
                      "Schema tokens should contain at least one field description")
    }

    func testStructureTokensWithoutDescriptions() throws {
        // Skip if tokenizer not available
        let tokenizerPath = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("gliner2-base-v1")
        guard FileManager.default.fileExists(atPath: tokenizerPath.path) else {
            throw XCTSkip("Tokenizer not available at \(tokenizerPath.path)")
        }

        let processor = try SchemaTransformer.createFromLocalDirectory(directoryUrl: tokenizerPath)

        // Build schema without field descriptions
        let schema = Schema()
            .structure("person")
            .field("name")
            .field("age")
            .done()
        let schemaDict = schema.build()

        // Transform text
        let record = processor.transform(text: "John is 30 years old.", schema: schemaDict)

        // Verify schema tokens do NOT include descriptions
        let schemaTokensList = record.schemaTokensList
        let structureSchemaTokens = schemaTokensList[0]
        let joined = structureSchemaTokens.joined(separator: " ")

        // Check that [DESCRIPTION] token is NOT present
        XCTAssertFalse(joined.contains("[DESCRIPTION]"), "Schema tokens should NOT contain [DESCRIPTION] token when no descriptions provided")
    }
}

// MARK: - Entity Descriptions Tests

final class EntityDescriptionsTests: XCTestCase {

    func testSchemaTokensIncludeDescriptions() throws {
        // Skip if tokenizer not available
        let tokenizerPath = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("gliner2-base-v1")
        guard FileManager.default.fileExists(atPath: tokenizerPath.path) else {
            throw XCTSkip("Tokenizer not available at \(tokenizerPath.path)")
        }

        let processor = try SchemaTransformer.createFromLocalDirectory(directoryUrl: tokenizerPath)

        // Build schema with descriptions
        let schema = Schema()
            .entities([
                "person": "A human being's name",
                "company": "A business organization"
            ])
        let schemaDict = schema.build()

        // Transform text
        let record = processor.transform(text: "Tim Cook is CEO of Apple.", schema: schemaDict)

        // Verify schema tokens include descriptions
        let schemaTokensList = record.schemaTokensList
        XCTAssertEqual(schemaTokensList.count, 1, "Should have exactly 1 schema (entities)")

        let entitySchemaTokens = schemaTokensList[0]
        let joined = entitySchemaTokens.joined(separator: " ")

        // Check that [DESCRIPTION] token is present
        XCTAssertTrue(joined.contains("[DESCRIPTION]"), "Schema tokens should contain [DESCRIPTION] token")

        // Check that descriptions are included
        XCTAssertTrue(joined.contains("person: A human being's name") || joined.contains("company: A business organization"),
                      "Schema tokens should contain at least one description")
    }

    func testSchemaTokensWithoutDescriptions() throws {
        // Skip if tokenizer not available
        let tokenizerPath = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("gliner2-base-v1")
        guard FileManager.default.fileExists(atPath: tokenizerPath.path) else {
            throw XCTSkip("Tokenizer not available at \(tokenizerPath.path)")
        }

        let processor = try SchemaTransformer.createFromLocalDirectory(directoryUrl: tokenizerPath)

        // Build schema without descriptions (using simple array)
        let schema = Schema()
            .entities(["person", "company"])
        let schemaDict = schema.build()

        // Transform text
        let record = processor.transform(text: "Tim Cook is CEO of Apple.", schema: schemaDict)

        // Verify schema tokens do NOT include descriptions
        let schemaTokensList = record.schemaTokensList
        XCTAssertEqual(schemaTokensList.count, 1, "Should have exactly 1 schema (entities)")

        let entitySchemaTokens = schemaTokensList[0]
        let joined = entitySchemaTokens.joined(separator: " ")

        // Check that [DESCRIPTION] token is NOT present
        XCTAssertFalse(joined.contains("[DESCRIPTION]"), "Schema tokens should NOT contain [DESCRIPTION] token when no descriptions provided")
    }
}
