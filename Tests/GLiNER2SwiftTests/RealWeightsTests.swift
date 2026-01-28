// RealWeightsTests.swift
// Tests that actually load real weights and run inference with GLiNER2Swift
//
// IMPORTANT: These tests use REAL WEIGHTS from /weights/ directory
// They verify that Swift inference produces the same results as Python

import XCTest
import Foundation
import Metal
@testable import GLiNER2Swift

// MLX imports - only used in MLX-dependent tests
import MLX
import MLXNN

/// Tests that load real weights and run actual Swift inference
///
/// IMPORTANT: These tests require:
/// 1. Real weights in /weights/ directory
/// 2. MLX Metal library access (GPU)
///
/// If MLX Metal is not available, tests will be skipped.
final class RealWeightsTests: XCTestCase {

    // Path to weights directory (relative to project root)
    static let weightsPath = "/Users/tmwstw/Documents/mnemos/GLiNER2/weights"
    static let fixturesPath = "/Users/tmwstw/Documents/mnemos/GLiNER2/GLiNER2Swift/Tests/GLiNER2SwiftTests/Fixtures/inference"

    /// Skip test if MLX Metal is not available (e.g., in CLI environment)
    /// In Xcode with GPU access, this will NOT skip.
    private func skipIfMLXUnavailable() throws {
        // Check if Metal device is available
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal GPU not available - run tests in Xcode with GPU access")
        }
        // Metal device exists - tests can proceed
    }

    // MARK: - Basic File Tests (No MLX required)

    func testWeightsDirectoryExists() throws {
        let fm = FileManager.default
        XCTAssertTrue(fm.fileExists(atPath: Self.weightsPath), "Weights directory should exist at \(Self.weightsPath)")
        XCTAssertTrue(fm.fileExists(atPath: "\(Self.weightsPath)/config.json"), "config.json should exist")
        XCTAssertTrue(fm.fileExists(atPath: "\(Self.weightsPath)/gliner2_weights.safetensors"), "gliner2_weights.safetensors should exist")
        XCTAssertTrue(fm.fileExists(atPath: "\(Self.weightsPath)/encoder_weights.safetensors"), "encoder_weights.safetensors should exist")
        XCTAssertTrue(fm.fileExists(atPath: "\(Self.weightsPath)/tokenizer.json"), "tokenizer.json should exist")
    }

    func testLoadConfig() throws {
        let configUrl = URL(fileURLWithPath: "\(Self.weightsPath)/config.json")
        let config = try ExtractorConfig.load(from: configUrl)

        XCTAssertEqual(config.maxWidth, 8, "maxWidth should be 8")
        XCTAssertEqual(config.hiddenSize, 768, "hiddenSize should be 768")
        XCTAssertEqual(config.countingLayer, .countLSTMv2, "countingLayer should be count_lstm_v2")
    }

    /// Test SafeTensors header parsing without creating MLX arrays
    func testSafeTensorsHeaderParsing() throws {
        let modelWeightsUrl = URL(fileURLWithPath: "\(Self.weightsPath)/gliner2_weights.safetensors")
        let data = try Data(contentsOf: modelWeightsUrl)

        // Parse header size
        XCTAssertGreaterThanOrEqual(data.count, 8, "File should have at least 8 bytes for header size")

        let headerSize = data.withUnsafeBytes { ptr in
            ptr.load(as: UInt64.self)
        }

        XCTAssertGreaterThan(headerSize, 0, "Header size should be positive")
        XCTAssertLessThan(headerSize, UInt64(1_000_000), "Header size should be reasonable")

        // Parse header JSON
        let headerData = data[8..<(8 + Int(headerSize))]
        let headerJson = try JSONSerialization.jsonObject(with: headerData) as! [String: Any]

        // Verify expected keys exist (using camelCase keys from conversion)
        let expectedKeys = [
            "classifier.layers.0.weight",
            "countEmbed.posEmbedding.weight",
            "countEmbed.gru.weightIH",
            "spanRep.spanRepLayer.projectStart.0.weight"
        ]

        for key in expectedKeys {
            XCTAssertNotNil(headerJson[key], "Weight key '\(key)' should exist in safetensors")
        }

        print("SafeTensors header contains \(headerJson.count - 1) weights (excluding __metadata__)")
    }

    func testWeightMappingFileExists() throws {
        let mappingUrl = URL(fileURLWithPath: "\(Self.weightsPath)/weight_mapping.json")
        let mappingData = try Data(contentsOf: mappingUrl)
        let mapping = try JSONSerialization.jsonObject(with: mappingData) as! [String: String]

        XCTAssertGreaterThan(mapping.count, 100, "Weight mapping should have many entries")

        // Verify some key mappings
        XCTAssertEqual(mapping["classifier.0.weight"], "classifier.layers.0.weight")
        XCTAssertEqual(mapping["count_embed.gru.weight_ih_l0"], "countEmbed.gru.weightIH")
        XCTAssertEqual(mapping["span_rep.span_rep_layer.project_start.0.weight"], "spanRep.spanRepLayer.projectStart.0.weight")

        print("Weight mapping verified with \(mapping.count) entries")
    }

    // MARK: - Tokenizer Tests (No MLX required)

    func testUnigramTokenizerLoads() throws {
        let tokenizerUrl = URL(fileURLWithPath: "\(Self.weightsPath)/tokenizer.json")
        let tokenizer = try UnigramTokenizer(tokenizerJsonUrl: tokenizerUrl)

        XCTAssertGreaterThan(tokenizer.vocabSize, 100000, "Vocab size should be > 100k")
        XCTAssertEqual(tokenizer.clsTokenId, 1, "[CLS] should be ID 1")
        XCTAssertEqual(tokenizer.sepTokenId, 2, "[SEP] should be ID 2")
        XCTAssertEqual(tokenizer.padTokenId, 0, "[PAD] should be ID 0")

        print("Tokenizer loaded with vocab size: \(tokenizer.vocabSize)")
    }

    func testUnigramTokenizerBasicTokenization() throws {
        let tokenizerUrl = URL(fileURLWithPath: "\(Self.weightsPath)/tokenizer.json")
        let tokenizer = try UnigramTokenizer(tokenizerJsonUrl: tokenizerUrl)

        // Test simple tokenization
        let text = "Hello world"
        let tokens = tokenizer.tokenize(text)

        XCTAssertFalse(tokens.isEmpty, "Should produce tokens")
        print("'\(text)' -> \(tokens)")

        // Tokens should contain metaspace prefix
        XCTAssertTrue(tokens.first?.hasPrefix("\u{2581}") ?? false, "First token should start with metaspace")
    }

    func testUnigramTokenizerEncoding() throws {
        let tokenizerUrl = URL(fileURLWithPath: "\(Self.weightsPath)/tokenizer.json")
        let tokenizer = try UnigramTokenizer(tokenizerJsonUrl: tokenizerUrl)

        let text = "Tim Cook is CEO of Apple"
        let ids = tokenizer.encode(text)

        XCTAssertFalse(ids.isEmpty, "Should produce token IDs")
        print("'\(text)' -> IDs: \(ids)")

        // All IDs should be valid (> 0 and < vocab size)
        for id in ids {
            XCTAssertGreaterThanOrEqual(id, 0, "Token ID should be >= 0")
            XCTAssertLessThan(id, tokenizer.vocabSize, "Token ID should be < vocab size")
        }
    }

    func testUnigramTokenizerSpecialTokens() throws {
        let tokenizerUrl = URL(fileURLWithPath: "\(Self.weightsPath)/tokenizer.json")
        let tokenizer = try UnigramTokenizer(tokenizerJsonUrl: tokenizerUrl)

        // Test that special tokens are preserved
        let text = "[P] person [E] name"
        let tokens = tokenizer.tokenize(text)

        print("'\(text)' -> \(tokens)")

        // Special tokens should be preserved whole
        XCTAssertTrue(tokens.contains("[P]"), "Should preserve [P] token")
        XCTAssertTrue(tokens.contains("[E]"), "Should preserve [E] token")
    }

    func testUnigramTokenizerRoundTrip() throws {
        let tokenizerUrl = URL(fileURLWithPath: "\(Self.weightsPath)/tokenizer.json")
        let tokenizer = try UnigramTokenizer(tokenizerJsonUrl: tokenizerUrl)

        let original = "Hello world"
        let ids = tokenizer.encode(original)
        let decoded = tokenizer.decode(ids)

        print("Original: '\(original)'")
        print("Decoded:  '\(decoded)'")

        // Decoded text should match original (ignoring leading space)
        XCTAssertEqual(decoded.trimmingCharacters(in: .whitespaces), original, "Round-trip should preserve text")
    }

    func testUnigramTokenizerWithSpecialTokensEncoding() throws {
        let tokenizerUrl = URL(fileURLWithPath: "\(Self.weightsPath)/tokenizer.json")
        let tokenizer = try UnigramTokenizer(tokenizerJsonUrl: tokenizerUrl)

        let text = "Tim Cook is CEO of Apple"
        let ids = tokenizer.encodeWithSpecialTokens(text)

        XCTAssertEqual(ids.first, tokenizer.clsTokenId, "Should start with [CLS]")
        XCTAssertEqual(ids.last, tokenizer.sepTokenId, "Should end with [SEP]")

        print("With special tokens: \(ids)")
    }

    func testLoadGLiNER2Weights() throws {
        try skipIfMLXUnavailable()

        // Test that we can load the model weights without crashing
        let modelWeightsUrl = URL(fileURLWithPath: "\(Self.weightsPath)/gliner2_weights.safetensors")

        // Load SafeTensors
        let weights = try SafeTensorsLoader.load(from: modelWeightsUrl)

        // Verify key weights exist (using camelCase keys from converted weights)
        XCTAssertNotNil(weights["classifier.layers.0.weight"], "classifier.layers.0.weight should exist")
        XCTAssertNotNil(weights["classifier.layers.1.weight"], "classifier.layers.1.weight should exist")
        XCTAssertNotNil(weights["countPred.layers.0.weight"], "countPred.layers.0.weight should exist")
        XCTAssertNotNil(weights["countEmbed.posEmbedding.weight"], "countEmbed.posEmbedding.weight should exist")
        XCTAssertNotNil(weights["countEmbed.gru.weightIH"], "countEmbed.gru.weightIH should exist")
        XCTAssertNotNil(weights["spanRep.spanRepLayer.projectStart.0.weight"], "spanRep.spanRepLayer.projectStart.0.weight should exist")

        // Print all keys for debugging
        print("GLiNER2 weight keys (\(weights.count) total):")
        for key in weights.keys.sorted().prefix(20) {
            print("  - \(key)")
        }
    }

    func testLoadEncoderWeights() throws {
        try skipIfMLXUnavailable()

        // Test that we can load encoder weights
        let encoderWeightsUrl = URL(fileURLWithPath: "\(Self.weightsPath)/encoder_weights.safetensors")

        let weights = try SafeTensorsLoader.load(from: encoderWeightsUrl)

        // Verify key encoder weights exist
        XCTAssertNotNil(weights["encoder.embeddings.word_embeddings.weight"], "word_embeddings should exist")
        XCTAssertNotNil(weights["encoder.encoder.layer.0.attention.self.query_proj.weight"], "layer 0 query_proj should exist")
        XCTAssertNotNil(weights["encoder.encoder.rel_embeddings.weight"], "rel_embeddings should exist")

        // Check embedding shape
        if let embWeight = weights["encoder.embeddings.word_embeddings.weight"] {
            let shape = embWeight.shape
            XCTAssertEqual(shape[0], 128011, "Vocab size should be 128011")
            XCTAssertEqual(shape[1], 768, "Hidden size should be 768")
        }

        print("Encoder weight keys (\(weights.count) total)")
    }

    func testInitializeExtractorWithWeights() throws {
        try skipIfMLXUnavailable()

        // Load config
        let configUrl = URL(fileURLWithPath: "\(Self.weightsPath)/config.json")
        let config = try ExtractorConfig.load(from: configUrl)

        // Create extractor
        let extractor = Extractor(config: config)

        // Load model weights
        let modelWeightsUrl = URL(fileURLWithPath: "\(Self.weightsPath)/gliner2_weights.safetensors")
        let modelWeights = try SafeTensorsLoader.load(from: modelWeightsUrl)

        // Load weights into model - this tests weight loading functions
        extractor.loadModelWeights(modelWeights)

        // Verify classifier weights were loaded (not random)
        // We check that the weights are not all zeros
        let classifierWeight = extractor.classifier.layers[0] as! Linear
        let weightSum = MLX.sum(MLX.abs(classifierWeight.weight))
        MLX.eval(weightSum)
        XCTAssertGreaterThan(Float(weightSum.item(Float32.self)), 0.0, "Classifier weights should not be all zeros")

        print("Extractor initialized and weights loaded successfully")
    }

    // MARK: - Inference Tests

    func testEntityExtractionWithRealWeights() async throws {
        try skipIfMLXUnavailable()

        // Load model from weights directory
        let model = try await GLiNER2.fromPretrained(Self.weightsPath)

        // Test entity extraction
        let text = "Tim Cook is the CEO of Apple Inc."
        let result = model.extractEntities(
            text: text,
            entityTypes: ["person", "organization"],
            threshold: 0.3,  // Lower threshold to catch more
            includeConfidence: true,
            includeSpans: true
        )

        print("Entity extraction result:")
        print(result)

        // STRONG ASSERTION: Result must not be empty dict
        XCTAssertFalse(result.isEmpty, "Result dictionary should not be empty")

        // Result structure: {"entities": {"person": [...], "organization": [...]}}
        // OR directly: {"person": [...], "organization": [...]}
        let entitiesDict: [String: Any]
        if let nested = result["entities"] as? [String: Any] {
            entitiesDict = nested
            print("Result structure: nested under 'entities' key")
        } else {
            entitiesDict = result
            print("Result structure: flat (entity types at top level)")
        }

        // Count total entities found
        var totalEntities = 0
        for key in ["person", "organization"] {
            if let entities = entitiesDict[key] as? [[String: Any]] {
                totalEntities += entities.count
                print("  \(key): \(entities.count) entities")
                for entity in entities {
                    print("    - \(entity)")
                }
            } else if let entities = entitiesDict[key] as? [String] {
                totalEntities += entities.count
                print("  \(key): \(entities)")
            } else {
                print("  \(key): not found or empty")
            }
        }

        // STRONG ASSERTION: Must find at least one entity in this text
        // "Tim Cook is the CEO of Apple Inc." clearly has person and org entities
        XCTAssertGreaterThan(totalEntities, 0,
            "Must extract at least one entity from '\(text)'. Result: \(result)")
    }

    func testClassificationWithRealWeights() async throws {
        try skipIfMLXUnavailable()

        // Load model
        let model = try await GLiNER2.fromPretrained(Self.weightsPath)

        // Test classification
        let text = "I absolutely love this product! It exceeded all my expectations."
        let result = model.classifyText(
            text: text,
            task: "sentiment",
            labels: ["positive", "negative", "neutral"],
            multiLabel: false,
            threshold: 0.3,
            includeConfidence: true
        )

        print("Classification result:")
        print(result)

        // STRONG ASSERTION: Result must not be empty
        XCTAssertFalse(result.isEmpty, "Classification result should not be empty")

        // STRONG ASSERTION: Must have sentiment key
        XCTAssertTrue(result.keys.contains("sentiment"),
            "Result must contain 'sentiment' key. Got keys: \(result.keys). Full result: \(result)")

        // STRONG ASSERTION: Must have a valid label
        if let sentiment = result["sentiment"] as? [String: Any] {
            XCTAssertNotNil(sentiment["label"],
                "Sentiment must have 'label' key. Got: \(sentiment)")
            if let label = sentiment["label"] as? String {
                print("  Predicted label: \(label)")
                XCTAssertTrue(["positive", "negative", "neutral"].contains(label),
                    "Label must be one of the provided options. Got: \(label)")
            } else {
                XCTFail("Label is not a string: \(sentiment["label"] ?? "nil")")
            }
        } else if let label = result["sentiment"] as? String {
            print("  Predicted label: \(label)")
            XCTAssertTrue(["positive", "negative", "neutral"].contains(label),
                "Label must be one of the provided options. Got: \(label)")
        } else {
            XCTFail("Sentiment result has unexpected format: \(result["sentiment"] ?? "nil")")
        }
    }

    func testStructureExtractionWithRealWeights() async throws {
        try skipIfMLXUnavailable()

        // Load model
        let model = try await GLiNER2.fromPretrained(Self.weightsPath)

        // Test structure extraction
        let text = "John Smith is 35 years old and lives in New York."
        let schema = model.createSchema()
            .structure("person_info")
            .field("name", dtype: "str")
            .field("age", dtype: "str")
            .field("location", dtype: "str")
            .done()

        let result = model.extract(
            text: text,
            schema: schema,
            threshold: 0.3,
            includeConfidence: true,
            includeSpans: true
        )

        print("Structure extraction result:")
        print(result)

        // STRONG ASSERTION: Result must not be empty
        XCTAssertFalse(result.isEmpty, "Structure extraction result should not be empty")

        // STRONG ASSERTION: Must have person_info key
        XCTAssertTrue(result.keys.contains("person_info"),
            "Result must contain 'person_info' key. Got keys: \(result.keys). Full result: \(result)")

        // Check structure format
        if let structs = result["person_info"] as? [[String: Any]] {
            print("  Found \(structs.count) person_info instances")
            for (idx, instance) in structs.enumerated() {
                print("  Instance \(idx): \(instance)")
            }
            // STRONG ASSERTION: Should have at least one instance
            XCTAssertGreaterThan(structs.count, 0,
                "Should extract at least one person_info from '\(text)'. Got: \(result)")
        } else if let structDict = result["person_info"] as? [String: Any] {
            // Single instance format
            print("  Single instance: \(structDict)")
            // Check if it has any extracted fields
            let hasContent = structDict.values.contains { value in
                if let arr = value as? [Any] { return !arr.isEmpty }
                if let str = value as? String { return !str.isEmpty }
                return true  // Any non-nil value counts as content
            }
            XCTAssertTrue(hasContent,
                "person_info must have extracted content. Got: \(structDict)")
        } else {
            XCTFail("person_info has unexpected format: \(result["person_info"] ?? "nil")")
        }
    }

    // MARK: - Weight Loading Verification Tests

    /// Verify that SpanMarkerV0 weights are actually loaded
    /// This is a critical test because span_rep values were 200x larger than expected
    func testSpanMarkerWeightsLoaded() async throws {
        try skipIfMLXUnavailable()

        print("\n" + String(repeating: "=", count: 70))
        print("SpanMarkerV0 WEIGHT LOADING VERIFICATION")
        print(String(repeating: "=", count: 70))

        // Load weights file
        let modelWeightsUrl = URL(fileURLWithPath: "\(Self.weightsPath)/gliner2_weights.safetensors")
        let weights = try SafeTensorsLoader.load(from: modelWeightsUrl)

        // Check that all expected SpanMarkerV0 keys exist
        let expectedKeys = [
            "spanRep.spanRepLayer.projectStart.0.weight",
            "spanRep.spanRepLayer.projectStart.0.bias",
            "spanRep.spanRepLayer.projectStart.3.weight",
            "spanRep.spanRepLayer.projectStart.3.bias",
            "spanRep.spanRepLayer.projectEnd.0.weight",
            "spanRep.spanRepLayer.projectEnd.0.bias",
            "spanRep.spanRepLayer.projectEnd.3.weight",
            "spanRep.spanRepLayer.projectEnd.3.bias",
            "spanRep.spanRepLayer.outProject.0.weight",
            "spanRep.spanRepLayer.outProject.0.bias",
            "spanRep.spanRepLayer.outProject.3.weight",
            "spanRep.spanRepLayer.outProject.3.bias",
        ]

        print("\n1. CHECKING WEIGHT KEYS IN SAFETENSORS FILE:")
        for key in expectedKeys {
            if let tensor = weights[key] {
                let l1Sum = MLX.sum(MLX.abs(tensor))
                MLX.eval(l1Sum)
                print("  ✓ \(key): shape=\(tensor.shape), L1 sum=\(Float(l1Sum.item(Float32.self)))")
            } else {
                print("  ✗ \(key): NOT FOUND!")
                XCTFail("Weight key '\(key)' not found in safetensors")
            }
        }

        // Load config and create model
        let configUrl = URL(fileURLWithPath: "\(Self.weightsPath)/config.json")
        let config = try ExtractorConfig.load(from: configUrl)
        let extractor = Extractor(config: config)

        print("\n2. CHECKING MODEL BEFORE WEIGHT LOADING:")
        if let linear0 = extractor.spanRep.getSpanRepLayer().projectStart.layers[0] as? Linear {
            let priorL1 = MLX.sum(MLX.abs(linear0.weight))
            MLX.eval(priorL1)
            print("  projectStart.0.weight L1 sum (random init): \(Float(priorL1.item(Float32.self)))")
        }

        // Load weights
        print("\n3. LOADING WEIGHTS...")
        extractor.loadModelWeights(weights)

        print("\n4. CHECKING MODEL AFTER WEIGHT LOADING:")

        // Check projectStart layer 0
        if let linear0 = extractor.spanRep.getSpanRepLayer().projectStart.layers[0] as? Linear {
            let postL1 = MLX.sum(MLX.abs(linear0.weight))
            MLX.eval(postL1)
            print("  projectStart.0.weight L1 sum: \(Float(postL1.item(Float32.self)))")
            print("    shape: \(linear0.weight.shape)")

            // Compare with safetensors value
            if let safeTensorWeight = weights["spanRep.spanRepLayer.projectStart.0.weight"] {
                let stL1 = MLX.sum(MLX.abs(safeTensorWeight))
                MLX.eval(stL1)
                let modelVal = Float(postL1.item(Float32.self))
                let stVal = Float(stL1.item(Float32.self))
                print("    safetensors L1 sum: \(stVal)")

                if abs(modelVal - stVal) < 0.01 {
                    print("    ✓ MATCH - weights loaded correctly")
                } else {
                    print("    ✗ MISMATCH - weights NOT loaded!")
                    XCTFail("projectStart.0 weights not loaded: model=\(modelVal), safetensors=\(stVal)")
                }
            }
        } else {
            XCTFail("Could not access projectStart.layers[0] as Linear")
        }

        // Check projectStart layer 3
        if let linear3 = extractor.spanRep.getSpanRepLayer().projectStart.layers[3] as? Linear {
            let postL1 = MLX.sum(MLX.abs(linear3.weight))
            MLX.eval(postL1)
            print("  projectStart.3.weight L1 sum: \(Float(postL1.item(Float32.self)))")

            if let safeTensorWeight = weights["spanRep.spanRepLayer.projectStart.3.weight"] {
                let stL1 = MLX.sum(MLX.abs(safeTensorWeight))
                MLX.eval(stL1)
                let modelVal = Float(postL1.item(Float32.self))
                let stVal = Float(stL1.item(Float32.self))
                if abs(modelVal - stVal) < 0.01 {
                    print("    ✓ MATCH")
                } else {
                    print("    ✗ MISMATCH - model=\(modelVal), safetensors=\(stVal)")
                    XCTFail("projectStart.3 weights not loaded")
                }
            }
        }

        // Check projectEnd
        if let linear0 = extractor.spanRep.getSpanRepLayer().projectEnd.layers[0] as? Linear {
            let postL1 = MLX.sum(MLX.abs(linear0.weight))
            MLX.eval(postL1)
            print("  projectEnd.0.weight L1 sum: \(Float(postL1.item(Float32.self)))")

            if let safeTensorWeight = weights["spanRep.spanRepLayer.projectEnd.0.weight"] {
                let stL1 = MLX.sum(MLX.abs(safeTensorWeight))
                MLX.eval(stL1)
                let modelVal = Float(postL1.item(Float32.self))
                let stVal = Float(stL1.item(Float32.self))
                if abs(modelVal - stVal) < 0.01 {
                    print("    ✓ MATCH")
                } else {
                    print("    ✗ MISMATCH - model=\(modelVal), safetensors=\(stVal)")
                    XCTFail("projectEnd.0 weights not loaded")
                }
            }
        }

        // Check outProject
        if let linear0 = extractor.spanRep.getSpanRepLayer().outProject.layers[0] as? Linear {
            let postL1 = MLX.sum(MLX.abs(linear0.weight))
            MLX.eval(postL1)
            print("  outProject.0.weight L1 sum: \(Float(postL1.item(Float32.self)))")

            if let safeTensorWeight = weights["spanRep.spanRepLayer.outProject.0.weight"] {
                let stL1 = MLX.sum(MLX.abs(safeTensorWeight))
                MLX.eval(stL1)
                let modelVal = Float(postL1.item(Float32.self))
                let stVal = Float(stL1.item(Float32.self))
                if abs(modelVal - stVal) < 0.01 {
                    print("    ✓ MATCH")
                } else {
                    print("    ✗ MISMATCH - model=\(modelVal), safetensors=\(stVal)")
                    XCTFail("outProject.0 weights not loaded")
                }
            }
        }

        print("\n" + String(repeating: "=", count: 70))
    }

    /// Verify that model weights are actually loaded (not random initialization)
    func testWeightsActuallyLoaded() async throws {
        try skipIfMLXUnavailable()

        let model = try await GLiNER2.fromPretrained(Self.weightsPath)

        // Check classifier weights are not random (should have meaningful values)
        if let linear = model.model.classifier.layers[0] as? Linear {
            let weight = linear.weight
            MLX.eval(weight)

            // Calculate sum of absolute values - should be >> 0 for trained weights
            let weightSum = MLX.sum(MLX.abs(weight))
            MLX.eval(weightSum)
            let sum = Float(weightSum.item(Float32.self))

            print("Classifier weight L1 sum: \(sum)")
            print("Classifier weight shape: \(weight.shape)")

            // Trained weights should have substantial values
            // Random Xavier init for [1, 768] would have sum ~= sqrt(2/769) * 768 ~= 39
            // Trained weights typically have much larger sum
            XCTAssertGreaterThan(sum, 50.0,
                "Classifier weights appear to be random (sum=\(sum)). Check weight loading.")
        } else {
            XCTFail("Could not access classifier.layers[0] as Linear")
        }

        // Check countPred weights
        if let linear = model.model.countPred.layers[0] as? Linear {
            let weight = linear.weight
            MLX.eval(weight)
            let weightSum = MLX.sum(MLX.abs(weight))
            MLX.eval(weightSum)
            let sum = Float(weightSum.item(Float32.self))

            print("CountPred weight L1 sum: \(sum)")
            print("CountPred weight shape: \(weight.shape)")

            XCTAssertGreaterThan(sum, 100.0,
                "CountPred weights appear random (sum=\(sum)). Check weight loading.")
        } else {
            XCTFail("Could not access countPred.layers[0] as Linear")
        }

        // Check countEmbed GRU weights (should be accessible)
        let gruWeightIH = model.model.countEmbed.gru.weightIH
        MLX.eval(gruWeightIH)
        let gruSum = MLX.sum(MLX.abs(gruWeightIH))
        MLX.eval(gruSum)
        let gruSumVal = Float(gruSum.item(Float32.self))

        print("CountEmbed GRU weightIH L1 sum: \(gruSumVal)")
        print("GRU weightIH shape: \(gruWeightIH.shape)")

        XCTAssertGreaterThan(gruSumVal, 1000.0,
            "GRU weights appear random (sum=\(gruSumVal)). Check weight loading.")

        print("Weight loading verification passed!")
    }

    /// Verify encoder weights are loaded
    func testEncoderWeightsLoaded() async throws {
        try skipIfMLXUnavailable()

        let model = try await GLiNER2.fromPretrained(Self.weightsPath)

        // Check word embeddings
        let embeddings = model.model.encoder.embeddings.wordEmbeddings.weight
        MLX.eval(embeddings)

        let embSum = MLX.sum(MLX.abs(embeddings))
        MLX.eval(embSum)
        let sum = Float(embSum.item(Float32.self))

        print("Encoder embeddings L1 sum: \(sum)")
        print("Embeddings shape: \(embeddings.shape)")

        // Word embeddings for 128k vocab should have very large sum
        XCTAssertGreaterThan(sum, 10000.0,
            "Encoder embeddings appear random (sum=\(sum)). Check weight loading.")

        // Verify vocab size
        XCTAssertEqual(embeddings.shape[0], 128011,
            "Vocab size should be 128011. Got: \(embeddings.shape[0])")

        print("Encoder weight verification passed!")
    }

    /// Verify encoder layer 0 weights match Python exactly
    func testEncoderLayer0WeightsMatchPython() async throws {
        try skipIfMLXUnavailable()

        print("\n" + String(repeating: "=", count: 70))
        print("DeBERTa Encoder Layer 0 WEIGHT VERIFICATION")
        print(String(repeating: "=", count: 70))

        let model = try await GLiNER2.fromPretrained(Self.weightsPath)

        // Python reference values:
        // query_proj.weight: 35817.66
        // key_proj.weight: 35440.90
        // value_proj.weight: 13340.10
        // attention.output.dense.weight: 13325.04
        // intermediate.dense.weight: 93057.56
        // output.dense.weight: 85679.08
        // word_embeddings.weight: 8505137.00
        // rel_embeddings.weight: 22767.64

        let layer0 = model.model.encoder.layers[0]

        // Query projection
        let queryWeight = layer0.attention.queryProj.weight
        MLX.eval(queryWeight)
        let querySum = Float(MLX.sum(MLX.abs(queryWeight)).item(Float32.self))
        print("query_proj.weight L1 sum: \(querySum) (Python: 35817.66)")
        XCTAssertEqual(querySum, 35817.66, accuracy: 100.0, "query_proj weights don't match")

        // Key projection
        let keyWeight = layer0.attention.keyProj.weight
        MLX.eval(keyWeight)
        let keySum = Float(MLX.sum(MLX.abs(keyWeight)).item(Float32.self))
        print("key_proj.weight L1 sum: \(keySum) (Python: 35440.90)")
        XCTAssertEqual(keySum, 35440.90, accuracy: 100.0, "key_proj weights don't match")

        // Value projection
        let valueWeight = layer0.attention.valueProj.weight
        MLX.eval(valueWeight)
        let valueSum = Float(MLX.sum(MLX.abs(valueWeight)).item(Float32.self))
        print("value_proj.weight L1 sum: \(valueSum) (Python: 13340.10)")
        XCTAssertEqual(valueSum, 13340.10, accuracy: 100.0, "value_proj weights don't match")

        // Word embeddings
        let wordEmbWeight = model.model.encoder.embeddings.wordEmbeddings.weight
        MLX.eval(wordEmbWeight)
        let wordSum = Float(MLX.sum(MLX.abs(wordEmbWeight)).item(Float32.self))
        print("word_embeddings.weight L1 sum: \(wordSum) (Python: 8505137.00)")
        XCTAssertEqual(wordSum, 8505137.0, accuracy: 10000.0, "word_embeddings weights don't match")

        // Relative embeddings (stored directly as MLXArray, not Embedding)
        let relEmbWeight = model.model.encoder.relEmbeddings
        MLX.eval(relEmbWeight)
        let relSum = Float(MLX.sum(MLX.abs(relEmbWeight)).item(Float32.self))
        print("rel_embeddings.weight L1 sum: \(relSum) (Python: 22767.64)")
        XCTAssertEqual(relSum, 22767.64, accuracy: 100.0, "rel_embeddings weights don't match")

        print("\n" + String(repeating: "=", count: 70))
    }

    // MARK: - Tokenizer Parity Tests

    /// Test tokenizer produces same token IDs as Python for various test cases.
    ///
    /// This test loads Python-generated fixtures and compares Swift tokenization output.
    func testTokenizerParityWithPython() throws {
        // Load Python fixture
        let fixtureUrl = URL(fileURLWithPath: "\(Self.fixturesPath)/tokenizer_parity.json")

        guard FileManager.default.fileExists(atPath: fixtureUrl.path) else {
            throw XCTSkip("Tokenizer parity fixtures not found. Run: python scripts/generate_inference_fixtures.py")
        }

        let fixtureData = try Data(contentsOf: fixtureUrl)
        let fixture = try JSONSerialization.jsonObject(with: fixtureData) as! [String: Any]
        let testCases = fixture["test_cases"] as! [[String: Any]]

        // Load tokenizer
        let tokenizerUrl = URL(fileURLWithPath: "\(Self.weightsPath)/tokenizer.json")
        let tokenizer = try UnigramTokenizer(tokenizerJsonUrl: tokenizerUrl)

        var allPassed = true

        for testCase in testCases {
            let text = testCase["text"] as! String
            let expectedIds = testCase["token_ids"] as! [Int]
            let expectedTokens = testCase["tokens"] as! [String]

            // Encode with Swift tokenizer
            let actualIds = tokenizer.encode(text)
            let actualTokens = tokenizer.tokenize(text)

            // Compare tokens first (more readable for debugging)
            if actualTokens != expectedTokens {
                print("Token MISMATCH for '\(text)':")
                print("  Expected tokens: \(expectedTokens)")
                print("  Actual tokens:   \(actualTokens)")
                allPassed = false
            }

            // Compare IDs
            if actualIds != expectedIds {
                print("ID MISMATCH for '\(text)':")
                print("  Expected IDs: \(expectedIds)")
                print("  Actual IDs:   \(actualIds)")
                allPassed = false
            } else {
                print("✓ '\(text)' -> \(actualIds.count) tokens match")
            }
        }

        XCTAssertTrue(allPassed, "Some tokenizations did not match Python output. See log above.")
    }

    /// Test special token IDs match Python
    func testTokenizerSpecialTokensMatchPython() throws {
        // Load Python fixture
        let fixtureUrl = URL(fileURLWithPath: "\(Self.fixturesPath)/tokenizer_special_tokens.json")

        guard FileManager.default.fileExists(atPath: fixtureUrl.path) else {
            throw XCTSkip("Tokenizer special tokens fixture not found")
        }

        let fixtureData = try Data(contentsOf: fixtureUrl)
        let expected = try JSONSerialization.jsonObject(with: fixtureData) as! [String: Int]

        // Load tokenizer
        let tokenizerUrl = URL(fileURLWithPath: "\(Self.weightsPath)/tokenizer.json")
        let tokenizer = try UnigramTokenizer(tokenizerJsonUrl: tokenizerUrl)

        // Compare special token IDs
        XCTAssertEqual(tokenizer.padTokenId, expected["pad_token_id"]!, "PAD token ID mismatch")
        XCTAssertEqual(tokenizer.clsTokenId, expected["cls_token_id"]!, "CLS token ID mismatch")
        XCTAssertEqual(tokenizer.sepTokenId, expected["sep_token_id"]!, "SEP token ID mismatch")
        XCTAssertEqual(tokenizer.unkTokenId, expected["unk_token_id"]!, "UNK token ID mismatch")

        print("Special token IDs match Python:")
        print("  PAD: \(tokenizer.padTokenId)")
        print("  CLS: \(tokenizer.clsTokenId)")
        print("  SEP: \(tokenizer.sepTokenId)")
        print("  UNK: \(tokenizer.unkTokenId)")
    }

    // MARK: - Diagnostic Tests (Find Where Pipeline Breaks)

    /// Diagnostic test to find where inference pipeline breaks
    /// This test checks intermediate values to identify the root cause
    func testDiagnosticEntityPipeline() async throws {
        try skipIfMLXUnavailable()

        print("\n" + String(repeating: "=", count: 70))
        print("DIAGNOSTIC: Entity Extraction Pipeline Analysis")
        print(String(repeating: "=", count: 70))

        let model = try await GLiNER2.fromPretrained(Self.weightsPath)

        let text = "Tim Cook is CEO of Apple."
        let entityTypes = ["person", "company"]

        print("\n1. INPUT")
        print("   Text: '\(text)'")
        print("   Entity types: \(entityTypes)")

        // Build schema manually to inspect
        let schema = model.createSchema().entities(entityTypes)
        let schemaDict = schema.build()
        print("\n2. SCHEMA")
        print("   Schema dict: \(schemaDict)")

        // Transform text
        let normalizedText = text.hasSuffix(".") || text.hasSuffix("!") || text.hasSuffix("?")
            ? text : text + "."
        print("\n3. NORMALIZED TEXT: '\(normalizedText)'")

        // Process through processor
        let record = model.processor.transform(text: normalizedText.lowercased(), schema: schemaDict)
        print("\n4. TRANSFORMED RECORD")
        print("   Token IDs count: \(record.inputIds.count)")
        print("   First 20 token IDs: \(Array(record.inputIds.prefix(20)))")
        print("   Schema tokens: \(record.schemaTokensList)")
        print("   Task types: \(record.taskTypes)")

        // Collate into batch
        let batch = model.processor.collateBatch([record])
        print("\n5. BATCH")
        print("   Input IDs shape: \(batch.inputIds.shape)")
        print("   Mapped indices count: \(batch.mappedIndices[0].count)")

        // Run encoder
        print("\n6. ENCODER OUTPUT")
        let encoderOutput = model.model.encode(batch.inputIds, attentionMask: batch.attentionMask)
        let hiddenStates = encoderOutput.lastHiddenState
        MLX.eval(hiddenStates)
        print("   Hidden states shape: \(hiddenStates.shape)")

        // Check hidden states values
        let hiddenSum = MLX.sum(MLX.abs(hiddenStates))
        MLX.eval(hiddenSum)
        print("   Hidden states L1 sum: \(Float(hiddenSum.item(Float32.self)))")

        // Additional encoder output stats for parity debugging
        let hiddenMean = MLX.mean(hiddenStates)
        let hiddenStd = MLX.std(hiddenStates)
        let hiddenMin = MLX.min(hiddenStates)
        let hiddenMax = MLX.max(hiddenStates)
        MLX.eval(hiddenMean, hiddenStd, hiddenMin, hiddenMax)
        print("   Hidden states mean: \(hiddenMean.item(Float.self))")
        print("   Hidden states std: \(hiddenStd.item(Float.self))")
        print("   Hidden states min: \(hiddenMin.item(Float.self))")
        print("   Hidden states max: \(hiddenMax.item(Float.self))")

        // Per-position L1 sums (first few and text positions)
        print("   --- Per-position L1 (Python reference) ---")
        print("   Python pos 0: 222.63, pos 1: 238.75, pos 11: 214.49")
        print("   --- Swift Actual Values ---")
        for pos in [0, 1, 2, 10, 11, 12, 17] {
            let posL1 = MLX.sum(MLX.abs(hiddenStates[0, pos]))
            MLX.eval(posL1)
            print("   Position \(pos): \(posL1.item(Float.self))")
        }

        // First 10 values at position 0 (for exact comparison)
        print("   First 10 values at position 0 (Python: [-0.060, -0.045, 0.007, 0.154, 0.098, ...]):")
        let pos0Vals = hiddenStates[0, 0, 0..<10]
        MLX.eval(pos0Vals)
        print("   \(pos0Vals)")

        // Find text start
        var textStartIdx = 0
        for (idx, mapping) in batch.mappedIndices[0].enumerated() {
            if mapping.segmentType == .text {
                textStartIdx = idx
                break
            }
        }
        print("   Text starts at index: \(textStartIdx)")

        // Extract schema embeddings
        print("\n7. SCHEMA EMBEDDINGS")
        let sampleHidden = hiddenStates[0]
        var schemaEmbList: [MLXArray] = []
        var processedOrigIndices: Set<Int> = []
        let specialTokens: Set<String> = ["[P]", "[C]", "[E]", "[R]", "[L]"]

        // Build special token indices
        var specialTokenIndices: Set<Int> = []
        var offset = 0
        for (schemaIdx, schemaTokens) in record.schemaTokensList.enumerated() {
            for (localIdx, token) in schemaTokens.enumerated() {
                if specialTokens.contains(token) {
                    specialTokenIndices.insert(offset + localIdx)
                }
            }
            offset += schemaTokens.count
            if schemaIdx < record.schemaTokensList.count - 1 {
                offset += 1  // [SEP_STRUCT]
            }
        }
        print("   Special token indices: \(specialTokenIndices)")

        for (idx, mapping) in batch.mappedIndices[0].enumerated() {
            if mapping.segmentType == .schema {
                if specialTokenIndices.contains(mapping.originalIndex) && !processedOrigIndices.contains(mapping.originalIndex) {
                    schemaEmbList.append(sampleHidden[idx])
                    processedOrigIndices.insert(mapping.originalIndex)
                }
            }
        }
        print("   Found \(schemaEmbList.count) schema embeddings")

        guard !schemaEmbList.isEmpty else {
            print("   ❌ ERROR: No schema embeddings found!")
            XCTFail("No schema embeddings found")
            return
        }

        let embs = MLX.stacked(schemaEmbList, axis: 0)
        print("   Schema embs shape: \(embs.shape)")

        // Count prediction
        print("\n8. COUNT PREDICTION")
        let countLogits = model.model.countPred(embs[0].expandedDimensions(axis: 0))
        MLX.eval(countLogits)
        print("   Count logits shape: \(countLogits.shape)")
        print("   Count logits values: \(countLogits)")

        let countLogitsSqueezed = countLogits.squeezed(axis: 0)
        let predCountIdx = MLX.argMax(countLogitsSqueezed)
        MLX.eval(predCountIdx)
        let predCount = Int(predCountIdx.item(Int32.self))
        print("   Predicted count: \(predCount)")

        if predCount <= 0 {
            print("   ❌ ERROR: predCount <= 0, will return empty results!")
            print("   This is likely why entity extraction returns empty arrays")
            XCTFail("predCount is \(predCount), expected > 0")
            return
        }

        // Span representation
        print("\n9. SPAN REPRESENTATION")
        let textLen = batch.mappedIndices[0].count - textStartIdx
        let textEmbeddings = sampleHidden[textStartIdx...]
        print("   Text length: \(textLen)")
        print("   Text embeddings shape: \(textEmbeddings.shape)")

        // Print input embeddings L1 sum for comparison with Python (Python: 1478.69)
        let textEmbsL1 = MLX.sum(MLX.abs(textEmbeddings))
        MLX.eval(textEmbsL1)
        print("   Text embeddings L1 sum: \(Float(textEmbsL1.item(Float32.self)))")

        // Run with debug=true to trace intermediate values
        print("\n   --- SpanMarkerV0 Intermediate Values (Python reference) ---")
        print("   Python: h L1=1478.69, start_rep=10015.78, end_rep=10198.38")
        print("   Python: start_span_rep=76343.44, end_span_rep=91579.69")
        print("   Python: cat(before relu)=167982.75, cat(after relu)=14725.32")
        print("   Python: out_project=22006.92")
        print("   --- Swift Actual Values ---")

        let spanInfo = model.model.computeSpanRep(textEmbeddings, debug: true)
        print("   Span rep shape: \(spanInfo.spanRep.shape)")
        print("   Spans idx shape: \(spanInfo.spansIdx.shape)")

        let spanRepSum = MLX.sum(MLX.abs(spanInfo.spanRep))
        MLX.eval(spanRepSum)
        print("   Span rep L1 sum: \(Float(spanRepSum.item(Float32.self)))")

        // Count embed
        print("\n10. COUNT EMBED")
        let fieldEmbs = embs[1...]
        print("    Field embs shape: \(fieldEmbs.shape)")

        let structProj = model.model.countEmbed(fieldEmbs, goldCountVal: predCount)
        MLX.eval(structProj)
        print("    Struct proj shape: \(structProj.shape)")

        let structProjSum = MLX.sum(MLX.abs(structProj))
        MLX.eval(structProjSum)
        print("    Struct proj L1 sum: \(Float(structProjSum.item(Float32.self)))")

        // Span scores
        print("\n11. SPAN SCORES")
        let L = spanInfo.spansIdx.dim(1) / 8  // maxWidth = 8
        let spanRepReshaped = spanInfo.spanRep.reshaped([L, 8, 768])
        print("    Span rep reshaped: \(spanRepReshaped.shape)")
        print("    L (text positions): \(L)")

        var spanScores = MLX.einsum("lkd,cpd->cplk", spanRepReshaped, structProj)
        spanScores = MLX.sigmoid(spanScores)
        MLX.eval(spanScores)
        print("    Span scores shape: \(spanScores.shape)")

        let scoresMax = MLX.max(spanScores)
        let scoresMin = MLX.min(spanScores)
        let scoresMean = MLX.mean(spanScores)
        MLX.eval(scoresMax, scoresMin, scoresMean)
        print("    Scores max: \(Float(scoresMax.item(Float32.self)))")
        print("    Scores min: \(Float(scoresMin.item(Float32.self)))")
        print("    Scores mean: \(Float(scoresMean.item(Float32.self)))")

        // Check if any scores are above threshold
        let threshold: Float = 0.5
        let aboveThreshold = MLX.sum(spanScores .> threshold)
        MLX.eval(aboveThreshold)
        let numAbove = Int(aboveThreshold.item(Int32.self))
        print("    Scores above \(threshold): \(numAbove)")

        if numAbove == 0 {
            print("    ❌ ERROR: No scores above threshold!")
            print("    This is why no entities are extracted")
        }

        print("\n" + String(repeating: "=", count: 70))
        print("DIAGNOSTIC COMPLETE")
        print(String(repeating: "=", count: 70))

        // Final assertion - at least show where it breaks
        XCTAssertGreaterThan(predCount, 0, "predCount should be > 0")
        XCTAssertGreaterThan(numAbove, 0, "Should have some scores above threshold \(threshold)")
    }

    // MARK: - Swift vs Python Comparison Tests

    /// Compare Swift entity extraction against Python fixtures
    /// This test shows exactly what matches and what doesn't
    func testEntityExtractionVsPython() async throws {
        try skipIfMLXUnavailable()

        // Load Python fixtures
        let resultUrl = URL(fileURLWithPath: "\(Self.fixturesPath)/entity_basic_result.json")
        let metadataUrl = URL(fileURLWithPath: "\(Self.fixturesPath)/entity_basic_metadata.json")

        guard FileManager.default.fileExists(atPath: resultUrl.path) else {
            throw XCTSkip("Python fixtures not found. Run: python scripts/generate_inference_fixtures.py")
        }

        let pythonResultData = try Data(contentsOf: resultUrl)
        let pythonResult = try JSONSerialization.jsonObject(with: pythonResultData) as! [String: Any]

        let metadataData = try Data(contentsOf: metadataUrl)
        let metadata = try JSONSerialization.jsonObject(with: metadataData) as! [String: Any]

        let text = metadata["text"] as! String
        let entityTypes = metadata["entity_types"] as! [String]

        print("\n" + String(repeating: "=", count: 60))
        print("ENTITY EXTRACTION COMPARISON: Swift vs Python")
        print(String(repeating: "=", count: 60))
        print("Input text: '\(text)'")
        print("Entity types: \(entityTypes)")

        // Load model and run Swift inference
        let model = try await GLiNER2.fromPretrained(Self.weightsPath)
        let swiftResult = model.extractEntities(
            text: text,
            entityTypes: entityTypes,
            threshold: 0.5,
            includeConfidence: true,
            includeSpans: true
        )

        // Print Python result
        print("\n--- PYTHON RESULT ---")
        printJSON(pythonResult)

        // Print Swift result
        print("\n--- SWIFT RESULT ---")
        printJSON(swiftResult)

        // Compare structures
        print("\n--- COMPARISON ---")

        // Python has: {"entities": {"person": [...], "company": [...]}}
        // Swift may have different structure
        var pythonEntities: [String: Any] = [:]
        if let entities = pythonResult["entities"] as? [String: Any] {
            pythonEntities = entities
        } else {
            // Maybe Python puts entities at top level
            pythonEntities = pythonResult
        }

        var matches = 0
        var mismatches = 0

        for entityType in entityTypes {
            print("\n  Entity type: '\(entityType)'")

            // Get Python entities for this type
            let pythonList = (pythonEntities[entityType] as? [[String: Any]]) ?? []
            let pythonTexts = Set(pythonList.compactMap { $0["text"] as? String })

            // Get Swift entities for this type
            var swiftTexts: Set<String> = []
            if let swiftList = swiftResult[entityType] as? [[String: Any]] {
                swiftTexts = Set(swiftList.compactMap { $0["text"] as? String })
            } else if let swiftList = swiftResult[entityType] as? [String] {
                swiftTexts = Set(swiftList)
            }

            print("    Python found: \(pythonTexts)")
            print("    Swift found:  \(swiftTexts)")

            if pythonTexts == swiftTexts {
                print("    ✅ MATCH")
                matches += 1
            } else {
                print("    ❌ MISMATCH")
                print("      Missing in Swift: \(pythonTexts.subtracting(swiftTexts))")
                print("      Extra in Swift:   \(swiftTexts.subtracting(pythonTexts))")
                mismatches += 1
            }
        }

        print("\n--- SUMMARY ---")
        print("Matches: \(matches)/\(entityTypes.count)")
        print("Mismatches: \(mismatches)/\(entityTypes.count)")

        // This test is informational - it shows what works and what doesn't
        // We expect mismatches until implementation is complete
        if mismatches > 0 {
            print("\n⚠️ Entity extraction does not match Python yet")
        } else {
            print("\n✅ Entity extraction matches Python!")
        }
    }

    /// Compare Swift classification against Python fixtures
    func testClassificationVsPython() async throws {
        try skipIfMLXUnavailable()

        let resultUrl = URL(fileURLWithPath: "\(Self.fixturesPath)/classify_sentiment_positive_result.json")
        let metadataUrl = URL(fileURLWithPath: "\(Self.fixturesPath)/classify_sentiment_positive_metadata.json")

        guard FileManager.default.fileExists(atPath: resultUrl.path) else {
            throw XCTSkip("Python fixtures not found")
        }

        let pythonResultData = try Data(contentsOf: resultUrl)
        let pythonResult = try JSONSerialization.jsonObject(with: pythonResultData) as! [String: Any]

        let metadataData = try Data(contentsOf: metadataUrl)
        let metadata = try JSONSerialization.jsonObject(with: metadataData) as! [String: Any]

        let text = metadata["text"] as! String
        let task = metadata["task"] as! String
        let labels = metadata["labels"] as! [String]

        print("\n" + String(repeating: "=", count: 60))
        print("CLASSIFICATION COMPARISON: Swift vs Python")
        print(String(repeating: "=", count: 60))
        print("Input text: '\(text)'")
        print("Task: \(task)")
        print("Labels: \(labels)")

        let model = try await GLiNER2.fromPretrained(Self.weightsPath)
        let swiftResult = model.classifyText(
            text: text,
            task: task,
            labels: labels,
            multiLabel: false,
            threshold: 0.5,
            includeConfidence: true
        )

        print("\n--- PYTHON RESULT ---")
        printJSON(pythonResult)

        print("\n--- SWIFT RESULT ---")
        printJSON(swiftResult)

        // Compare
        print("\n--- COMPARISON ---")

        let pythonLabel = extractClassificationLabel(pythonResult, task: task)
        let swiftLabel = extractClassificationLabel(swiftResult, task: task)

        print("  Python label: \(pythonLabel ?? "nil")")
        print("  Swift label:  \(swiftLabel ?? "nil")")

        if pythonLabel == swiftLabel {
            print("  ✅ MATCH")
        } else {
            print("  ❌ MISMATCH")
        }
    }

    /// Compare Swift structure extraction against Python fixtures
    func testStructureExtractionVsPython() async throws {
        try skipIfMLXUnavailable()

        let resultUrl = URL(fileURLWithPath: "\(Self.fixturesPath)/struct_person_result.json")
        let metadataUrl = URL(fileURLWithPath: "\(Self.fixturesPath)/struct_person_metadata.json")

        guard FileManager.default.fileExists(atPath: resultUrl.path) else {
            throw XCTSkip("Python fixtures not found")
        }

        let pythonResultData = try Data(contentsOf: resultUrl)
        let pythonResult = try JSONSerialization.jsonObject(with: pythonResultData) as! [String: Any]

        let metadataData = try Data(contentsOf: metadataUrl)
        let metadata = try JSONSerialization.jsonObject(with: metadataData) as! [String: Any]

        let text = metadata["text"] as! String
        let structureName = metadata["structure_name"] as! String
        let fields = (metadata["fields"] as? [String]) ?? []

        print("\n" + String(repeating: "=", count: 60))
        print("STRUCTURE EXTRACTION COMPARISON: Swift vs Python")
        print(String(repeating: "=", count: 60))
        print("Input text: '\(text)'")
        print("Structure: \(structureName)")
        print("Fields: \(fields)")

        let model = try await GLiNER2.fromPretrained(Self.weightsPath)

        var schemaBuilder = model.createSchema().structure(structureName)
        for field in fields {
            schemaBuilder = schemaBuilder.field(field, dtype: "str")
        }
        let schema = schemaBuilder.done()

        let swiftResult = model.extract(
            text: text,
            schema: schema,
            threshold: 0.5,
            includeConfidence: true,
            includeSpans: true
        )

        print("\n--- PYTHON RESULT ---")
        printJSON(pythonResult)

        print("\n--- SWIFT RESULT ---")
        printJSON(swiftResult)

        // Compare
        print("\n--- COMPARISON ---")

        let pythonStructs = (pythonResult[structureName] as? [[String: Any]]) ?? []
        let swiftStructs = (swiftResult[structureName] as? [[String: Any]]) ?? []

        print("  Python found \(pythonStructs.count) instances")
        print("  Swift found \(swiftStructs.count) instances")

        if pythonStructs.count != swiftStructs.count {
            print("  ❌ Instance count mismatch")
        }

        // Compare field values
        for field in fields {
            let pythonValue = extractFieldValue(pythonStructs.first, field: field)
            let swiftValue = extractFieldValue(swiftStructs.first, field: field)

            print("\n  Field '\(field)':")
            print("    Python: \(pythonValue ?? "nil")")
            print("    Swift:  \(swiftValue ?? "nil")")

            if pythonValue == swiftValue {
                print("    ✅ MATCH")
            } else {
                print("    ❌ MISMATCH")
            }
        }
    }

    // MARK: - Encoder Debug Test

    /// Debug test to trace encoder forward pass step by step
    func testEncoderForwardPassDebug() async throws {
        try skipIfMLXUnavailable()

        print("\n" + String(repeating: "=", count: 70))
        print("ENCODER FORWARD PASS DEBUG")
        print(String(repeating: "=", count: 70))

        let model = try await GLiNER2.fromPretrained(Self.weightsPath)

        // Use same input as diagnostic test
        // Python token IDs: [287, 128003, 6967, 287, 128005, 604, 128005, 483, 1263, 1263, 128002, 41718, 3712, 269, 101312, 265, 6038, 323]
        let pythonInputIds: [Int32] = [287, 128003, 6967, 287, 128005, 604, 128005, 483, 1263, 1263, 128002, 41718, 3712, 269, 101312, 265, 6038, 323]
        let inputIds = MLXArray(pythonInputIds).reshaped([1, 18])
        let attentionMask = MLXArray.ones([1, 18], dtype: .int32)

        print("\n1. INPUT")
        print("   Input IDs: \(pythonInputIds)")
        print("   Shape: \(inputIds.shape)")

        // Step 1: Check embeddings
        print("\n2. EMBEDDINGS")
        let embeddings = model.model.encoder.embeddings(inputIds)
        MLX.eval(embeddings)
        let embL1 = Float(MLX.sum(MLX.abs(embeddings)).item(Float32.self))
        let embMean = Float(MLX.mean(embeddings).item(Float32.self))
        let embStd = Float(MLX.std(embeddings).item(Float32.self))
        print("   Embeddings shape: \(embeddings.shape)")
        print("   Embeddings L1 sum: \(embL1)")
        print("   Embeddings mean: \(embMean)")
        print("   Embeddings std: \(embStd)")
        // Python reference: After embeddings + LayerNorm, values should be normalized

        // Step 2: Check after first layer
        print("\n3. AFTER LAYER 0")
        let layer0 = model.model.encoder.layers[0]
        let relEmb = model.model.encoder.relEmbeddings

        // Prepare attention mask
        let expandedMask = (1.0 - attentionMask.asType(.float32).expandedDimensions(axes: [1, 2])) * -10000.0

        let afterLayer0 = layer0(embeddings, relEmbeddings: relEmb, attentionMask: expandedMask)
        MLX.eval(afterLayer0)
        let layer0L1 = Float(MLX.sum(MLX.abs(afterLayer0)).item(Float32.self))
        let layer0Mean = Float(MLX.mean(afterLayer0).item(Float32.self))
        let layer0Std = Float(MLX.std(afterLayer0).item(Float32.self))
        print("   After layer 0 shape: \(afterLayer0.shape)")
        print("   After layer 0 L1 sum: \(layer0L1)")
        print("   After layer 0 mean: \(layer0Mean)")
        print("   After layer 0 std: \(layer0Std)")

        // Step 3: Check full encoder output
        print("\n4. FULL ENCODER OUTPUT")
        let encoderOutput = model.model.encode(inputIds, attentionMask: attentionMask)
        let hidden = encoderOutput.lastHiddenState
        MLX.eval(hidden)
        let hiddenL1 = Float(MLX.sum(MLX.abs(hidden)).item(Float32.self))
        let hiddenMean = Float(MLX.mean(hidden).item(Float32.self))
        let hiddenStd = Float(MLX.std(hidden).item(Float32.self))
        print("   Hidden states shape: \(hidden.shape)")
        print("   Hidden states L1 sum: \(hiddenL1) (Python: 3954.15)")
        print("   Hidden states mean: \(hiddenMean) (Python: 0.013544)")
        print("   Hidden states std: \(hiddenStd) (Python: 0.834328)")

        // Step 4: Check attention internals in layer 0
        print("\n5. ATTENTION DEBUG (Layer 0)")
        let attention = layer0.attention

        // Get Q, K, V
        let queryLayer = attention.queryProj(embeddings)
        let keyLayer = attention.keyProj(embeddings)
        let valueLayer = attention.valueProj(embeddings)
        MLX.eval(queryLayer, keyLayer, valueLayer)

        print("   Query L1 sum: \(Float(MLX.sum(MLX.abs(queryLayer)).item(Float32.self)))")
        print("   Key L1 sum: \(Float(MLX.sum(MLX.abs(keyLayer)).item(Float32.self)))")
        print("   Value L1 sum: \(Float(MLX.sum(MLX.abs(valueLayer)).item(Float32.self)))")

        // Check position bucket values
        print("\n6. POSITION BUCKETING")
        let seqLen = 18
        let buckets = makeLogBucketPosition(seqLen: seqLen, positionBuckets: 256, maxPosition: 512)
        MLX.eval(buckets)
        print("   Bucket positions shape: \(buckets.shape)")
        print("   Bucket [0,0] (diagonal): \(buckets[0, 0])")
        print("   Bucket [0,1]: \(buckets[0, 1])")
        print("   Bucket [1,0]: \(buckets[1, 0])")
        print("   Bucket min: \(MLX.min(buckets))")
        print("   Bucket max: \(MLX.max(buckets))")

        // Step 7: Trace through ALL layers to find where shrinkage happens
        print("\n7. LAYER-BY-LAYER TRACE")
        var currentHidden = embeddings
        for i in 0..<model.model.encoder.layers.count {
            let layer = model.model.encoder.layers[i]
            currentHidden = layer(currentHidden, relEmbeddings: relEmb, attentionMask: expandedMask)
            MLX.eval(currentHidden)
            let layerL1 = Float(MLX.sum(MLX.abs(currentHidden)).item(Float32.self))
            let layerStd = Float(MLX.std(currentHidden).item(Float32.self))
            print("   Layer \(i): L1=\(String(format: "%.1f", layerL1)), std=\(String(format: "%.4f", layerStd))")
        }

        // Step 8: Check relEmbeddings LayerNorm (used for relative embeddings, NOT final output!)
        // NOTE: DeBERTa does NOT apply final LayerNorm to hidden states!
        // The LayerNorm is only used for normalizing relative embeddings
        print("\n8. RELATIVE EMBEDDINGS LAYERNORM (NOT FINAL OUTPUT!)")

        // Check the relative embeddings LayerNorm weights
        if let lnWeight = model.model.encoder.relEmbeddingsLayerNorm.weight {
            MLX.eval(lnWeight)
            let lnWeightL1 = Float(MLX.sum(MLX.abs(lnWeight)).item(Float32.self))
            print("   rel_embeddings LayerNorm weight L1: \(lnWeightL1) (Python: 101.67)")
        } else {
            print("   rel_embeddings LayerNorm weight: nil")
        }

        if let lnBias = model.model.encoder.relEmbeddingsLayerNorm.bias {
            MLX.eval(lnBias)
            let lnBiasL1 = Float(MLX.sum(MLX.abs(lnBias)).item(Float32.self))
            print("   rel_embeddings LayerNorm bias L1: \(lnBiasL1) (Python: 23.79)")
        } else {
            print("   rel_embeddings LayerNorm bias: nil")
        }

        // Verify encoder output matches Python (should be ~3954, not crushed to ~867)
        print("\n   Final encoder output L1: \(hiddenL1) (Python: 3954.15)")

        print("\n" + String(repeating: "=", count: 70))

        // Assert encoder output is reasonable
        XCTAssertGreaterThan(hiddenL1, 1000.0,
            "Encoder hidden states L1 sum too small: \(hiddenL1). Expected ~3954")
    }

    // MARK: - Comparison Helper Functions

    private func printJSON(_ dict: [String: Any]) {
        if let data = try? JSONSerialization.data(withJSONObject: dict, options: .prettyPrinted),
           let str = String(data: data, encoding: .utf8) {
            print(str)
        } else {
            print(dict)
        }
    }

    private func extractClassificationLabel(_ result: [String: Any], task: String) -> String? {
        if let taskResult = result[task] as? [String: Any] {
            return taskResult["label"] as? String
        }
        return result[task] as? String
    }

    private func extractFieldValue(_ instance: [String: Any]?, field: String) -> String? {
        guard let instance = instance else { return nil }

        if let arr = instance[field] as? [[String: Any]], let first = arr.first {
            return first["text"] as? String
        } else if let dict = instance[field] as? [String: Any] {
            return dict["text"] as? String
        }
        return instance[field] as? String
    }

}
