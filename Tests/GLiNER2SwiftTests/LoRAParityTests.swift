// LoRAParityTests.swift
// Parity tests for LoRA adapter support in GLiNER2Swift
//
// These tests verify that the Swift LoRA merge produces identical results
// to the Python implementation.
//
// Prerequisites:
//   1. Base model at GLINER2_WEIGHTS_PATH (or <project_root>/weights/)
//   2. Adapter at GLINER2_ADAPTER_PATH (or <base_model>/final/)
//   3. Fixtures at:   Tests/GLiNER2SwiftTests/Fixtures/lora/
//      Generate with: uv run python GLiNER2Swift/scripts/generate_lora_fixtures.py

import XCTest
import Foundation
import Metal
@testable import GLiNER2Swift
import MLX
import MLXNN

final class LoRAParityTests: XCTestCase {

    // Paths – resolved from env vars, falling back to project-relative defaults
    static let baseModelPath: String = {
        if let envPath = ProcessInfo.processInfo.environment["GLINER2_WEIGHTS_PATH"] {
            return envPath
        }
        return URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("weights").path
    }()

    static let adapterPath: String = {
        if let envPath = ProcessInfo.processInfo.environment["GLINER2_ADAPTER_PATH"] {
            return envPath
        }
        return URL(fileURLWithPath: baseModelPath)
            .appendingPathComponent("final").path
    }()

    // Fixture loader
    var loader: InferenceFixtureLoader!

    override func setUp() {
        super.setUp()
        loader = InferenceFixtureLoader(subdir: "lora")
    }

    // MARK: - Skip Helpers

    private func skipIfNoGPU() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal GPU not available")
        }
    }

    private func skipIfNoModel() throws {
        guard FileManager.default.fileExists(atPath: Self.baseModelPath + "/model.safetensors") else {
            throw XCTSkip("Base model not found at \(Self.baseModelPath)")
        }
    }

    private func skipIfNoAdapter() throws {
        guard FileManager.default.fileExists(atPath: Self.adapterPath + "/adapter_config.json") else {
            throw XCTSkip("Adapter not found at \(Self.adapterPath)")
        }
    }

    // MARK: - Test 1: Config Loading

    func testAdapterConfigLoading() throws {
        try skipIfNoAdapter()

        let config = try LoRAAdapterConfig.load(from: URL(fileURLWithPath: Self.adapterPath))

        XCTAssertEqual(config.adapterType, "lora")
        XCTAssertEqual(config.adapterVersion, "1.0")
        XCTAssertEqual(config.loraR, 16)
        XCTAssertEqual(config.loraAlpha, 32.0)
        XCTAssertEqual(config.loraDropout, 0.05)
        XCTAssertEqual(config.scaling, 2.0, "scaling should be alpha/r = 32/16 = 2.0")

        // Check target modules
        XCTAssertTrue(config.targetModules.contains("encoder"))
        XCTAssertTrue(config.targetModules.contains("classifier"))
        XCTAssertTrue(config.targetModules.contains("span_rep"))
        XCTAssertTrue(config.targetModules.contains("count_embed"))
        XCTAssertTrue(config.targetModules.contains("count_pred"))

        print("Adapter config loaded: r=\(config.loraR), alpha=\(config.loraAlpha), scaling=\(config.scaling)")
        print("Target modules: \(config.targetModules)")
    }

    // MARK: - Test 2: IsAdapterPath Detection

    func testIsAdapterPathDetection() throws {
        try skipIfNoAdapter()

        let adapterUrl = URL(fileURLWithPath: Self.adapterPath)
        let baseUrl = URL(fileURLWithPath: Self.baseModelPath)
        let fakeUrl = URL(fileURLWithPath: "/tmp/nonexistent_dir_12345")

        XCTAssertTrue(LoRAAdapterConfig.isAdapterPath(adapterUrl), "Adapter directory should be detected")
        XCTAssertFalse(LoRAAdapterConfig.isAdapterPath(fakeUrl), "Non-existent path should not be detected")
        // Base model directory doesn't have adapter_config.json at top level
        // (it has it in final/ subdirectory)
    }

    // MARK: - Test 3: Synthetic Merge Computation

    func testLoRAMergeComputation() throws {
        try skipIfNoGPU()

        // Create synthetic weights
        let baseData: [Float] = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ]
        let base = MLXArray(baseData, [4, 4])

        let loraAData: [Float] = [
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
        ]
        let loraA = MLXArray(loraAData, [2, 4])  // [2, 4]

        let loraBData: [Float] = [
            1.0, 0.0,
            0.0, 1.0,
            0.5, 0.5,
            1.0, 1.0,
        ]
        let loraB = MLXArray(loraBData, [4, 2])  // [4, 2]

        let config = try JSONDecoder().decode(LoRAAdapterConfig.self, from: """
        {
            "adapter_type": "lora",
            "adapter_version": "1.0",
            "lora_r": 2,
            "lora_alpha": 4.0,
            "lora_dropout": 0.0,
            "target_modules": ["test"]
        }
        """.data(using: .utf8)!)

        XCTAssertEqual(config.scaling, 2.0)

        var baseWeights: [String: MLXArray] = ["test.layer.weight": base]
        let adapterWeights: [String: MLXArray] = [
            "test.layer.lora_A": loraA,
            "test.layer.lora_B": loraB,
        ]

        let count = mergeLoRAWeights(into: &baseWeights, adapterWeights: adapterWeights, config: config)
        XCTAssertEqual(count, 1, "Should merge 1 weight")

        // Compute expected: base + (B @ A) * 2.0
        let delta = matmul(loraB, loraA) * MLXArray(Float(2.0))
        let expected = base + delta

        let merged = baseWeights["test.layer.weight"]!
        MLX.eval(merged, expected)

        let diff = MLX.abs(merged - expected)
        let maxDiff = MLX.max(diff).item(Float.self)
        XCTAssertLessThan(maxDiff, 1e-6, "Merged weight should match manual computation")

        print("Synthetic merge test passed, max diff: \(maxDiff)")
    }

    // MARK: - Test 4: LoRA Weight Merge Count

    func testLoRAWeightMergeCount() throws {
        try skipIfNoGPU()
        try skipIfNoModel()
        try skipIfNoAdapter()

        // Load base weights
        let baseUrl = URL(fileURLWithPath: Self.baseModelPath + "/model.safetensors")
        var baseWeights = try loadArrays(url: baseUrl)

        // Load adapter
        let adapterUrl = URL(fileURLWithPath: Self.adapterPath + "/adapter_weights.safetensors")
        let adapterWeights = try loadArrays(url: adapterUrl)
        let config = try LoRAAdapterConfig.load(from: URL(fileURLWithPath: Self.adapterPath))

        // Count pairs in adapter
        let loraAKeys = adapterWeights.keys.filter { $0.hasSuffix(".lora_A") }
        let loraBKeys = adapterWeights.keys.filter { $0.hasSuffix(".lora_B") }
        print("Adapter has \(loraAKeys.count) lora_A keys and \(loraBKeys.count) lora_B keys")
        XCTAssertEqual(loraAKeys.count, loraBKeys.count, "lora_A and lora_B counts should match")

        // Merge
        let mergedCount = mergeLoRAWeights(into: &baseWeights, adapterWeights: adapterWeights, config: config)
        print("Merged \(mergedCount) weight matrices (expected \(loraAKeys.count))")
        XCTAssertEqual(mergedCount, loraAKeys.count, "Should merge all LoRA pairs")
        XCTAssertEqual(mergedCount, 92, "Should merge 92 LoRA pairs (known from adapter inspection)")
    }

    // MARK: - Test 5: Encoder Output Parity

    func testLoRAEncoderOutputParity() throws {
        try skipIfNoGPU()
        try skipIfNoModel()
        try skipIfNoAdapter()

        guard loader.fixtureExists("lora_encoder_output") else {
            throw XCTSkip("Fixtures not generated. Run: uv run python GLiNER2Swift/scripts/generate_lora_fixtures.py")
        }

        // Load Python reference
        let pyEncoderOutput = try loader.loadNpy("lora_encoder_output")  // [1, 18, 768]
        let pyInputIds = try loader.loadNpy("lora_input_ids")            // [1, 18]
        let pyAttentionMask = try loader.loadNpy("lora_attention_mask")  // [1, 18]

        // Load config and create extractor
        let configUrl = URL(fileURLWithPath: Self.baseModelPath + "/config.json")
        let config = try ExtractorConfig.load(from: configUrl)
        let extractor = Extractor(config: config)

        // Load weights with LoRA merged
        let baseWeightsUrl = URL(fileURLWithPath: Self.baseModelPath + "/model.safetensors")
        let adapterUrl = URL(fileURLWithPath: Self.adapterPath)
        try extractor.loadWeightsWithLoRA(baseWeightsUrl: baseWeightsUrl, adapterPath: adapterUrl)
        extractor.train(false)

        // Run encoder with same inputs
        let inputIds = pyInputIds.asType(.int32)
        let attentionMask = pyAttentionMask.asType(.int32)
        let output = extractor.encode(inputIds, attentionMask: attentionMask)
        let swiftHidden = output.lastHiddenState
        MLX.eval(swiftHidden)

        // Compare
        let pyRef = pyEncoderOutput.asType(.float32)
        let diff = MLX.abs(swiftHidden - pyRef)
        let meanDiff = MLX.mean(diff).item(Float.self)
        let maxDiff = MLX.max(diff).item(Float.self)

        print("Encoder output parity:")
        print("  Shape: Swift=\(swiftHidden.shape), Python=\(pyRef.shape)")
        print("  Mean diff: \(meanDiff)")
        print("  Max diff:  \(maxDiff)")

        // The Python pre-merged model had mean diff ~0.0002, max ~0.01 vs adapter model.
        // Our Swift merge should be closer to the Python adapter model (exact merge),
        // but there will be PyTorch vs MLX numerical differences through 12 encoder layers.
        XCTAssertLessThan(meanDiff, 0.01, "Mean encoder diff should be < 0.01")
        XCTAssertLessThan(maxDiff, 1.0, "Max encoder diff should be < 1.0")
    }

    // MARK: - Test 6: Entity Extraction with LoRA

    func testLoRAEntityExtraction() async throws {
        try skipIfNoGPU()
        try skipIfNoModel()
        try skipIfNoAdapter()

        guard loader.fixtureExists("lora_entity_result", ext: "json") else {
            throw XCTSkip("Fixtures not generated")
        }

        // Load Python reference
        let pyResult = try loader.loadJSON("lora_entity_result")
        guard let pyEntities = (pyResult["result"] as? [String: Any])?["entities"] as? [String: Any] else {
            XCTFail("Invalid fixture format"); return
        }

        // Load model with adapter
        let model = try await GLiNER2.fromPretrained(Self.baseModelPath)
        try model.loadAdapter(from: Self.adapterPath)

        // Run entity extraction
        let text = pyResult["text"] as! String
        let entityTypes = pyResult["entity_types"] as! [String]
        let result = model.extractEntities(
            text: text,
            entityTypes: entityTypes,
            threshold: 0.5,
            includeConfidence: true,
            includeSpans: true
        )

        print("Swift LoRA entity result: \(result)")

        // Compare entities
        guard let swiftEntities = result["entities"] as? [String: Any] else {
            XCTFail("Missing entities in Swift result"); return
        }

        for entityType in entityTypes {
            let pyList = pyEntities[entityType] as? [[String: Any]] ?? []
            let swiftList = swiftEntities[entityType] as? [[String: Any]] ?? []

            XCTAssertEqual(
                swiftList.count, pyList.count,
                "\(entityType) count mismatch: Swift=\(swiftList.count), Python=\(pyList.count)"
            )

            for (i, pyItem) in pyList.enumerated() {
                guard i < swiftList.count else { break }
                let swiftItem = swiftList[i]

                // Text must match exactly
                let pyText = pyItem["text"] as? String ?? ""
                let swiftText = swiftItem["text"] as? String ?? ""
                XCTAssertEqual(swiftText, pyText,
                    "\(entityType)[\(i)] text mismatch: Swift='\(swiftText)', Python='\(pyText)'")

                // Confidence within tolerance
                if let pyConf = pyItem["confidence"] as? Double,
                   let swiftConf = swiftItem["confidence"] as? Double {
                    XCTAssertEqual(swiftConf, pyConf, accuracy: 0.01,
                        "\(entityType)[\(i)] confidence mismatch")
                }
            }
        }
    }

    // MARK: - Test 7: Classification with LoRA

    func testLoRAClassification() async throws {
        try skipIfNoGPU()
        try skipIfNoModel()
        try skipIfNoAdapter()

        guard loader.fixtureExists("lora_classify_result", ext: "json") else {
            throw XCTSkip("Fixtures not generated")
        }

        // Load Python reference
        let pyResult = try loader.loadJSON("lora_classify_result")
        guard let pyClassification = pyResult["result"] as? [String: Any],
              let pySentiment = pyClassification["sentiment"] as? [String: Any],
              let pyLabel = pySentiment["label"] as? String else {
            XCTFail("Invalid classification fixture format"); return
        }

        // Load model with adapter
        let model = try await GLiNER2.fromPretrained(Self.baseModelPath)
        try model.loadAdapter(from: Self.adapterPath)

        // Run classification
        let text = pyResult["text"] as! String
        let labels = pyResult["labels"] as! [String]
        let task = pyResult["task"] as! String

        let result = model.classifyText(
            text: text,
            task: task,
            labels: labels,
            includeConfidence: true
        )

        print("Swift LoRA classification result: \(result)")

        // Compare
        guard let swiftSentiment = result["sentiment"] as? [String: Any],
              let swiftLabel = swiftSentiment["label"] as? String else {
            XCTFail("Missing sentiment in Swift result"); return
        }

        XCTAssertEqual(swiftLabel, pyLabel, "Classification label should match Python")

        if let pyConf = pySentiment["confidence"] as? Double,
           let swiftConf = swiftSentiment["confidence"] as? Double {
            XCTAssertEqual(swiftConf, pyConf, accuracy: 0.01, "Classification confidence should match")
        }
    }

    // MARK: - Test 8: fromPretrained with adapter

    func testFromPretrainedWithAdapter() async throws {
        try skipIfNoGPU()
        try skipIfNoModel()
        try skipIfNoAdapter()

        // One-step loading
        let model = try await GLiNER2.fromPretrained(
            Self.baseModelPath,
            adapterPath: Self.adapterPath
        )

        // Basic entity extraction test
        let result = model.extractEntities(
            text: "Tim Cook is CEO of Apple.",
            entityTypes: ["person", "company"],
            threshold: 0.5,
            includeConfidence: true
        )

        guard let entities = result["entities"] as? [String: Any] else {
            XCTFail("Missing entities"); return
        }

        // Should find at least Tim Cook
        let persons = entities["person"] as? [[String: Any]] ?? []
        let personTexts = persons.compactMap { $0["text"] as? String }
        XCTAssertTrue(personTexts.contains("Tim Cook"), "Should extract 'Tim Cook' with LoRA adapter")

        print("fromPretrained with adapter: extracted \(personTexts)")
    }

    // MARK: - Test 9: Two-step loading matches one-step

    func testLoadAdapterTwoStep() async throws {
        try skipIfNoGPU()
        try skipIfNoModel()
        try skipIfNoAdapter()

        let text = "Tim Cook is CEO of Apple."
        let entityTypes = ["person", "company"]

        // Two-step
        let model1 = try await GLiNER2.fromPretrained(Self.baseModelPath)
        try model1.loadAdapter(from: Self.adapterPath)
        let result1 = model1.extractEntities(
            text: text, entityTypes: entityTypes,
            threshold: 0.5, includeConfidence: true
        )

        // One-step
        let model2 = try await GLiNER2.fromPretrained(
            Self.baseModelPath, adapterPath: Self.adapterPath
        )
        let result2 = model2.extractEntities(
            text: text, entityTypes: entityTypes,
            threshold: 0.5, includeConfidence: true
        )

        // Compare entity texts
        guard let entities1 = result1["entities"] as? [String: Any],
              let entities2 = result2["entities"] as? [String: Any] else {
            XCTFail("Missing entities in one or both results"); return
        }

        for entityType in entityTypes {
            let list1 = (entities1[entityType] as? [[String: Any]] ?? []).compactMap { $0["text"] as? String }
            let list2 = (entities2[entityType] as? [[String: Any]] ?? []).compactMap { $0["text"] as? String }
            XCTAssertEqual(list1, list2, "\(entityType) results should match between one-step and two-step loading")
        }

        print("Two-step and one-step loading produce same results")
    }

    // MARK: - Test 10: Adapter Load/Unload Timing

    func testAdapterLoadUnloadTiming() async throws {
        try skipIfNoGPU()
        try skipIfNoModel()
        try skipIfNoAdapter()

        let text = "Tim Cook is CEO of Apple."
        let entityTypes = ["person", "company"]

        // Load base model
        let model = try await GLiNER2.fromPretrained(Self.baseModelPath)

        // Baseline: extract without adapter
        let baseResult = model.extractEntities(
            text: text, entityTypes: entityTypes,
            threshold: 0.5, includeConfidence: true
        )
        let basePersons = ((baseResult["entities"] as? [String: Any])?["person"] as? [[String: Any]])?.compactMap { $0["text"] as? String } ?? []
        print("Base model persons: \(basePersons)")

        // Time: load adapter
        let loadStart = CFAbsoluteTimeGetCurrent()
        try model.loadAdapter(from: Self.adapterPath)
        let loadTime = CFAbsoluteTimeGetCurrent() - loadStart
        print("Load adapter: \(String(format: "%.3f", loadTime))s")

        // Verify adapter is active
        let adapterResult = model.extractEntities(
            text: text, entityTypes: entityTypes,
            threshold: 0.5, includeConfidence: true
        )
        let adapterPersons = ((adapterResult["entities"] as? [String: Any])?["person"] as? [[String: Any]])?.compactMap { $0["text"] as? String } ?? []
        print("Adapter model persons: \(adapterPersons)")
        XCTAssertTrue(adapterPersons.contains("Tim Cook"), "Adapter should extract Tim Cook")

        // Time: unload adapter (restore base weights)
        let unloadStart = CFAbsoluteTimeGetCurrent()
        try model.unloadAdapter()
        let unloadTime = CFAbsoluteTimeGetCurrent() - unloadStart
        print("Unload adapter: \(String(format: "%.3f", unloadTime))s")

        // Verify base weights restored
        let restoredResult = model.extractEntities(
            text: text, entityTypes: entityTypes,
            threshold: 0.5, includeConfidence: true
        )
        let restoredPersons = ((restoredResult["entities"] as? [String: Any])?["person"] as? [[String: Any]])?.compactMap { $0["text"] as? String } ?? []
        print("Restored model persons: \(restoredPersons)")
        XCTAssertEqual(restoredPersons, basePersons, "After unload, results should match original base model")

        // Print summary
        print("\n--- Adapter Load/Unload Timing ---")
        print("  Load adapter:   \(String(format: "%.3f", loadTime))s")
        print("  Unload adapter: \(String(format: "%.3f", unloadTime))s")
        print("  Total round-trip: \(String(format: "%.3f", loadTime + unloadTime))s")
    }
}
