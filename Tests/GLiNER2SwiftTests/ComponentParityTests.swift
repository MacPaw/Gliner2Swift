// ComponentParityTests.swift
// Component-level parity tests comparing Swift/MLX against Python/PyTorch
//
// These tests verify that each component of the Swift implementation produces
// the same numerical results as the Python implementation.
//
// Run fixtures generation first:
//   cd ../scripts && python generate_component_fixtures.py

import XCTest
import Foundation
import Metal
@testable import GLiNER2Swift

import MLX
import MLXNN

final class ComponentParityTests: XCTestCase {

    // MARK: - Configuration

    static let weightsPath = "/Users/tmwstw/Documents/mnemos/GLiNER2/weights"
    static let fixturesPath = URL(fileURLWithPath: #file)
        .deletingLastPathComponent()
        .appendingPathComponent("Fixtures")

    /// Tolerance for numerical comparison
    let tolerance: Float = 1e-4

    /// Tolerance for L1 sum comparison (allows for small accumulated errors)
    let l1Tolerance: Float = 1.0

    /// Skip test if MLX Metal is not available
    private func skipIfMLXUnavailable() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal GPU not available")
        }
    }

    // MARK: - Test 1: Embedding Lookup

    func testEmbeddingLookup() async throws {
        try skipIfMLXUnavailable()

        guard let fixture = loadNPZFixture("embeddings") else {
            throw XCTSkip("Embedding fixtures not found. Run: python scripts/generate_component_fixtures.py")
        }

        let model = try await GLiNER2.fromPretrained(Self.weightsPath)

        // Get input IDs from fixture
        let inputIds = fixture["input_ids"]!
        print("Input IDs: \(inputIds)")

        // Get embeddings from Swift model
        let swiftEmbeddings = model.model.encoder.embeddings.wordEmbeddings(inputIds)
        MLX.eval(swiftEmbeddings)

        // Compare with Python
        let pythonWordEmb = fixture["word_embeddings"]!
        let swiftL1 = Float(MLX.sum(MLX.abs(swiftEmbeddings)).item(Float32.self))
        let pythonL1 = Float(MLX.sum(MLX.abs(pythonWordEmb)).item(Float32.self))

        print("Word embeddings L1:")
        print("  Python: \(pythonL1)")
        print("  Swift:  \(swiftL1)")
        print("  Diff:   \(abs(swiftL1 - pythonL1))")

        XCTAssertEqual(swiftL1, pythonL1, accuracy: l1Tolerance,
            "Word embedding L1 mismatch: Swift=\(swiftL1), Python=\(pythonL1)")
    }

    func testEmbeddingWithLayerNorm() async throws {
        try skipIfMLXUnavailable()

        guard let fixture = loadNPZFixture("embeddings") else {
            throw XCTSkip("Embedding fixtures not found")
        }

        let model = try await GLiNER2.fromPretrained(Self.weightsPath)

        let inputIds = fixture["input_ids"]!
        let pythonAfterLN = fixture["after_layernorm"]!
        let pythonFullEmb = fixture["full_embeddings"]!

        // Run full embeddings forward pass (includes LayerNorm)
        let swiftFullEmb = model.model.encoder.embeddings(inputIds)
        MLX.eval(swiftFullEmb)

        let swiftL1 = Float(MLX.sum(MLX.abs(swiftFullEmb)).item(Float32.self))
        let pythonL1 = Float(MLX.sum(MLX.abs(pythonFullEmb)).item(Float32.self))

        print("Full embeddings (with LayerNorm) L1:")
        print("  Python: \(pythonL1)")
        print("  Swift:  \(swiftL1)")
        print("  Diff:   \(abs(swiftL1 - pythonL1))")

        XCTAssertEqual(swiftL1, pythonL1, accuracy: l1Tolerance,
            "Full embeddings L1 mismatch")
    }

    // MARK: - Test 2: Position Bucket Computation

    func testPositionBucketsSeq5() throws {
        guard let fixture = loadNPZFixture("position_buckets_seq5") else {
            throw XCTSkip("Position bucket fixtures not found")
        }

        let pythonBuckets = fixture["position_matrix"]!
        let swiftBuckets = makeLogBucketPosition(seqLen: 5, positionBuckets: 256, maxPosition: 512)
        MLX.eval(swiftBuckets)

        // Compare exact values
        let diff = MLX.sum(MLX.abs(swiftBuckets - pythonBuckets))
        MLX.eval(diff)
        let diffVal = Int(diff.item(Int32.self))

        print("Position buckets (seq=5):")
        print("  Python:\n\(pythonBuckets)")
        print("  Swift:\n\(swiftBuckets)")
        print("  Total diff: \(diffVal)")

        XCTAssertEqual(diffVal, 0, "Position buckets should match exactly")
    }

    func testPositionBucketsSeq18() throws {
        guard let fixture = loadNPZFixture("position_buckets_seq18") else {
            throw XCTSkip("Position bucket fixtures not found")
        }

        let pythonBuckets = fixture["position_matrix"]!
        let swiftBuckets = makeLogBucketPosition(seqLen: 18, positionBuckets: 256, maxPosition: 512)
        MLX.eval(swiftBuckets)

        let diff = MLX.sum(MLX.abs(swiftBuckets - pythonBuckets))
        MLX.eval(diff)
        let diffVal = Int(diff.item(Int32.self))

        print("Position buckets (seq=18):")
        print("  Range: [\(MLX.min(swiftBuckets).item(Int32.self)), \(MLX.max(swiftBuckets).item(Int32.self))]")
        print("  Total diff: \(diffVal)")

        XCTAssertEqual(diffVal, 0, "Position buckets should match exactly")
    }

    // MARK: - Test 3: Attention Components

    func testAttentionQKVProjections() async throws {
        try skipIfMLXUnavailable()

        guard let fixture = loadNPZFixture("attention_layer0") else {
            throw XCTSkip("Attention fixtures not found")
        }

        let model = try await GLiNER2.fromPretrained(Self.weightsPath)
        let layer0 = model.model.encoder.layers[0]

        // Get input hidden states
        let inputHidden = fixture["input_hidden"]!

        // Compute Q, K, V with Swift
        let swiftQuery = layer0.attention.queryProj(inputHidden)
        let swiftKey = layer0.attention.keyProj(inputHidden)
        let swiftValue = layer0.attention.valueProj(inputHidden)
        MLX.eval(swiftQuery, swiftKey, swiftValue)

        // Compare with Python
        let pythonQuery = fixture["query"]!
        let pythonKey = fixture["key"]!
        let pythonValue = fixture["value"]!

        let querySwiftL1 = Float(MLX.sum(MLX.abs(swiftQuery)).item(Float32.self))
        let queryPythonL1 = Float(MLX.sum(MLX.abs(pythonQuery)).item(Float32.self))

        let keySwiftL1 = Float(MLX.sum(MLX.abs(swiftKey)).item(Float32.self))
        let keyPythonL1 = Float(MLX.sum(MLX.abs(pythonKey)).item(Float32.self))

        let valueSwiftL1 = Float(MLX.sum(MLX.abs(swiftValue)).item(Float32.self))
        let valuePythonL1 = Float(MLX.sum(MLX.abs(pythonValue)).item(Float32.self))

        print("Q/K/V Projections L1:")
        print("  Query:  Python=\(queryPythonL1), Swift=\(querySwiftL1), diff=\(abs(querySwiftL1-queryPythonL1))")
        print("  Key:    Python=\(keyPythonL1), Swift=\(keySwiftL1), diff=\(abs(keySwiftL1-keyPythonL1))")
        print("  Value:  Python=\(valuePythonL1), Swift=\(valueSwiftL1), diff=\(abs(valueSwiftL1-valuePythonL1))")

        XCTAssertEqual(querySwiftL1, queryPythonL1, accuracy: l1Tolerance, "Query projection mismatch")
        XCTAssertEqual(keySwiftL1, keyPythonL1, accuracy: l1Tolerance, "Key projection mismatch")
        XCTAssertEqual(valueSwiftL1, valuePythonL1, accuracy: l1Tolerance, "Value projection mismatch")
    }

    func testC2CAttention() async throws {
        try skipIfMLXUnavailable()

        guard let fixture = loadNPZFixture("attention_layer0") else {
            throw XCTSkip("Attention fixtures not found")
        }

        let pythonC2C = fixture["c2c_scores"]!
        let pythonC2CL1 = Float(MLX.sum(MLX.abs(pythonC2C)).item(Float32.self))

        // Compute c2c from query_heads and key_heads
        let queryHeads = fixture["query_heads"]!
        let keyHeads = fixture["key_heads"]!

        let swiftC2C = MLX.matmul(queryHeads, keyHeads.transposed(0, 1, 3, 2))
        MLX.eval(swiftC2C)

        let swiftC2CL1 = Float(MLX.sum(MLX.abs(swiftC2C)).item(Float32.self))

        print("C2C Attention L1:")
        print("  Python: \(pythonC2CL1)")
        print("  Swift:  \(swiftC2CL1)")
        print("  Diff:   \(abs(swiftC2CL1 - pythonC2CL1))")

        XCTAssertEqual(swiftC2CL1, pythonC2CL1, accuracy: l1Tolerance, "C2C attention mismatch")
    }

    // MARK: - Test 4: Single Encoder Layer

    func testEncoderLayer0Output() async throws {
        try skipIfMLXUnavailable()

        guard let fixture = loadNPZFixture("encoder_layers") else {
            throw XCTSkip("Encoder layer fixtures not found")
        }

        let model = try await GLiNER2.fromPretrained(Self.weightsPath)

        // Get embeddings input
        let embeddings = fixture["embeddings"]!
        let relEmb = fixture["rel_embeddings"]!
        let pythonLayer0Output = fixture["layer0_output"]!

        // Run layer 0
        let layer0 = model.model.encoder.layers[0]

        // Create dummy attention mask (all ones = no masking)
        let seqLen = embeddings.dim(1)
        let attentionMask = MLXArray.zeros([1, 1, 1, seqLen])

        let swiftLayer0Output = layer0(embeddings, relEmbeddings: relEmb, attentionMask: attentionMask)
        MLX.eval(swiftLayer0Output)

        let swiftL1 = Float(MLX.sum(MLX.abs(swiftLayer0Output)).item(Float32.self))
        let pythonL1 = Float(MLX.sum(MLX.abs(pythonLayer0Output)).item(Float32.self))

        print("Encoder Layer 0 Output L1:")
        print("  Python: \(pythonL1)")
        print("  Swift:  \(swiftL1)")
        print("  Diff:   \(abs(swiftL1 - pythonL1))")
        print("  Diff %: \(abs(swiftL1 - pythonL1) / pythonL1 * 100)%")

        // Allow for some accumulated error in layer computation
        let layerTolerance: Float = pythonL1 * 0.01  // 1% tolerance
        XCTAssertEqual(swiftL1, pythonL1, accuracy: layerTolerance,
            "Layer 0 output L1 mismatch: Swift=\(swiftL1), Python=\(pythonL1)")
    }

    // MARK: - Test 5: Full Encoder

    func testFullEncoderOutput() async throws {
        try skipIfMLXUnavailable()

        guard let fixture = loadNPZFixture("full_encoder") else {
            throw XCTSkip("Full encoder fixtures not found")
        }

        let model = try await GLiNER2.fromPretrained(Self.weightsPath)

        // Get input IDs
        let inputIds = fixture["input_ids"]!
        let pythonHidden = fixture["hidden_states"]!

        // Run full encoder
        let attentionMask = MLXArray.ones([1, inputIds.dim(1)], dtype: .int32)
        let encoderOutput = model.model.encode(inputIds, attentionMask: attentionMask)
        let swiftHidden = encoderOutput.lastHiddenState
        MLX.eval(swiftHidden)

        let swiftL1 = Float(MLX.sum(MLX.abs(swiftHidden)).item(Float32.self))
        let pythonL1 = Float(MLX.sum(MLX.abs(pythonHidden)).item(Float32.self))

        print("Full Encoder Output L1:")
        print("  Python: \(pythonL1)")
        print("  Swift:  \(swiftL1)")
        print("  Diff:   \(abs(swiftL1 - pythonL1))")
        print("  Diff %: \(abs(swiftL1 - pythonL1) / pythonL1 * 100)%")

        // Per-position comparison
        print("\nPer-position L1 comparison:")
        for pos in [0, 1, 10, 11, 17] {
            let swiftPosL1 = Float(MLX.sum(MLX.abs(swiftHidden[0, pos])).item(Float32.self))
            let pythonPosL1 = Float(MLX.sum(MLX.abs(pythonHidden[0, pos])).item(Float32.self))
            let diff = abs(swiftPosL1 - pythonPosL1)
            let pct = diff / pythonPosL1 * 100
            print("  Position \(pos): Python=\(String(format: "%.2f", pythonPosL1)), Swift=\(String(format: "%.2f", swiftPosL1)), diff=\(String(format: "%.2f", diff)) (\(String(format: "%.1f", pct))%)")
        }

        // First 10 values at position 0
        print("\nFirst 10 values at position 0:")
        let swiftPos0 = swiftHidden[0, 0, 0..<10]
        let pythonPos0 = pythonHidden[0, 0, 0..<10]
        MLX.eval(swiftPos0, pythonPos0)
        print("  Python: \(pythonPos0)")
        print("  Swift:  \(swiftPos0)")

        // Calculate max difference
        let maxDiff = MLX.max(MLX.abs(swiftHidden - pythonHidden))
        MLX.eval(maxDiff)
        print("\nMax element-wise diff: \(maxDiff.item(Float.self))")

        // Allow for accumulated error across 12 layers
        let encoderTolerance: Float = pythonL1 * 0.05  // 5% tolerance
        XCTAssertEqual(swiftL1, pythonL1, accuracy: encoderTolerance,
            "Full encoder output L1 mismatch: Swift=\(swiftL1), Python=\(pythonL1)")
    }

    // MARK: - Test 6: Weight Loading Verification

    func testWeightL1SumsMatchPython() async throws {
        try skipIfMLXUnavailable()

        guard let data = try? Data(contentsOf: Self.fixturesPath.appendingPathComponent("weight_reference.json")),
              let reference = try? JSONSerialization.jsonObject(with: data) as? [String: [String: Any]] else {
            throw XCTSkip("Weight reference not found. Run: python scripts/verify_all_weights.py")
        }

        let model = try await GLiNER2.fromPretrained(Self.weightsPath)

        // Critical weights to verify
        let criticalWeights: [(String, () -> MLXArray)] = [
            ("encoder.embeddings.word_embeddings.weight", { model.model.encoder.embeddings.wordEmbeddings.weight }),
            ("encoder.encoder.rel_embeddings.weight", { model.model.encoder.relEmbeddings }),
            ("encoder.encoder.layer.0.attention.self.query_proj.weight", { model.model.encoder.layers[0].attention.queryProj.weight }),
            ("encoder.encoder.layer.0.attention.self.key_proj.weight", { model.model.encoder.layers[0].attention.keyProj.weight }),
            ("encoder.encoder.layer.0.attention.self.value_proj.weight", { model.model.encoder.layers[0].attention.valueProj.weight }),
        ]

        var allMatch = true
        print("Weight L1 Sum Verification:")
        print("-" * 60)

        for (key, getWeight) in criticalWeights {
            guard let refInfo = reference[key],
                  let pythonL1 = refInfo["l1_sum"] as? Double else {
                print("  \(key): REFERENCE NOT FOUND")
                continue
            }

            let weight = getWeight()
            MLX.eval(weight)
            let swiftL1 = Double(MLX.sum(MLX.abs(weight)).item(Float32.self))

            let diff = abs(swiftL1 - pythonL1)
            let pct = diff / pythonL1 * 100

            let status = pct < 0.01 ? "OK" : "MISMATCH"
            print("  \(key)")
            print("    Python: \(String(format: "%.4f", pythonL1))")
            print("    Swift:  \(String(format: "%.4f", swiftL1))")
            print("    Diff:   \(String(format: "%.4f", diff)) (\(String(format: "%.4f", pct))%) - \(status)")

            if pct >= 0.01 {
                allMatch = false
            }
        }

        print("-" * 60)
        XCTAssertTrue(allMatch, "Some weights don't match Python L1 sums")
    }

    // MARK: - Helper Functions

    /// Load fixture file (SafeTensors format)
    private func loadNPZFixture(_ name: String) -> [String: MLXArray]? {
        // Try SafeTensors first (converted from NPZ)
        let safetensorsUrl = Self.fixturesPath.appendingPathComponent("\(name).safetensors")
        if FileManager.default.fileExists(atPath: safetensorsUrl.path) {
            do {
                let arrays = try MLX.loadArrays(url: safetensorsUrl)
                return arrays
            } catch {
                print("Failed to load SafeTensors fixture \(name): \(error)")
                return nil
            }
        }

        // Fall back to NPZ (may not work with MLX)
        let npzUrl = Self.fixturesPath.appendingPathComponent("\(name).npz")
        guard FileManager.default.fileExists(atPath: npzUrl.path) else {
            print("Fixture not found: \(name).safetensors or \(name).npz")
            return nil
        }

        do {
            let arrays = try MLX.loadArrays(url: npzUrl)
            return arrays
        } catch {
            print("Failed to load NPZ fixture \(name): \(error)")
            print("Run: python scripts/convert_npz_to_safetensors.py to convert")
            return nil
        }
    }
}

// MARK: - String repeat operator

private func * (string: String, count: Int) -> String {
    return String(repeating: string, count: count)
}
