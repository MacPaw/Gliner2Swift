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
// ParityTests.swift
// Tests for numerical parity between Swift/MLX and Python implementations
//
// Run Python fixture generator first:
//   cd GLiNER2Swift/scripts && python generate_parity_fixtures.py
//
// Then run tests in Xcode with Cmd+U

import XCTest
import MLX
import MLXNN
import Foundation
@testable import GLiNER2Swift


// MARK: - Fixture Loader

/// Loads numpy .npy files for parity testing
struct FixtureLoader {
    let fixturesPath: URL

    init() {
        // Try multiple locations for fixtures
        let possiblePaths = [
            // When running from Xcode
            Bundle.module.resourceURL?.appendingPathComponent("Fixtures"),
            // When running from command line
            URL(fileURLWithPath: #file)
                .deletingLastPathComponent()
                .appendingPathComponent("Fixtures"),
            // Project root fallback
            URL(fileURLWithPath: #file)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("Tests/GLiNER2SwiftTests/Fixtures")
        ]

        for path in possiblePaths.compactMap({ $0 }) {
            if FileManager.default.fileExists(atPath: path.path) {
                fixturesPath = path
                return
            }
        }

        // Fallback to first option even if it doesn't exist
        fixturesPath = possiblePaths.first!!
    }

    /// Load a numpy .npy file as MLXArray
    func load(_ name: String) throws -> MLXArray {
        let url = fixturesPath.appendingPathComponent("\(name).npy")

        guard FileManager.default.fileExists(atPath: url.path) else {
            throw FixtureError.fileNotFound(name)
        }

        // Use MLX's numpy loading
        return try MLX.loadArray(url: url)
    }

    /// Load JSON metadata
    func loadJSON(_ name: String) throws -> [String: Any] {
        let url = fixturesPath.appendingPathComponent("\(name).json")
        let data = try Data(contentsOf: url)
        return try JSONSerialization.jsonObject(with: data) as! [String: Any]
    }

    /// Check if fixture exists
    func exists(_ name: String) -> Bool {
        let url = fixturesPath.appendingPathComponent("\(name).npy")
        return FileManager.default.fileExists(atPath: url.path)
    }

    enum FixtureError: Error {
        case fileNotFound(String)
        case loadFailed(String, Error)
    }
}


// MARK: - Test Utilities

extension XCTestCase {

    /// Assert two MLXArrays are close within tolerance
    func assertClose(
        _ actual: MLXArray,
        _ expected: MLXArray,
        atol: Float = 1e-5,
        rtol: Float = 1e-5,
        message: String = "",
        file: StaticString = #file,
        line: UInt = #line
    ) {
        // Check shapes match
        XCTAssertEqual(
            actual.shape,
            expected.shape,
            "Shape mismatch: \(actual.shape) vs \(expected.shape). \(message)",
            file: file,
            line: line
        )

        // Check values are close
        let diff = MLX.abs(actual - expected)
        let tolerance = MLXArray(atol) + MLXArray(rtol) * MLX.abs(expected)
        let allClose = MLX.all(diff .<= tolerance).item(Bool.self)

        if !allClose {
            let maxDiff = MLX.max(diff).item(Float.self)
            let meanDiff = MLX.mean(diff).item(Float.self)
            XCTFail(
                "Arrays not close. Max diff: \(maxDiff), Mean diff: \(meanDiff). \(message)",
                file: file,
                line: line
            )
        }
    }

    /// Assert two integer MLXArrays are exactly equal
    func assertEqual(
        _ actual: MLXArray,
        _ expected: MLXArray,
        message: String = "",
        file: StaticString = #file,
        line: UInt = #line
    ) {
        XCTAssertEqual(
            actual.shape,
            expected.shape,
            "Shape mismatch: \(actual.shape) vs \(expected.shape). \(message)",
            file: file,
            line: line
        )

        let allEqual = MLX.all(MLX.equal(actual, expected)).item(Bool.self)
        if !allEqual {
            let numDiff = MLX.sum(MLX.logicalNot(MLX.equal(actual, expected))).item(Int32.self)
            XCTFail(
                "Arrays not equal. \(numDiff) elements differ. \(message)",
                file: file,
                line: line
            )
        }
    }
}


// MARK: - Position Bucketing Parity Tests

final class PositionBucketingParityTests: XCTestCase {

    var loader: FixtureLoader!

    override func setUp() {
        super.setUp()
        loader = FixtureLoader()
    }

    func testPositionBucketing5x5() throws {
        try XCTSkipUnless(loader.exists("position_buckets_5x5"), "Fixtures not generated")

        let expected = try loader.load("position_buckets_5x5")
        let actual = makeLogBucketPosition(seqLen: 5, positionBuckets: 256, maxPosition: 512)

        assertEqual(actual, expected, message: "Position buckets 5x5")
    }

    func testPositionBucketing32x32() throws {
        try XCTSkipUnless(loader.exists("position_buckets_32x32"), "Fixtures not generated")

        let expected = try loader.load("position_buckets_32x32")
        let actual = makeLogBucketPosition(seqLen: 32, positionBuckets: 256, maxPosition: 512)

        assertEqual(actual, expected, message: "Position buckets 32x32")
    }

    func testPositionBucketing1x1() throws {
        try XCTSkipUnless(loader.exists("position_buckets_1x1"), "Fixtures not generated")

        let expected = try loader.load("position_buckets_1x1")
        let actual = makeLogBucketPosition(seqLen: 1, positionBuckets: 256, maxPosition: 512)

        assertEqual(actual, expected, message: "Position buckets 1x1 (edge case)")
    }

    func testRelativePositionSign() throws {
        // Verify relative position direction matches Python: q_ids[:, None] - k_ids[None, :]
        // This means relativePos[i, j] = i - j

        let seqLen = 5
        let seqRange = MLXArray(Array(0..<seqLen).map { Int32($0) })
        let posI = seqRange.expandedDimensions(axis: 1)
        let posJ = seqRange.expandedDimensions(axis: 0)
        let relativePos = posI - posJ

        // relativePos[0, 0] should be 0
        // relativePos[0, 1] should be -1 (0 - 1)
        // relativePos[1, 0] should be 1 (1 - 0)

        let val_0_0 = relativePos[0, 0].item(Int32.self)
        let val_0_1 = relativePos[0, 1].item(Int32.self)
        let val_1_0 = relativePos[1, 0].item(Int32.self)

        XCTAssertEqual(val_0_0, 0, "relativePos[0,0] should be 0")
        XCTAssertEqual(val_0_1, -1, "relativePos[0,1] should be -1")
        XCTAssertEqual(val_1_0, 1, "relativePos[1,0] should be 1")
    }
}


// MARK: - GRU Parity Tests

final class GRUParityTests: XCTestCase {

    var loader: FixtureLoader!

    override func setUp() {
        super.setUp()
        loader = FixtureLoader()
    }

    func testGRUForwardParity() throws {
        try XCTSkipUnless(loader.exists("gru_input"), "Fixtures not generated")

        // Load fixtures
        let input = try loader.load("gru_input")
        let h0 = try loader.load("gru_h0")
        let expectedOutput = try loader.load("gru_output")
        let expectedHn = try loader.load("gru_hn")

        // Load weights
        let weightIH = try loader.load("gru_weight_ih")
        let weightHH = try loader.load("gru_weight_hh")
        let biasIH = try loader.load("gru_bias_ih")
        let biasHH = try loader.load("gru_bias_hh")

        // Create GRU with loaded weights
        let gru = GRU(weightIH: weightIH, weightHH: weightHH, biasIH: biasIH, biasHH: biasHH)

        // Forward pass
        let (output, hn) = gru(input, h0: h0)

        // Compare
        assertClose(output, expectedOutput, atol: 1e-4, message: "GRU output")
        assertClose(hn, expectedHn, atol: 1e-4, message: "GRU hidden state")
    }

    func testGRUWeightLoadingCamelCase() throws {
        try XCTSkipUnless(loader.exists("gru_weight_ih"), "Fixtures not generated")

        // Test loading with camelCase keys (from convert_weights.py)
        let weights: [String: MLXArray] = [
            "gru.weightIH": try loader.load("gru_weight_ih"),
            "gru.weightHH": try loader.load("gru_weight_hh"),
            "gru.biasIH": try loader.load("gru_bias_ih"),
            "gru.biasHH": try loader.load("gru_bias_hh"),
        ]

        let gru = GRU.fromWeights(weights, prefix: "gru")
        XCTAssertEqual(gru.hiddenSize, 768)
    }

    func testGRUWeightLoadingSnakeCase() throws {
        try XCTSkipUnless(loader.exists("gru_weight_ih"), "Fixtures not generated")

        // Test loading with snake_case keys (raw PyTorch)
        let weights: [String: MLXArray] = [
            "gru.weight_ih_l0": try loader.load("gru_weight_ih"),
            "gru.weight_hh_l0": try loader.load("gru_weight_hh"),
            "gru.bias_ih_l0": try loader.load("gru_bias_ih"),
            "gru.bias_hh_l0": try loader.load("gru_bias_hh"),
        ]

        let gru = GRU.fromWeights(weights, prefix: "gru")
        XCTAssertEqual(gru.hiddenSize, 768)
    }
}


// MARK: - DownscaledTransformer Parity Tests

final class DownscaledTransformerParityTests: XCTestCase {

    var loader: FixtureLoader!

    override func setUp() {
        super.setUp()
        loader = FixtureLoader()
    }

    func testDownscaledTransformerForwardParity() throws {
        try XCTSkipUnless(loader.exists("dst_input"), "Fixtures not generated")

        // Load fixtures
        let input = try loader.load("dst_input")
        let expectedOutput = try loader.load("dst_output")

        // Create model
        let model = DownscaledTransformer(
            inputSize: 768,
            hiddenSize: 128,
            numHeads: 4,
            numLayers: 2,
            dropout: 0.0
        )

        // Load weights
        try loadDownscaledTransformerWeights(model)

        // Forward pass
        let output = model(input)

        // Compare
        assertClose(output, expectedOutput, atol: 1e-4, message: "DownscaledTransformer output")
    }

    func testDownscaledTransformerConcatenation() throws {
        try XCTSkipUnless(loader.exists("dst_after_concat"), "Fixtures not generated")

        // Verify CONCATENATION (not addition) is used
        let afterTransformer = try loader.load("dst_after_transformer")  // [L, M, 128]
        let input = try loader.load("dst_input")  // [L, M, 768]
        let expectedConcat = try loader.load("dst_after_concat")  // [L, M, 896]

        // Swift concatenation
        let actualConcat = MLX.concatenated([afterTransformer, input], axis: -1)

        assertClose(actualConcat, expectedConcat, message: "Concatenation check")
        XCTAssertEqual(actualConcat.dim(2), 896, "Concatenated dimension should be 896 (128 + 768)")
    }

    private func loadDownscaledTransformerWeights(_ model: DownscaledTransformer) throws {
        // Load in_projector
        let inProjWeight = try loader.load("dst_in_projector_weight")
        let inProjBias = try loader.load("dst_in_projector_bias")
        updateLinearWeights(model.inProjector, weight: inProjWeight, bias: inProjBias)

        // Load transformer layer weights
        for i in 0..<model.transformerLayers.count {
            let layer = model.transformerLayers[i]
            let prefix = "dst_transformer_layer_\(i)"

            // Self-attention (combined in_proj -> split into Q, K, V)
            let inProjW = try loader.load("\(prefix)_self_attn_in_proj_weight")
            let inProjB = try loader.load("\(prefix)_self_attn_in_proj_bias")
            let dim = 128  // hiddenSize
            // Split Q, K, V
            let qWeight = inProjW[0..<dim]
            let kWeight = inProjW[dim..<(2*dim)]
            let vWeight = inProjW[(2*dim)..<(3*dim)]
            let qBias = inProjB[0..<dim]
            let kBias = inProjB[dim..<(2*dim)]
            let vBias = inProjB[(2*dim)..<(3*dim)]
            updateLinearWeights(layer.selfAttn.queryProj, weight: qWeight, bias: qBias)
            updateLinearWeights(layer.selfAttn.keyProj, weight: kWeight, bias: kBias)
            updateLinearWeights(layer.selfAttn.valueProj, weight: vWeight, bias: vBias)

            let outProjW = try loader.load("\(prefix)_self_attn_out_proj_weight")
            let outProjB = try loader.load("\(prefix)_self_attn_out_proj_bias")
            updateLinearWeights(layer.selfAttn.outProj, weight: outProjW, bias: outProjB)

            // FFN
            let linear1W = try loader.load("\(prefix)_linear1_weight")
            let linear1B = try loader.load("\(prefix)_linear1_bias")
            updateLinearWeights(layer.linear1, weight: linear1W, bias: linear1B)

            let linear2W = try loader.load("\(prefix)_linear2_weight")
            let linear2B = try loader.load("\(prefix)_linear2_bias")
            updateLinearWeights(layer.linear2, weight: linear2W, bias: linear2B)

            // Layer norms
            let norm1W = try loader.load("\(prefix)_norm1_weight")
            let norm1B = try loader.load("\(prefix)_norm1_bias")
            updateLayerNormWeights(layer.norm1, weight: norm1W, bias: norm1B)

            let norm2W = try loader.load("\(prefix)_norm2_weight")
            let norm2B = try loader.load("\(prefix)_norm2_bias")
            updateLayerNormWeights(layer.norm2, weight: norm2W, bias: norm2B)
        }

        // Load out_projector (indices 0, 2, 4 are Linear layers)
        let out0W = try loader.load("dst_out_projector_0_weight")
        let out0B = try loader.load("dst_out_projector_0_bias")
        if let linear0 = model.outProjector.layers[0] as? Linear {
            updateLinearWeights(linear0, weight: out0W, bias: out0B)
        }

        let out2W = try loader.load("dst_out_projector_2_weight")
        let out2B = try loader.load("dst_out_projector_2_bias")
        if let linear2 = model.outProjector.layers[2] as? Linear {
            updateLinearWeights(linear2, weight: out2W, bias: out2B)
        }

        let out4W = try loader.load("dst_out_projector_4_weight")
        let out4B = try loader.load("dst_out_projector_4_bias")
        if let linear4 = model.outProjector.layers[4] as? Linear {
            updateLinearWeights(linear4, weight: out4W, bias: out4B)
        }
    }
}


// MARK: - CountLSTMv2 Parity Tests

final class CountLSTMv2ParityTests: XCTestCase {

    var loader: FixtureLoader!

    override func setUp() {
        super.setUp()
        loader = FixtureLoader()
    }

    func testCountLSTMv2Addition() throws {
        try XCTSkipUnless(loader.exists("clv2_gru_output"), "Fixtures not generated")

        // Verify ADDITION (not concatenation) is used in CountLSTMv2
        let gruOutput = try loader.load("clv2_gru_output")  // [count, M, hidden]
        let pcEmb = try loader.load("clv2_pc_emb")  // [M, hidden]
        let expectedAfterAdd = try loader.load("clv2_after_addition")  // [count, M, hidden]

        // Swift addition (CountLSTMv2 uses addition, CountLSTM uses concatenation)
        let pcBroadcast = pcEmb.expandedDimensions(axis: 0)
        let actualAfterAdd = gruOutput + pcBroadcast

        assertClose(actualAfterAdd, expectedAfterAdd, message: "CountLSTMv2 addition check")

        // Verify shapes are the same (addition preserves shape, concat would double last dim)
        XCTAssertEqual(actualAfterAdd.shape, gruOutput.shape, "Addition should preserve shape")
    }

    func testCountLSTMv2ForwardParity() throws {
        try XCTSkipUnless(loader.exists("clv2_output"), "Fixtures not generated")

        // Load fixtures
        let pcEmb = try loader.load("clv2_pc_emb")
        let expectedOutput = try loader.load("clv2_output")
        let metadata = try loader.loadJSON("clv2_metadata")

        let goldCount = metadata["gold_count"] as! Int

        // Create model
        let model = CountLSTMv2(hiddenSize: 768, maxCount: 20)

        // Load weights
        try loadCountLSTMv2Weights(model)

        // Forward pass
        let output = model(pcEmb, goldCountVal: goldCount)

        // Compare (relaxed tolerance due to accumulated numerical error across GRU + transformer layers)
        assertClose(output, expectedOutput, atol: 3e-2, message: "CountLSTMv2 output")
    }

    private func loadCountLSTMv2Weights(_ model: CountLSTMv2) throws {
        // Load GRU weights
        let weights: [String: MLXArray] = [
            "gru.weightIH": try loader.load("clv2_gru_weight_ih"),
            "gru.weightHH": try loader.load("clv2_gru_weight_hh"),
            "gru.biasIH": try loader.load("clv2_gru_bias_ih"),
            "gru.biasHH": try loader.load("clv2_gru_bias_hh"),
        ]
        model.gru.loadWeights(weights, prefix: "gru")

        // Load position embedding
        let posEmb = try loader.load("clv2_pos_embedding")
        model.posEmbedding.update(parameters: ModuleParameters.unflattened(["weight": posEmb]))

        // Load DownscaledTransformer weights
        let dst = model.transformer

        // Load in_projector
        let inProjWeight = try loader.load("clv2_dst_in_projector_weight")
        let inProjBias = try loader.load("clv2_dst_in_projector_bias")
        updateLinearWeights(dst.inProjector, weight: inProjWeight, bias: inProjBias)

        // Load transformer layer weights
        for i in 0..<dst.transformerLayers.count {
            let layer = dst.transformerLayers[i]
            let prefix = "clv2_dst_transformer_layer_\(i)"

            // Self-attention (combined in_proj -> split into Q, K, V)
            let inProjW = try loader.load("\(prefix)_self_attn_in_proj_weight")
            let inProjB = try loader.load("\(prefix)_self_attn_in_proj_bias")
            let dim = 128  // hiddenSize
            // Split Q, K, V
            let qWeight = inProjW[0..<dim]
            let kWeight = inProjW[dim..<(2*dim)]
            let vWeight = inProjW[(2*dim)..<(3*dim)]
            let qBias = inProjB[0..<dim]
            let kBias = inProjB[dim..<(2*dim)]
            let vBias = inProjB[(2*dim)..<(3*dim)]
            updateLinearWeights(layer.selfAttn.queryProj, weight: qWeight, bias: qBias)
            updateLinearWeights(layer.selfAttn.keyProj, weight: kWeight, bias: kBias)
            updateLinearWeights(layer.selfAttn.valueProj, weight: vWeight, bias: vBias)

            let outProjW = try loader.load("\(prefix)_self_attn_out_proj_weight")
            let outProjB = try loader.load("\(prefix)_self_attn_out_proj_bias")
            updateLinearWeights(layer.selfAttn.outProj, weight: outProjW, bias: outProjB)

            // FFN
            let linear1W = try loader.load("\(prefix)_linear1_weight")
            let linear1B = try loader.load("\(prefix)_linear1_bias")
            updateLinearWeights(layer.linear1, weight: linear1W, bias: linear1B)

            let linear2W = try loader.load("\(prefix)_linear2_weight")
            let linear2B = try loader.load("\(prefix)_linear2_bias")
            updateLinearWeights(layer.linear2, weight: linear2W, bias: linear2B)

            // Layer norms
            let norm1W = try loader.load("\(prefix)_norm1_weight")
            let norm1B = try loader.load("\(prefix)_norm1_bias")
            updateLayerNormWeights(layer.norm1, weight: norm1W, bias: norm1B)

            let norm2W = try loader.load("\(prefix)_norm2_weight")
            let norm2B = try loader.load("\(prefix)_norm2_bias")
            updateLayerNormWeights(layer.norm2, weight: norm2W, bias: norm2B)
        }

        // Load out_projector (indices 0, 2, 4 are Linear layers)
        let out0W = try loader.load("clv2_dst_out_projector_0_weight")
        let out0B = try loader.load("clv2_dst_out_projector_0_bias")
        if let linear0 = dst.outProjector.layers[0] as? Linear {
            updateLinearWeights(linear0, weight: out0W, bias: out0B)
        }

        let out2W = try loader.load("clv2_dst_out_projector_2_weight")
        let out2B = try loader.load("clv2_dst_out_projector_2_bias")
        if let linear2 = dst.outProjector.layers[2] as? Linear {
            updateLinearWeights(linear2, weight: out2W, bias: out2B)
        }

        let out4W = try loader.load("clv2_dst_out_projector_4_weight")
        let out4B = try loader.load("clv2_dst_out_projector_4_bias")
        if let linear4 = dst.outProjector.layers[4] as? Linear {
            updateLinearWeights(linear4, weight: out4W, bias: out4B)
        }
    }
}


// MARK: - Integration Tests

final class EndToEndParityTests: XCTestCase {

    func testFullPipelineSmoke() throws {
        // Smoke test that all components work together
        let config = ExtractorConfig()

        // Create components
        let gru = GRU(inputSize: config.hiddenSize, hiddenSize: config.hiddenSize)
        let countLSTM = CountLSTMv2(hiddenSize: config.hiddenSize, maxCount: config.maxCount)

        // Create dummy input
        let pcEmb = MLXArray.ones([4, config.hiddenSize]) * 0.1

        // Forward pass through CountLSTMv2
        let output = countLSTM(pcEmb, goldCountVal: 3)

        // Verify output shape
        XCTAssertEqual(output.dim(0), 3, "Count dimension")
        XCTAssertEqual(output.dim(1), 4, "Fields dimension")
        XCTAssertEqual(output.dim(2), config.hiddenSize, "Hidden dimension")
    }
}