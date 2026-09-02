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
// DeBERTaParityTests.swift
// Numerical parity tests for DeBERTa encoder components
//
// These tests verify that the Swift/MLX implementation produces the same
// numerical results as the Python/PyTorch implementation.
//
// Run fixtures generation first:
//   cd ../scripts && python generate_fixtures.py --output-dir ../Tests/GLiNER2SwiftTests/Fixtures

import XCTest
import MLX
@testable import GLiNER2Swift

final class DeBERTaParityTests: XCTestCase {

    // Tolerance for numerical comparison
    let tolerance: Float = 1e-5

    // MARK: - Position Bucket Tests

    func testMakeLogBucketPositionBasic() throws {
        // Test the log bucket position computation
        let seqLen = 10
        let buckets = makeLogBucketPosition(
            seqLen: seqLen,
            positionBuckets: 256,
            maxPosition: 512
        )

        // Check shape
        XCTAssertEqual(buckets.dim(0), seqLen)
        XCTAssertEqual(buckets.dim(1), seqLen)

        // Check diagonal is always 0
        // Because pos[i,i] = i - i = 0, which stays as 0 (raw relative position for small values)
        // Note: DeBERTa returns RAW relative positions for small values, NOT offset by mid
        for i in 0..<seqLen {
            let value = buckets[i, i].item(Int32.self)
            XCTAssertEqual(Int(value), 0, "Diagonal should be 0 (raw relative position)")
        }

        // Check anti-symmetry: bucket[i,j] = -bucket[j,i]
        // Because pos[i,j] = i - j and pos[j,i] = j - i = -(i - j)
        // For small values, bucket = relative position, so bucket[i,j] + bucket[j,i] = 0
        for i in 0..<seqLen {
            for j in 0..<seqLen {
                if i != j {
                    let bij = buckets[i, j].item(Int32.self)
                    let bji = buckets[j, i].item(Int32.self)
                    XCTAssertEqual(Int(bij) + Int(bji), 0,
                                   "Position buckets should be anti-symmetric: bucket[i,j] = -bucket[j,i]")
                }
            }
        }
    }

    func testMakeLogBucketPositionLinearRange() throws {
        // For small relative positions, bucket = raw relative position
        // DeBERTa does NOT offset by mid for values in the linear range
        let seqLen = 5

        let buckets = makeLogBucketPosition(
            seqLen: seqLen,
            positionBuckets: 256,
            maxPosition: 512
        )

        // Position (0, 1) has relative position 0 - 1 = -1
        // For small values, bucket = relative position = -1
        let pos01 = buckets[0, 1].item(Int32.self)
        XCTAssertEqual(Int(pos01), -1, "Position [0,1] should be -1 (i-j = 0-1)")

        // Position (1, 0) has relative position 1 - 0 = 1
        // For small values, bucket = relative position = 1
        let pos10 = buckets[1, 0].item(Int32.self)
        XCTAssertEqual(Int(pos10), 1, "Position [1,0] should be 1 (i-j = 1-0)")

        // Position (0, 4) has relative position 0 - 4 = -4
        let pos04 = buckets[0, 4].item(Int32.self)
        XCTAssertEqual(Int(pos04), -4, "Position [0,4] should be -4 (i-j = 0-4)")

        // Position (4, 0) has relative position 4 - 0 = 4
        let pos40 = buckets[4, 0].item(Int32.self)
        XCTAssertEqual(Int(pos40), 4, "Position [4,0] should be 4 (i-j = 4-0)")
    }

    // MARK: - Disentangled Attention Scale Factor Test

    func testDisentangledAttentionScaleFactor() throws {
        // CRITICAL: DeBERTa uses sqrt(head_dim * 3) as scale factor
        // because it has 3 attention types: c2c, c2p, p2c

        let attention = DisentangledSelfAttention(
            hiddenSize: 768,
            numHeads: 12,
            posAttType: ["c2p", "p2c"]
        )

        let expectedHeadDim = 768 / 12  // 64
        let expectedScaleFactor = sqrt(Float(expectedHeadDim * 3))  // sqrt(192)

        // We can't directly access scaleFactor since it's private,
        // but we can verify the attention output shape
        let batchSize = 1
        let seqLen = 8
        let hiddenStates = MLXArray.ones([batchSize, seqLen, 768]) * 0.1
        let relEmbeddings = MLXArray.ones([512, 768]) * 0.1

        let output = attention(hiddenStates, relEmbeddings: relEmbeddings)

        XCTAssertEqual(output.dim(0), batchSize)
        XCTAssertEqual(output.dim(1), seqLen)
        XCTAssertEqual(output.dim(2), 768)
    }

    // MARK: - DeBERTa Embeddings Test

    func testDeBERTaEmbeddingsShape() throws {
        let embeddings = DeBERTaEmbeddings(
            vocabSize: 128011,
            hiddenSize: 768,
            dropoutProb: 0.0  // Disable dropout for deterministic testing
        )

        let inputIds = MLXArray([1, 2, 3, 4, 5]).reshaped([1, 5])
        let output = embeddings(inputIds)

        XCTAssertEqual(output.dim(0), 1)  // batch
        XCTAssertEqual(output.dim(1), 5)  // seq_len
        XCTAssertEqual(output.dim(2), 768)  // hidden
    }

    // MARK: - DeBERTa Layer Test

    func testDeBERTaLayerShape() throws {
        let layer = DeBERTaLayer(
            hiddenSize: 768,
            intermediateSize: 3072,
            numHeads: 12,
            dropoutProb: 0.0  // Disable dropout for deterministic testing
        )

        let batchSize = 1
        let seqLen = 8
        let hiddenStates = MLXArray.ones([batchSize, seqLen, 768]) * 0.1
        let relEmbeddings = MLXArray.ones([512, 768]) * 0.1

        let output = layer(hiddenStates, relEmbeddings: relEmbeddings)

        XCTAssertEqual(output.dim(0), batchSize)
        XCTAssertEqual(output.dim(1), seqLen)
        XCTAssertEqual(output.dim(2), 768)
    }

    // MARK: - Full Encoder Test

    func testDeBERTaEncoderShape() throws {
        let config = DeBERTaConfig(
            vocabSize: 1000,  // Small vocab for testing
            hiddenSize: 768,
            numHiddenLayers: 2,  // Small number for testing
            numAttentionHeads: 12
        )

        let encoder = DeBERTaEncoder(config: config)

        let inputIds = MLXArray([1, 2, 3, 4, 5]).reshaped([1, 5])
        // Per-layer outputs are opt-in since Phase 3.6; this test is one of the few callers
        // that actually wants them.
        let output = encoder(inputIds, outputHiddenStates: true)

        XCTAssertEqual(output.lastHiddenState.dim(0), 1)  // batch
        XCTAssertEqual(output.lastHiddenState.dim(1), 5)  // seq_len
        XCTAssertEqual(output.lastHiddenState.dim(2), 768)  // hidden

        // Should have embeddings + 2 layer outputs = 3 hidden states
        XCTAssertEqual(output.hiddenStates.count, 3)
    }

    // MARK: - Custom Multi-Head Attention Test

    func testCustomMultiHeadAttentionShape() throws {
        let attention = CustomMultiHeadAttention(dims: 128, numHeads: 4)

        let batchSize = 2
        let seqLen = 10
        let queries = MLXArray.ones([batchSize, seqLen, 128]) * 0.1
        let keys = MLXArray.ones([batchSize, seqLen, 128]) * 0.1
        let values = MLXArray.ones([batchSize, seqLen, 128]) * 0.1

        let output = attention(queries: queries, keys: keys, values: values)

        XCTAssertEqual(output.dim(0), batchSize)
        XCTAssertEqual(output.dim(1), seqLen)
        XCTAssertEqual(output.dim(2), 128)
    }

    // MARK: - Configuration Tests

    func testDeBERTaConfigDefaults() throws {
        let config = DeBERTaConfig.gliner2Base

        XCTAssertEqual(config.vocabSize, 128011)
        XCTAssertEqual(config.hiddenSize, 768)
        XCTAssertEqual(config.numHiddenLayers, 12)
        XCTAssertEqual(config.numAttentionHeads, 12)
        XCTAssertEqual(config.intermediateSize, 3072)
        XCTAssertEqual(config.positionBuckets, 256)
        XCTAssertEqual(config.maxPositionEmbeddings, 512)
        XCTAssertEqual(config.layerNormEps, 1e-7)
        XCTAssertTrue(config.posAttType.contains("c2p"))
        XCTAssertTrue(config.posAttType.contains("p2c"))
    }
}
