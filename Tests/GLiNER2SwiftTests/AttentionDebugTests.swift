// AttentionDebugTests.swift
// Test c2p/p2c attention computation against Python reference values

import XCTest
import MLX
import MLXNN
@testable import GLiNER2Swift

final class AttentionDebugTests: XCTestCase {

    // MARK: - Test Position Bucket Computation

    func testPositionBuckets_SmallSequence() throws {
        // Test with seq_len=5, same as Python debug script
        let buckets = makeLogBucketPosition(
            seqLen: 5,
            positionBuckets: 256,
            maxPosition: 512
        )

        // Python reference:
        // tensor([[ 0, -1, -2, -3, -4],
        //         [ 1,  0, -1, -2, -3],
        //         [ 2,  1,  0, -1, -2],
        //         [ 3,  2,  1,  0, -1],
        //         [ 4,  3,  2,  1,  0]])

        // Flatten and reshape for MLXArray
        let expectedFlat: [Int32] = [
            0, -1, -2, -3, -4,
            1,  0, -1, -2, -3,
            2,  1,  0, -1, -2,
            3,  2,  1,  0, -1,
            4,  3,  2,  1,  0
        ]
        let expected = MLXArray(expectedFlat).reshaped([5, 5])

        print("Swift buckets:")
        print(buckets)
        print("\nExpected:")
        print(expected)

        // Compare
        let diff = MLX.abs(buckets - expected)
        let maxDiff = MLX.max(diff).item(Int32.self)
        XCTAssertEqual(maxDiff, 0, "Position buckets should match Python exactly")
    }

    func testC2PPositionIndices() throws {
        // For seq_len=5, compute c2p position indices
        // Python: c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
        // where att_span = 128

        let seqLen = 5
        let attSpan = Int32(128)

        // Raw relative positions (i - j)
        let seqRange = MLXArray(Array(0..<seqLen).map { Int32($0) })
        let posI = seqRange.expandedDimensions(axis: 1)
        let posJ = seqRange.expandedDimensions(axis: 0)
        let relPos = posI - posJ

        print("Raw relative positions (i - j):")
        print(relPos)

        // c2p indices
        let c2pPos = relPos + MLXArray(attSpan)
        print("\nc2p position indices (rel_pos + 128):")
        print(c2pPos)

        // Python reference:
        // tensor([[128, 127, 126, 125, 124],
        //         [129, 128, 127, 126, 125],
        //         [130, 129, 128, 127, 126],
        //         [131, 130, 129, 128, 127],
        //         [132, 131, 130, 129, 128]])

        let expectedFlat: [Int32] = [
            128, 127, 126, 125, 124,
            129, 128, 127, 126, 125,
            130, 129, 128, 127, 126,
            131, 130, 129, 128, 127,
            132, 131, 130, 129, 128
        ]
        let expected = MLXArray(expectedFlat).reshaped([5, 5])

        let diff = MLX.abs(c2pPos - expected)
        let maxDiff = MLX.max(diff).item(Int32.self)
        XCTAssertEqual(maxDiff, 0, "c2p position indices should match Python exactly")
    }

    func testP2CPositionIndices() throws {
        // For p2c, Python uses the SAME c2p_pos indices (NOT negated!)
        // Then transposes the gathered result

        let seqLen = 5
        let attSpan = Int32(128)

        let seqRange = MLXArray(Array(0..<seqLen).map { Int32($0) })
        let posI = seqRange.expandedDimensions(axis: 1)
        let posJ = seqRange.expandedDimensions(axis: 0)
        let relPos = posI - posJ

        // p2c uses SAME indices as c2p
        let p2cPos = relPos + MLXArray(attSpan)

        // Python reference (should be same as c2p):
        let expectedFlat: [Int32] = [
            128, 127, 126, 125, 124,
            129, 128, 127, 126, 125,
            130, 129, 128, 127, 126,
            131, 130, 129, 128, 127,
            132, 131, 130, 129, 128
        ]
        let expected = MLXArray(expectedFlat).reshaped([5, 5])

        print("p2c position indices (same as c2p, then transpose result):")
        print(p2cPos)

        let diff = MLX.abs(p2cPos - expected)
        let maxDiff = MLX.max(diff).item(Int32.self)
        XCTAssertEqual(maxDiff, 0, "p2c position indices should be same as c2p")
    }

    // MARK: - Test Full Attention with Real Weights

    func testAttentionLayer0_RealWeights() async throws {
        // Load model weights
        let weightsPath = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub/models--fastino--gliner2-base-v1/snapshots")

        let fm = FileManager.default
        guard let snapshots = try? fm.contentsOfDirectory(atPath: weightsPath.path),
              let snapshot = snapshots.first else {
            throw XCTSkip("Model weights not found")
        }

        let modelPath = weightsPath.appendingPathComponent(snapshot)

        // Load safetensors
        let safetensorsPath = modelPath.appendingPathComponent("model.safetensors")
        guard fm.fileExists(atPath: safetensorsPath.path) else {
            throw XCTSkip("model.safetensors not found")
        }

        let weights = try MLX.loadArrays(url: safetensorsPath)

        // Create attention module
        let attention = DisentangledSelfAttention(
            hiddenSize: 768,
            numHeads: 12,
            positionBuckets: 256,
            maxPosition: 512,
            dropoutProb: 0.0,  // Disable dropout for determinism
            posAttType: ["c2p", "p2c"]
        )

        // Load weights
        attention.loadWeights(weights, prefix: "encoder.encoder.layer.0.attention.self")

        // Get rel_embeddings
        guard let relEmbWeight = weights["encoder.encoder.rel_embeddings.weight"] else {
            throw XCTSkip("rel_embeddings not found in weights")
        }

        print("rel_embeddings shape: \(relEmbWeight.shape)")
        print("rel_embeddings L1: \(MLX.sum(MLX.abs(relEmbWeight)).item(Float.self))")

        // Create test input (small sequence)
        let batchSize = 1
        let seqLen = 5
        let hiddenSize = 768

        // Use specific values for reproducibility
        MLX.GPU.set(cacheLimit: 0)

        // Create deterministic hidden states
        let hiddenStates = MLXArray.ones([batchSize, seqLen, hiddenSize]) * 0.1

        print("\nInput hidden states L1: \(MLX.sum(MLX.abs(hiddenStates)).item(Float.self))")

        // Run attention
        let output = attention(hiddenStates, relEmbeddings: relEmbWeight)

        print("Output shape: \(output.shape)")
        print("Output L1: \(MLX.sum(MLX.abs(output)).item(Float.self))")

        // Print first few values
        print("\nOutput first 5 values at position 0:")
        for i in 0..<5 {
            print("  [\(i)]: \(output[0, 0, i].item(Float.self))")
        }
    }

    func testAttentionScoresBreakdown() async throws {
        // Load model weights
        let weightsPath = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub/models--fastino--gliner2-base-v1/snapshots")

        let fm = FileManager.default
        guard let snapshots = try? fm.contentsOfDirectory(atPath: weightsPath.path),
              let snapshot = snapshots.first else {
            throw XCTSkip("Model weights not found")
        }

        let modelPath = weightsPath.appendingPathComponent(snapshot)
        let safetensorsPath = modelPath.appendingPathComponent("model.safetensors")
        guard fm.fileExists(atPath: safetensorsPath.path) else {
            throw XCTSkip("model.safetensors not found")
        }

        let weights = try MLX.loadArrays(url: safetensorsPath)

        // Create attention with debug output
        print("=== ATTENTION COMPUTATION BREAKDOWN ===")

        // Load projections manually
        let queryWeight = weights["encoder.encoder.layer.0.attention.self.query_proj.weight"]!
        let queryBias = weights["encoder.encoder.layer.0.attention.self.query_proj.bias"]!
        let keyWeight = weights["encoder.encoder.layer.0.attention.self.key_proj.weight"]!
        let keyBias = weights["encoder.encoder.layer.0.attention.self.key_proj.bias"]!
        let relEmbWeight = weights["encoder.encoder.rel_embeddings.weight"]!

        print("Query weight L1: \(MLX.sum(MLX.abs(queryWeight)).item(Float.self))")
        print("Key weight L1: \(MLX.sum(MLX.abs(keyWeight)).item(Float.self))")
        print("Rel embeddings L1: \(MLX.sum(MLX.abs(relEmbWeight)).item(Float.self))")

        // Test input
        let batchSize = 1
        let seqLen = 5
        let hiddenSize = 768
        let numHeads = 12
        let headDim = hiddenSize / numHeads

        // Use embedding values from Python (from debug_c2p_p2c.py)
        // Python: Embeddings L1: 1107.3037
        let hiddenStates = MLXArray.ones([batchSize, seqLen, hiddenSize]) * 0.1
        print("\nInput L1: \(MLX.sum(MLX.abs(hiddenStates)).item(Float.self))")

        // Manual Q, K computation
        // Q = hidden @ W_q.T + b_q
        let query = MLX.matmul(hiddenStates, queryWeight.transposed()) + queryBias
        let key = MLX.matmul(hiddenStates, keyWeight.transposed()) + keyBias

        print("Query L1: \(MLX.sum(MLX.abs(query)).item(Float.self))")
        print("Key L1: \(MLX.sum(MLX.abs(key)).item(Float.self))")

        // Reshape for multi-head
        let q = query.reshaped([batchSize, seqLen, numHeads, headDim]).transposed(0, 2, 1, 3)
        let k = key.reshaped([batchSize, seqLen, numHeads, headDim]).transposed(0, 2, 1, 3)

        // c2c attention
        let c2c = MLX.matmul(q, k.transposed(0, 1, 3, 2))
        print("\nC2C attention L1: \(MLX.sum(MLX.abs(c2c)).item(Float.self))")

        // Position key
        let posKey = MLX.matmul(relEmbWeight, keyWeight.transposed()) + keyBias
        print("pos_key L1: \(MLX.sum(MLX.abs(posKey)).item(Float.self))")

        // Reshape pos_key for multi-head
        let numBuckets = posKey.dim(0)
        let posKeyHeads = posKey.reshaped([numBuckets, numHeads, headDim])
        let posKeyT = posKeyHeads.transposed(1, 0, 2)  // [heads, num_buckets, head_dim]

        // c2p attention (full)
        let c2pFull = MLX.einsum("bhsd,hpd->bhsp", q, posKeyT)
        print("c2p_att (full) L1: \(MLX.sum(MLX.abs(c2pFull)).item(Float.self))")

        // Compute relative positions
        let relPos = makeLogBucketPosition(seqLen: seqLen, positionBuckets: 256, maxPosition: 512)
        print("\nRelative positions:")
        print(relPos)

        // c2p position indices
        let attSpan = Int32(128)
        let c2pPos = relPos + MLXArray(attSpan)
        let c2pPosClamped = MLX.clip(c2pPos, min: 0, max: 255)

        print("\nc2p indices (clamped to 0-255):")
        print(c2pPosClamped)

        // Gather for c2p
        let c2pPosExpanded = c2pPosClamped.expandedDimensions(axes: [0, 1])
        let c2pPosBroadcast = MLX.broadcast(c2pPosExpanded, to: [batchSize, numHeads, seqLen, seqLen])
        let c2p = takeAlong(c2pFull, c2pPosBroadcast.asType(.int32), axis: -1)
        print("c2p (gathered) L1: \(MLX.sum(MLX.abs(c2p)).item(Float.self))")

        // Print c2p scores for head 0
        print("\nc2p scores [0,0] (head 0):")
        for i in 0..<seqLen {
            var row = "  ["
            for j in 0..<seqLen {
                row += String(format: "%.4f", c2p[0, 0, i, j].item(Float.self))
                if j < seqLen - 1 { row += ", " }
            }
            row += "]"
            print(row)
        }

        // Python reference for c2p[0,0]:
        // tensor([[-1.8464, -1.7349, -1.5732, -0.8126, -1.5864],
        //         [ 5.2399,  6.0339,  5.8472,  6.2480,  5.9089],
        //         [12.0564, 11.9222, 11.7271, 11.6377, 11.9029],
        //         [-1.7086, -0.5591, -1.7706, -1.8464, -1.7349],
        //         [ 2.5006,  2.7334,  2.0259,  2.8696,  3.0902]])
        print("\nPython c2p reference [0,0]:")
        print("  [[-1.8464, -1.7349, -1.5732, -0.8126, -1.5864],")
        print("   [ 5.2399,  6.0339,  5.8472,  6.2480,  5.9089],")
        print("   [12.0564, 11.9222, 11.7271, 11.6377, 11.9029],")
        print("   [-1.7086, -0.5591, -1.7706, -1.8464, -1.7349],")
        print("   [ 2.5006,  2.7334,  2.0259,  2.8696,  3.0902]]")
    }

    // MARK: - Test with Python Fixtures

    func testC2P_WithPythonEmbeddings() async throws {
        // Load Python fixtures - try multiple possible paths
        let possiblePaths = [
            "/Users/tmwstw/Documents/mnemos/GLiNER2/GLiNER2Swift/Tests/GLiNER2SwiftTests/Fixtures/attention_layer0.safetensors",
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("Tests/GLiNER2SwiftTests/Fixtures/attention_layer0.safetensors").path,
            URL(fileURLWithPath: #file)
                .deletingLastPathComponent()
                .appendingPathComponent("Fixtures/attention_layer0.safetensors").path
        ]

        var fixturesPath: URL? = nil
        for path in possiblePaths {
            if FileManager.default.fileExists(atPath: path) {
                fixturesPath = URL(fileURLWithPath: path)
                break
            }
        }

        guard let fixturesPath = fixturesPath else {
            throw XCTSkip("attention_layer0.safetensors not found. Run generate_attention_fixtures.py first.")
        }

        let fixtures = try MLX.loadArrays(url: fixturesPath)

        // Load model weights
        let weightsPath = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub/models--fastino--gliner2-base-v1/snapshots")

        let fm = FileManager.default
        guard let snapshots = try? fm.contentsOfDirectory(atPath: weightsPath.path),
              let snapshot = snapshots.first else {
            throw XCTSkip("Model weights not found")
        }

        let modelPath = weightsPath.appendingPathComponent(snapshot)
        let safetensorsPath = modelPath.appendingPathComponent("model.safetensors")
        let weights = try MLX.loadArrays(url: safetensorsPath)

        // Get Python reference values
        let pythonQuery = fixtures["query"]!
        let pythonC2p = fixtures["c2p"]!
        let pythonC2pPos = fixtures["c2p_pos"]!

        print("=== TESTING C2P WITH PYTHON EMBEDDINGS ===")
        print("Python query shape: \(pythonQuery.shape)")
        print("Python query L1: \(MLX.sum(MLX.abs(pythonQuery)).item(Float.self))")
        print("Python c2p shape: \(pythonC2p.shape)")
        print("Python c2p L1: \(MLX.sum(MLX.abs(pythonC2p)).item(Float.self))")

        // Load weights
        let keyWeight = weights["encoder.encoder.layer.0.attention.self.key_proj.weight"]!
        let keyBias = weights["encoder.encoder.layer.0.attention.self.key_proj.bias"]!
        let relEmbWeight = weights["encoder.encoder.rel_embeddings.weight"]!

        // Compute pos_key = key_proj(rel_embeddings)
        let posKey = MLX.matmul(relEmbWeight, keyWeight.transposed()) + keyBias
        print("\npos_key shape: \(posKey.shape)")
        print("pos_key L1: \(MLX.sum(MLX.abs(posKey)).item(Float.self))")

        // Reshape for multi-head
        let numHeads = 12
        let headDim = 64
        let numBuckets = posKey.dim(0)
        let posKeyHeads = posKey.reshaped([numBuckets, numHeads, headDim])
        let posKeyT = posKeyHeads.transposed(1, 0, 2)  // [heads, num_buckets, head_dim]

        // Reshape query for multi-head
        let batchSize = 1
        let seqLen = 5
        let qHeads = pythonQuery.reshaped([batchSize, seqLen, numHeads, headDim]).transposed(0, 2, 1, 3)

        print("q_heads shape: \(qHeads.shape)")
        print("posKeyT shape: \(posKeyT.shape)")

        // Compute c2p_full using einsum
        let c2pFull = MLX.einsum("bhsd,hpd->bhsp", qHeads, posKeyT)
        print("\nc2p_full shape: \(c2pFull.shape)")
        print("c2p_full L1: \(MLX.sum(MLX.abs(c2pFull)).item(Float.self))")
        print("Python c2p_full L1: 336025.4062")

        // Gather using c2p_pos
        let c2pPosExpanded = pythonC2pPos.expandedDimensions(axes: [0, 1])
        let c2pPosBroadcast = MLX.broadcast(c2pPosExpanded, to: [batchSize, numHeads, seqLen, seqLen])
        let c2p = takeAlong(c2pFull, c2pPosBroadcast.asType(.int32), axis: -1)

        print("\nc2p shape: \(c2p.shape)")
        print("c2p L1: \(MLX.sum(MLX.abs(c2p)).item(Float.self))")
        print("Python c2p L1: 3305.4348")

        // Compare c2p values
        print("\nSwift c2p[0,0] (head 0):")
        for i in 0..<seqLen {
            var row = "  ["
            for j in 0..<seqLen {
                row += String(format: "%.4f", c2p[0, 0, i, j].item(Float.self))
                if j < seqLen - 1 { row += ", " }
            }
            row += "]"
            print(row)
        }

        print("\nPython c2p[0,0] (head 0):")
        for i in 0..<seqLen {
            var row = "  ["
            for j in 0..<seqLen {
                row += String(format: "%.4f", pythonC2p[0, 0, i, j].item(Float.self))
                if j < seqLen - 1 { row += ", " }
            }
            row += "]"
            print(row)
        }

        // Check max difference
        let diff = MLX.abs(c2p - pythonC2p)
        let maxDiff = MLX.max(diff).item(Float.self)
        print("\nMax difference: \(maxDiff)")

        XCTAssertLessThan(maxDiff, 1e-4, "c2p should match Python within 1e-4")
    }

    func testP2C_WithPythonEmbeddings() async throws {
        // Load Python fixtures
        let possiblePaths = [
            "/Users/tmwstw/Documents/mnemos/GLiNER2/GLiNER2Swift/Tests/GLiNER2SwiftTests/Fixtures/attention_layer0.safetensors",
            URL(fileURLWithPath: #file)
                .deletingLastPathComponent()
                .appendingPathComponent("Fixtures/attention_layer0.safetensors").path
        ]

        var fixturesPath: URL? = nil
        for path in possiblePaths {
            if FileManager.default.fileExists(atPath: path) {
                fixturesPath = URL(fileURLWithPath: path)
                break
            }
        }

        guard let fixturesPath = fixturesPath else {
            throw XCTSkip("attention_layer0.safetensors not found")
        }

        let fixtures = try MLX.loadArrays(url: fixturesPath)

        // Load model weights
        let weightsPath = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub/models--fastino--gliner2-base-v1/snapshots")

        let fm = FileManager.default
        guard let snapshots = try? fm.contentsOfDirectory(atPath: weightsPath.path),
              let snapshot = snapshots.first else {
            throw XCTSkip("Model weights not found")
        }

        let modelPath = weightsPath.appendingPathComponent(snapshot)
        let safetensorsPath = modelPath.appendingPathComponent("model.safetensors")
        let weights = try MLX.loadArrays(url: safetensorsPath)

        // Get Python reference values
        let pythonKey = fixtures["key"]!
        let pythonP2c = fixtures["p2c"]!
        let pythonC2pPos = fixtures["c2p_pos"]!  // p2c uses same indices!

        print("=== TESTING P2C WITH PYTHON EMBEDDINGS ===")
        print("Python key L1: \(MLX.sum(MLX.abs(pythonKey)).item(Float.self))")
        print("Python p2c L1: \(MLX.sum(MLX.abs(pythonP2c)).item(Float.self))")

        // Load weights
        let queryWeight = weights["encoder.encoder.layer.0.attention.self.query_proj.weight"]!
        let queryBias = weights["encoder.encoder.layer.0.attention.self.query_proj.bias"]!
        let relEmbWeight = weights["encoder.encoder.rel_embeddings.weight"]!

        // Compute pos_query = query_proj(rel_embeddings)
        let posQuery = MLX.matmul(relEmbWeight, queryWeight.transposed()) + queryBias
        print("pos_query L1: \(MLX.sum(MLX.abs(posQuery)).item(Float.self))")

        // Reshape for multi-head
        let numHeads = 12
        let headDim = 64
        let numBuckets = posQuery.dim(0)
        let posQueryHeads = posQuery.reshaped([numBuckets, numHeads, headDim])
        let posQueryT = posQueryHeads.transposed(1, 0, 2)

        // Reshape key for multi-head
        let batchSize = 1
        let seqLen = 5
        let kHeads = pythonKey.reshaped([batchSize, seqLen, numHeads, headDim]).transposed(0, 2, 1, 3)

        // Compute p2c_full using einsum
        let p2cFull = MLX.einsum("bhsd,hpd->bhsp", kHeads, posQueryT)
        print("p2c_full L1: \(MLX.sum(MLX.abs(p2cFull)).item(Float.self))")
        print("Python p2c_full L1: 261997.0781")

        // Gather using c2p_pos (SAME indices as c2p!)
        let c2pPosExpanded = pythonC2pPos.expandedDimensions(axes: [0, 1])
        let c2pPosBroadcast = MLX.broadcast(c2pPosExpanded, to: [batchSize, numHeads, seqLen, seqLen])
        let p2cGathered = takeAlong(p2cFull, c2pPosBroadcast.asType(.int32), axis: -1)

        // Transpose the result (Python: permute(0, 1, 3, 2))
        let p2c = p2cGathered.transposed(0, 1, 3, 2)

        print("\np2c L1: \(MLX.sum(MLX.abs(p2c)).item(Float.self))")
        print("Python p2c L1: 2416.1960")

        // Compare p2c values
        print("\nSwift p2c[0,0] (head 0):")
        for i in 0..<seqLen {
            var row = "  ["
            for j in 0..<seqLen {
                row += String(format: "%.4f", p2c[0, 0, i, j].item(Float.self))
                if j < seqLen - 1 { row += ", " }
            }
            row += "]"
            print(row)
        }

        print("\nPython p2c[0,0] (head 0):")
        for i in 0..<seqLen {
            var row = "  ["
            for j in 0..<seqLen {
                row += String(format: "%.4f", pythonP2c[0, 0, i, j].item(Float.self))
                if j < seqLen - 1 { row += ", " }
            }
            row += "]"
            print(row)
        }

        // Check max difference
        let diff = MLX.abs(p2c - pythonP2c)
        let maxDiff = MLX.max(diff).item(Float.self)
        print("\nMax difference: \(maxDiff)")

        XCTAssertLessThan(maxDiff, 1e-4, "p2c should match Python within 1e-4")
    }

    func testRelEmbeddingsLayerNorm() async throws {
        // Load model weights
        let weightsPath = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub/models--fastino--gliner2-base-v1/snapshots")

        let fm = FileManager.default
        guard let snapshots = try? fm.contentsOfDirectory(atPath: weightsPath.path),
              let snapshot = snapshots.first else {
            throw XCTSkip("Model weights not found")
        }

        let modelPath = weightsPath.appendingPathComponent(snapshot)
        let safetensorsPath = modelPath.appendingPathComponent("model.safetensors")
        let weights = try MLX.loadArrays(url: safetensorsPath)

        // Get rel_embeddings
        let relEmb = weights["encoder.encoder.rel_embeddings.weight"]!
        let lnWeight = weights["encoder.encoder.LayerNorm.weight"]!
        let lnBias = weights["encoder.encoder.LayerNorm.bias"]!

        print("=== TESTING REL_EMBEDDINGS LAYERNORM ===")
        print("Raw rel_embeddings L1: \(MLX.sum(MLX.abs(relEmb)).item(Float.self))")
        print("Python raw L1: 22767.6406")

        print("\nLayerNorm weight L1: \(MLX.sum(MLX.abs(lnWeight)).item(Float.self))")
        print("Python LN weight L1: 101.6684")
        print("LayerNorm bias L1: \(MLX.sum(MLX.abs(lnBias)).item(Float.self))")
        print("Python LN bias L1: 23.7871")

        // Create LayerNorm and load weights
        let ln = LayerNorm(dimensions: 768, eps: 1e-7)
        ln.update(parameters: ModuleParameters.unflattened([
            "weight": lnWeight,
            "bias": lnBias
        ]))

        // Apply LayerNorm
        let normalizedRelEmb = ln(relEmb)
        print("\nNormalized rel_embeddings L1: \(MLX.sum(MLX.abs(normalizedRelEmb)).item(Float.self))")
        print("Python normalized L1: 33938.0156")

        // Check if they're close
        let l1Diff = abs(MLX.sum(MLX.abs(normalizedRelEmb)).item(Float.self) - 33938.0156)
        print("L1 difference: \(l1Diff)")

        XCTAssertLessThan(l1Diff, 1.0, "Normalized rel_embeddings should match Python")
    }

    func testEmbeddingsLayer() async throws {
        // Load model weights
        let weightsPath = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub/models--fastino--gliner2-base-v1/snapshots")

        let fm = FileManager.default
        guard let snapshots = try? fm.contentsOfDirectory(atPath: weightsPath.path),
              let snapshot = snapshots.first else {
            throw XCTSkip("Model weights not found")
        }

        let modelPath = weightsPath.appendingPathComponent(snapshot)
        let safetensorsPath = modelPath.appendingPathComponent("model.safetensors")
        let weights = try MLX.loadArrays(url: safetensorsPath)

        // Create embeddings layer
        let embeddings = DeBERTaEmbeddings(
            vocabSize: 128011,
            hiddenSize: 768,
            dropoutProb: 0.0,  // Disable dropout for determinism
            layerNormEps: 1e-7
        )

        // Load weights
        embeddings.loadWeights(weights, prefix: "encoder.embeddings")

        // Test input
        let inputIds = MLXArray([287, 128003, 6967, 287, 128005, 483, 128005, 604, 1263, 1263, 128002, 41718, 3712, 269, 101312, 265, 6038, 323]).reshaped([1, 18])

        print("=== TESTING EMBEDDINGS LAYER ===")

        // Get word embeddings
        let wordEmb = embeddings.wordEmbeddings(inputIds)
        print("Word embeddings L1: \(MLX.sum(MLX.abs(wordEmb)).item(Float.self))")
        print("Python word emb L1: 956.9072")

        // Get final embeddings (after LayerNorm)
        let finalEmb = embeddings(inputIds)
        print("\nFinal embeddings L1: \(MLX.sum(MLX.abs(finalEmb)).item(Float.self))")
        print("Python final emb L1: 4009.2480")

        let wordEmbDiff = abs(MLX.sum(MLX.abs(wordEmb)).item(Float.self) - 956.9072)
        let finalEmbDiff = abs(MLX.sum(MLX.abs(finalEmb)).item(Float.self) - 4009.2480)

        print("\nWord emb L1 diff: \(wordEmbDiff)")
        print("Final emb L1 diff: \(finalEmbDiff)")

        XCTAssertLessThan(wordEmbDiff, 1.0, "Word embeddings should match Python")
        XCTAssertLessThan(finalEmbDiff, 1.0, "Final embeddings should match Python")
    }

    func testLayer0Output() async throws {
        // Load fixtures
        let possiblePaths = [
            "/Users/tmwstw/Documents/mnemos/GLiNER2/GLiNER2Swift/Tests/GLiNER2SwiftTests/Fixtures/layer0_debug.safetensors",
            URL(fileURLWithPath: #file)
                .deletingLastPathComponent()
                .appendingPathComponent("Fixtures/layer0_debug.safetensors").path
        ]

        var fixturesPath: URL? = nil
        for path in possiblePaths {
            if FileManager.default.fileExists(atPath: path) {
                fixturesPath = URL(fileURLWithPath: path)
                break
            }
        }

        guard let fixturesPath = fixturesPath else {
            throw XCTSkip("layer0_debug.safetensors not found")
        }

        let fixtures = try MLX.loadArrays(url: fixturesPath)

        // Load model weights
        let weightsPath = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub/models--fastino--gliner2-base-v1/snapshots")

        let fm = FileManager.default
        guard let snapshots = try? fm.contentsOfDirectory(atPath: weightsPath.path),
              let snapshot = snapshots.first else {
            throw XCTSkip("Model weights not found")
        }

        let modelPath = weightsPath.appendingPathComponent(snapshot)
        let safetensorsPath = modelPath.appendingPathComponent("model.safetensors")
        let weights = try MLX.loadArrays(url: safetensorsPath)

        // Get Python reference values
        let pythonEmbeddings = fixtures["embeddings"]!
        let pythonRelEmbNormalized = fixtures["rel_embeddings_normalized"]!
        let pythonLayer0Output = fixtures["layer0_output"]!
        let pythonQuery = fixtures["query"]!

        print("=== TESTING LAYER 0 ===")
        print("Python embeddings L1: \(MLX.sum(MLX.abs(pythonEmbeddings)).item(Float.self))")
        print("Python rel_emb_norm L1: \(MLX.sum(MLX.abs(pythonRelEmbNormalized)).item(Float.self))")
        print("Python layer0 output L1: \(MLX.sum(MLX.abs(pythonLayer0Output)).item(Float.self))")

        // Create layer 0
        let layer0 = DeBERTaLayer(
            hiddenSize: 768,
            intermediateSize: 3072,
            numHeads: 12,
            positionBuckets: 256,
            maxPosition: 512,
            dropoutProb: 0.0,
            layerNormEps: 1e-7
        )

        // Load weights
        layer0.loadWeights(weights, prefix: "encoder.encoder.layer.0")

        // Test query projection
        let query = layer0.attention.queryProj(pythonEmbeddings)
        let queryL1 = MLX.sum(MLX.abs(query)).item(Float.self)
        print("\nSwift query L1: \(queryL1)")
        print("Python query L1: \(MLX.sum(MLX.abs(pythonQuery)).item(Float.self))")
        print("Query diff: \(abs(queryL1 - 11567.4492))")

        // Run layer 0 with Python inputs
        // CRITICAL: Python uses all-ones attention mask, not nil!
        // Create attention mask: all ones (1 = valid position)
        let seqLen = pythonEmbeddings.dim(1)
        let attentionMask = MLXArray.ones([1, seqLen]).asType(.int32)
        // Expand to [batch, 1, 1, seq] and convert: 1 -> 0.0, 0 -> -10000.0
        let expandedMask = attentionMask.expandedDimensions(axes: [1, 2])
        let preparedMask = (1.0 - expandedMask.asType(.float32)) * -10000.0

        let layer0Output = layer0(pythonEmbeddings, relEmbeddings: pythonRelEmbNormalized, attentionMask: preparedMask)

        let layer0L1 = MLX.sum(MLX.abs(layer0Output)).item(Float.self)
        print("\nSwift layer0 output L1: \(layer0L1)")
        print("Python layer0 output L1: 4593.7393")
        print("Layer0 diff: \(abs(layer0L1 - 4593.7393))")

        // Compare per-position
        print("\nPer-position comparison:")
        for pos in [0, 1, 11] {
            let swiftL1 = MLX.sum(MLX.abs(layer0Output[0, pos])).item(Float.self)
            let pythonL1 = MLX.sum(MLX.abs(pythonLayer0Output[0, pos])).item(Float.self)
            print("  Position \(pos): Swift=\(swiftL1), Python=\(pythonL1), diff=\(abs(swiftL1 - pythonL1))")
        }

        // Max element-wise difference
        let diff = MLX.abs(layer0Output - pythonLayer0Output)
        let maxDiff = MLX.max(diff).item(Float.self)
        print("\nMax element-wise difference: \(maxDiff)")

        XCTAssertLessThan(maxDiff, 1e-3, "Layer 0 output should match Python within 1e-3")
    }
}
