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
// DisentangledAttention.swift
// DeBERTa's disentangled attention mechanism
//
// Matches Python: transformers/models/deberta_v2/modeling_deberta_v2.py:DisentangledSelfAttention
//
// CRITICAL IMPLEMENTATION NOTES (from Double-Check Review):
// 1. share_att_key=true: REUSE key_proj/query_proj on rel_embeddings (NO separate pos_key_proj!)
// 2. Scale factor: sqrt(head_dim * 3) because 3 attention types (c2c + c2p + p2c)
// 3. Position buckets: MUST implement logarithmic position bucketing
// 4. p2c transpose: REQUIRED transpose after gather
// 5. LayerNorm eps: MUST use 1e-7

import Foundation
import MLX
import MLXNN

// MARK: - Position Bucket Computation

/// Compute relative position bucket indices using logarithmic bucketing.
///
/// DeBERTa uses bucketed relative positions to handle longer sequences
/// while keeping the position embedding table manageable.
///
/// MATCHES EXACTLY: transformers/models/deberta_v2/modeling_deberta_v2.py:make_log_bucket_position
///
/// Python implementation:
/// ```python
/// sign = torch.sign(relative_pos)
/// mid = bucket_size // 2
/// abs_pos = torch.where(
///     (relative_pos < mid) & (relative_pos > -mid),
///     torch.tensor(mid - 1).type_as(relative_pos),
///     torch.abs(relative_pos),
/// )
/// log_pos = (
///     torch.ceil(torch.log(abs_pos / mid) / torch.log(torch.tensor((max_position - 1) / mid)) * (mid - 1)) + mid
/// )
/// bucket_pos = torch.where(abs_pos <= mid, relative_pos.type_as(log_pos), log_pos * sign)
/// ```
///
/// - Parameters:
///   - seqLen: Sequence length
///   - positionBuckets: Number of position buckets (256 for DeBERTa v3)
///   - maxPosition: Maximum relative position (512 for DeBERTa v3)
/// - Returns: Position bucket indices [seqLen, seqLen]
public func makeLogBucketPosition(
    seqLen: Int,
    positionBuckets: Int = 256,
    maxPosition: Int = 512
) -> MLXArray {
    // Build position difference matrix: positions[i, j] = i - j (q_ids[:, None] - k_ids[None, :])
    let seqRange = MLXArray(Array(0..<seqLen).map { Int32($0) })
    let posI = seqRange.expandedDimensions(axis: 1)  // [seqLen, 1]
    let posJ = seqRange.expandedDimensions(axis: 0)  // [1, seqLen]
    let relativePos = posI - posJ  // [seqLen, seqLen] - Note: i - j, not j - i

    let mid = Int32(positionBuckets / 2)
    let midArray = MLXArray(mid)

    // sign = torch.sign(relative_pos)
    let sign = MLX.sign(relativePos)

    // abs_pos = torch.where(
    //     (relative_pos < mid) & (relative_pos > -mid),
    //     torch.tensor(mid - 1),
    //     torch.abs(relative_pos),
    // )
    // Note: positions in range (-mid, mid) get clamped to mid - 1
    let negativeMid = MLXArray(-mid)
    let lessThanMid = relativePos .< midArray
    let greaterThanNegMid = relativePos .> negativeMid
    let inLinearRange = lessThanMid .&& greaterThanNegMid
    let absPos = MLX.where(
        inLinearRange,
        MLXArray(mid - 1),
        MLX.abs(relativePos)
    )

    // log_pos = (
    //     torch.ceil(torch.log(abs_pos / mid) / torch.log(torch.tensor((max_position - 1) / mid)) * (mid - 1)) + mid
    // )
    // CRITICAL: Uses (max_position - 1) / mid, NOT max_position / mid
    let logBase = Float(maxPosition - 1) / Float(mid)
    let logBaseValue = Foundation.log(logBase)
    let midMinusOne = Float(mid - 1)

    let absPosFloat = absPos.asType(.float32)
    let midFloat = MLXArray(Float(mid))

    // log(abs_pos / mid) / log((max_position - 1) / mid) * (mid - 1) + mid
    let logRatio = MLX.log(absPosFloat / midFloat)
    let logPos = MLX.ceil(logRatio / MLXArray(logBaseValue) * MLXArray(midMinusOne)) + midFloat

    // bucket_pos = torch.where(abs_pos <= mid, relative_pos, log_pos * sign)
    // For positions with abs_pos <= mid, use relative_pos directly
    // For positions with abs_pos > mid, use log_pos * sign
    let linearMask = absPos .<= midArray
    let bucketPos = MLX.where(
        linearMask,
        relativePos.asType(.float32),
        logPos * sign.asType(.float32)
    ).asType(.int32)

    return bucketPos
}

// MARK: - Disentangled Self-Attention

/// DeBERTa's Disentangled Self-Attention mechanism.
///
/// Computes attention using three components:
/// 1. Content-to-content (c2c): Standard attention
/// 2. Content-to-position (c2p): Query attends to relative positions
/// 3. Position-to-content (p2c): Positions attend to content
///
/// CRITICAL: Uses share_att_key=true, meaning query_proj and key_proj are reused
/// for both content and position embeddings (no separate pos_key_proj/pos_query_proj).
/// Holder for derived tensors that must NOT be visible to `Module` reflection.
///
/// Any stored `MLXArray` property on a `Module` — even a private one — is captured as a
/// parameter when the module's reflection cache is built at init, which would put these
/// derived values into `parameters()` and `update(parameters:)`. A plain (non-Module,
/// non-MLXArray) class is classified as `.other` and ignored entirely, so it is the safe
/// place to memoize.
final class DisentangledAttentionCache {
    /// `keyProj(relEmbeddings)` reshaped to [heads, buckets, headDim].
    var posKey: MLXArray?
    /// `queryProj(relEmbeddings)` reshaped to [heads, buckets, headDim].
    var posQuery: MLXArray?
    /// Log-bucket relative-position matrix, keyed by the sequence length it was built for.
    var relPos: (seqLen: Int, value: MLXArray)?

    func reset() {
        posKey = nil
        posQuery = nil
        relPos = nil
    }
}

public class DisentangledSelfAttention: Module {

    /// Memoized projections of the (frozen) relative-position embedding table.
    ///
    /// At inference the table and the projection weights are constant, so
    /// `keyProj(relEmbeddings)` and `queryProj(relEmbeddings)` — a [512,768]x[768,768]
    /// matmul each — produce byte-identical results on every call of every layer. Twelve
    /// layers x two projections is ~14.5 GFLOP of pure repetition per inference,
    /// independent of input length.
    ///
    /// Invalidated by `resetCaches()`, which weight loading must call.
    let cache = DisentangledAttentionCache()

    /// Discard memoized projections. Call after any weight mutation.
    public func resetCaches() {
        cache.reset()
    }

    /// Hidden size
    public let hiddenSize: Int

    /// Number of attention heads
    public let numHeads: Int

    /// Dimension per head
    public let headDim: Int

    /// Number of position buckets
    public let positionBuckets: Int

    /// Maximum relative position
    public let maxPosition: Int

    /// Query projection (shared for content and position)
    public let queryProj: Linear

    /// Key projection (shared for content and position)
    public let keyProj: Linear

    /// Value projection
    public let valueProj: Linear

    // NOTE: Output projection is handled by DeBERTaLayer.attentionOutputDense
    // DO NOT add outProj here - it causes double projection with random weights!

    /// Dropout
    public let dropout: Dropout

    /// Position attention types enabled
    public let posAttType: Set<String>

    /// Scale factor: sqrt(head_dim * scale_factor)
    /// scale_factor = 3 because c2c + c2p + p2c
    private let scaleFactor: Float

    /// Initialize disentangled self-attention
    ///
    /// - Parameters:
    ///   - hiddenSize: Model hidden size (768)
    ///   - numHeads: Number of attention heads (12)
    ///   - positionBuckets: Number of position buckets (256)
    ///   - maxPosition: Maximum relative position (512)
    ///   - dropoutProb: Attention dropout probability
    ///   - posAttType: Position attention types (["c2p", "p2c"])
    public init(
        hiddenSize: Int,
        numHeads: Int,
        positionBuckets: Int = 256,
        maxPosition: Int = 512,
        dropoutProb: Float = 0.1,
        posAttType: Set<String> = ["c2p", "p2c"]
    ) {
        self.hiddenSize = hiddenSize
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        self.positionBuckets = positionBuckets
        self.maxPosition = maxPosition
        self.posAttType = posAttType

        // CRITICAL: Scale factor is 3 when both c2p and p2c are enabled
        var numAttTypes = 1  // c2c always
        if posAttType.contains("c2p") { numAttTypes += 1 }
        if posAttType.contains("p2c") { numAttTypes += 1 }
        self.scaleFactor = Foundation.sqrt(Float(headDim * numAttTypes))

        // Projections (shared for content and position when share_att_key=true)
        self.queryProj = Linear(hiddenSize, hiddenSize)
        self.keyProj = Linear(hiddenSize, hiddenSize)
        self.valueProj = Linear(hiddenSize, hiddenSize)
        // NOTE: No outProj - handled by DeBERTaLayer.attentionOutputDense

        self.dropout = Dropout(p: dropoutProb)
    }

    /// Forward pass
    ///
    /// - Parameters:
    ///   - hiddenStates: Input [batch, seq, hidden]
    ///   - relEmbeddings: Relative position embeddings [2*max_pos or buckets, hidden]
    ///   - attentionMask: Optional attention mask [batch, 1, 1, seq] or [batch, 1, seq, seq]
    /// - Returns: Output [batch, seq, hidden]
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        relEmbeddings: MLXArray,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let seqLen = hiddenStates.dim(1)

        // Project to queries, keys, values
        let queryLayer = queryProj(hiddenStates)  // [batch, seq, hidden]
        let keyLayer = keyProj(hiddenStates)
        let valueLayer = valueProj(hiddenStates)

        // Reshape for multi-head attention
        // [batch, seq, hidden] -> [batch, heads, seq, head_dim]
        let query = reshapeForHeads(queryLayer, batchSize: batchSize, seqLen: seqLen)
        let key = reshapeForHeads(keyLayer, batchSize: batchSize, seqLen: seqLen)
        let value = reshapeForHeads(valueLayer, batchSize: batchSize, seqLen: seqLen)

        // Position bucket indices depend only on seqLen and fixed config, so memoize by
        // length: the matrix is otherwise rebuilt from ~15 small kernels on every layer of
        // every call.
        let relPos: MLXArray
        if let cached = cache.relPos, cached.seqLen == seqLen {
            relPos = cached.value
        } else {
            relPos = makeLogBucketPosition(
                seqLen: seqLen,
                positionBuckets: positionBuckets,
                maxPosition: maxPosition
            )
            MLX.eval(relPos)
            cache.relPos = (seqLen, relPos)
        }

        // DeBERTa's three components all carry the same 1/scaleFactor: Python divides the
        // key before the c2c matmul and divides the c2p and p2c scores after computing
        // them. That makes the whole pre-softmax expression
        //     (q·kᵀ + c2p + p2c) / scaleFactor + attentionMask,
        // which is exactly SDPA's `scale` plus an additive mask — so the position terms go
        // in as the bias and the fused kernel does the rest. Regrouping the divisions is a
        // floating-point reassociation, not a change of formula.
        var positionBias: MLXArray? = nil

        // Content-to-position attention
        if posAttType.contains("c2p") {
            positionBias = computeC2P(
                query: query,
                relEmbeddings: relEmbeddings,
                relPos: relPos,
                batchSize: batchSize,
                seqLen: seqLen
            )
        }

        // Position-to-content attention
        if posAttType.contains("p2c") {
            let p2cScores = computeP2C(
                key: key,
                relEmbeddings: relEmbeddings,
                relPos: relPos,
                batchSize: batchSize,
                seqLen: seqLen
            )
            positionBias = positionBias.map { $0 + p2cScores } ?? p2cScores
        }

        // The bias must share the q/k/v dtype — SDPA throws on an f32 mask over f16
        // inputs. It does here by construction: the position scores come from einsums on
        // query/key, the attention mask is built in the hidden-state dtype, and dividing
        // by a Swift scalar preserves dtype.
        var bias = positionBias.map { $0 / scaleFactor }
        if let mask = attentionMask {
            // 0 for valid positions, a large negative value for masked ones.
            bias = bias.map { $0 + mask } ?? mask
        }

        // Dropout is intentionally absent: the model runs `train(false)`, where it is the
        // identity, and the fused kernel has nowhere to put it.
        var output = MLXFast.scaledDotProductAttention(
            queries: query,
            keys: key,
            values: value,
            scale: 1 / scaleFactor,
            mask: bias
        )

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        output = output.transposed(0, 2, 1, 3)
        output = output.reshaped([batchSize, seqLen, hiddenSize])

        // NOTE: No output projection here - handled by DeBERTaLayer.attentionOutputDense
        return output
    }

    // MARK: - Private Helpers

    /// Reshape tensor for multi-head attention
    /// [batch, seq, hidden] -> [batch, heads, seq, head_dim]
    private func reshapeForHeads(_ x: MLXArray, batchSize: Int, seqLen: Int) -> MLXArray {
        let reshaped = x.reshaped([batchSize, seqLen, numHeads, headDim])
        return reshaped.transposed(0, 2, 1, 3)  // [batch, heads, seq, head_dim]
    }

    /// Compute content-to-position attention scores
    ///
    /// CRITICAL: Python does matmul-then-gather, NOT gather-then-matmul!
    /// c2p_att = query @ pos_key.T  (attention to ALL positions)
    /// c2p = gather(c2p_att, rel_pos)  (select relevant scores)
    ///
    /// CRITICAL: share_att_key=true means we use keyProj on rel_embeddings
    private func computeC2P(
        query: MLXArray,
        relEmbeddings: MLXArray,
        relPos: MLXArray,
        batchSize: Int,
        seqLen: Int
    ) -> MLXArray {
        // Project relative embeddings using KEY projection (share_att_key=true).
        // Memoized: constant for the lifetime of the loaded weights (see `cache`).
        let numBuckets = relEmbeddings.dim(0)
        let posKeyTransposed: MLXArray
        if let cached = cache.posKey {
            posKeyTransposed = cached
        } else {
            let posKey = keyProj(relEmbeddings)  // [num_buckets, hidden]
            posKeyTransposed = posKey
                .reshaped([numBuckets, numHeads, headDim])
                .transposed(1, 0, 2)             // [heads, num_buckets, head_dim]
            MLX.eval(posKeyTransposed)           // materialize once, not per use
            cache.posKey = posKeyTransposed
        }

        // Step 1: Compute attention scores to ALL position embeddings
        // query: [batch, heads, seq, head_dim]
        // posKeyTransposed: [heads, num_buckets, head_dim]
        // Result: [batch, heads, seq, num_buckets]
        let c2pFull = MLX.einsum("bhsd,hpd->bhsp", query, posKeyTransposed)

        // Step 2: Add offset to relPos to handle negative indices
        // relPos contains values like -17 to +17 for seq_len=18
        // We need to map these to valid indices [0, att_span*2-1]
        // att_span = pos_ebd_size = position_buckets = 256 (NOT position_buckets/2!)
        let attSpan = Int32(positionBuckets)
        let relPosOffset = relPos + MLXArray(attSpan)  // Now in range [0, 511]

        // Clamp to valid range [0, num_buckets - 1]
        let relPosClamped = MLX.clip(relPosOffset, min: 0, max: Int32(numBuckets - 1))

        // Step 3: Gather the relevant scores using relPos indices
        // c2pFull: [batch, heads, seq_i, num_buckets]
        // relPosClamped: [seq_i, seq_j] -> need to gather along last dim
        // Result: [batch, heads, seq_i, seq_j]

        // Expand relPos for gathering: [1, 1, seq_i, seq_j]
        let relPosExpanded = relPosClamped.expandedDimensions(axes: [0, 1])
        // Broadcast to [batch, heads, seq_i, seq_j]
        let relPosBroadcast = MLX.broadcast(relPosExpanded, to: [batchSize, numHeads, seqLen, seqLen])

        // Use takeAlong to gather (MLX's equivalent of torch.gather/take_along_axis)
        let c2pScores = takeAlong(c2pFull, relPosBroadcast.asType(.int32), axis: -1)

        return c2pScores
    }

    /// Compute position-to-content attention scores
    ///
    /// CRITICAL: Python uses NEGATED indices for p2c, then transposes!
    /// p2c_att = key @ pos_query.T  (attention from keys to all positions)
    /// p2c_pos = clamp(-relative_pos + att_span, 0, att_span*2-1)  (NEGATED!)
    /// p2c = gather(p2c_att, p2c_pos)
    /// p2c = p2c.permute(0,1,3,2)  (transpose at the end)
    ///
    /// Note: p2c_pos == c2p_pos.T (transpose relationship due to negation)
    ///
    /// CRITICAL: share_att_key=true means we use queryProj on rel_embeddings
    private func computeP2C(
        key: MLXArray,
        relEmbeddings: MLXArray,
        relPos: MLXArray,
        batchSize: Int,
        seqLen: Int
    ) -> MLXArray {
        // Project relative embeddings using QUERY projection (share_att_key=true).
        // Memoized: constant for the lifetime of the loaded weights (see `cache`).
        let numBuckets = relEmbeddings.dim(0)
        let posQueryTransposed: MLXArray
        if let cached = cache.posQuery {
            posQueryTransposed = cached
        } else {
            let posQuery = queryProj(relEmbeddings)  // [num_buckets, hidden]
            posQueryTransposed = posQuery
                .reshaped([numBuckets, numHeads, headDim])
                .transposed(1, 0, 2)               // [heads, num_buckets, head_dim]
            MLX.eval(posQueryTransposed)
            cache.posQuery = posQueryTransposed
        }

        // Step 1: Compute attention scores from keys to ALL position embeddings
        // key: [batch, heads, seq, head_dim]
        // posQueryTransposed: [heads, num_buckets, head_dim]
        // Result: [batch, heads, seq, num_buckets]
        let p2cFull = MLX.einsum("bhsd,hpd->bhsp", key, posQueryTransposed)

        // Step 2: Use NEGATED relative position for p2c indices
        // Python: p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
        // att_span = pos_ebd_size = position_buckets = 256
        let attSpan = Int32(positionBuckets)
        let relPosOffset = MLXArray(attSpan) - relPos  // NEGATED: att_span - rel_pos

        // Clamp to valid range [0, num_buckets - 1]
        let relPosClamped = MLX.clip(relPosOffset, min: 0, max: Int32(numBuckets - 1))

        // Step 3: Gather using c2p_pos indices
        // p2cFull: [batch, heads, seq, num_buckets]
        // relPosClamped: [seq_i, seq_j]

        // Expand for gathering
        let relPosExpanded = relPosClamped.expandedDimensions(axes: [0, 1])
        let relPosBroadcast = MLX.broadcast(relPosExpanded, to: [batchSize, numHeads, seqLen, seqLen])

        let p2cGathered = takeAlong(p2cFull, relPosBroadcast.asType(.int32), axis: -1)

        // Step 4: Transpose the result (Python: permute(0, 1, 3, 2))
        // This swaps seq_i and seq_j dimensions
        let p2cResult = p2cGathered.transposed(0, 1, 3, 2)

        return p2cResult
    }
}

// MARK: - Weight Loading

extension DisentangledSelfAttention {
    /// Load weights from a dictionary
    ///
    /// Expected keys (with prefix):
    /// - {prefix}.query_proj.weight: [hidden, hidden]
    /// - {prefix}.query_proj.bias: [hidden]
    /// - {prefix}.key_proj.weight: [hidden, hidden]
    /// - {prefix}.key_proj.bias: [hidden]
    /// - {prefix}.value_proj.weight: [hidden, hidden]
    /// - {prefix}.value_proj.bias: [hidden]
    /// - {prefix}.out_proj.weight or {prefix}.dense.weight: [hidden, hidden]
    /// - {prefix}.out_proj.bias or {prefix}.dense.bias: [hidden]
    ///
    /// NOTE: DeBERTa v2/v3 uses in_proj style (query/key/value_proj) not combined
    public func loadWeights(_ weights: [String: MLXArray], prefix: String) {
        let p = prefix.isEmpty ? "" : "\(prefix)."

        // Helper to update a Linear module
        func updateLinear(_ module: Linear, weightKey: String, biasKey: String) {
            var params: [String: MLXArray] = [:]
            if let w = weights[weightKey] {
                params["weight"] = w
            }
            if let b = weights[biasKey] {
                params["bias"] = b
            }
            if !params.isEmpty {
                module.update(parameters: ModuleParameters.unflattened(params))
            }
        }

        // Query projection
        updateLinear(queryProj, weightKey: "\(p)query_proj.weight", biasKey: "\(p)query_proj.bias")

        // Key projection
        updateLinear(keyProj, weightKey: "\(p)key_proj.weight", biasKey: "\(p)key_proj.bias")

        // Value projection
        updateLinear(valueProj, weightKey: "\(p)value_proj.weight", biasKey: "\(p)value_proj.bias")

        // NOTE: Output projection is handled by DeBERTaLayer.attentionOutputDense
        // which loads from "attention.output.dense.weight/bias"
    }
}
