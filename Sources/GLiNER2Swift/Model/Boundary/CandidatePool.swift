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
// CandidatePool.swift
// GLiNER2.5 shared candidate pool, pool scorer, and the boundary head that ties the
// modules together.
//
// Matches Python: gliner2/models/boundary/pool.py (DocumentCandidatePool,
// SharedPoolScorer), content.py (SpanContentPooler), model.py (BoundaryHead.forward on
// the `candidate_pool == "shared"` path).

import Foundation
import MLX
import MLXNN

// MARK: - Span content pooler

/// Mean of projected word states over a span, via an fp32 prefix sum, then LayerNorm.
///
/// Prefix `…contentPooler.*`.
public final class SpanContentPooler: Module {
    public let valueProjection: Linear
    public let layerNorm: LayerNorm

    public init(hiddenSize: Int, contentDim: Int) {
        self.valueProjection = Linear(hiddenSize, contentDim)
        self.layerNorm = LayerNorm(dimensions: contentDim, eps: 1e-5)
        super.init()
    }

    /// `[B, N, c]` fp32 prefix sums with a leading zero row.
    public func prefix(tokenStates: MLXArray, textMask: MLXArray) -> MLXArray {
        let values = valueProjection(tokenStates) * textMask.asType(tokenStates.dtype).expandedDimensions(axis: -1)
        let B = values.dim(0), c = values.dim(2)
        return MLX.concatenated(
            [MLXArray.zeros([B, 1, c], dtype: .float32), values.asType(.float32).cumsum(axis: 1)],
            axis: 1)
    }

    /// Pool `[B, K]` spans given the prefix: `(prefix[e] - prefix[s]) / max(e - s, 1)`,
    /// cast to `dtype`, then LayerNorm → `[B, K, c]`.
    public func pool(prefix: MLXArray, starts: MLXArray, ends: MLXArray, dtype: DType) -> MLXArray {
        let sum = BoundaryOps.gatherRows(prefix, ends) - BoundaryOps.gatherRows(prefix, starts)
        let length = MLX.clip(ends - starts, min: Int32(1)).asType(.float32).expandedDimensions(axis: -1)
        return layerNorm((sum / length).asType(dtype))
    }

    /// Pool `[B, Q, C]` spans → `[B, Q, C, c]`.
    public func poolPerQuery(prefix: MLXArray, starts: MLXArray, ends: MLXArray, dtype: DType) -> MLXArray {
        let sum = BoundaryOps.gatherStates(prefix, ends) - BoundaryOps.gatherStates(prefix, starts)
        let length = MLX.clip(ends - starts, min: Int32(1)).asType(.float32).expandedDimensions(axis: -1)
        return layerNorm((sum / length).asType(dtype))
    }
}

// MARK: - Rotary boundary embedding

/// Rotate endpoint vectors so dot products encode relative boundary distance.
/// Interleaved pairs `(2i, 2i+1)`, angles in fp32 (`rotary.py`).
public struct RotaryBoundaryEmbedding {
    let dim: Int
    let invFreq: MLXArray   // [dim/2] fp32

    public init(dim: Int, base: Float) {
        self.dim = dim
        let half = dim / 2
        self.invFreq = MLXArray((0..<half).map { i in
            1.0 / powf(base, Float(2 * i) / Float(dim))
        })
    }

    /// - Parameter states: `[B, N, dim]`; positions are the boundary indices `0 ..< N`.
    public func callAsFunction(_ states: MLXArray) -> MLXArray {
        let B = states.dim(0), N = states.dim(1)
        let half = dim / 2
        let positions = BoundaryOps.arange(N).asType(.float32).reshaped([1, N, 1])
        let angle = positions * invFreq.reshaped([1, 1, half])            // [1, N, half]
        let cos = MLX.cos(angle), sin = MLX.sin(angle)
        let pairs = states.asType(.float32).reshaped([B, N, half, 2])
        let even = pairs[0..., 0..., 0..., 0]
        let odd = pairs[0..., 0..., 0..., 1]
        let rotated = MLX.stacked([even * cos - odd * sin, even * sin + odd * cos], axis: -1)
        return rotated.reshaped([B, N, dim]).asType(states.dtype)
    }
}

// MARK: - Document candidate pool (shared_pool_builder)

/// One document-level pool of `poolSize` candidate spans shared by every query.
///
/// Prefix `boundaryHead.sharedPoolBuilder.*`.
///
/// Every selection step upstream is a stable `torch.sort`; here each one is a single
/// `argSort` over an exact int64 composite key (see `BoundaryOps`), so the tie protocol —
/// score descending, then boundary index / span key ascending, best copy of a duplicate
/// key first — holds without assuming anything about MLX's sort. All scores that decide
/// membership are computed in fp32 regardless of the model dtype: the per-query quota
/// band (`5000 + rank bonus`) does not survive fp16 spacing.
public final class DocumentCandidatePool: Module {
    public let startProjection: Linear
    public let endProjection: Linear

    let poolSize: Int
    let topK: Int
    let quota: Int
    let scale: Float

    public init(boundaryDim: Int, settings: BoundaryHeadSettings) {
        self.startProjection = Linear(boundaryDim, boundaryDim)
        self.endProjection = Linear(boundaryDim, boundaryDim)
        self.poolSize = settings.poolSize
        self.topK = settings.poolBoundaryTopK
        self.quota = settings.minPoolPerQuery
        self.scale = 1.0 / Float(boundaryDim).squareRoot()
        super.init()
    }

    public struct Pooled {
        /// `[B, P, 2]` int32 `(start, end)` boundary indices, `(0, 0)` where invalid.
        public let indices: MLXArray
        /// `[B, P]` bool.
        public let mask: MLXArray
        /// `[B, P]` fp32: compat + union start + union end marginals, -1e4 where invalid.
        public let proposalLogits: MLXArray
        /// `[B, P]` fp32: query-agnostic endpoint compatibility, 0 where invalid.
        public let compatLogits: MLXArray
    }

    /// Top-`k` positions of each row by (score desc, index asc); invalid rows get score
    /// 0 and index 0, exactly like `select_top_boundaries`.
    private func selectTop(_ logits: MLXArray, valid: MLXArray, k: Int)
        -> (indices: MLXArray, scores: MLXArray, valid: MLXArray)
    {
        let masked = MLX.which(valid, logits, MLXArray(BoundaryOps.maskLogit))
        let order = BoundaryOps.argsortScoreDesc(masked)[0..., 0..<k]
        let scores = MLX.takeAlong(masked, order, axis: -1)
        let v = MLX.takeAlong(valid, order, axis: -1)
        return (MLX.which(v, order.asType(.int32), Int32(0)), MLX.which(v, scores, Float(0)), v)
    }

    /// - Parameters:
    ///   - boundaryStates: `[B, N, d]`
    ///   - boundaryMask: `[B, N]` bool
    ///   - queryMask: `[B, Q]` bool
    ///   - startLogits / endLogits: `[B, Q, N]` (any float dtype; used in fp32 here)
    public func callAsFunction(
        boundaryStates: MLXArray, boundaryMask: MLXArray, queryMask: MLXArray,
        startLogits: MLXArray, endLogits: MLXArray
    ) -> Pooled {
        let B = boundaryStates.dim(0), N = boundaryStates.dim(1)
        let Q = queryMask.dim(1)
        let maskLogit = MLXArray(BoundaryOps.maskLogit)
        let startF = startLogits.asType(.float32)
        let endF = endLogits.asType(.float32)

        // 1. Query-union marginals.
        let qBoundary = BoundaryOps.and(boundaryMask.expandedDimensions(axis: 1),
                                        queryMask.expandedDimensions(axis: -1))     // [B, Q, N]
        let unionStart = MLX.which(qBoundary, startF, maskLogit).max(axis: 1)          // [B, N]
        let unionEnd = MLX.which(qBoundary, endF, maskLogit).max(axis: 1)
        let unionValid = BoundaryOps.and(boundaryMask, queryMask.any(axis: -1, keepDims: true))

        // 2. Top-K starts and ends.
        let k = min(topK, N)
        let starts = selectTop(unionStart, valid: unionValid, k: k)
        let ends = selectTop(unionEnd, valid: unionValid, k: k)

        // 3. Cartesian pairing → [B, K²].
        let kk = k * k
        let pairS = MLX.broadcast(starts.indices.reshaped([B, k, 1]), to: [B, k, k]).reshaped([B, kk])
        let pairE = MLX.broadcast(ends.indices.reshaped([B, 1, k]), to: [B, k, k]).reshaped([B, kk])
        let pairValid = BoundaryOps.and(
            BoundaryOps.and(
                MLX.broadcast(starts.valid.reshaped([B, k, 1]), to: [B, k, k]),
                MLX.broadcast(ends.valid.reshaped([B, 1, k]), to: [B, k, k])),
            ends.indices.reshaped([B, 1, k]) .> starts.indices.reshaped([B, k, 1])
        ).reshaped([B, kk])

        // 4. Query-agnostic endpoint compatibility (fp32 for the selection math).
        let startAll = startProjection(boundaryStates)                                  // [B, N, d]
        let endAll = endProjection(boundaryStates)
        let compat = (BoundaryOps.gatherRows(startAll, pairS) * BoundaryOps.gatherRows(endAll, pairE))
            .asType(.float32).sum(axis: -1) * scale                                     // [B, K²]
        let unionPairScore = compat
            + MLX.takeAlong(unionStart, pairS, axis: -1)
            + MLX.takeAlong(unionEnd, pairE, axis: -1)

        // 5. Per-query quota: each query's top `quota` pairs get a priority band.
        let pairKeys = pairS * Int32(N) + pairE                                          // [B, K²] int32
        var allKeys = pairKeys
        var allScores = unionPairScore
        var allValid = pairValid
        if quota > 0 && Q > 0 {
            let pairS3 = MLX.broadcast(pairS.reshaped([B, 1, kk]), to: [B, Q, kk])
            let pairE3 = MLX.broadcast(pairE.reshaped([B, 1, kk]), to: [B, Q, kk])
            let perQuery = MLX.takeAlong(startF, pairS3, axis: -1)
                + MLX.takeAlong(endF, pairE3, axis: -1)
                + compat.expandedDimensions(axis: 1)                                    // [B, Q, K²]
            let perQueryValid = BoundaryOps.and(pairValid.expandedDimensions(axis: 1),
                                                queryMask.expandedDimensions(axis: -1))
            let q = min(quota, kk)
            let ranked = BoundaryOps.argsortScoreDesc(
                MLX.which(perQueryValid, perQuery, maskLogit))[0..., 0..., 0..<q]       // [B, Q, q]
            let quotaKeys = MLX.takeAlong(
                MLX.broadcast(pairKeys.reshaped([B, 1, kk]), to: [B, Q, kk]), ranked, axis: -1)
                .reshaped([B, Q * q])
            let quotaValid = MLX.takeAlong(perQueryValid, ranked, axis: -1).reshaped([B, Q * q])
            // -MASK_LOGIT * 0.5 + rank bonus (q .. 1), in fp32.
            let bonus = MLXArray((0..<q).map { Float(q - $0) }) + (-BoundaryOps.maskLogit * 0.5)
            let quotaScores = MLX.broadcast(bonus.reshaped([1, 1, q]), to: [B, Q, q]).reshaped([B, Q * q])
            allKeys = MLX.concatenated([quotaKeys, pairKeys], axis: -1)
            allScores = MLX.concatenated([quotaScores, unionPairScore], axis: -1)
            allValid = MLX.concatenated([quotaValid, pairValid], axis: -1)
        }

        // Always at least `poolSize` slots so the dedup output has a fixed width.
        let width = allKeys.dim(-1)
        if width < poolSize {
            let pad = poolSize - width
            allKeys = MLX.concatenated([allKeys, MLXArray.zeros([B, pad], dtype: .int32)], axis: -1)
            allScores = MLX.concatenated([allScores, MLXArray.zeros([B, pad], dtype: .float32)], axis: -1)
            allValid = MLX.concatenated([allValid, MLXArray.zeros([B, pad], dtype: .bool)], axis: -1)
        }

        // 6. Deduplicate by span key, keep the best copy, rank, truncate to P.
        let invalidKey = Int32(N * N)
        var keys = MLX.which(allValid, allKeys, invalidKey)
        var scores = MLX.which(allValid, allScores, maskLogit)
        var valid = allValid

        let byScore = BoundaryOps.argsortScoreDesc(scores)
        keys = MLX.takeAlong(keys, byScore, axis: -1)
        scores = MLX.takeAlong(scores, byScore, axis: -1)
        valid = MLX.takeAlong(valid, byScore, axis: -1)

        let byKey = BoundaryOps.argsortKeyAscScoreDesc(keys: keys, scores: scores)
        keys = MLX.takeAlong(keys, byKey, axis: -1)
        scores = MLX.takeAlong(scores, byKey, axis: -1)
        valid = MLX.takeAlong(valid, byKey, axis: -1)

        let M = keys.dim(-1)
        let first = MLX.concatenated(
            [MLXArray.ones([B, 1], dtype: .bool), keys[0..., 1...] .!= keys[0..., 0..<(M - 1)]],
            axis: -1)
        let keep = BoundaryOps.and(valid, first)

        let order = BoundaryOps.argsortScoreDescKeyAsc(
            scores: MLX.which(keep, scores, maskLogit), keys: keys)[0..., 0..<poolSize]
        let selectedKeys = MLX.takeAlong(keys, order, axis: -1)                        // [B, P]
        let selectedValid = MLX.takeAlong(keep, order, axis: -1)

        // 7. Decode keys, recompute the retained pairs' scores.
        let s = MLX.floorDivide(selectedKeys, Int32(N))
        let e = selectedKeys - s * Int32(N)
        let sSafe = MLX.which(selectedValid, s, Int32(0))
        let eSafe = MLX.which(selectedValid, e, Int32(0))
        let indices = MLX.stacked([sSafe, eSafe], axis: -1)                             // [B, P, 2]

        let selectedCompat = (BoundaryOps.gatherRows(startAll, sSafe) * BoundaryOps.gatherRows(endAll, eSafe))
            .asType(.float32).sum(axis: -1) * scale
        let proposal = selectedCompat
            + MLX.takeAlong(unionStart, sSafe, axis: -1)
            + MLX.takeAlong(unionEnd, eSafe, axis: -1)

        return Pooled(
            indices: indices,
            mask: selectedValid,
            proposalLogits: MLX.which(selectedValid, proposal, maskLogit),
            compatLogits: MLX.which(selectedValid, selectedCompat, Float(0)))
    }
}

// MARK: - Shared pool scorer

/// Reranks every (candidate, query) pair of the shared pool.
///
/// Prefix `boundaryHead.sharedPoolScorer.*`.
public final class SharedPoolScorer: Module {
    public let startProjection: Linear
    public let endProjection: Linear
    public let lengthProjection: Linear
    public let priorProjection: Linear
    public let contentPooler: SpanContentPooler
    public let contentProjection: Linear
    public let candidateNorm: LayerNorm
    public let queryProjection: Linear
    public let film: Linear
    public let filmOutput: TwoLayerMLP

    let pairDim: Int
    let scale: Float

    public init(hiddenSize: Int, settings: BoundaryHeadSettings) {
        let d = settings.boundaryDim, p = settings.pairDim, c = settings.contentDim
        self.startProjection = Linear(d, p)
        self.endProjection = Linear(d, p)
        self.lengthProjection = Linear(3, p)
        self.priorProjection = Linear(1, p)
        self.contentPooler = SpanContentPooler(hiddenSize: hiddenSize, contentDim: c)
        self.contentProjection = Linear(c, p)
        self.candidateNorm = LayerNorm(dimensions: p, eps: 1e-5)
        self.queryProjection = Linear(hiddenSize, p)
        self.film = Linear(p, 2 * p)
        self.filmOutput = TwoLayerMLP(p, 64, 1, activation: .gelu)
        self.pairDim = p
        self.scale = 1.0 / Float(p).squareRoot()
        super.init()
    }

    /// - Returns: `score [B, P, Q]` (masked to -1e4) and `candidate [B, P, p]`.
    public func callAsFunction(
        pooled: DocumentCandidatePool.Pooled,
        boundaryStates: MLXArray,
        tokenStates: MLXArray, textMask: MLXArray, textLengths: MLXArray,
        queryStates: MLXArray, queryMask: MLXArray,
        marginals: BoundaryQueryHead.Marginals
    ) -> (score: MLXArray, candidate: MLXArray) {
        let dtype = boundaryStates.dtype
        let B = boundaryStates.dim(0), N = boundaryStates.dim(1)
        let P = pooled.indices.dim(1), Q = queryStates.dim(1)
        let starts = pooled.indices[0..., 0..., 0]                                       // [B, P] int32
        let ends = pooled.indices[0..., 0..., 1]

        // Endpoint representations.
        let startRep = BoundaryOps.gatherRows(startProjection(boundaryStates), starts)  // [B, P, p]
        let endRep = BoundaryOps.gatherRows(endProjection(boundaryStates), ends)

        // Length features in the activation dtype.
        let length = MLX.clip(ends - starts, min: Int32(1)).asType(dtype)
        let textLen = MLX.clip(textLengths.asType(.int32), min: Int32(1)).asType(dtype).reshaped([B, 1])
        let lengthFeatures = MLX.stacked(
            [MLX.log1p(length), length / textLen, MLX.rsqrt(length)], axis: -1)           // [B, P, 3]

        var candidate = startRep + endRep
            + lengthProjection(lengthFeatures)
            + priorProjection(pooled.compatLogits.asType(dtype).expandedDimensions(axis: -1))

        // Span content.
        let prefix = contentPooler.prefix(tokenStates: tokenStates, textMask: textMask)     // [B, N, c] fp32
        let content = contentPooler.pool(prefix: prefix, starts: starts, ends: ends, dtype: dtype)
        candidate = candidate + contentProjection(content)
        candidate = candidateNorm(candidate) * pooled.mask.asType(dtype).expandedDimensions(axis: -1)

        // Query interaction + FiLM.
        let query = queryProjection(queryStates)                                            // [B, Q, p]
        var score = MLX.matmul(candidate, query.transposed(0, 2, 1)) * scale                // [B, P, Q]
        let filmParams = film(query)
        let gamma = filmParams[0..., 0..., 0..<pairDim]
        let beta = filmParams[0..., 0..., pairDim...]
        let conditioned = candidate.expandedDimensions(axis: 2) * (gamma.expandedDimensions(axis: 1) + 1)
            + beta.expandedDimensions(axis: 1)                                              // [B, P, Q, p]
        score = score + filmOutput(conditioned).squeezed(axis: -1)

        // Boundary marginals of each pair, per query.
        let sIdx = MLX.broadcast(MLX.clip(starts, min: Int32(0), max: Int32(N - 1)).reshaped([B, 1, P]), to: [B, Q, P])
        let eIdx = MLX.broadcast(MLX.clip(ends, min: Int32(0), max: Int32(N - 1)).reshaped([B, 1, P]), to: [B, Q, P])
        score = score
            + MLX.takeAlong(marginals.startLogits, sIdx, axis: -1).transposed(0, 2, 1)
            + MLX.takeAlong(marginals.endLogits, eIdx, axis: -1).transposed(0, 2, 1)

        // Inside evidence from the fp32 prefix, normalised by sqrt(length).
        let span = (eIdx - sIdx).asType(.float32)
        let interval = MLX.takeAlong(marginals.insidePrefix, eIdx, axis: -1)
            - MLX.takeAlong(marginals.insidePrefix, sIdx, axis: -1)
            + marginals.insidePrefixMean * span                                             // [B, Q, P] fp32
        let inside = interval / MLX.sqrt(MLX.clip(span, min: Float(1)))
        score = score + inside.transposed(0, 2, 1).asType(dtype)

        let maskLogit = MLXArray(BoundaryOps.maskLogit).asType(dtype)
        score = MLX.which(pooled.mask.expandedDimensions(axis: -1), score, maskLogit)
        score = MLX.which(queryMask.expandedDimensions(axis: 1), score, maskLogit)
        return (score, candidate)
    }
}

// MARK: - Per-query modules (parameters; used by score_explicit_spans)

/// `boundaryHead.boundaryProposer.*` — only the parameters are needed on the shared-pool
/// path; `scoreExplicitPairs` serves choice fields and entity attributes.
public final class SparseBoundaryProposer: Module {
    public let endKeyProjection: Linear
    public let startPairProjection: Linear
    public let startQueryProjection: Linear
    let rotary: RotaryBoundaryEmbedding
    let boundaryDim: Int

    public init(hiddenSize: Int, boundaryDim: Int, rotaryBase: Float) {
        self.endKeyProjection = Linear(boundaryDim, boundaryDim)
        self.startPairProjection = Linear(boundaryDim, boundaryDim)
        self.startQueryProjection = Linear(hiddenSize, boundaryDim / 2)
        self.rotary = RotaryBoundaryEmbedding(dim: boundaryDim, base: rotaryBase)
        self.boundaryDim = boundaryDim
        super.init()
    }

    /// The marginal-free proposal prior for explicit `[B, Q, C, 2]` spans → `[B, Q, C]`
    /// (`score_explicit_pairs`), zero where invalid.
    public func scoreExplicitPairs(
        boundaryStates: MLXArray, queryStates: MLXArray, indices: MLXArray, validMask: MLXArray
    ) -> MLXArray {
        let N = boundaryStates.dim(1)
        let startAll = rotary(startPairProjection(boundaryStates))
        let endAll = rotary(endKeyProjection(boundaryStates))
        let gate = MLX.repeated(MLX.sigmoid(startQueryProjection(queryStates)), count: 2, axis: -1)
        let starts = MLX.clip(indices[.ellipsis, 0], min: Int32(0), max: Int32(N - 1))
        let ends = MLX.clip(indices[.ellipsis, 1], min: Int32(0), max: Int32(N - 1))
        let startStates = BoundaryOps.gatherStates(startAll, starts) * gate.expandedDimensions(axis: 2)
        let endStates = BoundaryOps.gatherStates(endAll, ends)
        let compat = (startStates * endStates).sum(axis: -1) / Float(boundaryDim).squareRoot()
        return MLX.which(validMask, compat, Float(0))
    }
}

/// `boundaryHead.pairScorer.*` — the per-query reranker.
public final class SparseBoundaryPairScorer: Module {
    public let startEndpointProjection: Linear
    public let endEndpointProjection: Linear
    public let queryGate: Linear
    public let lengthQueryProjection: Linear
    public let insideWeight: Linear
    public let endpointDifferenceProjection: Linear
    public let contentPooler: SpanContentPooler
    public let contentQueryProjection: Linear
    public let contentBias: Linear
    public let compatMix: Linear

    let rotary: RotaryBoundaryEmbedding
    let pairDim: Int
    let heads: Int

    public init(hiddenSize: Int, settings: BoundaryHeadSettings) {
        let d = settings.boundaryDim, p = settings.pairDim, c = settings.contentDim
        self.startEndpointProjection = Linear(d, p)
        self.endEndpointProjection = Linear(d, p)
        self.queryGate = Linear(hiddenSize, p / 2)
        self.lengthQueryProjection = Linear(hiddenSize, 3)
        self.insideWeight = Linear(hiddenSize, 1)
        self.endpointDifferenceProjection = Linear(2 * p, 1)
        self.contentPooler = SpanContentPooler(hiddenSize: hiddenSize, contentDim: c)
        self.contentQueryProjection = Linear(hiddenSize, c)
        self.contentBias = Linear(c, 1)
        self.compatMix = Linear(settings.multiheadPairCompatHeads, 1)
        self.rotary = RotaryBoundaryEmbedding(dim: p, base: settings.rotaryBase)
        self.pairDim = p
        self.heads = settings.multiheadPairCompatHeads
        super.init()
    }

    /// Rerank explicit `[B, Q, C, 2]` spans (`SparseBoundaryPairScorer.forward` on the
    /// "external proposer" path) → `[B, Q, C]` logits, -1e4 where invalid.
    public func callAsFunction(
        boundaryStates: MLXArray, queryStates: MLXArray,
        indices: MLXArray, validMask: MLXArray, compatLogits: MLXArray,
        marginals: BoundaryQueryHead.Marginals, textLengths: MLXArray,
        tokenStates: MLXArray, textMask: MLXArray
    ) -> MLXArray {
        let dtype = boundaryStates.dtype
        let B = indices.dim(0), Q = indices.dim(1), C = indices.dim(2)
        let N = boundaryStates.dim(1)
        let starts = indices[.ellipsis, 0]                                            // [B, Q, C]
        let ends = indices[.ellipsis, 1]
        let scale = 1.0 / Float(pairDim).squareRoot()

        // Endpoint compatibility (rotary endpoints, gated, multi-head mix).
        let startAll = rotary(startEndpointProjection(boundaryStates))
        let endAll = rotary(endEndpointProjection(boundaryStates))
        let sProj = BoundaryOps.gatherStates(startAll, starts)                        // [B, Q, C, p]
        let eProj = BoundaryOps.gatherStates(endAll, ends)
        let gate = MLX.repeated(MLX.sigmoid(queryGate(queryStates)), count: 2, axis: -1)
            .expandedDimensions(axis: 2)                                               // [B, Q, 1, p]
        let perHead = (sProj * gate * eProj).reshaped([B, Q, C, heads, pairDim / heads]).sum(axis: -1)
        var compat = compatMix(perHead).squeezed(axis: -1) * scale
        let difference = MLX.concatenated([sProj - eProj, MLX.abs(sProj - eProj)], axis: -1)
        compat = compat + endpointDifferenceProjection(difference).squeezed(axis: -1)

        // Marginals once, plus the marginal-free proposal prior.
        let sIdx = MLX.clip(starts, min: Int32(0), max: Int32(N - 1))
        let eIdx = MLX.clip(ends, min: Int32(0), max: Int32(N - 1))
        let a = MLX.takeAlong(marginals.startLogits, sIdx, axis: -1)
        let bmarg = MLX.takeAlong(marginals.endLogits, eIdx, axis: -1)
        let prior = MLX.which(validMask, compatLogits, Float(0)).asType(dtype)
        var score = compat + a + bmarg + prior

        // Span content.
        let prefix = contentPooler.prefix(tokenStates: tokenStates, textMask: textMask)
        let spanContent = contentPooler.poolPerQuery(prefix: prefix, starts: starts, ends: ends, dtype: dtype)
        let coefficient = contentQueryProjection(queryStates).expandedDimensions(axis: 2)
        let contentScale = 1.0 / Float(spanContent.dim(-1)).squareRoot()
        score = score + (spanContent * coefficient).sum(axis: -1) * contentScale
        score = score + contentBias(spanContent).squeezed(axis: -1)

        // Inside evidence, query-conditioned weight.
        let span = (eIdx - sIdx).asType(.float32)
        let interval = (MLX.takeAlong(marginals.insidePrefix, eIdx, axis: -1)
            - MLX.takeAlong(marginals.insidePrefix, sIdx, axis: -1)
            + marginals.insidePrefixMean * span).asType(dtype)
        let denom = MLX.sqrt(MLX.clip(ends - starts, min: Int32(1)).asType(dtype))
        let weight = insideWeight(queryStates)                                          // [B, Q, 1]
        score = score + weight * (interval / denom)

        // Continuous length features.
        let length = MLX.clip(ends - starts, min: Int32(1)).asType(.float32)
        let tl = MLX.clip(textLengths.asType(.float32), min: Float(1)).reshaped([B, 1, 1])
        let feats = MLX.stacked([MLX.log1p(length), length / tl, MLX.rsqrt(length)], axis: -1).asType(dtype)
        let lengthCoeff = lengthQueryProjection(queryStates).expandedDimensions(axis: 2)   // [B, Q, 1, 3]
        score = score + (feats * lengthCoeff).sum(axis: -1)

        return MLX.which(validMask, score, MLXArray(BoundaryOps.maskLogit).asType(dtype))
    }
}

// MARK: - Boundary head

/// The GLiNER2.5 decoder head on the shared-pool path.
///
/// Prefix `boundaryHead.*`.
public final class BoundaryHead: Module {
    public let boundaryEncoder: BoundaryEncoder
    public let boundaryQueryHead: BoundaryQueryHead
    public let sharedPoolBuilder: DocumentCandidatePool
    public let sharedPoolScorer: SharedPoolScorer
    public let boundaryProposer: SparseBoundaryProposer
    public let pairScorer: SparseBoundaryPairScorer
    public let candidateEncoder: Linear
    public let nullProjection: Linear
    public let countHead: Linear

    public let settings: BoundaryHeadSettings

    public init(hiddenSize: Int, settings: BoundaryHeadSettings) {
        let d = settings.boundaryDim
        self.settings = settings
        self.boundaryEncoder = BoundaryEncoder(hiddenSize: hiddenSize, settings: settings)
        self.boundaryQueryHead = BoundaryQueryHead(hiddenSize: hiddenSize, boundaryDim: d)
        self.sharedPoolBuilder = DocumentCandidatePool(boundaryDim: d, settings: settings)
        self.sharedPoolScorer = SharedPoolScorer(hiddenSize: hiddenSize, settings: settings)
        self.boundaryProposer = SparseBoundaryProposer(hiddenSize: hiddenSize, boundaryDim: d,
                                                       rotaryBase: settings.rotaryBase)
        self.pairScorer = SparseBoundaryPairScorer(hiddenSize: hiddenSize, settings: settings)
        self.candidateEncoder = Linear(2 * d, hiddenSize)
        self.nullProjection = Linear(hiddenSize, 1)
        self.countHead = Linear(hiddenSize, 1)
        super.init()
    }

    /// Everything the decoder needs, plus the intermediates the parity tests pin.
    public struct Output {
        /// `[B, Q, P]` reranker logits, -1e4 where the pair or query is invalid.
        public let pairLogits: MLXArray
        /// `[B, P, 2]` int32 `(start, end)` boundary indices of the shared pool.
        public let indices: MLXArray
        /// `[B, P]` bool.
        public let validMask: MLXArray
        /// `[B, Q]` abstention logits.
        public let nullLogits: MLXArray
        /// `[B, Q]` Poisson log-rates (unused unless `adaptive_threshold`).
        public let countLogRates: MLXArray
        /// `[B, P, H]` candidate states for the record head, zero where invalid.
        public let candidateStates: MLXArray
        public let boundary: BoundaryEncoder.Output
        public let marginals: BoundaryQueryHead.Marginals
        public let pooled: DocumentCandidatePool.Pooled
        /// `[B, P, Q]` scorer output before the transpose.
        public let scorerScore: MLXArray
        /// `[B, P, p]` scorer candidate vectors.
        public let scorerCandidate: MLXArray
    }

    /// - Parameters:
    ///   - tokenStates: `[B, L, H]` word states (zero where masked)
    ///   - textMask: `[B, L]` bool
    ///   - queryStates: `[B, Q, H]` query-marker states (zero where masked)
    ///   - queryMask: `[B, Q]` bool
    public func callAsFunction(
        tokenStates: MLXArray, textMask: MLXArray,
        queryStates: MLXArray, queryMask: MLXArray
    ) -> Output {
        let textLengths = textMask.asType(.int32).sum(axis: -1)                       // [B]
        let boundary = boundaryEncoder(tokenStates, textLengths: textLengths)
        let marginals = boundaryQueryHead(
            boundaryStates: boundary.states, boundaryMask: boundary.mask,
            tokenStates: tokenStates, textMask: textMask,
            queryStates: queryStates, queryMask: queryMask)
        let pooled = sharedPoolBuilder(
            boundaryStates: boundary.states, boundaryMask: boundary.mask, queryMask: queryMask,
            startLogits: marginals.startLogits, endLogits: marginals.endLogits)
        let scored = sharedPoolScorer(
            pooled: pooled, boundaryStates: boundary.states,
            tokenStates: tokenStates, textMask: textMask, textLengths: textLengths,
            queryStates: queryStates, queryMask: queryMask, marginals: marginals)

        let starts = pooled.indices[0..., 0..., 0]
        let ends = pooled.indices[0..., 0..., 1]
        let candidateStates = candidateEncoder(MLX.concatenated(
            [BoundaryOps.gatherRows(boundary.states, starts),
             BoundaryOps.gatherRows(boundary.states, ends)], axis: -1))
            * pooled.mask.asType(tokenStates.dtype).expandedDimensions(axis: -1)

        return Output(
            pairLogits: scored.score.transposed(0, 2, 1),
            indices: pooled.indices,
            validMask: pooled.mask,
            nullLogits: nullProjection(queryStates).squeezed(axis: -1),
            countLogRates: countHead(queryStates).squeezed(axis: -1),
            candidateStates: candidateStates,
            boundary: boundary,
            marginals: marginals,
            pooled: pooled,
            scorerScore: scored.score,
            scorerCandidate: scored.candidate)
    }

    /// Score caller-provided half-open spans for every query (`score_explicit_spans`):
    /// the per-query proposer prior plus the pair reranker, bypassing top-k selection.
    ///
    /// - Parameters:
    ///   - indices: `[B, Q, C, 2]` int32 `(start, end)` boundary indices
    ///   - validMask: optional `[B, Q, C]`
    /// - Returns: `[B, Q, C]` logits, -1e4 where the span is illegal or masked.
    public func scoreExplicitSpans(
        tokenStates: MLXArray, textMask: MLXArray,
        queryStates: MLXArray, queryMask: MLXArray,
        indices: MLXArray, validMask: MLXArray? = nil
    ) -> MLXArray {
        let B = indices.dim(0)
        let textLengths = textMask.asType(.int32).sum(axis: -1)
        let starts = indices[.ellipsis, 0]
        let ends = indices[.ellipsis, 1]
        var legal = BoundaryOps.and(
            BoundaryOps.and(starts .>= Int32(0), ends .> starts),
            BoundaryOps.and(ends .<= textLengths.reshaped([B, 1, 1]), queryMask.expandedDimensions(axis: -1)))
        if let validMask { legal = BoundaryOps.and(legal, validMask) }

        let boundary = boundaryEncoder(tokenStates, textLengths: textLengths)
        let marginals = boundaryQueryHead(
            boundaryStates: boundary.states, boundaryMask: boundary.mask,
            tokenStates: tokenStates, textMask: textMask,
            queryStates: queryStates, queryMask: queryMask)
        let compat = boundaryProposer.scoreExplicitPairs(
            boundaryStates: boundary.states, queryStates: queryStates, indices: indices, validMask: legal)
        return pairScorer(
            boundaryStates: boundary.states, queryStates: queryStates,
            indices: indices, validMask: legal, compatLogits: compat,
            marginals: marginals, textLengths: textLengths,
            tokenStates: tokenStates, textMask: textMask)
    }
}
