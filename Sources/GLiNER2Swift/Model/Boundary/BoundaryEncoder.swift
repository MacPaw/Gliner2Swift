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
// BoundaryEncoder.swift
// GLiNER2.5 boundary encoder and boundary/query marginals.
//
// Matches Python: gliner2/models/boundary/encoding.py (BoundaryEncoder,
// BoundaryAttentionBlock, ResidualSwiGLU) and heads.py (BoundaryQueryHead).
// Reference: gliner2/models/boundary/encoding.py and heads.py.
//
// Notation: B batch, L padded words, N = L + 1 boundaries (boundary i sits between word
// i-1 and word i; 0 is BOS, textLength is EOS), Q queries, H encoder hidden, d = 128.

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Shared helpers

/// Small tensor helpers shared by the boundary modules.
enum BoundaryOps {
    /// Finite masking sentinel, chosen upstream to survive fp16 arithmetic. Never replace
    /// with `-inf`: several sentinels are summed and differenced downstream.
    static let maskLogit: Float = -1.0e4

    /// `0 ..< n` as an int32 array.
    static func arange(_ n: Int) -> MLXArray {
        MLXArray((0..<n).map { Int32($0) })
    }

    /// Gather rows of `[B, N, D]` at `[B, K]` indices → `[B, K, D]`.
    ///
    /// Flattens to `[B*N, D]` and gathers with per-batch offsets: one `take` instead of a
    /// `takeAlong` with the index expanded over `D`.
    static func gatherRows(_ rows: MLXArray, _ indices: MLXArray) -> MLXArray {
        let B = rows.dim(0), N = rows.dim(1), D = rows.dim(2)
        let K = indices.dim(1)
        let offsets = (arange(B) * Int32(N)).reshaped([B, 1])
        let flat = (indices.asType(.int32) + offsets).reshaped([B * K])
        return MLX.take(rows.reshaped([B * N, D]), flat, axis: 0).reshaped([B, K, D])
    }

    /// Gather `[B, N, D]` at `[B, Q, C]` indices → `[B, Q, C, D]` (`gather_states`).
    static func gatherStates(_ rows: MLXArray, _ indices: MLXArray) -> MLXArray {
        let B = indices.dim(0), Q = indices.dim(1), C = indices.dim(2)
        let N = rows.dim(1), D = rows.dim(2)
        let clamped = MLX.clip(indices.asType(.int32), min: Int32(0), max: Int32(N - 1))
        return gatherRows(rows, clamped.reshaped([B, Q * C])).reshaped([B, Q, C, D])
    }

    /// Order-preserving int64 image of fp32 scores: `a < b ⇔ orderedBits(a) < orderedBits(b)`
    /// for finite values. Bit-casts the float and folds the negative half so that more
    /// negative floats map to smaller integers.
    static func orderedBits(_ scores: MLXArray) -> MLXArray {
        let v = scores.asType(.float32).view(dtype: .int32).asType(.int64)
        // For negative floats the raw bit pattern orders in reverse; -v - 2^31 - 1 flips it
        // while keeping every negative float below every non-negative one.
        return MLX.which(v .>= 0, v, (v * -1) - Int64(2_147_483_649))
    }

    /// Positions that sort a row by (score descending, index ascending) — the order a
    /// stable descending `torch.sort` produces — with no dependence on MLX sort stability:
    /// one exact int64 composite key per element, `argSort`ed ascending.
    static func argsortScoreDesc(_ scores: MLXArray) -> MLXArray {
        let n = scores.dim(-1)
        let index = arange(n).asType(.int64)
        let key = (orderedBits(scores) * -1) * Int64(4_294_967_296) + index
        return MLX.argSort(key, axis: -1)
    }

    /// Positions that sort by (key ascending, score descending). Keys must be < 2^30.
    static func argsortKeyAscScoreDesc(keys: MLXArray, scores: MLXArray) -> MLXArray {
        let secondary = Int64(2_147_483_648) - orderedBits(scores)   // in (0, 2^32]
        let composite = keys.asType(.int64) * Int64(8_589_934_592) + secondary
        return MLX.argSort(composite, axis: -1)
    }

    /// Positions that sort by (score descending, key ascending). Keys must be < 2^32.
    static func argsortScoreDescKeyAsc(scores: MLXArray, keys: MLXArray) -> MLXArray {
        let composite = (orderedBits(scores) * -1) * Int64(4_294_967_296) + keys.asType(.int64)
        return MLX.argSort(composite, axis: -1)
    }

    /// Boolean `a && b` on MLX arrays.
    static func and(_ a: MLXArray, _ b: MLXArray) -> MLXArray { MLX.logicalAnd(a, b) }
    static func or(_ a: MLXArray, _ b: MLXArray) -> MLXArray { MLX.logicalOr(a, b) }
}

// MARK: - Two-Linear MLP

/// `Linear → activation → Linear`, with the two Linear layers stored as `layers.0` and
/// `layers.1` so the converted key convention (`film_output.0` → `layers.0`,
/// `film_output.3` → `layers.1`) matches the parameter paths directly.
public final class TwoLayerMLP: Module {
    public let layers: [Linear]
    public let activation: ActivationType

    public init(_ inputDim: Int, _ hiddenDim: Int, _ outputDim: Int, activation: ActivationType) {
        self.layers = [Linear(inputDim, hiddenDim), Linear(hiddenDim, outputDim)]
        self.activation = activation
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = layers[0](x)
        let activated: MLXArray
        switch activation {
        case .gelu: activated = MLXNN.gelu(h)          // exact erf, as nn.GELU()
        case .relu: activated = MLX.maximum(h, 0)
        case .tanh: activated = MLX.tanh(h)
        case .sigmoid: activated = MLX.sigmoid(h)
        case .leakyRelu: activated = MLX.maximum(h, h * 0.01)
        }
        return layers[1](activated)
    }
}

// MARK: - Boundary attention block

/// Pre-norm windowed self-attention over boundary states.
///
/// ```python
/// qkv = qkv_projection(norm(states)).view(b, n, 3, heads, head_dim)
/// query, key, value = qkv.permute(2, 0, 3, 1, 4)
/// allowed = key_padding & (|i - j| <= window) | eye(n)
/// attended = F.scaled_dot_product_attention(query, key, value, attn_mask=allowed)
/// return (states + output_projection(attended)) * mask[..., None]
/// ```
public final class BoundaryAttentionBlock: Module {
    public let norm: LayerNorm
    public let qkvProjection: Linear
    public let outputProjection: Linear
    let heads: Int
    let headDim: Int

    public init(dim: Int, heads: Int) {
        self.norm = LayerNorm(dimensions: dim, eps: 1e-5)
        self.qkvProjection = Linear(dim, 3 * dim)
        self.outputProjection = Linear(dim, dim)
        self.heads = heads
        self.headDim = dim / heads
        super.init()
    }

    /// - Parameters:
    ///   - states: `[B, N, d]`
    ///   - maskF: `[B, N, 1]` float mask (1 for valid boundaries)
    ///   - additiveMask: `[B, 1, N, N]` additive attention mask (0 allowed / -1e4 blocked)
    public func callAsFunction(_ states: MLXArray, maskF: MLXArray, additiveMask: MLXArray) -> MLXArray {
        let B = states.dim(0), N = states.dim(1), d = states.dim(2)
        let qkv = qkvProjection(norm(states))
            .reshaped([B, N, 3, heads, headDim])
            .transposed(2, 0, 3, 1, 4)                       // [3, B, heads, N, headDim]
        let attended = MLXFast.scaledDotProductAttention(
            queries: qkv[0], keys: qkv[1], values: qkv[2],
            scale: 1.0 / Float(headDim).squareRoot(),
            mask: additiveMask
        )                                                    // [B, heads, N, headDim]
        let merged = attended.transposed(0, 2, 1, 3).reshaped([B, N, d])
        return (states + outputProjection(merged)) * maskF
    }
}

// MARK: - Residual SwiGLU

/// `states + output_projection(value * silu(gate))` with `value, gate = chunk(input_projection(norm(states)), 2)`.
public final class ResidualSwiGLU: Module {
    public let norm: LayerNorm
    public let inputProjection: Linear
    public let outputProjection: Linear
    let hidden: Int

    public init(dim: Int, hidden: Int) {
        self.norm = LayerNorm(dimensions: dim, eps: 1e-5)
        self.inputProjection = Linear(dim, 2 * hidden)
        self.outputProjection = Linear(hidden, dim)
        self.hidden = hidden
        super.init()
    }

    public func callAsFunction(_ states: MLXArray) -> MLXArray {
        let projected = inputProjection(norm(states))
        let value = projected[0..., 0..., 0..<hidden]
        let gate = projected[0..., 0..., hidden...]
        return states + outputProjection(value * MLXNN.silu(gate))
    }
}

// MARK: - Boundary encoder

/// Builds one state per word boundary from the word states either side of it, then
/// refines with windowed attention and a SwiGLU block.
///
/// Prefix `boundaryHead.boundaryEncoder.*`.
public final class BoundaryEncoder: Module {
    public var bosState: MLXArray
    public var eosState: MLXArray
    public let leftProjection: Linear
    public let rightProjection: Linear
    public let outputProjection: Linear
    public let layerNorm: LayerNorm
    public let attentionBlocks: [BoundaryAttentionBlock]
    public let refinementBlocks: [ResidualSwiGLU]

    let window: Int
    let dim: Int

    /// Window-plus-diagonal masks by boundary count; the `[N, N]` boolean is the only
    /// quadratic tensor on the head, so it is built once per shape.
    private let windowMasks = WindowMaskCache()

    public init(hiddenSize: Int, settings: BoundaryHeadSettings) {
        let d = settings.boundaryDim
        self.dim = d
        self.window = settings.boundaryAttentionWindow
        self.bosState = MLXArray.zeros([hiddenSize])
        self.eosState = MLXArray.zeros([hiddenSize])
        self.leftProjection = Linear(hiddenSize, d)
        self.rightProjection = Linear(hiddenSize, d)
        self.outputProjection = Linear(2 * d, d)
        self.layerNorm = LayerNorm(dimensions: d, eps: 1e-5)
        self.attentionBlocks = (0..<settings.boundaryAttentionLayers).map { _ in
            BoundaryAttentionBlock(dim: d, heads: settings.boundaryAttentionHeads)
        }
        let ffnHidden = Int((Float(d) * settings.boundaryFfnMultiplier).rounded())
        self.refinementBlocks = (0..<settings.boundaryRefinementLayers).map { _ in
            ResidualSwiGLU(dim: d, hidden: ffnHidden)
        }
        super.init()
    }

    /// Boundary states and their validity mask.
    public struct Output {
        /// `[B, N, d]`, zero on invalid boundaries.
        public let states: MLXArray
        /// `[B, N]` bool: boundary `i` is valid iff `i <= textLength`.
        public let mask: MLXArray
    }

    /// - Parameters:
    ///   - tokenStates: `[B, L, H]` word states, zero where `textMask` is false
    ///   - textLengths: `[B]` int32 word counts
    public func callAsFunction(_ tokenStates: MLXArray, textLengths: MLXArray) -> Output {
        let B = tokenStates.dim(0), L = tokenStates.dim(1), H = tokenStates.dim(2)
        let N = L + 1
        let dtype = tokenStates.dtype

        let bos = MLX.broadcast(bosState.asType(dtype).reshaped([1, 1, H]), to: [B, 1, H])
        let eos = MLX.broadcast(eosState.asType(dtype).reshaped([1, 1, H]), to: [B, 1, H])
        let positions = BoundaryOps.arange(N).reshaped([1, N])
        let lengths = textLengths.asType(.int32).reshaped([B, 1])

        // left[i] = word i-1 (BOS at 0); right[i] = word i (EOS at i == textLength).
        let left = MLX.concatenated([bos, tokenStates], axis: 1)
        var right = MLX.concatenated([tokenStates, eos], axis: 1)
        let isEos = (positions .== lengths).expandedDimensions(axis: -1)         // [B, N, 1]
        right = MLX.which(isEos, eos, right)

        var states = outputProjection(
            MLX.concatenated([leftProjection(left), rightProjection(right)], axis: -1))
        states = layerNorm(states)

        let mask = positions .<= lengths                                           // [B, N]
        let maskF = mask.asType(dtype).expandedDimensions(axis: -1)               // [B, N, 1]

        if !attentionBlocks.isEmpty {
            // allowed[b, i, j] = mask[b, j] & |i-j| <= window | i == j
            let windowOrEye = windowMasks.mask(n: N, window: window)              // [N, N] bool
            let keyMask = mask.reshaped([B, 1, 1, N])
            let eye = windowMasks.eye(n: N)
            let allowed = BoundaryOps.or(
                BoundaryOps.and(keyMask, windowOrEye.reshaped([1, 1, N, N])),
                eye.reshaped([1, 1, N, N]))
            let additive = MLX.which(allowed, MLXArray(Float(0)), MLXArray(BoundaryOps.maskLogit))
                .asType(dtype)                                                     // [B, 1, N, N]
            for block in attentionBlocks {
                states = block(states, maskF: maskF, additiveMask: additive)
            }
        }
        for block in refinementBlocks {
            states = block(states)
        }
        states = states * maskF
        return Output(states: states, mask: mask)
    }

    /// Plain Swift class so `Module` reflection ignores the cached arrays.
    final class WindowMaskCache {
        private var masks: [Int: MLXArray] = [:]
        private var eyes: [Int: MLXArray] = [:]

        func mask(n: Int, window: Int) -> MLXArray {
            if let cached = masks[n] { return cached }
            let pos = BoundaryOps.arange(n)
            let diff = MLX.abs(pos.reshaped([n, 1]) - pos.reshaped([1, n]))
            let m = window > 0 ? (diff .<= Int32(window)) : MLXArray.ones([n, n], dtype: .bool)
            masks[n] = m
            return m
        }

        func eye(n: Int) -> MLXArray {
            if let cached = eyes[n] { return cached }
            let pos = BoundaryOps.arange(n)
            let e = pos.reshaped([n, 1]) .== pos.reshaped([1, n])
            eyes[n] = e
            return e
        }
    }
}

// MARK: - Boundary query head

/// Start / end / inside marginals of every query against every boundary / word.
///
/// Prefix `boundaryHead.boundaryQueryHead.*`.
public final class BoundaryQueryHead: Module {
    public let startBoundaryProjection: Linear
    public let startQueryProjection: Linear
    public let endBoundaryProjection: Linear
    public let endQueryProjection: Linear
    public let insideTextProjection: Linear
    public let insideQueryProjection: Linear
    let scale: Float

    public init(hiddenSize: Int, boundaryDim: Int) {
        self.startBoundaryProjection = Linear(boundaryDim, boundaryDim)
        self.startQueryProjection = Linear(hiddenSize, boundaryDim)
        self.endBoundaryProjection = Linear(boundaryDim, boundaryDim)
        self.endQueryProjection = Linear(hiddenSize, boundaryDim)
        self.insideTextProjection = Linear(hiddenSize, boundaryDim)
        self.insideQueryProjection = Linear(hiddenSize, boundaryDim)
        self.scale = 1.0 / Float(boundaryDim).squareRoot()
        super.init()
    }

    public struct Marginals {
        /// `[B, Q, N]`, masked to -1e4.
        public let startLogits: MLXArray
        /// `[B, Q, N]`, masked to -1e4.
        public let endLogits: MLXArray
        /// `[B, Q, L]`, masked to -1e4.
        public let insideLogits: MLXArray
        /// `[B, Q, N]` fp32 prefix sums of the mean-centred inside logits.
        public let insidePrefix: MLXArray
        /// `[B, Q, 1]` fp32 per-query mean of the inside logits over valid words.
        public let insidePrefixMean: MLXArray
    }

    /// - Parameters:
    ///   - boundaryStates: `[B, N, d]`
    ///   - boundaryMask: `[B, N]` bool
    ///   - tokenStates: `[B, L, H]`
    ///   - textMask: `[B, L]` bool
    ///   - queryStates: `[B, Q, H]`
    ///   - queryMask: `[B, Q]` bool
    public func callAsFunction(
        boundaryStates: MLXArray, boundaryMask: MLXArray,
        tokenStates: MLXArray, textMask: MLXArray,
        queryStates: MLXArray, queryMask: MLXArray
    ) -> Marginals {
        let maskLogit = MLXArray(BoundaryOps.maskLogit)

        let sb = startBoundaryProjection(boundaryStates)                  // [B, N, d]
        let sq = startQueryProjection(queryStates)                        // [B, Q, d]
        let eb = endBoundaryProjection(boundaryStates)
        let eq = endQueryProjection(queryStates)
        let it = insideTextProjection(tokenStates)                        // [B, L, d]
        let iq = insideQueryProjection(queryStates)

        var startLogits = MLX.matmul(sq, sb.transposed(0, 2, 1)) * scale  // [B, Q, N]
        var endLogits = MLX.matmul(eq, eb.transposed(0, 2, 1)) * scale
        var insideLogits = MLX.matmul(iq, it.transposed(0, 2, 1)) * scale // [B, Q, L]

        let bKeep = BoundaryOps.and(boundaryMask.expandedDimensions(axis: 1),
                                    queryMask.expandedDimensions(axis: -1))   // [B, Q, N]
        let tKeep = BoundaryOps.and(textMask.expandedDimensions(axis: 1),
                                    queryMask.expandedDimensions(axis: -1))   // [B, Q, L]
        startLogits = MLX.which(bKeep, startLogits, maskLogit.asType(startLogits.dtype))
        endLogits = MLX.which(bKeep, endLogits, maskLogit.asType(endLogits.dtype))
        insideLogits = MLX.which(tKeep, insideLogits, maskLogit.asType(insideLogits.dtype))

        // Inside prefix, explicitly fp32 and centred per query (Python: .float()).
        let x = MLX.which(tKeep, insideLogits, MLXArray(Float(0))).asType(.float32)
        let keepF = tKeep.asType(.float32)
        let count = MLX.clip(keepF.sum(axis: -1, keepDims: true), min: Float(1))
        let mean = x.sum(axis: -1, keepDims: true) / count                       // [B, Q, 1]
        let centred = (x - mean) * keepF
        let B = x.dim(0), Q = x.dim(1)
        let prefix = MLX.concatenated(
            [MLXArray.zeros([B, Q, 1], dtype: .float32), centred.cumsum(axis: -1)],
            axis: -1)                                                             // [B, Q, N]

        return Marginals(
            startLogits: startLogits, endLogits: endLogits, insideLogits: insideLogits,
            insidePrefix: prefix, insidePrefixMean: mean)
    }
}
