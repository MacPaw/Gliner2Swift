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
// RelationScorer.swift
// GLiNER2.5 sparse relation scoring.
//
// Matches Python: gliner2/models/boundary/relations.py:SparseRelationScorer. The typed
// pair generator that feeds it runs on the host (Decoding/RelationDecoder.swift).

import Foundation
import MLX
import MLXNN

/// One proposed (head span, tail span) pair for a relation type of one sample.
public struct RelationPair: Sendable, Equatable {
    public let relationIndex: Int
    public let headStart: Int
    public let headEnd: Int
    public let tailStart: Int
    public let tailEnd: Int
    public let headProbability: Float
    public let tailProbability: Float
}

/// Scores proposed relation pairs from four endpoint word states, the relation query
/// (head-role ⧺ tail-role marker states), relative order and normalised distance, plus a
/// gated bilinear / linear term over span-mean content.
///
/// Prefix `relationScorer.*`.
public final class SparseRelationScorer: Module {
    public let mlp: TwoLayerMLP
    public let headContentProjection: Linear
    public let tailContentProjection: Linear
    public let relationContentGate: Linear
    public let contentLinear: Linear
    let hiddenSize: Int

    public init(hiddenSize: Int, relationQueryDim: Int) {
        self.hiddenSize = hiddenSize
        self.mlp = TwoLayerMLP(4 * hiddenSize + relationQueryDim + 2, hiddenSize, 1, activation: .gelu)
        self.headContentProjection = Linear(hiddenSize, hiddenSize)
        self.tailContentProjection = Linear(hiddenSize, hiddenSize)
        self.relationContentGate = Linear(relationQueryDim, hiddenSize)
        self.contentLinear = Linear(2 * hiddenSize + relationQueryDim, 1)
        super.init()
    }

    /// Score one sample's pairs.
    ///
    /// - Parameters:
    ///   - wordStates: `[L, H]` the sample's word states, **unpadded**. Python normalises the
    ///     head/tail distance by the padded width of the batch it happens to be in; with a
    ///     single-text call that is the true word count, which is what is used here so
    ///     results do not depend on batching.
    ///   - relationQueryStates: `[R, 2H]`
    /// - Returns: `[P]` logits.
    public func callAsFunction(
        wordStates: MLXArray, relationQueryStates: MLXArray, pairs: [RelationPair]
    ) -> MLXArray {
        let dtype = wordStates.dtype
        let L = wordStates.dim(0)
        let P = pairs.count
        func positions(_ f: (RelationPair) -> Int) -> MLXArray {
            MLXArray(pairs.map { Int32(min(max(f($0), 0), max(L - 1, 0))) })
        }
        let hStart = MLX.take(wordStates, positions { $0.headStart }, axis: 0)          // [P, H]
        let hEnd = MLX.take(wordStates, positions { $0.headEnd - 1 }, axis: 0)
        let tStart = MLX.take(wordStates, positions { $0.tailStart }, axis: 0)
        let tEnd = MLX.take(wordStates, positions { $0.tailEnd - 1 }, axis: 0)
        let rel = MLX.take(relationQueryStates, MLXArray(pairs.map { Int32($0.relationIndex) }), axis: 0)

        let delta = MLXArray(pairs.map { Float($0.tailStart - $0.headStart) }).asType(dtype)
        let order = MLX.sign(delta).reshaped([P, 1])
        let dist = (MLX.abs(delta) / Float(max(L, 1))).reshaped([P, 1])

        let feats = MLX.concatenated([hStart, hEnd, tStart, tEnd, rel, order, dist], axis: -1)
        var score = mlp(feats).squeezed(axis: -1)                                         // [P]

        // Span-mean content through a prefix sum (cast back to the activation dtype, as
        // the reference does).
        let prefix = MLX.concatenated(
            [MLXArray.zeros([1, hiddenSize], dtype: dtype),
             wordStates.asType(.float32).cumsum(axis: 0).asType(dtype)], axis: 0)          // [L+1, H]
        func pool(_ start: (RelationPair) -> Int, _ end: (RelationPair) -> Int) -> MLXArray {
            let s = MLXArray(pairs.map { Int32(min(max(start($0), 0), L)) })
            let e = MLXArray(pairs.map { Int32(min(max(end($0), 0), L)) })
            let sum = MLX.take(prefix, e, axis: 0) - MLX.take(prefix, s, axis: 0)
            let width = MLXArray(pairs.map { Float(max(end($0) - start($0), 1)) }).asType(dtype).reshaped([P, 1])
            return sum / width
        }
        let headContent = headContentProjection(pool({ $0.headStart }, { $0.headEnd }))
        let tailContent = tailContentProjection(pool({ $0.tailStart }, { $0.tailEnd }))
        let gate = MLX.sigmoid(relationContentGate(rel))
        let biaffine = (headContent * gate * tailContent).sum(axis: -1) / Float(hiddenSize).squareRoot()
        let linear = contentLinear(MLX.concatenated([headContent, tailContent, rel], axis: -1)).squeezed(axis: -1)
        score = score + biaffine + linear
        return score
    }
}
