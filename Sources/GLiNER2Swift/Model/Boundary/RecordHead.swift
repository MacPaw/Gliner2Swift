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
// RecordHead.swift
// GLiNER2.5 record (instance formation) head: natural / latent / anchorless instances
// with null-aware field assignment.
//
// Matches Python: gliner2/models/boundary/records.py:RecordHead (`forward_group`,
// `_assign_logits`, `_anchorless_states`). Decoding lives in Decoding/RecordDecoder.swift.

import Foundation
import MLX
import MLXNN

/// How many mentions a field may bind within one record instance.
public enum FieldCardinality: String, Sendable {
    case optionalOne = "optional_one"   // 0 or 1 (scalar, may be ABSENT)
    case requiredOne = "required_one"   // exactly 1 (scalar, never ABSENT)
    case zeroOrMore = "zero_or_more"    // list, may be empty
    case oneOrMore = "one_or_more"      // list, at least one

    public var isScalar: Bool { self == .optionalOne || self == .requiredOne }
    public var allowsAbsent: Bool { self == .optionalOne || self == .zeroOrMore }
}

/// One field of a record schema, bound to its boundary query id.
public struct RecordFieldSpec: Sendable, Equatable {
    public let queryId: Int
    public let name: String
    public let roleIndex: Int
    public let cardinality: FieldCardinality
    public let isAnchor: Bool
    public let exclusive: Bool

    public var isScalar: Bool { cardinality.isScalar }
    public var allowsAbsent: Bool { cardinality.allowsAbsent }
}

/// A record / event schema group compiled against a concrete query layout.
public struct RecordSpec: Sendable, Equatable {
    public let taskIndex: Int
    public let taskName: String
    /// `"natural"`, `"latent"` or `"anchorless"`.
    public let mode: String
    public let fields: [RecordFieldSpec]
    public let anchorQueryId: Int?
}

/// One decoded record group, read back to the host.
struct RecordGroupOutput {
    let spec: RecordSpec
    /// `[Ni]` object / anchor logits.
    let objectLogits: [Float]
    /// Per field: `[Ni][1 + Cf]` assignment logits (column 0 = ABSENT).
    let assignLogits: [[[Float]]]
    /// Per field: the surviving candidate spans, in pool order.
    let fieldSpans: [[(start: Int, end: Int)]]
    /// Per instance: the anchor span (natural mode), else nil.
    let instanceSpans: [(start: Int, end: Int)?]
    var instanceCount: Int { objectLogits.count }
}

/// Prefix `recordDecoder.*`.
public final class RecordHead: Module {
    public let instProj: Linear
    public let fieldProj: Linear
    public let candProj: Linear
    public var nullEmbed: MLXArray
    public let objectHead: Linear
    public let latentSeedHead: Linear
    public var instanceEmbed: MLXArray
    public let qProj: Linear
    public let kProj: Linear
    public let vProj: Linear
    let recordDim: Int
    let hiddenSize: Int

    public init(hiddenSize: Int, recordDim: Int, instanceQueries: Int) {
        self.hiddenSize = hiddenSize
        self.recordDim = recordDim
        self.instProj = Linear(hiddenSize, recordDim)
        self.fieldProj = Linear(hiddenSize, recordDim)
        self.candProj = Linear(hiddenSize, recordDim)
        self.nullEmbed = MLXArray.zeros([recordDim])
        self.objectHead = Linear(hiddenSize, 1)
        self.latentSeedHead = Linear(hiddenSize, 1)
        self.instanceEmbed = MLXArray.zeros([instanceQueries, hiddenSize])
        self.qProj = Linear(hiddenSize, recordDim)
        self.kProj = Linear(hiddenSize, recordDim)
        self.vProj = Linear(hiddenSize, hiddenSize)
        super.init()
    }

    /// Decode one record group for one sample (`forward_group`), reading the tiny
    /// per-group tensors back to the host.
    ///
    /// - Parameters:
    ///   - queryStates: `[Q, H]` this sample's query states
    ///   - candidateStates: `[P, H]` this sample's candidate states
    ///   - readback: the sample's read-back candidates (spans, validity, raw logits)
    func forwardGroup(
        spec: RecordSpec, queryStates: MLXArray, candidateStates: MLXArray,
        readback: BoundaryReadback, sample b: Int
    ) -> RecordGroupOutput {
        let dtype = queryStates.dtype
        let queryCount = queryStates.dim(0)

        // Per-field surviving candidates.
        var fieldStates: [MLXArray] = []
        var fieldSpans: [[(start: Int, end: Int)]] = []
        var fieldLogits: [[Float]] = []
        for field in spec.fields {
            let valid = field.queryId >= 0 && field.queryId < queryCount && field.queryId < readback.queries
            let qid = min(max(field.queryId, 0), max(queryCount - 1, 0))
            var keep: [Int32] = []
            var spans: [(start: Int, end: Int)] = []
            var logits: [Float] = []
            if valid {
                for c in 0..<readback.pool where readback.isValid(sample: b, candidate: c) {
                    keep.append(Int32(c))
                    spans.append(readback.span(sample: b, candidate: c))
                    logits.append(readback.pairLogit(sample: b, query: qid, candidate: c))
                }
            }
            fieldStates.append(keep.isEmpty
                               ? MLXArray.zeros([0, hiddenSize], dtype: dtype)
                               : MLX.take(candidateStates, MLXArray(keep), axis: 0))
            fieldSpans.append(spans)
            fieldLogits.append(logits)
        }
        let fieldQueryIds = MLXArray(spec.fields.map { Int32(min(max($0.queryId, 0), max(queryCount - 1, 0))) })
        let fq = MLX.take(queryStates, fieldQueryIds, axis: 0)                        // [F, H]

        // Instance formation.
        var instStates: MLXArray
        var objectLogits: MLXArray
        var instanceSpans: [(start: Int, end: Int)?] = []
        switch spec.mode {
        case "natural":
            let anchorField = spec.fields.firstIndex { $0.queryId == spec.anchorQueryId } ?? 0
            instStates = fieldStates[anchorField]
            objectLogits = MLXArray(fieldLogits[anchorField])
            instanceSpans = fieldSpans[anchorField].map { Optional($0) }
        case "latent":
            let nonEmpty = fieldStates.enumerated().filter { $0.element.dim(0) > 0 }
            if nonEmpty.isEmpty {
                instStates = MLXArray.zeros([0, hiddenSize], dtype: dtype)
                objectLogits = MLXArray.zeros([0], dtype: dtype)
            } else {
                instStates = MLX.concatenated(nonEmpty.map(\.element), axis: 0)
                objectLogits = latentSeedHead(instStates).squeezed(axis: -1)
                for (f, _) in nonEmpty { instanceSpans += fieldSpans[f].map { Optional($0) } }
            }
        default:  // anchorless
            let ctx = fieldStates.filter { $0.dim(0) > 0 }
            let inst = instanceEmbed.asType(dtype)
            if ctx.isEmpty {
                instStates = inst
            } else {
                let context = MLX.concatenated(ctx, axis: 0)                              // [M, H]
                let q = qProj(inst), k = kProj(context), v = vProj(context)
                let weights = MLX.softmax(MLX.matmul(q, k.transposed(1, 0)) / Float(recordDim).squareRoot(), axis: -1)
                instStates = inst + MLX.matmul(weights, v)
            }
            objectLogits = objectHead(instStates).squeezed(axis: -1)
            instanceSpans = Array(repeating: nil, count: instStates.dim(0))
        }

        // Null-aware assignment logits: [Ni, 1 + Cf] per field.
        let instQ = instProj(instStates)                                                // [Ni, D]
        let fieldQ = fieldProj(fq)                                                      // [F, D]
        var assign: [MLXArray] = []
        for (f, cand) in fieldStates.enumerated() {
            let query = instQ + fieldQ[f].expandedDimensions(axis: 0)                   // [Ni, D]
            let nullCol = MLX.matmul(query, nullEmbed.asType(dtype).reshaped([recordDim, 1]))   // [Ni, 1]
            if cand.dim(0) == 0 {
                assign.append(nullCol)
            } else {
                let scores = MLX.matmul(query, candProj(cand).transposed(1, 0))         // [Ni, Cf]
                assign.append(MLX.concatenated([nullCol, scores], axis: -1))
            }
        }

        MLX.eval([objectLogits] + assign)
        let ni = objectLogits.dim(0)
        let objectValues = objectLogits.asType(.float32).asArray(Float.self)
        var assignValues: [[[Float]]] = []
        for a in assign {
            let width = a.dim(1)
            let flat = a.asType(.float32).asArray(Float.self)
            assignValues.append((0..<ni).map { Array(flat[($0 * width)..<(($0 + 1) * width)]) })
        }
        return RecordGroupOutput(
            spec: spec, objectLogits: objectValues, assignLogits: assignValues,
            fieldSpans: fieldSpans, instanceSpans: instanceSpans)
    }
}
