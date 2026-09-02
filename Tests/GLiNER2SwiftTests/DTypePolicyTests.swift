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
// DTypePolicyTests.swift
// Phase 4: dtype pinning and the compute-dtype cross-check.
//
// MLX has no global compute dtype — activations inherit theirs from the weights that
// produce them — so a single strongly typed float32 constant anywhere on the path
// silently promotes everything downstream of it. That is not visible in any output; it
// only shows up as lost speed and doubled memory. These tests pin the dtype end to end so
// a reintroduced promotion fails loudly.

import XCTest
import MLX
@testable import GLiNER2Swift

final class DTypePolicyTests: XCTestCase {

    // MARK: - Helpers

    /// Runs the span-score pipeline for one text/schema and returns the final scores.
    ///
    /// Mirrors what `extractFromBatch` does, up to the readback: encode, pool, span
    /// representations, count-aware projections, einsum, sigmoid.
    private func spanScores(_ model: GLiNER2, text: String, schema: Schema) -> MLXArray {
        let record = model.processor.transform(text: text, schema: schema.build())
        let batch = model.processor.collateBatch([record])

        let hidden = model.model.encode(batch.inputIds, attentionMask: batch.attentionMask)
            .lastHiddenState[0]

        let start = batch.textStartIndices[0]
        let pooled = model.poolTextEmbeddings(
            subwordEmbeddings: hidden[start ..< batch.mappedIndices[0].count],
            wordFirstIndices: batch.wordFirstIndices[0],
            wordSubwordCounts: batch.wordSubwordCounts[0],
            poolingType: model.processor.tokenPooling
        )

        let info = model.model.computeSpanRep(pooled)
        let markers = batch.schemaMarkerPositions[0][0]
        let embs = MLX.take(hidden, MLXArray(markers.map { Int32($0) }), axis: 0)

        return model.model.computeSpanScores(spanInfo: info, schemaEmb: embs, predCount: 1)
    }

    private func nerSchema() -> Schema {
        Schema().entities(["person", "organization", "location"])
    }

    private static let sampleTexts = [
        "Tim Cook, the CEO of Apple Inc., spoke in Cupertino on Tuesday.",
        "Maria Gonzalez joined Siemens in Munich after ten years at Bosch.",
        "The summit in Nairobi brought together delegates from Kenya and Rwanda."
    ]

    // MARK: - Pinning

    func testCanonicalSnapshotKeepsFloat16Weights() async throws {
        try TestModel.requireGPU()
        let path = try TestModel.requireFP16Model()
        let model = try await GLiNER2.fromPretrained(path)

        let floating = model.model.parameters().flattened()
            .filter { $0.1.dtype.isFloatingPoint }
        XCTAssertGreaterThan(floating.count, 200, "Expected the full parameter set")

        let promoted = floating.filter { $0.1.dtype != .float16 }.map(\.0)
        XCTAssertTrue(promoted.isEmpty,
                      "`.auto` must preserve the checkpoint dtype; promoted: \(promoted.prefix(5))")
    }

    /// The regression trap: any float32 constant reintroduced on the path shows up here,
    /// because the span scores are the last thing computed before readback.
    func testSpanScoresStayFloat16() async throws {
        try TestModel.requireGPU()
        let path = try TestModel.requireFP16Model()
        let model = try await GLiNER2.fromPretrained(path)

        let scores = spanScores(model, text: Self.sampleTexts[0], schema: nerSchema())
        XCTAssertEqual(scores.dtype, .float16,
                       "Span scores promoted to \(scores.dtype) — something on the path "
                       + "creates a strongly typed float32 array (see Phase 4.1)")
    }

    func testFloat32PolicyCastsEveryFloatingWeight() async throws {
        try TestModel.requireGPU()
        let path = try TestModel.requireFP16Model()
        let model = try await GLiNER2.fromPretrained(path, dtype: .float32)

        let floating = model.model.parameters().flattened()
            .filter { $0.1.dtype.isFloatingPoint }
        let notCast = floating.filter { $0.1.dtype != .float32 }.map(\.0)
        XCTAssertTrue(notCast.isEmpty, "`.float32` must cast every float weight; missed: \(notCast.prefix(5))")

        let scores = spanScores(model, text: Self.sampleTexts[0], schema: nerSchema())
        XCTAssertEqual(scores.dtype, .float32)
    }

    // MARK: - Compute-dtype cross-check

    /// Same weights, different compute dtype.
    ///
    /// Deliberately NOT a comparison against the fp32 reference checkpoint: the snapshot's
    /// weights are already fp16-rounded, so that comparison would fail from weight
    /// quantization alone and prove nothing about the code. Casting the snapshot to fp32
    /// isolates the one variable that matters here.
    func testComputeDtypeDoesNotChangePredictions() async throws {
        try TestModel.requireGPU()
        let path = try TestModel.requireFP16Model()

        let half = try await GLiNER2.fromPretrained(path)
        let full = try await GLiNER2.fromPretrained(path, dtype: .float32)

        var maxDelta: Float = 0
        var compared = 0

        for text in Self.sampleTexts {
            for threshold in [Float(0.3), 0.5, 0.7] {
                let a = half.extract(text: text, schema: nerSchema(), threshold: threshold,
                                     includeConfidence: true, includeSpans: true)
                let b = full.extract(text: text, schema: nerSchema(), threshold: threshold,
                                     includeConfidence: true, includeSpans: true)

                let entitiesA = a["entities"] as? [String: [Any]] ?? [:]
                let entitiesB = b["entities"] as? [String: [Any]] ?? [:]
                XCTAssertEqual(Set(entitiesA.keys), Set(entitiesB.keys))

                for (label, spansA) in entitiesA {
                    let spansB = entitiesB[label] ?? []
                    let byKeyA = Self.confidencesByKey(spansA)
                    let byKeyB = Self.confidencesByKey(spansB)

                    // The extracted set is what parity is defined on (text, start, end).
                    XCTAssertEqual(Set(byKeyA.keys), Set(byKeyB.keys),
                                   "fp16 and fp32 extract different spans for \(label) at "
                                   + "threshold \(threshold): \(text)")

                    for (key, ca) in byKeyA {
                        guard let cb = byKeyB[key] else { continue }
                        maxDelta = max(maxDelta, abs(ca - cb))
                        compared += 1
                    }

                    // Order within the list comes from sorting by confidence, so two spans
                    // whose scores are separated by less than fp16's resolution can legally
                    // swap. Anything but a near-tie swapping is a real bug, so the gap is
                    // asserted rather than the order.
                    let orderA = spansA.map { Self.spanKey($0) }
                    let orderB = spansB.map { Self.spanKey($0) }
                    if orderA != orderB {
                        let gap = Self.minimumAdjacentGap(spansA)
                        print("order flip in \(label) @\(threshold): \(orderA) vs \(orderB), "
                              + "closest confidences differ by \(gap)")
                        XCTAssertLessThan(gap, 0.01,
                                          "\(label) reordered at threshold \(threshold) without a "
                                          + "confidence tie — not explainable by compute dtype")
                    }
                }
            }
        }

        XCTAssertGreaterThan(compared, 0, "Corpus produced no confidences to compare")
        print("compute-dtype cross-check: \(compared) confidences, max |fp16 - fp32| = \(maxDelta)")
        // Tier C in the plan allows +-0.02 against Python fp32. Compute dtype alone should
        // stay well inside that; this asserts the stated bound rather than the observation,
        // so a real precision regression is caught.
        XCTAssertLessThan(maxDelta, 0.02, "fp16 confidences drifted further than tier C allows")
    }

    private static func confidencesByKey(_ values: [Any]) -> [String: Float] {
        var result: [String: Float] = [:]
        for value in values {
            guard let dict = value as? [String: Any],
                  let confidence = dict["confidence"] as? Float else { continue }
            result[spanKey(value)] = confidence
        }
        return result
    }

    /// Smallest confidence difference between consecutive entries of a sorted list.
    private static func minimumAdjacentGap(_ values: [Any]) -> Float {
        let confidences = values.compactMap { ($0 as? [String: Any])?["confidence"] as? Float }
        guard confidences.count >= 2 else { return .greatestFiniteMagnitude }
        return zip(confidences, confidences.dropFirst())
            .map { abs($0 - $1) }
            .min() ?? .greatestFiniteMagnitude
    }

    private static func spanKey(_ value: Any) -> String {
        guard let dict = value as? [String: Any] else { return "\(value)" }
        let text = dict["text"] as? String ?? ""
        let start = dict["start"] as? Int ?? -1
        let end = dict["end"] as? Int ?? -1
        return "\(text)|\(start)|\(end)"
    }
}
