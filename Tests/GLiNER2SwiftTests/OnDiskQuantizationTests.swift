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
// OnDiskQuantizationTests.swift
// Loading a pre-quantized model directory (config `quantization` block + packed
// .weight/.scales/.biases) must be numerically identical to quantizing the fp16 model at
// load. That equivalence is the whole correctness contract for on-disk int8 loading.
//
// Produce the directories with:
//   scripts/convert_weights.py --model fastino/gliner2-base-v1 --output <fp16> --dtype fp16
//   scripts/convert_weights.py --model fastino/gliner2-base-v1 --output <int8> --quantize int8
// then set GLINER2_INT8_MODEL (and GLINER2_FP16_MODEL) to run this.

import XCTest
@testable import GLiNER2Swift

final class OnDiskQuantizationTests: XCTestCase {

    private func requireInt8Model() throws -> String {
        guard let path = ProcessInfo.processInfo.environment["GLINER2_INT8_MODEL"] else {
            throw XCTSkip("Set GLINER2_INT8_MODEL to a directory produced by "
                          + "`convert_weights.py --quantize int8` to run this.")
        }
        return path
    }

    private static let texts = [
        "Tim Cook, the CEO of Apple Inc., spoke in Cupertino on Tuesday.",
        "Maria Gonzalez joined Siemens in Munich after ten years at Bosch.",
        "The summit in Nairobi brought together delegates from Kenya and Rwanda.",
        "Dr. Chen published results in Nature while working at Stanford in 2021.",
    ]

    /// A pre-quantized directory loads as a quantized model without a runtime policy.
    func testPreQuantizedDirectoryLoadsQuantized() async throws {
        try TestModel.requireGPU()
        let model = try await GLiNER2.fromPretrained(try requireInt8Model())
        XCTAssertTrue(model.model.isQuantized,
                      "a directory with a `quantization` config block must load quantized")
        XCTAssertNotNil(model.config.quantization)
        XCTAssertEqual(model.config.quantization?.bits, 8)
    }

    /// The load-from-disk path and the quantize-at-load path must agree bit for bit — same
    /// affine quantization, same weights, so the predictions and confidences are identical.
    func testOnDiskInt8EqualsRuntimeInt8() async throws {
        try TestModel.requireGPU()
        let fp16Path = try TestModel.requireFP16Model()
        let int8Path = try requireInt8Model()

        let runtime = try await GLiNER2.fromPretrained(fp16Path, quantization: .int8)
        let onDisk = try await GLiNER2.fromPretrained(int8Path)

        var maxDelta: Float = 0
        for text in Self.texts {
            for threshold in [Float(0.3), 0.5, 0.7] {
                let a = runtime.extract(
                    text: text,
                    schema: runtime.createSchema().entities(["person", "organization", "location"]),
                    threshold: threshold, includeConfidence: true, includeSpans: true)
                let b = onDisk.extract(
                    text: text,
                    schema: onDisk.createSchema().entities(["person", "organization", "location"]),
                    threshold: threshold, includeConfidence: true, includeSpans: true)

                let ea = a["entities"] as? [String: [Any]] ?? [:]
                let eb = b["entities"] as? [String: [Any]] ?? [:]
                XCTAssertEqual(Set(ea.keys), Set(eb.keys), "labels differ for: \(text)")
                for (label, spansA) in ea {
                    let spansB = eb[label] ?? []
                    XCTAssertEqual(spansA.map(Self.spanKey), spansB.map(Self.spanKey),
                                   "spans differ on \(label) @\(threshold): \(text)")
                    for (x, y) in zip(spansA, spansB) {
                        if let ca = (x as? [String: Any])?["confidence"] as? Float,
                           let cb = (y as? [String: Any])?["confidence"] as? Float {
                            maxDelta = max(maxDelta, abs(ca - cb))
                        }
                    }
                }
            }
        }
        XCTAssertLessThan(maxDelta, 1e-4,
                          "on-disk int8 and runtime int8 must be numerically identical "
                          + "(observed max confidence delta \(maxDelta))")
    }

    private static func spanKey(_ value: Any) -> String {
        guard let d = value as? [String: Any] else { return "\(value)" }
        return "\(d["text"] as? String ?? "")|\(d["start"] as? Int ?? -1)|\(d["end"] as? Int ?? -1)"
    }
}
