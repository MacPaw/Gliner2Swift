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
// CompileProbeTests.swift
// Phase 5.1 feasibility probe: is MLX.compile worth integrating?
//
// Integrating compile means bucket-padding every batch for shape stability, priming the
// relative-position caches before tracing (they call MLX.eval internally, which cannot
// happen inside a traced function), and invalidating the compiled closure whenever weights
// change. That is a lot of machinery to add on faith, so this measures the ceiling first:
// encoder-only latency, compiled versus not, at a fixed shape.
//
//   GLINER2_RUN_COMPILE_PROBE=1 xcodebuild test -scheme GLiNER2Swift \
//     -destination 'platform=macOS' -only-testing:GLiNER2SwiftTests/CompileProbeTests

import XCTest
import MLX
import MLXNN
@testable import GLiNER2Swift

final class CompileProbeTests: XCTestCase {

    func testCompiledEncoderCeiling() async throws {
        guard ProcessInfo.processInfo.environment["GLINER2_RUN_COMPILE_PROBE"] != nil else {
            throw XCTSkip("Set GLINER2_RUN_COMPILE_PROBE=1 to run the Phase 5.1 probe")
        }
        try TestModel.requireGPU()
        let path = try TestModel.requireFP16Model()
        let model = try await GLiNER2.fromPretrained(path)

        let text = "Tim Cook, the CEO of Apple Inc., spoke in Cupertino on Tuesday about the "
            + "company's plans for Austin, Texas and the wider United States market."
        let schema = Schema().entities(["person", "organization", "location"])
        let record = model.processor.transform(text: text, schema: schema.build())
        let batch = model.processor.collateBatch([record])
        let ids = batch.inputIds
        let mask = batch.attentionMask

        let encoder = model.model.encoder

        // Prime the relative-position caches. They memoize with MLX.eval inside, which is
        // not legal during tracing, so they must be populated before compile sees them.
        MLX.eval(encoder(ids, attentionMask: mask).lastHiddenState)

        let compiled = compile(inputs: [encoder], outputs: [encoder]) {
            (ids: MLXArray, mask: MLXArray) in
            encoder(ids, attentionMask: mask).lastHiddenState
        }

        // First call traces and compiles; excluded from timing.
        let compiledFirst = compiled(ids, mask)
        MLX.eval(compiledFirst)

        // Equivalence before speed — a faster wrong answer is not interesting.
        let reference = encoder(ids, attentionMask: mask).lastHiddenState
        MLX.eval(reference)
        let maxDiff = MLX.max(MLX.abs(reference - compiledFirst)).item(Float.self)

        func p50(_ body: () -> MLXArray) -> Double {
            for _ in 0..<5 { MLX.eval(body()) }
            var samples: [Double] = []
            for _ in 0..<50 {
                let start = DispatchTime.now().uptimeNanoseconds
                MLX.eval(body())
                samples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
            }
            samples.sort()
            return samples[samples.count / 2]
        }

        let plainMs = p50 { encoder(ids, attentionMask: mask).lastHiddenState }
        let compiledMs = p50 { compiled(ids, mask) }

        print("""

        === Phase 5.1 probe: encoder forward, seq \(ids.dim(1)) ===
        plain:      \(String(format: "%.2f", plainMs)) ms
        compiled:   \(String(format: "%.2f", compiledMs)) ms \
        (\(String(format: "%+.1f", (compiledMs / plainMs - 1) * 100)) %)
        max |diff|: \(maxDiff)

        """)

        XCTAssertLessThan(maxDiff, 0.01, "compiled encoder does not match the plain one")
    }
}
