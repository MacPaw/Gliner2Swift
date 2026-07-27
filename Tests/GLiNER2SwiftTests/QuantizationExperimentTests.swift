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
// QuantizationExperimentTests.swift
// Phase 4.4: does 8-bit QuantizedLinear over the encoder pay for itself?
//
// A measurement, not a feature. Each configuration is loaded, measured and released on
// its own, because holding two models at once makes the memory figures meaningless.
// Run explicitly:
//
//   xcodebuild test -scheme GLiNER2Swift -destination 'platform=macOS' \
//     -only-testing:GLiNER2SwiftTests/QuantizationExperimentTests

import XCTest
import MLX
import MLXNN
@testable import GLiNER2Swift

final class QuantizationExperimentTests: XCTestCase {

    private static let texts = [
        "Tim Cook, the CEO of Apple Inc., spoke in Cupertino on Tuesday.",
        "Maria Gonzalez joined Siemens in Munich after ten years at Bosch.",
        "The summit in Nairobi brought together delegates from Kenya and Rwanda.",
        "Dr. Chen published the results in Nature while working at Stanford."
    ]

    private static let longText = String(
        repeating: "The regional council met in Lisbon to review the proposal from Banco Santander "
            + "before Ana Ribeiro presented the findings to delegates from Porto and Madrid. ",
        count: 6)

    private func schema() -> Schema {
        Schema().entities(["person", "organization", "location"])
    }

    private struct Sample {
        var shortMs: Double
        var longMs: Double
        var activeMB: Int
        var peakMB: Int
        var predictions: [[String: Set<String>]]
    }

    private func p50(iterations: Int = 50, _ body: () -> Void) -> Double {
        for _ in 0..<5 { body() }
        var samples: [Double] = []
        for _ in 0..<iterations {
            let start = DispatchTime.now().uptimeNanoseconds
            body()
            samples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
        }
        samples.sort()
        return samples[samples.count / 2]
    }

    /// Loads a model, optionally quantizes it, measures, and releases it before returning.
    enum Mode: String { case fp16, int8Linear, int8LinearAndEmbedding }

    private func measure(mode: Mode, path: String) async throws -> Sample {
        MLX.Memory.clearCache()
        MLX.GPU.resetPeakMemory()

        var model: GLiNER2? = try await GLiNER2.fromPretrained(path)
        defer { model = nil }
        guard let model else { fatalError("unreachable") }

        if mode != .fp16 {
            // 8-bit, groupSize 64 (768 % 64 == 0 for every dim). The embedding table is
            // 128011 x 768 — by far the largest single tensor — so it is measured both
            // ways: including it is where the memory win mostly lives, but it also changes
            // the token representations themselves rather than just a projection.
            let includeEmbedding = mode == .int8LinearAndEmbedding
            MLXNN.quantize(model: model.model.encoder, groupSize: 64, bits: 8) { _, module in
                module is Linear || (includeEmbedding && module is Embedding)
            }
            // The relative-position projections were memoized from the fp16 weights.
            model.model.encoder.resetCaches()
        }

        let predictions = Self.texts.map { text -> [String: Set<String>] in
            let result = model.extract(text: text, schema: schema(), threshold: 0.5,
                                       includeConfidence: true, includeSpans: true)
            let entities = result["entities"] as? [String: [Any]] ?? [:]
            return entities.mapValues { Set($0.map { Self.key($0) }) }
        }

        let shortMs = p50 { _ = model.extract(text: Self.texts[0], schema: self.schema()) }
        let longMs = p50(iterations: 20) {
            _ = model.extract(text: Self.longText, schema: self.schema())
        }

        MLX.Memory.clearCache()
        let snapshot = MLX.Memory.snapshot()
        return Sample(shortMs: shortMs, longMs: longMs,
                      activeMB: snapshot.activeMemory / 1_048_576,
                      peakMB: snapshot.peakMemory / 1_048_576,
                      predictions: predictions)
    }

    func testEightBitEncoderQuantization() async throws {
        // Opt-in: this loads three models, runs a few hundred inferences, and asserts on
        // predictions that quantization noise can legitimately move. It belongs in a
        // deliberate measurement run, not in every suite run.
        guard ProcessInfo.processInfo.environment["GLINER2_RUN_QUANT_EXPERIMENT"] != nil else {
            throw XCTSkip("Set GLINER2_RUN_QUANT_EXPERIMENT=1 to run the Phase 4.4 measurement")
        }
        try TestModel.requireGPU()
        let path = try TestModel.requireFP16Model()

        let half = try await measure(mode: .fp16, path: path)
        let int8 = try await measure(mode: .int8Linear, path: path)
        let int8Emb = try await measure(mode: .int8LinearAndEmbedding, path: path)

        var total = 0
        var matching = 0
        for (index, want) in half.predictions.enumerated() {
            let got = int8.predictions[index]
            for (label, wantKeys) in want {
                total += 1
                if got[label] == wantKeys { matching += 1 }
                else {
                    print("  \(label) in \"\(Self.texts[index].prefix(38))…\": "
                          + "fp16 \(wantKeys.sorted()) vs int8 \((got[label] ?? []).sorted())")
                }
            }
        }

        print("""

        === Phase 4.4: 8-bit encoder quantization (each config measured in isolation) ===
        short text p50:   fp16 \(fmt(half.shortMs)) ms -> int8 \(fmt(int8.shortMs)) ms \
        (\(pct(int8.shortMs, half.shortMs)))
        long text p50:    fp16 \(fmt(half.longMs)) ms -> int8 \(fmt(int8.longMs)) ms \
        (\(pct(int8.longMs, half.longMs)))
        active memory:    fp16 \(half.activeMB) MB -> int8 \(int8.activeMB) MB
        peak memory:      fp16 \(half.peakMB) MB -> int8 \(int8.peakMB) MB
        label sets equal: \(matching)/\(total)

        with the embedding table also quantized:
        short text p50:   \(fmt(int8Emb.shortMs)) ms (\(pct(int8Emb.shortMs, half.shortMs)))
        long text p50:    \(fmt(int8Emb.longMs)) ms (\(pct(int8Emb.longMs, half.longMs)))
        active memory:    \(int8Emb.activeMB) MB
        peak memory:      \(int8Emb.peakMB) MB
        label sets equal: \(agreement(half, int8Emb))

        """)

        // These four sentences agree; the 58-case parity corpus does NOT — it drops to
        // 53/55 under quantization (one borderline entity, one confidence 0.021 off).
        // That is the number the decision rests on; see QuantizationPolicy. This assertion
        // only guards against a gross regression in the quantization path itself.
        XCTAssertEqual(matching, total,
                       "8-bit quantization changed predictions even on the easy sample — "
                       + "that indicates a broken quantization path, not precision loss")
    }

    /// "matching/total" label-set agreement between two runs.
    private func agreement(_ reference: Sample, _ other: Sample) -> String {
        var total = 0
        var matching = 0
        for (index, want) in reference.predictions.enumerated() {
            for (label, keys) in want {
                total += 1
                if other.predictions[index][label] == keys { matching += 1 }
            }
        }
        return "\(matching)/\(total)"
    }

    private func fmt(_ value: Double) -> String { String(format: "%.2f", value) }
    private func pct(_ new: Double, _ old: Double) -> String {
        String(format: "%+.1f %%", (new / old - 1) * 100)
    }

    private static func key(_ value: Any) -> String {
        guard let dict = value as? [String: Any] else { return "\(value)" }
        return "\(dict["text"] as? String ?? "")|\(dict["start"] as? Int ?? -1)"
    }
}
