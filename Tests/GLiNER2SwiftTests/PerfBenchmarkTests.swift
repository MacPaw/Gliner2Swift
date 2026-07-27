// Benchmark harness for execution time + MLX memory (IMPLEMENTATION_PLAN.md §10).
//
// Run the SAME file before and after a change and diff the printed tables. Every perf PR
// pastes the table for the scenarios it plausibly affects; benchmarks/BASELINE.md is
// appended to (never overwritten) at each phase boundary.
//
// Must be run through Xcode / xcodebuild — bare `swift test` cannot build mlx-swift's
// Metal shaders, so every MLX test aborts with "Failed to load the default metallib":
//   xcodebuild test -scheme GLiNER2Swift -destination 'platform=macOS' \
//     -only-testing:GLiNER2SwiftTests/PerfBenchmarkTests
//
// Model selection (remember the TEST_RUNNER_ prefix when passing through xcodebuild):
//   TEST_RUNNER_GLINER2_MODEL=/path/to/local/model/dir    (offline, recommended)
// Defaults to the canonical fp16 snapshot discovered by TestModel, else the fp32 hub id.

import XCTest
import Foundation
import MLX
@testable import GLiNER2Swift

final class PerfBenchmarkTests: XCTestCase {

    /// Explicit override, else the canonical fp16 snapshot, else the public fp32 base model.
    static let modelPath = ProcessInfo.processInfo.environment["GLINER2_MODEL"]
        ?? TestModel.fp16ModelPath
        ?? "fastino/gliner2-base-v1"

    private static let warmupIterations = 5
    private static let timedIterations = 100

    // MARK: - Inputs

    /// Decode-heavy: many entity types over a sentence dense with entities.
    private static let denseText = """
    Tim Cook is the CEO of Apple in Cupertino. Satya Nadella leads Microsoft in Redmond, \
    while Sundar Pichai runs Google in Mountain View. Jensen Huang founded NVIDIA in Santa \
    Clara, and Lisa Su is CEO of AMD in Austin. Elon Musk runs Tesla and SpaceX.
    """

    private static let nerLabels =
        ["person", "company", "location", "product", "title", "city", "role", "org"]

    /// ~450 words — exercises the encoder's quadratic attention and the long-text CPU paths.
    private static let longText = String(
        repeating: "The quarterly review meeting was held in San Francisco on March 15, "
            + "where Tim Cook and Satya Nadella discussed a joint venture between "
            + "Apple and Microsoft. ",
        count: 12)

    private static let batchTexts: [String] = (0..<32).map { index in
        "Report \(index): Tim Cook met Satya Nadella in Redmond to discuss Apple and Microsoft."
    }

    // MARK: - Measurement

    private struct Measurement {
        let name: String
        let detail: String
        let samplesMs: [Double]
        let memory: MLX.Memory.Snapshot
    }

    /// Runs `body` warmup+timed times and captures latency percentiles and peak memory.
    private func measure(
        _ name: String, detail: String,
        iterations: Int = PerfBenchmarkTests.timedIterations,
        body: () -> Bool
    ) -> Measurement {
        for _ in 0..<Self.warmupIterations { _ = body() }

        MLX.GPU.resetPeakMemory()
        var samplesMs: [Double] = []
        samplesMs.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let start = DispatchTime.now().uptimeNanoseconds
            let ok = body()
            let elapsed = DispatchTime.now().uptimeNanoseconds - start
            XCTAssertTrue(ok, "\(name): produced an empty result")  // also blocks DCE
            samplesMs.append(Double(elapsed) / 1_000_000)
        }
        return Measurement(name: name, detail: detail,
                           samplesMs: samplesMs.sorted(), memory: MLX.Memory.snapshot())
    }

    private func report(_ measurements: [Measurement], modelDescription: String) {
        func pct(_ samples: [Double], _ p: Double) -> Double {
            samples[min(samples.count - 1, Int(p * Double(samples.count)))]
        }
        func mb(_ bytes: Int) -> String { String(format: "%.1f", Double(bytes) / 1_048_576) }

        var table = """

        ================== GLiNER2Swift benchmark ==================
        model: \(modelDescription)
        iterations: \(Self.timedIterations) timed (+\(Self.warmupIterations) warmup)

        | scenario | min | p50 | p90 | mean | max | peak MB |
        |---|---|---|---|---|---|---|

        """
        for measurement in measurements {
            let samples = measurement.samplesMs
            let mean = samples.reduce(0, +) / Double(samples.count)
            table += String(
                format: "| %@ | %.2f | %.2f | %.2f | %.2f | %.2f | %@ |\n",
                measurement.name, samples.first!, pct(samples, 0.50), pct(samples, 0.90),
                mean, samples.last!, mb(measurement.memory.peakMemory))
        }
        table += "\nScenario detail:\n"
        for measurement in measurements {
            table += "  \(measurement.name): \(measurement.detail)\n"
        }
        let snapshot = measurements.last?.memory
        table += """

        MLX unified memory after the run:
          active (steady-state held): \(mb(snapshot?.activeMemory ?? 0)) MB
          cache:                      \(mb(snapshot?.cacheMemory ?? 0)) MB
        ============================================================

        """
        print(table)
    }

    // MARK: - Scenarios (plan §10)

    func testBenchmark() async throws {
        try TestModel.requireGPU()

        // Scenario 5 first: cold start is only meaningful before anything is warm.
        let coldStart = DispatchTime.now().uptimeNanoseconds
        // GLINER2_COMPILE_ENCODER=1 measures the Phase 5.1 configuration (compiled
        // encoder + bucket-padded sequence lengths) through the same scenarios.
        let compileEncoder = ProcessInfo.processInfo.environment["GLINER2_COMPILE_ENCODER"] != nil
        let model = try await GLiNER2.fromPretrained(
            Self.modelPath, compileEncoder: compileEncoder)
        let firstCall = model.extractEntities(text: Self.denseText, entityTypes: Self.nerLabels)
        let coldMs = Double(DispatchTime.now().uptimeNanoseconds - coldStart) / 1_000_000
        XCTAssertFalse(firstCall.isEmpty)

        let dtype = TestModel.isFloat16Checkpoint(at: Self.modelPath) ? "fp16" : "fp32/mixed"

        var measurements: [Measurement] = []

        // 1. single text x 8-label NER (decode-heavy; the original A/B case)
        measurements.append(measure(
            "1-ner-8-labels",
            detail: "single dense sentence x \(Self.nerLabels.count) entity types"
        ) {
            !model.extractEntities(text: Self.denseText, entityTypes: Self.nerLabels).isEmpty
        })

        // 2. mixed schema: entities + a 4-field structure + classification
        let mixedSchema = model.createSchema()
            .entities(["person", "company", "location"])
            .structure("employment")
            .field("employee")
            .field("employer")
            .field("city")
            .field("role")
            .done()
            .classification(task: "sentiment", labels: ["positive", "negative"])
        measurements.append(measure(
            "2-mixed-schema",
            detail: "3 entity types + 4-field structure + 2-label classification"
        ) {
            !model.extract(text: Self.denseText, schema: mixedSchema).isEmpty
        })

        // 3. long single text (~450 words) — encoder-bound
        measurements.append(measure(
            "3-long-text",
            detail: "~450-word document x 3 entity types",
            iterations: max(10, Self.timedIterations / 5)
        ) {
            !model.extractEntities(text: Self.longText,
                                   entityTypes: ["person", "company", "location"]).isEmpty
        })

        // 4. batch throughput
        measurements.append(measure(
            "4-batch-32",
            detail: "batchExtract over \(Self.batchTexts.count) texts, batchSize 8",
            iterations: max(5, Self.timedIterations / 20)
        ) {
            let schema = model.createSchema().entities(["person", "company", "location"])
            return model.batchExtract(texts: Self.batchTexts, schema: schema,
                                      batchSize: 8).count == Self.batchTexts.count
        })

        report(measurements, modelDescription: "\(Self.modelPath) [\(dtype)]")
        print(String(format: "5-cold-start: %.1f ms (fromPretrained + first inference)\n", coldMs))
    }
}
