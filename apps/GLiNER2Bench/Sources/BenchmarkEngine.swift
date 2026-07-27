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
// BenchmarkEngine.swift
// Platform-agnostic benchmark harness: loads a GLiNER2 model at a chosen precision,
// runs a fixed set of scenarios, and collects latency + memory telemetry.
//
// Deliberately UI-free (Foundation + GLiNER2Swift + MLX only) so it compiles and can be
// exercised on macOS as well as iOS. The SwiftUI layer just calls `run` and renders
// `BenchmarkReport`.

import Foundation
import MLX
import GLiNER2Swift

// MARK: - Precision

public enum Precision: String, CaseIterable, Sendable, Codable {
    case fp16
    case int8

    var policy: QuantizationPolicy {
        switch self {
        case .fp16: return .none
        case .int8: return .int8
        }
    }

    public var label: String {
        switch self {
        case .fp16: return "fp16"
        case .int8: return "int8 (encoder)"
        }
    }
}

// MARK: - Results (Codable so the UI can export/share them)

public struct ScenarioResult: Identifiable, Sendable, Codable {
    public var id: String { name }
    public let name: String
    public let detail: String
    public let iterations: Int
    public let p50Ms: Double
    public let p90Ms: Double
    public let meanMs: Double
    public let minMs: Double
    public let maxMs: Double
    public let tokens: Int          // input tokens processed per iteration
    public let tokensPerSecond: Double
    public let itemsFound: Int      // entities/structures/etc. surfaced (sanity signal)
    public let peakMemoryMB: Double // MLX peak during this scenario
}

public struct BenchmarkReport: Sendable, Codable {
    public let precision: Precision
    public let device: String
    public let osVersion: String
    public let processorCount: Int
    public let modelLoadMs: Double
    public let scenarios: [ScenarioResult]

    // Memory, MB. MLX unified-memory figures plus the app's resident footprint.
    public let mlxActiveMB: Double     // steady-state held after a cache clear
    public let mlxPeakMB: Double       // high-water mark across the whole run
    public let mlxCacheMB: Double
    public let processResidentMB: Double

    /// A compact Markdown table — the "send me your results" payload.
    public func markdown() -> String {
        var out = """
        ## GLiNER2Swift on-device benchmark — \(precision.label)

        - Device: \(device), \(processorCount) cores, iOS \(osVersion)
        - Model load: \(fmt(modelLoadMs)) ms
        - Memory: MLX active \(fmt(mlxActiveMB)) MB · MLX peak \(fmt(mlxPeakMB)) MB · app resident \(fmt(processResidentMB)) MB

        | scenario | p50 ms | p90 ms | mean ms | tokens/s | found | peak MB |
        |---|---|---|---|---|---|---|

        """
        for s in scenarios {
            out += "| \(s.name) | \(fmt(s.p50Ms)) | \(fmt(s.p90Ms)) | \(fmt(s.meanMs)) "
                + "| \(fmt(s.tokensPerSecond, 0)) | \(s.itemsFound) | \(fmt(s.peakMemoryMB)) |\n"
        }
        return out
    }

    private func fmt(_ v: Double, _ places: Int = 1) -> String {
        String(format: "%.\(places)f", v)
    }
}

// MARK: - Engine

public actor BenchmarkEngine {

    public enum Phase: Sendable, Equatable {
        case idle
        case loading
        case warming(String)
        case running(scenario: String, precision: Precision)
        case done
        case failed(String)
    }

    public init() {}

    /// Load `precision`, run every scenario, and return a report. `progress` is called on
    /// the main actor with human-readable phase updates.
    public func run(
        modelPath: String,
        precision: Precision,
        warmup: Int = 3,
        iterations: Int = 20,
        progress: @Sendable @escaping (Phase) -> Void
    ) async throws -> BenchmarkReport {
        progress(.loading)

        // Start from a clean allocator so load-time memory is comparable run to run.
        MLX.Memory.clearCache()
        MLX.GPU.resetPeakMemory()

        let loadStart = DispatchTime.now().uptimeNanoseconds
        let model = try await GLiNER2.fromPretrained(modelPath, quantization: precision.policy)
        // Force the weights resident by running one tiny inference before timing the clock.
        _ = model.extractEntities(text: "Warm up.", entityTypes: ["thing"])
        let loadMs = elapsedMs(since: loadStart)

        var results: [ScenarioResult] = []
        for scenario in Scenarios.all {
            progress(.warming(scenario.name))
            for _ in 0..<warmup { _ = scenario.run(model) }

            progress(.running(scenario: scenario.name, precision: precision))
            MLX.GPU.resetPeakMemory()

            var samples: [Double] = []
            samples.reserveCapacity(iterations)
            var found = 0
            let iters = scenario.heavy ? max(6, iterations / 3) : iterations
            for _ in 0..<iters {
                let t0 = DispatchTime.now().uptimeNanoseconds
                found = scenario.run(model)
                samples.append(elapsedMs(since: t0))
            }
            samples.sort()

            let peak = Double(MLX.Memory.snapshot().peakMemory) / 1_048_576
            let mean = samples.reduce(0, +) / Double(samples.count)
            let tokens = scenario.tokenCount(model)
            let p50 = percentile(samples, 0.50)
            results.append(ScenarioResult(
                name: scenario.name,
                detail: scenario.detail,
                iterations: iters,
                p50Ms: p50,
                p90Ms: percentile(samples, 0.90),
                meanMs: mean,
                minMs: samples.first ?? 0,
                maxMs: samples.last ?? 0,
                tokens: tokens,
                tokensPerSecond: p50 > 0 ? Double(tokens) / (p50 / 1000) : 0,
                itemsFound: found,
                peakMemoryMB: peak
            ))
        }

        // Steady-state memory: what's still held once transient buffers are released.
        MLX.Memory.clearCache()
        let snap = MLX.Memory.snapshot()

        progress(.done)
        return BenchmarkReport(
            precision: precision,
            device: DeviceInfo.model,
            osVersion: DeviceInfo.osVersion,
            processorCount: ProcessInfo.processInfo.processorCount,
            modelLoadMs: loadMs,
            scenarios: results,
            mlxActiveMB: Double(snap.activeMemory) / 1_048_576,
            mlxPeakMB: Double(snap.peakMemory) / 1_048_576,
            mlxCacheMB: Double(snap.cacheMemory) / 1_048_576,
            processResidentMB: DeviceInfo.residentMemoryMB()
        )
    }

    private func elapsedMs(since start: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
    }

    private func percentile(_ sorted: [Double], _ p: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        return sorted[min(sorted.count - 1, Int(p * Double(sorted.count)))]
    }
}

// MARK: - Scenarios

struct Scenario {
    let name: String
    let detail: String
    let heavy: Bool
    /// Runs one extraction, returns a count of surfaced items (sanity / anti-DCE).
    let run: (GLiNER2) -> Int
    /// Input token count for throughput (schema tokens excluded — just the text).
    let tokenCount: (GLiNER2) -> Int
}

enum Scenarios {
    static let denseText = """
    Tim Cook is the CEO of Apple in Cupertino. Satya Nadella leads Microsoft in Redmond, \
    while Sundar Pichai runs Google in Mountain View. Jensen Huang founded NVIDIA in Santa \
    Clara, and Lisa Su is CEO of AMD in Austin. Elon Musk runs Tesla and SpaceX.
    """

    static let nerLabels = ["person", "company", "location", "product", "title", "city", "role", "org"]

    static let longText = String(
        repeating: "The quarterly review meeting was held in San Francisco on March 15, "
            + "where Tim Cook and Satya Nadella discussed a joint venture between "
            + "Apple and Microsoft. ",
        count: 12)

    static let batchTexts: [String] = (0..<16).map {
        "Report \($0): Tim Cook met Satya Nadella in Redmond to discuss Apple and Microsoft."
    }

    static func countEntities(_ result: [String: Any]) -> Int {
        guard let entities = result["entities"] as? [String: [Any]] else { return 0 }
        return entities.values.reduce(0) { $0 + $1.count }
    }

    static let all: [Scenario] = [
        Scenario(
            name: "ner-8-labels",
            detail: "single dense sentence × 8 entity types",
            heavy: false,
            run: { countEntities($0.extractEntities(text: denseText, entityTypes: nerLabels)) },
            tokenCount: { $0.processor.tokenizer.encode(denseText).count }
        ),
        Scenario(
            name: "mixed-schema",
            detail: "3 entities + 4-field structure + classification",
            heavy: false,
            run: { model in
                let schema = model.createSchema()
                    .entities(["person", "company", "location"])
                    .structure("employment")
                    .field("employee").field("employer").field("city").field("role")
                    .done()
                    .classification(task: "sentiment", labels: ["positive", "negative"])
                let r = model.extract(text: denseText, schema: schema)
                return countEntities(r) + ((r["employment"] as? [[String: Any]])?.count ?? 0)
            },
            tokenCount: { $0.processor.tokenizer.encode(denseText).count }
        ),
        Scenario(
            name: "long-text",
            detail: "~450-word document × 3 entity types",
            heavy: true,
            run: { countEntities($0.extractEntities(text: longText, entityTypes: ["person", "company", "location"])) },
            tokenCount: { $0.processor.tokenizer.encode(longText).count }
        ),
        Scenario(
            name: "batch-16",
            detail: "batchExtract over 16 short texts",
            heavy: true,
            run: { model in
                let schema = model.createSchema().entities(["person", "company", "location"])
                let results = model.batchExtract(texts: batchTexts, schema: schema)
                return results.reduce(0) { $0 + countEntities($1) }
            },
            tokenCount: { model in batchTexts.reduce(0) { $0 + model.processor.tokenizer.encode($1).count } }
        ),
    ]
}
