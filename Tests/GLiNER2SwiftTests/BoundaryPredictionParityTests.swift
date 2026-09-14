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
// BoundaryPredictionParityTests.swift
// End-to-end prediction parity of the GLiNER2.5 checkpoints against Python gliner2 2.0.0
// — the master gate for the boundary port.
//
// The corpus is the v1 corpus' texts and schemas re-recorded through `AutoExtractor`:
//   Tests/GLiNER2SwiftTests/Fixtures/prediction_corpus_boundary/<size>/corpus.json
//   (regenerate: <gliner2-ref>/.venv/bin/python scripts/generate_prediction_corpus.py
//    --architecture boundary --model fastino/gliner2.5-<size>-v1)
//
// Model selection per size (small / base / multi):
//   GLINER25_MODEL_DIR_<SIZE>   any loadable directory (raw fp32 snapshot, or a converted
//                               fp16 / int8 directory from scripts/convert_weights.py)
//   otherwise the raw HuggingFace snapshot from the cache (fp32).
//   GLINER25_PARITY_SIZES       comma-separated subset to run (default: every size whose
//                               corpus and model are present)
//   GLINER2_QUANTIZE=int8       runtime encoder quantization, as for the v1 gate
//   GLINER2_COMPILE_ENCODER=1   compiled encoder, as for the v1 gate

import XCTest
import Foundation
@testable import GLiNER2Swift

final class BoundaryPredictionParityTests: XCTestCase {

    /// Tier-C: confidences may drift this far from the Python fp32 reference (fp16 models).
    private let confidenceTolerance: Double = 0.02

    private static func corpusURL(_ size: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/prediction_corpus_boundary/\(size)/corpus.json")
    }

    private static func modelDirectory(_ size: String) -> String? {
        if let env = ProcessInfo.processInfo.environment["GLINER25_MODEL_DIR_\(size.uppercased())"] {
            return env
        }
        return TestModel.boundaryModelPath(size)
    }

    /// Cases that exercise decoders not yet ported: choice fields (Phase 5) and relations
    /// (Phase 6). A tagged case that starts passing fails the suite so the list stays honest.
    private static func pendingPhase(for spec: PredictionParityTests.SchemaSpec) -> String? {
        nil   // every decoder of the boundary path is ported; nothing is pending
    }

    func testBoundaryPredictionParityWithPython() async throws {
        try TestModel.requireGPU()
        let requested = ProcessInfo.processInfo.environment["GLINER25_PARITY_SIZES"]?
            .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        let sizes = requested ?? ["small", "base", "multi"]

        let quantization: QuantizationPolicy
        switch ProcessInfo.processInfo.environment["GLINER2_QUANTIZE"] {
        case "int8": quantization = .int8
        case "int8-linear": quantization = .int8(includeEmbeddings: false)
        default: quantization = .none
        }
        let compileEncoder = ProcessInfo.processInfo.environment["GLINER2_COMPILE_ENCODER"] != nil

        var ran = 0
        var report: [String] = []
        for size in sizes {
            let url = Self.corpusURL(size)
            guard FileManager.default.fileExists(atPath: url.path) else {
                print("  [skip] no boundary corpus for \(size) at \(url.path)")
                continue
            }
            guard let modelPath = Self.modelDirectory(size) else {
                print("  [skip] no gliner2.5-\(size) model directory")
                continue
            }
            let corpus = try JSONDecoder().decode(PredictionParityTests.Corpus.self, from: Data(contentsOf: url))
            let model = try await GLiNER2.fromPretrained(
                modelPath, quantization: quantization, compileEncoder: compileEncoder, strict: true)
            XCTAssertEqual(model.config.architecture, .boundary, "\(size): not a boundary checkpoint")
            ran += 1

            var failures: [String] = []
            var expected: [String] = []
            var unexpectedPasses: [String] = []
            for testCase in corpus.cases {
                let schema = PredictionParityTests.buildSchema(testCase.schema)
                let actual = model.extract(
                    text: testCase.text, schema: schema, threshold: Float(testCase.threshold),
                    includeConfidence: true, includeSpans: true)
                let mismatches = PredictionParityTests.diff(
                    expected: testCase.expected, actual: JSONValue(swiftValue: actual),
                    path: testCase.name, confidenceTolerance: confidenceTolerance)
                let phase = Self.pendingPhase(for: testCase.schema)
                if mismatches.isEmpty {
                    if let phase { unexpectedPasses.append("\(size)/\(testCase.name) [\(phase)]") }
                } else if let phase {
                    expected.append("\(size)/\(testCase.name) [\(phase)]: \(mismatches.count) mismatch(es)")
                } else {
                    failures.append("\(size)/\(testCase.name):\n        "
                                    + mismatches.prefix(6).joined(separator: "\n        "))
                }
            }
            let summary = """

                === GLiNER2.5 prediction parity vs Python (\(corpus.model)) ===
                  model dir:         \(modelPath)
                  quantization:      \(quantization)  compiled encoder: \(compileEncoder)
                  cases:             \(corpus.cases.count)
                  passing:           \(corpus.cases.count - failures.count - expected.count)
                  expected failures: \(expected.count)
                  unexpected passes: \(unexpectedPasses.count)
                  FAILURES:          \(failures.count)
                """
            print(summary)
            for line in expected { print("  [expected] \(line)") }
            for line in unexpectedPasses { print("  [pending, passing] \(line)") }
            report.append(summary)
            // A pending case can legitimately pass (Python found no relation either); it is
            // reported, not failed, until the relation head lands and the tag goes away.
            XCTAssertTrue(failures.isEmpty,
                          "\(size): Swift predictions diverge from Python in \(failures.count) case(s):\n\n"
                          + failures.joined(separator: "\n\n"))
        }
        if ran == 0 {
            throw XCTSkip("no GLiNER2.5 corpus + model pair available; see the header for the env vars")
        }
    }
}
