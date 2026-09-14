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
// BoundaryLoRAParityTests.swift
// LoRA adapter merging on a GLiNER2.5 checkpoint: Swift's load-time merge must produce
// the same predictions as Python's `peft` merge_and_unload().
//
// Adapter + expectations come from a random-weight adapter built with gliner2 2.0.0
// (targets: encoder, boundary head, classifier, relation scorer) and saved in the legacy
// `adapter_weights.safetensors` + `adapter_config.json` layout:
//   GLINER25_LORA_TEST_DIR=<dir with adapter/ and expected.json>   (default: out/lora25_test)

import XCTest
import Foundation
@testable import GLiNER2Swift

final class BoundaryLoRAParityTests: XCTestCase {

    private static var testDir: URL {
        if let env = ProcessInfo.processInfo.environment["GLINER25_LORA_TEST_DIR"] {
            return URL(fileURLWithPath: env)
        }
        return TestModel.repoRoot.appendingPathComponent("out/lora25_test")
    }

    private struct Expected: Decodable {
        let texts: [String]
        let expected: [JSONValue]
    }

    private func run(modelPath: String, label: String) async throws {
        let dir = Self.testDir
        let expectedURL = dir.appendingPathComponent("expected.json")
        guard FileManager.default.fileExists(atPath: expectedURL.path) else {
            throw XCTSkip("no LoRA test bundle at \(dir.path)")
        }
        let expected = try JSONDecoder().decode(Expected.self, from: Data(contentsOf: expectedURL))
        let model = try await GLiNER2.fromPretrained(
            modelPath, adapterPath: dir.appendingPathComponent("adapter").path)
        XCTAssertEqual(model.config.architecture, .boundary)

        let nerCls = model.createSchema().entities(["person", "company", "location"])
            .classification(task: "sentiment", labels: ["positive", "negative"])
        let rel = model.createSchema().entities(["person", "company"]).relations(["works_for"])
        let actual: [[String: Any]] = [
            model.extract(text: expected.texts[0], schema: nerCls, threshold: 0.5, includeConfidence: true, includeSpans: true),
            model.extract(text: expected.texts[1], schema: nerCls, threshold: 0.5, includeConfidence: true, includeSpans: true),
            model.extract(text: expected.texts[0], schema: rel, threshold: 0.5, includeConfidence: true, includeSpans: true),
        ]
        var problems: [String] = []
        for (i, exp) in expected.expected.enumerated() where i < actual.count {
            problems += PredictionParityTests.diff(
                expected: exp, actual: JSONValue(swiftValue: actual[i]),
                path: "\(label)[\(i)]", confidenceTolerance: 0.02)
        }
        XCTAssertTrue(problems.isEmpty, "\(label): LoRA-merged predictions diverge from Python:\n"
                      + problems.prefix(10).joined(separator: "\n"))
    }

    /// Raw fp32 snapshot: adapter keys match the checkpoint keys directly.
    func testAdapterOnRawCheckpointMatchesPython() async throws {
        try TestModel.requireGPU()
        try await run(modelPath: try TestModel.requireBoundaryModel("small"), label: "raw")
    }

    /// Converted fp16 directory: head deltas must be remapped onto the camelCase keys.
    func testAdapterOnConvertedDirectoryMatchesPython() async throws {
        try TestModel.requireGPU()
        let converted = TestModel.repoRoot.appendingPathComponent("out/gliner25_small_mlx_fp16").path
        guard FileManager.default.fileExists(atPath: converted + "/model.safetensors") else {
            throw XCTSkip("no converted small fp16 directory at \(converted)")
        }
        try await run(modelPath: converted, label: "fp16")
    }
}
