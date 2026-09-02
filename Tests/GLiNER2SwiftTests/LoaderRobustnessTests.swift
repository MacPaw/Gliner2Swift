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
// LoaderRobustnessTests.swift
// Phase 6.7 — strict load fails loudly on a missing/incomplete checkpoint instead of
// silently keeping random init.

import XCTest
import MLX
@testable import GLiNER2Swift

final class LoaderRobustnessTests: XCTestCase {

    /// A complete critical-key set (dummy arrays) with each varying-index family present.
    private func completeWeights() -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        for key in Extractor.requiredCriticalKeys() { weights[key] = MLXArray(0) }
        for family in [
            "spanRep.spanRepLayer.projectStart.0.weight",
            "spanRep.spanRepLayer.projectEnd.0.weight",
            "spanRep.spanRepLayer.outProject.0.weight",
            "classifier.layers.0.weight",
            "countPred.layers.0.weight",
            "countEmbed.transformer.inProjector.weight",
        ] { weights[family] = MLXArray(0) }
        return weights
    }

    func testStrictVerifyPassesOnCompleteKeySet() {
        XCTAssertNoThrow(try Extractor.verifyRequired(completeWeights()))
    }

    func testStrictVerifyThrowsWhenACriticalKeyIsMissing() {
        var weights = completeWeights()
        weights["encoder.embeddings.word_embeddings.weight"] = nil
        XCTAssertThrowsError(try Extractor.verifyRequired(weights)) { error in
            XCTAssertTrue("\(error)".contains("word_embeddings"),
                          "error should name the missing key: \(error)")
        }
    }

    func testStrictVerifyThrowsWhenAnEncoderLayerIsMissing() {
        var weights = completeWeights()
        weights["encoder.encoder.layer.7.attention.self.query_proj.weight"] = nil
        XCTAssertThrowsError(try Extractor.verifyRequired(weights))
    }

    func testStrictVerifyThrowsWhenAHeadFamilyIsMissing() {
        var weights = completeWeights()
        for key in weights.keys where key.hasPrefix("classifier.layers.") { weights[key] = nil }
        XCTAssertThrowsError(try Extractor.verifyRequired(weights)) { error in
            XCTAssertTrue("\(error)".contains("classifier.layers"), "\(error)")
        }
    }

    /// The shipped model really does have every critical key — strict load must not throw.
    func testRealModelLoadsUnderStrict() async throws {
        try TestModel.requireGPU()
        let path = try TestModel.requireFP16Model()
        let model = try await GLiNER2.fromPretrained(path, strict: true)
        // Sanity: it actually runs.
        let result = model.extractEntities(text: "Tim Cook leads Apple.", entityTypes: ["person"])
        XCTAssertNotNil(result["entities"])
    }
}
