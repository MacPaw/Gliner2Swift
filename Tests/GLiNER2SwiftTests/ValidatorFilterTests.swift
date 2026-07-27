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
// ValidatorFilterTests.swift
// Consumes the Python-generated validator fixtures (IMPLEMENTATION_PLAN.md §1.4).
//
// `scripts/generate_inference_fixtures.py` has emitted validator_filter_cases.json and
// regex_engine_matrix.json for some time, but nothing on the Swift side read them. These
// pin the span-filtering semantics that structure fields rely on: spans failing ANY
// validator are dropped before formatting (Python engine.py:668).

import XCTest
import Foundation
@testable import GLiNER2Swift

final class ValidatorFilterTests: XCTestCase {

    private static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent("Fixtures/inference")

    private struct ValidatorSpec: Decodable {
        let pattern: String
        let mode: String
        let exclude: Bool
    }

    private struct SpanSpec: Decodable {
        let text: String
        let confidence: Float
        let charStart: Int
        let charEnd: Int
    }

    private struct FilterCase: Decodable {
        let name: String
        let validators: [ValidatorSpec]
        let spans: [SpanSpec]
        /// The spans Python keeps after applying every validator.
        let filtered: [SpanSpec]
    }

    private struct FilterFixture: Decodable {
        let cases: [FilterCase]
    }

    private struct EnginePattern: Decodable {
        let pattern: String
        let mode: String
    }

    /// `results[patternIndex][textIndex]` — Python's verdict for every pattern x text pair.
    private struct EngineFixture: Decodable {
        let patterns: [EnginePattern]
        let texts: [String]
        let results: [[Bool]]
    }

    private func load<T: Decodable>(_ type: T.Type, _ fileName: String) throws -> T {
        let url = Self.fixturesDir.appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("""
                \(fileName) not found. Regenerate with: \
                python scripts/generate_inference_fixtures.py
                """)
        }
        return try JSONDecoder().decode(type, from: Data(contentsOf: url))
    }

    private func makeValidator(
        pattern: String, mode: String, exclude: Bool = false
    ) throws -> RegexValidator {
        let matchMode: RegexValidator.MatchMode = mode == "full" ? .full : .partial
        return try RegexValidator(pattern: pattern, mode: matchMode, exclude: exclude)
    }

    private func makeValidator(_ spec: ValidatorSpec) throws -> RegexValidator {
        try makeValidator(pattern: spec.pattern, mode: spec.mode, exclude: spec.exclude)
    }

    /// The exact filter the structure-field decode path applies.
    func testValidatorFilteringMatchesPython() throws {
        let fixture = try load(FilterFixture.self, "validator_filter_cases.json")
        XCTAssertFalse(fixture.cases.isEmpty)

        for testCase in fixture.cases {
            let validators = try testCase.validators.map(makeValidator)
            let spans = testCase.spans.map {
                ExtractedSpan(text: $0.text, confidence: $0.confidence,
                              charStart: $0.charStart, charEnd: $0.charEnd)
            }

            let kept = spans
                .filter { span in validators.allSatisfy { $0.validate(span.text) } }
                .map { [$0.text, String($0.charStart), String($0.charEnd)] }
            let expected = testCase.filtered
                .map { [$0.text, String($0.charStart), String($0.charEnd)] }

            XCTAssertEqual(kept, expected,
                           "validator filtering diverged for case '\(testCase.name)'")
        }
    }

    /// Per-pattern behaviour of the regex engine itself (full vs partial matching).
    ///
    /// Swift's NSRegularExpression and Python's `re` are different engines; this pins the
    /// subset of syntax the validators rely on to identical verdicts.
    func testRegexEngineMatrixMatchesPython() throws {
        let fixture = try load(EngineFixture.self, "regex_engine_matrix.json")
        XCTAssertFalse(fixture.patterns.isEmpty)
        XCTAssertEqual(fixture.results.count, fixture.patterns.count)

        for (patternIndex, spec) in fixture.patterns.enumerated() {
            let validator = try makeValidator(pattern: spec.pattern, mode: spec.mode)
            let expectedRow = fixture.results[patternIndex]
            XCTAssertEqual(expectedRow.count, fixture.texts.count)

            for (textIndex, text) in fixture.texts.enumerated() {
                XCTAssertEqual(
                    validator.validate(text), expectedRow[textIndex],
                    "pattern \(spec.pattern) (mode=\(spec.mode)) vs \"\(text)\"")
            }
        }
    }
}
