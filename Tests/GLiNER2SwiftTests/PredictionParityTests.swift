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
// PredictionParityTests.swift
// End-to-end prediction parity against Python GLiNER2 — the master gate for the
// performance work (IMPLEMENTATION_PLAN.md §0.4).
//
// Every perf refactor must leave this suite green. Predictions (text / start / end /
// label) must match Python exactly; confidences must match within the tier-C tolerance
// (§9). If a prediction ever differs, that is a bug — never a reason to loosen a bound.
//
// Regenerate the corpus with:
//   <GLiNER2>/.venv/bin/python scripts/generate_prediction_corpus.py

import XCTest
import Foundation
@testable import GLiNER2Swift

final class PredictionParityTests: XCTestCase {

    // MARK: - Tolerances

    /// Tier-C: confidences may drift this far from the Python fp32 reference.
    private let confidenceTolerance: Double = 0.02

    // MARK: - Expected failures
    //
    // Divergences that are known today and already assigned to a plan phase. Keyed by
    // case name (not tag) so the list states exactly what is broken. Delete an entry when
    // its phase lands: a case that passes while still listed here FAILS the suite, so the
    // list cannot rot into hidden regressions.
    //
    // Progress against fastino/gliner2-base-v1, 58 cases:
    //   Phase 0 baseline        35 passing, 23 expected failures
    //   PR2 (Phase 1.1 + 1.4)   45 passing, 13 expected failures
    //   PR3 (tokenizer offsets) 45 passing, 13 expected failures
    //   PR4 (Phase 1.3 + 1.5)   55 passing,  3 expected failures

    private static let expectedFailures: [String: String] = [
        // Non-ASCII residue. NOT a charsmap problem: Python's inference tokenizer is a
        // fast DebertaV2Tokenizer normalizing with Replace+NFC+Strip, not the
        // Precompiled charsmap the model directory's tokenizer.json declares (see
        // UnigramTokenizer.normalize). After the scalar-offset and NFC fixes, Swift's
        // word segmentation and character offsets match Python EXACTLY for the fullwidth
        // and NBSP/ZWSP cases; what remains is residual token-id divergence.
        //
        // nonascii_nfd_accents additionally segments differently: ICU's \w matches
        // combining marks, so Swift splits "zoe\u{301}" as one word where Python's `re`
        // yields "zoe" + "\u{301}". Reconciling the two regex engines is separate work.
        "nonascii_nfd_accents": "regex \\w semantics (ICU vs Python re) on combining marks",
        "nonascii_fullwidth": "residual token-id divergence on fullwidth forms",
        "nonascii_nbsp_zwsp": "residual token-id divergence on zero-width space",
    ]

    // MARK: - Corpus model

    private struct Corpus: Decodable {
        let model: String
        let cases: [Case]
    }

    private struct Case: Decodable {
        let name: String
        let text: String
        let threshold: Double
        let schema: SchemaSpec
        let tags: [String]
        let expected: JSONValue
    }

    private struct SchemaSpec: Decodable {
        struct Classification: Decodable {
            let task: String
            let labels: [String]
            let multiLabel: Bool?
            let clsThreshold: Double?
            enum CodingKeys: String, CodingKey {
                case task, labels
                case multiLabel = "multi_label"
                case clsThreshold = "cls_threshold"
            }
        }
        struct Field: Decodable {
            let name: String
            let dtype: String?
            let choices: [String]?
            let description: String?
            let threshold: Double?
        }
        struct Structure: Decodable {
            let name: String
            let fields: [Field]
        }
        /// Either `["person", ...]` or `{"person": "description", ...}`.
        let entities: EntitySpec?
        let classification: Classification?
        let structures: [Structure]?
        let relations: [String]?
    }

    private enum EntitySpec: Decodable {
        case list([String])
        case described([String: String])

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let list = try? container.decode([String].self) {
                self = .list(list)
            } else {
                self = .described(try container.decode([String: String].self))
            }
        }
    }

    // MARK: - Loading

    private static let corpusURL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent("Fixtures/prediction_corpus/corpus.json")

    private func loadCorpus() throws -> Corpus {
        guard FileManager.default.fileExists(atPath: Self.corpusURL.path) else {
            throw XCTSkip("""
                Prediction corpus not found. Regenerate with: \
                <GLiNER2>/.venv/bin/python scripts/generate_prediction_corpus.py
                """)
        }
        return try JSONDecoder().decode(Corpus.self, from: Data(contentsOf: Self.corpusURL))
    }

    // MARK: - Schema construction (mirrors generate_prediction_corpus.py::build_schema)

    private func buildSchema(_ spec: SchemaSpec, model: GLiNER2) -> Schema {
        var schema = model.createSchema()

        if let entities = spec.entities {
            switch entities {
            case .list(let names):          schema = schema.entities(names)
            case .described(let described): schema = schema.entities(described)
            }
        }

        for structure in spec.structures ?? [] {
            var builder = schema.structure(structure.name)
            for field in structure.fields {
                builder = builder.field(
                    field.name,
                    dtype: field.dtype ?? "list",
                    choices: field.choices,
                    description: field.description,
                    threshold: field.threshold.map { Float($0) }
                )
            }
            schema = builder.done()
        }

        if let relations = spec.relations {
            schema = schema.relations(relations)
        }

        if let cls = spec.classification {
            schema = schema.classification(
                task: cls.task,
                labels: cls.labels,
                multiLabel: cls.multiLabel ?? false,
                threshold: Float(cls.clsThreshold ?? 0.5)
            )
        }

        return schema
    }

    // MARK: - The gate

    func testPredictionParityWithPython() async throws {
        try TestModel.requireGPU()
        let corpus = try loadCorpus()
        let modelPath = try TestModel.requireFP16Model()
        let model = try await GLiNER2.fromPretrained(modelPath)

        var failures: [String] = []          // unexpected divergence -> test fails
        var expectedFailures: [String] = []  // known divergence, tagged to a plan phase
        var unexpectedPasses: [String] = []  // tagged but now passing -> update the list

        for testCase in corpus.cases {
            let schema = buildSchema(testCase.schema, model: model)
            let actual = model.extract(
                text: testCase.text,
                schema: schema,
                threshold: Float(testCase.threshold),
                includeConfidence: true,
                includeSpans: true
            )

            let mismatches = Self.diff(
                expected: testCase.expected,
                actual: JSONValue(swiftValue: actual),
                path: testCase.name,
                confidenceTolerance: confidenceTolerance
            )

            let knownPhase = Self.expectedFailures[testCase.name]
            if mismatches.isEmpty {
                if let phase = knownPhase {
                    unexpectedPasses.append("\(testCase.name) [\(phase)]")
                }
            } else if let phase = knownPhase {
                expectedFailures.append("\(testCase.name) [\(phase)]: \(mismatches.count) mismatch(es)")
            } else {
                failures.append("""
                    \(testCase.name):
                        \(mismatches.prefix(6).joined(separator: "\n        "))
                    """)
            }
        }

        print("""

            === Prediction parity vs Python (\(corpus.model)) ===
              cases:             \(corpus.cases.count)
              passing:           \(corpus.cases.count - failures.count - expectedFailures.count)
              expected failures: \(expectedFailures.count)
              unexpected passes: \(unexpectedPasses.count)
              FAILURES:          \(failures.count)
            """)
        for line in expectedFailures { print("  [expected] \(line)") }
        for line in unexpectedPasses { print("  [now passing — remove its tag] \(line)") }

        // A tagged case that starts passing means a phase landed: tighten the list so it
        // cannot silently rot back into a real regression.
        XCTAssertTrue(
            unexpectedPasses.isEmpty,
            "These cases now pass and must be removed from expectedFailureTags:\n"
                + unexpectedPasses.joined(separator: "\n")
        )

        XCTAssertTrue(
            failures.isEmpty,
            "Swift predictions diverge from Python in \(failures.count) case(s):\n\n"
                + failures.joined(separator: "\n\n")
        )
    }
}

// MARK: - JSON comparison

/// A minimal JSON tree used to compare Python's recorded output against Swift's
/// `[String: Any]` result without depending on `NSDictionary` bridging behaviour.
enum JSONValue: Decodable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode([String: JSONValue].self))
        }
    }

    /// Wrap the `[String: Any]` tree that `GLiNER2.extract` returns.
    init(swiftValue: Any) {
        switch swiftValue {
        case is NSNull:
            self = .null
        case let value as JSONValue:
            self = value
        case let value as String:
            self = .string(value)
        case let value as Bool:
            self = .bool(value)
        case let value as Int:
            self = .number(Double(value))
        case let value as Float:
            self = .number(Double(value))
        case let value as Double:
            self = .number(value)
        case let value as NSNumber:
            self = .number(value.doubleValue)
        case let value as [Any]:
            self = .array(value.map { JSONValue(swiftValue: $0) })
        case let value as [String: Any]:
            self = .object(value.mapValues { JSONValue(swiftValue: $0) })
        default:
            self = .string(String(describing: swiftValue))
        }
    }

    /// Stable key for order-insensitive list comparison. Confidences are excluded so that
    /// a tiny numeric drift cannot reorder otherwise-identical elements.
    var identityKey: String {
        switch self {
        case .null:            return "null"
        case .bool(let b):     return "b:\(b)"
        case .number(let n):   return "n:\(n)"
        case .string(let s):   return "s:\(s)"
        case .array(let a):    return "[" + a.map(\.identityKey).joined(separator: ",") + "]"
        case .object(let o):
            let parts = o.filter { $0.key != "confidence" }
                .sorted { $0.key < $1.key }
                .map { "\($0.key)=\($0.value.identityKey)" }
            return "{" + parts.joined(separator: ",") + "}"
        }
    }
}

extension PredictionParityTests {

    /// Structural diff. Predictions must match exactly; `confidence` leaves get a
    /// tolerance. Lists are compared order-insensitively (the plan specifies set
    /// equality) so that confidence ties cannot cause spurious failures.
    static func diff(
        expected: JSONValue, actual: JSONValue, path: String, confidenceTolerance: Double
    ) -> [String] {
        switch (expected, actual) {
        case (.null, .null):
            return []

        case let (.string(lhs), .string(rhs)):
            return lhs == rhs ? [] : ["\(path): expected \"\(lhs)\", got \"\(rhs)\""]

        case let (.bool(lhs), .bool(rhs)):
            return lhs == rhs ? [] : ["\(path): expected \(lhs), got \(rhs)"]

        case let (.number(lhs), .number(rhs)):
            let isConfidence = path.hasSuffix("confidence")
            let tolerance = isConfidence ? confidenceTolerance : 0
            if abs(lhs - rhs) <= tolerance { return [] }
            let kind = isConfidence ? "confidence" : "value"
            return ["\(path): expected \(kind) \(lhs), got \(rhs)"]

        case let (.object(lhs), .object(rhs)):
            var problems: [String] = []
            let missing = lhs.keys.filter { rhs[$0] == nil }.sorted()
            let extra = rhs.keys.filter { lhs[$0] == nil }.sorted()
            if !missing.isEmpty { problems.append("\(path): missing key(s) \(missing)") }
            if !extra.isEmpty { problems.append("\(path): unexpected key(s) \(extra)") }
            for key in lhs.keys.sorted() {
                guard let rhsValue = rhs[key] else { continue }
                problems += diff(expected: lhs[key]!, actual: rhsValue,
                                 path: "\(path).\(key)",
                                 confidenceTolerance: confidenceTolerance)
            }
            return problems

        case let (.array(lhs), .array(rhs)):
            guard lhs.count == rhs.count else {
                return ["\(path): expected \(lhs.count) element(s), got \(rhs.count)"]
            }
            // Pair elements by identity so ordering differences between equal sets pass.
            let sortedLHS = lhs.sorted { $0.identityKey < $1.identityKey }
            let sortedRHS = rhs.sorted { $0.identityKey < $1.identityKey }
            var problems: [String] = []
            for (index, pair) in zip(sortedLHS, sortedRHS).enumerated() {
                problems += diff(expected: pair.0, actual: pair.1,
                                 path: "\(path)[\(index)]",
                                 confidenceTolerance: confidenceTolerance)
            }
            return problems

        default:
            return ["\(path): type mismatch — expected \(expected.typeName), got \(actual.typeName)"]
        }
    }
}

extension JSONValue {
    var typeName: String {
        switch self {
        case .null:   return "null"
        case .bool:   return "bool"
        case .number: return "number"
        case .string: return "string"
        case .array:  return "array"
        case .object: return "object"
        }
    }
}
