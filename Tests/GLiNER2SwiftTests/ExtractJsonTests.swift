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
// ExtractJsonTests.swift
// Tests for the extractJson() shortcut API and the parseFieldSpec helper.
//
// Parity with Python: gliner2/inference/engine.py:extract_json, _parse_field_spec

import XCTest
import Foundation
import Metal
@testable import GLiNER2Swift
import MLX

final class ExtractJsonTests: XCTestCase {

    static let weightsPath = "/Users/tmwstw/Documents/mnemos/GLiNER2/weights_attempt3_f16"

    private func skipIfNoGPU() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal GPU not available")
        }
    }

    private func skipIfNoWeights() throws {
        guard FileManager.default.fileExists(
            atPath: Self.weightsPath + "/model.safetensors"
        ) else {
            throw XCTSkip("Converted weights not found at \(Self.weightsPath)")
        }
    }

    // MARK: - parseFieldSpec: name-only

    func testParseFieldSpecNameOnly() {
        let p = GLiNER2.parseFieldSpec("restaurant")
        XCTAssertEqual(p.name, "restaurant")
        XCTAssertEqual(p.dtype, "list")  // default
        XCTAssertNil(p.choices)
        XCTAssertNil(p.description)
    }

    // MARK: - parseFieldSpec: name + dtype

    func testParseFieldSpecStrDtype() {
        let p = GLiNER2.parseFieldSpec("name::str")
        XCTAssertEqual(p.name, "name")
        XCTAssertEqual(p.dtype, "str")
        XCTAssertNil(p.choices)
        XCTAssertNil(p.description)
    }

    func testParseFieldSpecListDtype() {
        let p = GLiNER2.parseFieldSpec("tags::list")
        XCTAssertEqual(p.name, "tags")
        XCTAssertEqual(p.dtype, "list")
    }

    // MARK: - parseFieldSpec: Python docstring examples

    func testPythonExampleRestaurant() {
        let p = GLiNER2.parseFieldSpec("restaurant::str::Restaurant name")
        XCTAssertEqual(p.name, "restaurant")
        XCTAssertEqual(p.dtype, "str")
        XCTAssertNil(p.choices)
        XCTAssertEqual(p.description, "Restaurant name")
    }

    func testPythonExampleSeatingWithChoicesImplicitStr() {
        // Python: choices present + dtype NOT explicitly set → dtype defaults to "str"
        let p = GLiNER2.parseFieldSpec("seating::[indoor|outdoor|bar]::Seating preference")
        XCTAssertEqual(p.name, "seating")
        XCTAssertEqual(p.dtype, "str")
        XCTAssertEqual(p.choices, ["indoor", "outdoor", "bar"])
        XCTAssertEqual(p.description, "Seating preference")
    }

    func testPythonExampleDietaryWithChoicesExplicitList() {
        // Explicit "list" must override choices' implicit "str" default
        let p = GLiNER2.parseFieldSpec("dietary::[vegetarian|vegan|gluten-free|none]::list::Dietary restrictions")
        XCTAssertEqual(p.name, "dietary")
        XCTAssertEqual(p.dtype, "list")
        XCTAssertEqual(p.choices, ["vegetarian", "vegan", "gluten-free", "none"])
        XCTAssertEqual(p.description, "Dietary restrictions")
    }

    // MARK: - parseFieldSpec: financial (user's example)

    func testFinancialSpecChoicesBeforeDtype() {
        let p = GLiNER2.parseFieldSpec(
            "type::[equity|bond|option|future|forex]::str::Type of financial instrument"
        )
        XCTAssertEqual(p.name, "type")
        XCTAssertEqual(p.dtype, "str")
        XCTAssertEqual(p.choices, ["equity", "bond", "option", "future", "forex"])
        XCTAssertEqual(p.description, "Type of financial instrument")
    }

    func testFinancialSpecSimpleStr() {
        let p = GLiNER2.parseFieldSpec(
            "broker::str::Financial institution or brokerage firm"
        )
        XCTAssertEqual(p.name, "broker")
        XCTAssertEqual(p.dtype, "str")
        XCTAssertNil(p.choices)
        XCTAssertEqual(p.description, "Financial institution or brokerage firm")
    }

    // MARK: - parseFieldSpec: order independence (dtype before choices)

    func testDtypeBeforeChoicesDoesNotOverride() {
        // Python: dtype explicitly set first; later choices must NOT default-override it
        let p = GLiNER2.parseFieldSpec("kind::list::[a|b|c]::Kind of thing")
        XCTAssertEqual(p.name, "kind")
        XCTAssertEqual(p.dtype, "list")
        XCTAssertEqual(p.choices, ["a", "b", "c"])
        XCTAssertEqual(p.description, "Kind of thing")
    }

    // MARK: - parseFieldSpec: choice trimming

    func testChoicesWhitespaceTrimmed() {
        let p = GLiNER2.parseFieldSpec("x::[ a | b |c ]")
        XCTAssertEqual(p.choices, ["a", "b", "c"])
    }

    // MARK: - parseFieldSpec: description only (no dtype, no choices)

    func testDescriptionOnlyImpliesListDtype() {
        let p = GLiNER2.parseFieldSpec("notes::General notes about the item")
        XCTAssertEqual(p.name, "notes")
        XCTAssertEqual(p.dtype, "list")  // no dtype, no choices → stays list
        XCTAssertNil(p.choices)
        XCTAssertEqual(p.description, "General notes about the item")
    }

    // MARK: - End-to-end extractJson (requires converted weights)

    func testExtractJsonFinancialEndToEnd() async throws {
        try skipIfNoGPU()
        try skipIfNoWeights()

        let model = try await GLiNER2.fromPretrained(Self.weightsPath)

        let financialText = """
        On 2024-03-15, Fidelity Brokerage executed a BUY order for 100 shares of \
        AAPL at $175.32 per share. Commission charged: $4.95. Status: Completed.
        """

        let result = model.extractJson(
            text: financialText,
            structures: [
                "transaction": [
                    "broker::str::Financial institution or brokerage firm",
                    "amount::str::Transaction amount with currency",
                    "security::str::Stock, bond, or financial instrument",
                    "date::str::Transaction date",
                    "commission::str::Fees or commission charged",
                    "status::str::Transaction status",
                    "type::[equity|bond|option|future|forex]::str::Type of financial instrument"
                ]
            ],
            threshold: 0.3,
            includeConfidence: true
        )

        print("extractJson financial result: \(result)")

        guard let instances = result["transaction"] as? [[String: Any]] else {
            XCTFail("Missing 'transaction' instances in result: \(result)")
            return
        }

        XCTAssertFalse(instances.isEmpty, "Should extract at least one transaction")

        for (i, inst) in instances.enumerated() {
            print("  instance \(i):")
            for (k, v) in inst {
                print("    \(k): \(v)")
            }
        }
    }

    func testExtractJsonMatchesBuilderOutput() async throws {
        try skipIfNoGPU()
        try skipIfNoWeights()

        let model = try await GLiNER2.fromPretrained(Self.weightsPath)
        let text = "John Smith is 35 years old and lives in New York."

        // Path A: shortcut API
        let shortcut = model.extractJson(
            text: text,
            structures: [
                "person_info": [
                    "name::str::Full name",
                    "age::str::Age in years",
                    "city::str::City of residence"
                ]
            ],
            threshold: 0.3,
            includeConfidence: true
        )

        // Path B: explicit fluent builder (equivalent schema)
        let schema = model.createSchema()
            .structure("person_info")
            .field("name", dtype: "str", description: "Full name")
            .field("age", dtype: "str", description: "Age in years")
            .field("city", dtype: "str", description: "City of residence")
            .done()

        let fluent = model.extract(
            text: text,
            schema: schema,
            threshold: 0.3,
            includeConfidence: true
        )

        // Both paths should produce equivalent outputs. We compare JSON strings
        // to dodge the NSDictionary ordering tax.
        let shortcutJson = try JSONSerialization.data(
            withJSONObject: shortcut, options: [.sortedKeys]
        )
        let fluentJson = try JSONSerialization.data(
            withJSONObject: fluent, options: [.sortedKeys]
        )
        XCTAssertEqual(
            String(data: shortcutJson, encoding: .utf8),
            String(data: fluentJson, encoding: .utf8),
            "extractJson should produce the same result as the fluent builder"
        )
    }
}