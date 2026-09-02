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
// StructureGroupingTests.swift
// Phase 6.6 — repeated `.structure(name)` merges into one schema block with the union of
// fields (first-seen order), matching Python processor.py:639-651.

import XCTest
@testable import GLiNER2Swift

final class StructureGroupingTests: XCTestCase {

    func testRepeatedStructureNameMergesFields() {
        let schema = Schema()
            .structure("person").field("name").field("age").done()
            .structure("person").field("age").field("email").done()   // age repeats
        let dict = schema.build()

        let structures = dict["json_structures"] as? [[String: Any]] ?? []
        // One block for "person", not two.
        XCTAssertEqual(structures.count, 1)
        let fields = structures.first?["person"] as? [String: Any] ?? [:]
        XCTAssertEqual(Set(fields.keys), ["name", "age", "email"])

        // Field order is first-seen union: name, age, email (age not duplicated).
        let order = (dict["_field_orders"] as? [String: [String]])?["person"]
        XCTAssertEqual(order, ["name", "age", "email"])
    }

    func testMergedStructureProducesOneSchemaBlock() throws {
        let dir = try TestModel.requireTokenizerDirectory()
        let processor = try SchemaTransformer.createFromLocalDirectory(directoryUrl: dir)
        let schema = Schema()
            .structure("person").field("name").done()
            .structure("person").field("age").done()
            .build()

        let record = processor.transform(text: "John is 30.", schema: schema)

        // Exactly one structure schema, and it carries both fields.
        let structureBlocks = record.schemaTokensList.filter { $0.contains("[C]") }
        XCTAssertEqual(structureBlocks.count, 1, "same-named structures must not produce two blocks")
        let joined = structureBlocks.first?.joined(separator: " ") ?? ""
        XCTAssertTrue(joined.contains("name"))
        XCTAssertTrue(joined.contains("age"))
    }

    func testMergedDescriptionsCombine() {
        let schema = Schema()
            .structure("person").field("name", description: "full name").done()
            .structure("person").field("age", description: "years old").done()
            .build()
        let descriptions = (schema["json_descriptions"] as? [String: [String: String]])?["person"]
        XCTAssertEqual(descriptions?["name"], "full name")
        XCTAssertEqual(descriptions?["age"], "years old")
    }
}
