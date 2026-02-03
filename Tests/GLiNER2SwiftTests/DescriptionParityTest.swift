// DescriptionParityTest.swift
// Parity test for schema descriptions between Python and Swift

import XCTest
@testable import GLiNER2Swift

final class DescriptionParityTests: XCTestCase {

    func testEntityDescriptionsParity() throws {
        let tokenizerPath = URL(fileURLWithPath: "/Users/tmwstw/Documents/mnemos/GLiNER2/GLiNER2Swift/gliner2-base-v1")
        guard FileManager.default.fileExists(atPath: tokenizerPath.path) else {
            throw XCTSkip("Tokenizer not available")
        }

        let processor = try SchemaTransformer.createFromLocalDirectory(directoryUrl: tokenizerPath)
        let text = "Tim Cook is CEO of Apple."

        // Entity with descriptions
        let entitySchema = Schema().entities([
            "person": "A human being's name",
            "company": "A business organization"
        ])
        let result = processor.transform(text: text, schema: entitySchema.build())

        print("\n=== SWIFT ENTITY WITH DESCRIPTIONS ===")
        for (i, tokens) in result.schemaTokensList.enumerated() {
            print("Schema \(i): \(tokens)")
        }

        // Python output:
        // Schema 0: ['(', '[P]', "entities [DESCRIPTION] person: A human being's name [DESCRIPTION] company: A business organization", '(', '[E]', 'person', '[E]', 'company', ')', ')']

        let tokens = result.schemaTokensList[0]
        XCTAssertEqual(tokens[0], "(")
        XCTAssertEqual(tokens[1], "[P]")
        XCTAssertTrue(tokens[2].contains("[DESCRIPTION]"), "Should contain [DESCRIPTION] token")
        XCTAssertTrue(tokens[2].hasPrefix("entities"), "Should start with 'entities'")
        XCTAssertTrue(tokens[2].contains("person: A human being's name"), "Should contain person description")
        XCTAssertTrue(tokens[2].contains("company: A business organization"), "Should contain company description")
        XCTAssertEqual(tokens[3], "(")
        // Check for [E] tokens (order may vary due to dictionary)
        XCTAssertTrue(tokens.contains("[E]"))
        XCTAssertTrue(tokens.contains("person"))
        XCTAssertTrue(tokens.contains("company"))
    }

    func testStructureDescriptionsParity() throws {
        let tokenizerPath = URL(fileURLWithPath: "/Users/tmwstw/Documents/mnemos/GLiNER2/GLiNER2Swift/gliner2-base-v1")
        guard FileManager.default.fileExists(atPath: tokenizerPath.path) else {
            throw XCTSkip("Tokenizer not available")
        }

        let processor = try SchemaTransformer.createFromLocalDirectory(directoryUrl: tokenizerPath)
        let text = "Tim Cook is CEO of Apple."

        // Structure with field descriptions
        let structSchema = Schema()
            .structure("person")
            .field("name", description: "The person's full name")
            .field("age", description: "The person's age in years")
            .done()
        let result = processor.transform(text: text, schema: structSchema.build())

        print("\n=== SWIFT STRUCTURE WITH DESCRIPTIONS ===")
        for (i, tokens) in result.schemaTokensList.enumerated() {
            print("Schema \(i): \(tokens)")
        }

        // Python output:
        // Schema 0: ['(', '[P]', "person [DESCRIPTION] name: The person's full name [DESCRIPTION] age: The person's age in years", '(', '[C]', 'age', '[C]', 'name', ')', ')']

        let tokens = result.schemaTokensList[0]
        XCTAssertEqual(tokens[0], "(")
        XCTAssertEqual(tokens[1], "[P]")
        XCTAssertTrue(tokens[2].contains("[DESCRIPTION]"), "Should contain [DESCRIPTION] token")
        XCTAssertTrue(tokens[2].hasPrefix("person"), "Should start with 'person'")
        XCTAssertTrue(tokens[2].contains("name: The person's full name"), "Should contain name description")
        XCTAssertTrue(tokens[2].contains("age: The person's age in years"), "Should contain age description")
        XCTAssertEqual(tokens[3], "(")
        // Check for [C] tokens
        XCTAssertTrue(tokens.contains("[C]"))
        XCTAssertTrue(tokens.contains("name"))
        XCTAssertTrue(tokens.contains("age"))
    }

    func testEntityWithoutDescriptionsParity() throws {
        let tokenizerPath = URL(fileURLWithPath: "/Users/tmwstw/Documents/mnemos/GLiNER2/GLiNER2Swift/gliner2-base-v1")
        guard FileManager.default.fileExists(atPath: tokenizerPath.path) else {
            throw XCTSkip("Tokenizer not available")
        }

        let processor = try SchemaTransformer.createFromLocalDirectory(directoryUrl: tokenizerPath)
        let text = "Tim Cook is CEO of Apple."

        // Entity without descriptions
        let entitySchema = Schema().entities(["person", "company"])
        let result = processor.transform(text: text, schema: entitySchema.build())

        print("\n=== SWIFT ENTITY WITHOUT DESCRIPTIONS ===")
        for (i, tokens) in result.schemaTokensList.enumerated() {
            print("Schema \(i): \(tokens)")
        }

        // Python output:
        // Schema 0: ['(', '[P]', 'entities', '(', '[E]', 'person', '[E]', 'company', ')', ')']

        let tokens = result.schemaTokensList[0]
        XCTAssertEqual(tokens[0], "(")
        XCTAssertEqual(tokens[1], "[P]")
        XCTAssertEqual(tokens[2], "entities")  // No descriptions
        XCTAssertFalse(tokens[2].contains("[DESCRIPTION]"), "Should NOT contain [DESCRIPTION] token")
        XCTAssertEqual(tokens[3], "(")
    }

    func testStructureWithoutDescriptionsParity() throws {
        let tokenizerPath = URL(fileURLWithPath: "/Users/tmwstw/Documents/mnemos/GLiNER2/GLiNER2Swift/gliner2-base-v1")
        guard FileManager.default.fileExists(atPath: tokenizerPath.path) else {
            throw XCTSkip("Tokenizer not available")
        }

        let processor = try SchemaTransformer.createFromLocalDirectory(directoryUrl: tokenizerPath)
        let text = "Tim Cook is CEO of Apple."

        // Structure without descriptions
        let structSchema = Schema()
            .structure("person")
            .field("name")
            .field("age")
            .done()
        let result = processor.transform(text: text, schema: structSchema.build())

        print("\n=== SWIFT STRUCTURE WITHOUT DESCRIPTIONS ===")
        for (i, tokens) in result.schemaTokensList.enumerated() {
            print("Schema \(i): \(tokens)")
        }

        // Python output:
        // Schema 0: ['(', '[P]', 'person', '(', '[C]', 'age', '[C]', 'name', ')', ')']

        let tokens = result.schemaTokensList[0]
        XCTAssertEqual(tokens[0], "(")
        XCTAssertEqual(tokens[1], "[P]")
        XCTAssertEqual(tokens[2], "person")  // No descriptions
        XCTAssertFalse(tokens[2].contains("[DESCRIPTION]"), "Should NOT contain [DESCRIPTION] token")
        XCTAssertEqual(tokens[3], "(")
    }
}
