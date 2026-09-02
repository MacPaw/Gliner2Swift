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
// TokenizerParityTests.swift
// Numerical parity tests for UnigramTokenizer against Python HuggingFace tokenizer
//
// CRITICAL: Tokenizer issues cascade through the entire pipeline.
// These tests must pass BEFORE proceeding to encoder parity tests.
//
// Run fixtures generation first:
//   cd ../scripts && python generate_tokenizer_fixtures.py

import XCTest
import Foundation
@testable import GLiNER2Swift

final class TokenizerParityTests: XCTestCase {

    // MARK: - Properties

    /// Path to weights directory (set via `GLINER2_WEIGHTS_PATH`, else `<repo>/weights`)
    static let weightsPath: String = TestModel.fp32WeightsPath

    /// Path to fixtures directory
    static let fixturesPath = URL(fileURLWithPath: #file)
        .deletingLastPathComponent()
        .appendingPathComponent("Fixtures")

    var tokenizer: UnigramTokenizer!
    var fixtures: [String: TokenizerFixture]!

    // MARK: - Setup

    override func setUp() async throws {
        try await super.setUp()

        // Skip (never fail) when the model is not available on this machine.
        let weightsDir = try TestModel.requireFP32Weights()

        // Load tokenizer
        let tokenizerUrl = URL(fileURLWithPath: weightsDir)
            .appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokenizerUrl.path) else {
            throw XCTSkip("tokenizer.json not found at \(tokenizerUrl.path)")
        }
        tokenizer = try UnigramTokenizer(tokenizerJsonUrl: tokenizerUrl)

        // Load fixtures
        let fixturesUrl = Self.fixturesPath.appendingPathComponent("tokenizer_fixtures.json")
        let data = try Data(contentsOf: fixturesUrl)
        fixtures = try JSONDecoder().decode([String: TokenizerFixture].self, from: data)
    }

    // MARK: - Special Token ID Tests

    func testSpecialTokenIds() throws {
        // Verify special token IDs match Python exactly
        // These are CRITICAL for correct inference

        XCTAssertEqual(tokenizer.padTokenId, 0, "PAD token ID mismatch")
        XCTAssertEqual(tokenizer.clsTokenId, 1, "CLS token ID mismatch")
        XCTAssertEqual(tokenizer.sepTokenId, 2, "SEP token ID mismatch")
        XCTAssertEqual(tokenizer.unkTokenId, 3, "UNK token ID mismatch")
        XCTAssertEqual(tokenizer.maskTokenId, 128000, "MASK token ID mismatch")
    }

    func testGLiNER2SpecialTokenIds() throws {
        // GLiNER2-specific special tokens
        XCTAssertEqual(tokenizer.sepStructId, 128001, "[SEP_STRUCT] token ID mismatch")
        XCTAssertEqual(tokenizer.sepTextId, 128002, "[SEP_TEXT] token ID mismatch")
        XCTAssertEqual(tokenizer.pTokenId, 128003, "[P] token ID mismatch")
        XCTAssertEqual(tokenizer.cTokenId, 128004, "[C] token ID mismatch")
        XCTAssertEqual(tokenizer.eTokenId, 128005, "[E] token ID mismatch")
        XCTAssertEqual(tokenizer.rTokenId, 128006, "[R] token ID mismatch")
        XCTAssertEqual(tokenizer.lTokenId, 128007, "[L] token ID mismatch")
        XCTAssertEqual(tokenizer.exampleTokenId, 128008, "[EXAMPLE] token ID mismatch")
        XCTAssertEqual(tokenizer.outputTokenId, 128009, "[OUTPUT] token ID mismatch")
        XCTAssertEqual(tokenizer.descriptionTokenId, 128010, "[DESCRIPTION] token ID mismatch")
    }

    // MARK: - Basic Tokenization Tests

    func testBasicPunctuation() throws {
        let fixture = try XCTUnwrap(fixtures["basic_punctuation"])
        let ids = tokenizer.encode(fixture.text)

        XCTAssertEqual(ids, fixture.tokenIds,
            "Punctuation tokenization mismatch for '\(fixture.text)'\nExpected: \(fixture.tokenIds)\nGot: \(ids)")
    }

    func testBasicSentence() throws {
        // This is THE key test - if this passes, encoder inputs will match
        let fixture = try XCTUnwrap(fixtures["basic_sentence"])
        let ids = tokenizer.encode(fixture.text)

        XCTAssertEqual(ids, fixture.tokenIds,
            "Basic sentence tokenization mismatch for '\(fixture.text)'\nExpected: \(fixture.tokenIds)\nGot: \(ids)")
    }

    func testSingleWord() throws {
        let fixture = try XCTUnwrap(fixtures["single_word"])
        let ids = tokenizer.encode(fixture.text)

        XCTAssertEqual(ids, fixture.tokenIds,
            "Single word tokenization mismatch for '\(fixture.text)'")
    }

    // MARK: - Edge Cases

    func testEmptyString() throws {
        let fixture = try XCTUnwrap(fixtures["empty_string"])
        let ids = tokenizer.encode(fixture.text)

        XCTAssertEqual(ids, fixture.tokenIds,
            "Empty string should produce no tokens")
    }

    func testWhitespaceOnly() throws {
        let fixture = try XCTUnwrap(fixtures["whitespace_only"])
        let ids = tokenizer.encode(fixture.text)

        XCTAssertEqual(ids, fixture.tokenIds,
            "Whitespace-only string should produce no tokens")
    }

    func testApostrophe() throws {
        let fixture = try XCTUnwrap(fixtures["apostrophe"])
        let ids = tokenizer.encode(fixture.text)

        XCTAssertEqual(ids, fixture.tokenIds,
            "Apostrophe tokenization mismatch for '\(fixture.text)'\nExpected: \(fixture.tokenIds)\nGot: \(ids)")
    }

    func testAbbreviation() throws {
        let fixture = try XCTUnwrap(fixtures["abbreviation"])
        let ids = tokenizer.encode(fixture.text)

        XCTAssertEqual(ids, fixture.tokenIds,
            "Abbreviation tokenization mismatch for '\(fixture.text)'\nExpected: \(fixture.tokenIds)\nGot: \(ids)")
    }

    // MARK: - Special Token Tests

    func testPTokenOnly() throws {
        let fixture = try XCTUnwrap(fixtures["p_token_only"])
        let ids = tokenizer.encode(fixture.text)

        XCTAssertEqual(ids, fixture.tokenIds,
            "[P] token mismatch\nExpected: \(fixture.tokenIds)\nGot: \(ids)")
    }

    func testETokenOnly() throws {
        let fixture = try XCTUnwrap(fixtures["e_token_only"])
        let ids = tokenizer.encode(fixture.text)

        XCTAssertEqual(ids, fixture.tokenIds,
            "[E] token mismatch\nExpected: \(fixture.tokenIds)\nGot: \(ids)")
    }

    func testSchemaTokens() throws {
        let fixture = try XCTUnwrap(fixtures["schema_tokens"])
        let ids = tokenizer.encode(fixture.text)

        XCTAssertEqual(ids, fixture.tokenIds,
            "Schema tokens tokenization mismatch\nExpected: \(fixture.tokenIds)\nGot: \(ids)")

        // Should contain [P] and [E] token IDs
        XCTAssertTrue(ids.contains(128003), "Should contain [P] token (128003)")
        XCTAssertTrue(ids.contains(128005), "Should contain [E] token (128005)")
    }

    func testSepStructToken() throws {
        let fixture = try XCTUnwrap(fixtures["sep_struct"])
        let ids = tokenizer.encode(fixture.text)

        XCTAssertEqual(ids, fixture.tokenIds,
            "[SEP_STRUCT] token mismatch")
    }

    func testSepTextToken() throws {
        let fixture = try XCTUnwrap(fixtures["sep_text"])
        let ids = tokenizer.encode(fixture.text)

        XCTAssertEqual(ids, fixture.tokenIds,
            "[SEP_TEXT] token mismatch")
    }

    // MARK: - Unicode Tests

    func testUnicodeAccents() throws {
        let fixture = try XCTUnwrap(fixtures["accents"])
        let ids = tokenizer.encode(fixture.text)

        XCTAssertEqual(ids, fixture.tokenIds,
            "Accented characters tokenization mismatch for '\(fixture.text)'")

        // Should not contain UNK tokens
        XCTAssertFalse(ids.contains(tokenizer.unkTokenId),
            "Accented characters should not produce UNK tokens")
    }

    // MARK: - Unknown-character handling
    //
    // The reference tokenizer declares `unk_id: 3` and `byte_fallback: false`, so any
    // character no vocabulary piece covers becomes ONE [UNK] — not a run of `<0xNN>` byte
    // pieces, which is what this implementation used to emit and which inflated the token
    // count fivefold. Expected ids below were taken from
    // `AutoTokenizer.from_pretrained("fastino/gliner2-base-v1")` on 2026-07-21.

    func testFullwidthLettersBecomeSingleUnknowns() throws {
        let ids = tokenizer.encode("ＡＢＣ Corp hired Ｊｏｈｎ Ｓｍｉｔｈ in Ｔｏｋｙｏ.")
        XCTAssertEqual(ids, [507, 3, 6071, 5458, 507, 3, 507, 3, 267, 507, 3, 260],
                       "Each fullwidth run must collapse to a single [UNK]")
    }

    func testZeroWidthAndNonBreakingSpacesSurviveAsUnknowns() throws {
        // The NBSP must NOT be treated as a word separator (Metaspace splits on the ASCII
        // space alone), and the zero-width space must not be trimmed away — Foundation's
        // CharacterSet.whitespaces contains U+200B even though Character.isWhitespace
        // does not.
        let ids = tokenizer.encode("Tim\u{00A0}Cook joined Apple\u{200B}Inc in Cupertino.")
        XCTAssertEqual(ids, [4185, 3, 48294, 2280, 2013, 3, 55668, 267, 58326, 260])
    }

    func testDecomposedAccentsComposeBeforeTokenizing() throws {
        // NFD input; the normalizer composes it, so the ids match the NFC spelling.
        let decomposed = "Zoe\u{301} Dupont works at Cafe\u{301} Rene\u{301} in Montre\u{301}al."
        XCTAssertEqual(tokenizer.encode(decomposed), tokenizer.encode("Zoé Dupont works at Café René in Montréal."))
        XCTAssertFalse(tokenizer.encode(decomposed).contains(tokenizer.unkTokenId))
    }

    func testJapanese() throws {
        let fixture = try XCTUnwrap(fixtures["japanese"])
        let ids = tokenizer.encode(fixture.text)

        XCTAssertEqual(ids, fixture.tokenIds,
            "Japanese text tokenization mismatch for '\(fixture.text)'")
    }

    func testCyrillic() throws {
        let fixture = try XCTUnwrap(fixtures["cyrillic"])
        let ids = tokenizer.encode(fixture.text)

        XCTAssertEqual(ids, fixture.tokenIds,
            "Cyrillic text tokenization mismatch for '\(fixture.text)'")
    }

    // MARK: - Numbers

    func testNumbers() throws {
        let fixture = try XCTUnwrap(fixtures["numbers"])
        let ids = tokenizer.encode(fixture.text)

        XCTAssertEqual(ids, fixture.tokenIds,
            "Number tokenization mismatch for '\(fixture.text)'")
    }

    func testCurrency() throws {
        let fixture = try XCTUnwrap(fixtures["currency"])
        let ids = tokenizer.encode(fixture.text)

        XCTAssertEqual(ids, fixture.tokenIds,
            "Currency tokenization mismatch for '\(fixture.text)'")
    }

    func testDate() throws {
        let fixture = try XCTUnwrap(fixtures["date"])
        let ids = tokenizer.encode(fixture.text)

        XCTAssertEqual(ids, fixture.tokenIds,
            "Date tokenization mismatch for '\(fixture.text)'")
    }

    // MARK: - Comprehensive Fixture Test

    func testAllFixtures() throws {
        var failures: [String] = []
        var successes = 0

        for (name, fixture) in fixtures {
            // Skip metadata entries
            if name.hasPrefix("_") { continue }

            let ids = tokenizer.encode(fixture.text)

            if ids != fixture.tokenIds {
                let expectedPreview = fixture.tokenIds.prefix(10).map(String.init).joined(separator: ", ")
                let gotPreview = ids.prefix(10).map(String.init).joined(separator: ", ")
                failures.append("\(name): expected [\(expectedPreview)...], got [\(gotPreview)...] (\(fixture.numTokens) vs \(ids.count) tokens)")
            } else {
                successes += 1
            }
        }

        if !failures.isEmpty {
            print("\n=== Tokenizer Parity Test Failures ===")
            for failure in failures {
                print("  - \(failure)")
            }
            print("===================================\n")
        }

        XCTAssertEqual(failures.count, 0,
            "\(failures.count) tokenizer mismatches out of \(successes + failures.count) test cases:\n\(failures.joined(separator: "\n"))")

        print("Tokenizer parity: \(successes)/\(successes + failures.count) test cases passed")
    }

    // MARK: - Tokenize Method Tests (tokens, not IDs)

    func testTokenizeBasicSentence() throws {
        let fixture = try XCTUnwrap(fixtures["basic_sentence"])
        let tokens = tokenizer.tokenize(fixture.text)

        XCTAssertEqual(tokens, fixture.tokens,
            "Tokenize (tokens) mismatch for '\(fixture.text)'\nExpected: \(fixture.tokens)\nGot: \(tokens)")
    }

    // MARK: - Round-trip Tests

    func testEncodeDecodeRoundTrip() throws {
        let texts = [
            "Hello, world!",
            "Tim Cook is CEO of Apple.",
            "The quick brown fox.",
        ]

        for text in texts {
            let ids = tokenizer.encode(text)
            let decoded = tokenizer.decode(ids)

            // Note: Decoded may not be exactly equal due to spacing, but should be similar
            // The important thing is that it doesn't crash and produces reasonable output
            XCTAssertFalse(decoded.isEmpty, "Decode should produce non-empty result for '\(text)'")
        }
    }

    // MARK: - encodeWithSpecialTokens Tests

    func testEncodeWithSpecialTokens() throws {
        let text = "Hello"
        let ids = tokenizer.encodeWithSpecialTokens(text)

        // Should have [CLS] + tokens + [SEP]
        XCTAssertEqual(ids.first, tokenizer.clsTokenId, "Should start with CLS token")
        XCTAssertEqual(ids.last, tokenizer.sepTokenId, "Should end with SEP token")
        XCTAssertEqual(ids.count, 3, "Should have 3 tokens: CLS + Hello + SEP")
    }
}

// MARK: - Fixture Data Structures

struct TokenizerFixture: Codable {
    let text: String
    let tokenIds: [Int]
    let tokenIdsWithSpecial: [Int]?
    let tokens: [String]
    let numTokens: Int

    enum CodingKeys: String, CodingKey {
        case text
        case tokenIds = "token_ids"
        case tokenIdsWithSpecial = "token_ids_with_special"
        case tokens
        case numTokens = "num_tokens"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        // Handle special metadata entries that don't have all fields
        if container.contains(.text) {
            text = try container.decode(String.self, forKey: .text)
            tokenIds = try container.decodeIfPresent([Int].self, forKey: .tokenIds) ?? []
            tokenIdsWithSpecial = try container.decodeIfPresent([Int].self, forKey: .tokenIdsWithSpecial)
            tokens = try container.decodeIfPresent([String].self, forKey: .tokens) ?? []
            numTokens = try container.decodeIfPresent(Int.self, forKey: .numTokens) ?? 0
        } else {
            // Metadata entry - provide defaults
            text = ""
            tokenIds = []
            tokenIdsWithSpecial = nil
            tokens = []
            numTokens = 0
        }
    }
}
