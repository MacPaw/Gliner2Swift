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
// WhitespaceTokenSplitter.swift
// Fast regex-based tokenizer for text splitting
//
// Matches Python: gliner2/processor.py:WhitespaceTokenSplitter

import Foundation

/// Token with its character span in original text
public struct Token: Sendable, Equatable {
    /// The token text
    public let text: String
    /// Start character index in original text
    public let start: Int
    /// End character index in original text (exclusive)
    public let end: Int
}

/// Fast regex-based tokenizer for text splitting.
///
/// Handles:
/// - URLs (http://, https://, www.)
/// - Email addresses
/// - Twitter/social handles (@username)
/// - Words with internal hyphens/underscores
/// - Single non-whitespace characters
public struct WhitespaceTokenSplitter: Sendable {
    /// Regex pattern for tokenization
    ///
    /// Matches (in order of priority):
    /// 1. URLs: https?://... or www....
    /// 2. Emails: word@domain.tld
    /// 3. Handles: @username
    /// 4. Words: word-word_word
    /// 5. Single characters: any non-whitespace
    private static let pattern: NSRegularExpression = {
        let patternString = """
        (?:https?://[^\\s]+|www\\.[^\\s]+)\
        |[a-z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]{2,}\
        |@[a-z0-9_]+\
        |\\w+(?:[-_]\\w+)*\
        |\\S
        """
        return try! NSRegularExpression(
            pattern: patternString,
            options: [.caseInsensitive, .allowCommentsAndWhitespace]
        )
    }()

    public init() {}

    /// Tokenize text into tokens with character spans
    ///
    /// - Parameters:
    ///   - text: Input text to tokenize
    ///   - lower: Whether to lowercase the text (default: true)
    /// - Returns: Array of tokens with their spans
    public func tokenize(_ text: String, lower: Bool = true) -> [Token] {
        let processedText = lower ? text.lowercased() : text

        let range = NSRange(processedText.startIndex..., in: processedText)
        let matches = Self.pattern.matches(in: processedText, options: [], range: range)

        return matches.compactMap { match in
            guard let swiftRange = Range(match.range, in: processedText) else {
                return nil
            }
            let tokenText = String(processedText[swiftRange])
            let startIndex = processedText.distance(
                from: processedText.startIndex,
                to: swiftRange.lowerBound
            )
            let endIndex = processedText.distance(
                from: processedText.startIndex,
                to: swiftRange.upperBound
            )
            return Token(text: tokenText, start: startIndex, end: endIndex)
        }
    }

    /// Tokenize and return only token strings
    ///
    /// - Parameters:
    ///   - text: Input text to tokenize
    ///   - lower: Whether to lowercase the text (default: true)
    /// - Returns: Array of token strings
    public func tokenizeText(_ text: String, lower: Bool = true) -> [String] {
        tokenize(text, lower: lower).map { $0.text }
    }
}

// MARK: - Convenience Extensions

extension Array where Element == Token {
    /// Get start character indices
    public var starts: [Int] {
        map { $0.start }
    }

    /// Get end character indices
    public var ends: [Int] {
        map { $0.end }
    }

    /// Get token texts
    public var texts: [String] {
        map { $0.text }
    }
}
