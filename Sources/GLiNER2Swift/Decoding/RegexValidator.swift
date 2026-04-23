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
// RegexValidator.swift
// Regex-based span filter for post-processing
//
// Matches Python: gliner2/inference/engine.py:RegexValidator

import Foundation

/// Errors thrown by `RegexValidator`.
public enum RegexValidatorError: Error, CustomStringConvertible {
    /// Pattern contains a construct with divergent semantics between Python `re`
    /// and `NSRegularExpression` (e.g. atomic groups, possessive quantifiers).
    case unsupportedConstruct(pattern: String, construct: String)

    public var description: String {
        switch self {
        case .unsupportedConstruct(let pattern, let construct):
            return "RegexValidator: pattern \(pattern.debugDescription) contains " +
                "unsupported construct \(construct.debugDescription) — this has " +
                "divergent semantics between Python re and NSRegularExpression and " +
                "cannot be used in a parity-sensitive validator."
        }
    }
}

/// Regex-based span filter for post-processing.
///
/// Use to filter extracted spans based on regex patterns.
/// Can be used in two modes:
/// - `full`: Pattern must match the entire span text
/// - `partial`: Pattern only needs to appear somewhere in the span text
///
/// ## Python parity
///
/// This mirrors `gliner2.inference.engine.RegexValidator`. Default options are
/// `.caseInsensitive`, matching Python's `flags=re.IGNORECASE`. Validators are
/// applied only to structure-field span extraction (not NER/relations), matching
/// Python behavior.
///
/// ## Engine differences
///
/// `NSRegularExpression` (ICU) is not a drop-in for Python `re`. Known
/// divergences the user should avoid if strict parity is required:
/// - Variable-width lookbehind: supported in Python 3.7+, rejected by ICU at
///   init time (Swift will throw).
/// - Possessive quantifiers (`*+`, `++`, `?+`, `}+`), atomic groups (`(?>...)`):
///   available in newer Python and ICU but may exhibit subtle backtracking
///   differences. Swift throws `RegexValidatorError.unsupportedConstruct` on
///   these at init time as a best-effort parity guard.
/// - `\w`, `\d`, `\b`: both engines default to Unicode semantics but exact
///   character sets differ at the edges.
///
/// Thread-safety: `NSRegularExpression`'s matching methods are documented
/// thread-safe by Apple, so `RegexValidator` is declared `@unchecked Sendable`.
///
/// Example:
/// ```swift
/// // Only keep spans that look like phone numbers
/// let phoneValidator = try RegexValidator(
///     pattern: #"\d{3}-\d{3}-\d{4}"#,
///     mode: .full
/// )
///
/// // Exclude spans containing "test"
/// let excludeTestValidator = try RegexValidator(
///     pattern: "test",
///     mode: .partial,
///     exclude: true
/// )
/// ```
public struct RegexValidator: @unchecked Sendable {
    /// The compiled regex pattern
    private let regex: NSRegularExpression

    /// Match mode: full or partial
    public let mode: MatchMode

    /// If true, the validator passes when the pattern does NOT match
    public let exclude: Bool

    /// Match mode options
    public enum MatchMode: Sendable {
        /// Pattern must match the entire span text
        case full
        /// Pattern only needs to appear somewhere in the span text
        case partial
    }

    /// Best-effort token checks for constructs with divergent semantics between
    /// Python `re` and `NSRegularExpression` (ICU). Matched as raw substrings —
    /// not perfect (won't exclude these tokens inside character classes) but
    /// catches the common cases.
    private static let unsupportedTokens: [String] = [
        "(?>",  // atomic group
        "*+", "++", "?+", "}+"  // possessive quantifiers
    ]

    /// Initialize a regex validator
    ///
    /// - Parameters:
    ///   - pattern: Regex pattern string
    ///   - mode: Match mode (default: .full)
    ///   - exclude: If true, invert the match result (default: false)
    ///   - options: Regex options (default: .caseInsensitive)
    /// - Throws:
    ///   - `RegexValidatorError.unsupportedConstruct` if the pattern contains
    ///     atomic groups or possessive quantifiers (parity guard).
    ///   - Any `NSError` thrown by `NSRegularExpression` for invalid syntax
    ///     (including variable-width lookbehind, which ICU rejects).
    public init(
        pattern: String,
        mode: MatchMode = .full,
        exclude: Bool = false,
        options: NSRegularExpression.Options = .caseInsensitive
    ) throws {
        for token in Self.unsupportedTokens where pattern.contains(token) {
            throw RegexValidatorError.unsupportedConstruct(pattern: pattern, construct: token)
        }
        self.regex = try NSRegularExpression(pattern: pattern, options: options)
        self.mode = mode
        self.exclude = exclude
    }

    /// Initialize with pre-compiled regex
    ///
    /// - Parameters:
    ///   - regex: Compiled NSRegularExpression
    ///   - mode: Match mode
    ///   - exclude: If true, invert the match result
    public init(
        regex: NSRegularExpression,
        mode: MatchMode = .full,
        exclude: Bool = false
    ) {
        self.regex = regex
        self.mode = mode
        self.exclude = exclude
    }

    /// Validate a span text
    ///
    /// - Parameter text: The span text to validate
    /// - Returns: true if the span passes validation
    public func validate(_ text: String) -> Bool {
        let range = NSRange(text.startIndex..., in: text)

        let matched: Bool
        switch mode {
        case .full:
            // Check if pattern matches entire string
            if let match = regex.firstMatch(in: text, options: [], range: range) {
                matched = match.range == range
            } else {
                matched = false
            }
        case .partial:
            // Check if pattern appears anywhere
            matched = regex.firstMatch(in: text, options: [], range: range) != nil
        }

        return exclude ? !matched : matched
    }

    /// Call as function (convenience)
    public func callAsFunction(_ text: String) -> Bool {
        validate(text)
    }
}

// MARK: - Filtering Extensions

extension Array where Element == ExtractedSpan {
    /// Filter spans using validators
    ///
    /// - Parameter validators: Array of validators (all must pass)
    /// - Returns: Filtered spans
    public func filtered(by validators: [RegexValidator]) -> [ExtractedSpan] {
        guard !validators.isEmpty else { return self }

        return filter { span in
            validators.allSatisfy { $0.validate(span.text) }
        }
    }
}

// MARK: - Common Validators

extension RegexValidator {
    /// Validator that only keeps numeric spans
    public static var numericOnly: RegexValidator {
        try! RegexValidator(pattern: #"^\d+$"#, mode: .full)
    }

    /// Validator that only keeps alphanumeric spans
    public static var alphanumericOnly: RegexValidator {
        try! RegexValidator(pattern: #"^[a-zA-Z0-9]+$"#, mode: .full)
    }

    /// Validator that excludes single-character spans
    public static var excludeSingleChar: RegexValidator {
        try! RegexValidator(pattern: #"^.$"#, mode: .full, exclude: true)
    }

    /// Validator for email-like patterns
    public static var emailLike: RegexValidator {
        try! RegexValidator(
            pattern: #"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"#,
            mode: .full
        )
    }

    /// Validator for phone number patterns (US format)
    public static var phoneNumberUS: RegexValidator {
        try! RegexValidator(
            pattern: #"^\(?[\d]{3}\)?[-.\s]?[\d]{3}[-.\s]?[\d]{4}$"#,
            mode: .full
        )
    }
}
