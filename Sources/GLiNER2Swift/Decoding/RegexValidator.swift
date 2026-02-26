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

/// Regex-based span filter for post-processing.
///
/// Use to filter extracted spans based on regex patterns.
/// Can be used in two modes:
/// - `full`: Pattern must match the entire span text
/// - `partial`: Pattern only needs to appear somewhere in the span text
///
/// Example:
/// ```swift
/// // Only keep spans that look like phone numbers
/// let phoneValidator = RegexValidator(
///     pattern: #"\d{3}-\d{3}-\d{4}"#,
///     mode: .full
/// )
///
/// // Exclude spans containing "test"
/// let excludeTestValidator = RegexValidator(
///     pattern: "test",
///     mode: .partial,
///     exclude: true
/// )
/// ```
public struct RegexValidator: Sendable {
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

    /// Initialize a regex validator
    ///
    /// - Parameters:
    ///   - pattern: Regex pattern string
    ///   - mode: Match mode (default: .full)
    ///   - exclude: If true, invert the match result (default: false)
    ///   - options: Regex options (default: .caseInsensitive)
    /// - Throws: Error if pattern is invalid
    public init(
        pattern: String,
        mode: MatchMode = .full,
        exclude: Bool = false,
        options: NSRegularExpression.Options = .caseInsensitive
    ) throws {
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
