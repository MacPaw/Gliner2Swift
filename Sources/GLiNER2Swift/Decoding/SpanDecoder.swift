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
// SpanDecoder.swift
// Span decoding utilities for GLiNER2
//
// Matches Python: gliner2/inference/engine.py:_find_spans, _format_spans

import MLX
import Foundation

// MARK: - Extracted Span

/// A span extracted from text with its metadata
public struct ExtractedSpan: Sendable, Equatable {
    /// The span text
    public let text: String

    /// Confidence score (after sigmoid)
    public let confidence: Float

    /// Character start position in original text
    public let charStart: Int

    /// Character end position in original text (exclusive)
    public let charEnd: Int

    public init(text: String, confidence: Float, charStart: Int, charEnd: Int) {
        self.text = text
        self.confidence = confidence
        self.charStart = charStart
        self.charEnd = charEnd
    }
}

// MARK: - Span Decoder

/// Decoder for extracting spans from model scores.
///
/// Key steps:
/// 1. Sigmoid is applied to raw scores
/// 2. For text spans: use `scores[:, :, -textLen:]`
/// 3. For choice fields: use `scores[:, :, :-textLen]`
/// 4. Greedy non-overlapping selection
public struct SpanDecoder {
    /// Maximum span width
    public let maxWidth: Int

    public init(maxWidth: Int) {
        self.maxWidth = maxWidth
    }

    // MARK: - Find Spans

    /// Find valid spans above threshold.
    ///
    /// - Parameters:
    ///   - scores: Span scores for a single field [textLen, maxWidth] (after sigmoid)
    ///   - threshold: Confidence threshold
    ///   - textLen: Number of text tokens
    ///   - text: Original text
    ///   - startMap: Start character positions for each token
    ///   - endMap: End character positions for each token
    /// - Returns: List of extracted spans (text, confidence, charStart, charEnd)
    public func findSpans(
        scores: MLXArray,
        threshold: Float,
        textLen: Int,
        text: String,
        startMap: [Int],
        endMap: [Int]
    ) -> [ExtractedSpan] {
        var spans: [ExtractedSpan] = []

        // Get dimensions
        let seqLen = scores.dim(0)
        let widthDim = scores.dim(1)

        // MLX doesn't have argWhere, so we iterate manually
        // First, evaluate the scores to Swift-accessible values
        MLX.eval(scores)

        // Iterate over all positions
        for start in 0..<seqLen {
            for width in 0..<widthDim {
                let score = Float(scores[start, width].item(Float32.self))

                guard score >= threshold else { continue }

                let end = start + width + 1  // end is exclusive

                // Validate bounds
                guard start >= 0 && start < textLen && end <= textLen else { continue }
                guard start < startMap.count && (end - 1) < endMap.count else { continue }

                // Get character positions
                let charStart = startMap[start]
                let charEnd = endMap[end - 1]

                // Extract text span
                guard charStart < text.count && charEnd <= text.count else { continue }
                let startIdx = text.index(text.startIndex, offsetBy: charStart)
                let endIdx = text.index(text.startIndex, offsetBy: charEnd)
                let spanText = String(text[startIdx..<endIdx]).trimmingCharacters(in: .whitespaces)

                guard !spanText.isEmpty else { continue }

                spans.append(ExtractedSpan(
                    text: spanText,
                    confidence: score,
                    charStart: charStart,
                    charEnd: charEnd
                ))
            }
        }

        return spans
    }

    // MARK: - Format Spans

    /// Format spans with greedy non-overlapping selection.
    ///
    /// Selects spans in order of confidence, skipping any that overlap
    /// with already-selected spans.
    ///
    /// - Parameters:
    ///   - spans: Raw extracted spans
    ///   - includeConfidence: Include confidence in output
    ///   - includeSpans: Include character positions in output
    /// - Returns: Formatted span dictionaries or strings
    public func formatSpans(
        _ spans: [ExtractedSpan],
        includeConfidence: Bool = false,
        includeSpans: Bool = false
    ) -> [Any] {
        guard !spans.isEmpty else { return [] }

        // Sort by confidence (descending)
        let sortedSpans = spans.sorted { $0.confidence > $1.confidence }

        var selected: [ExtractedSpan] = []

        for span in sortedSpans {
            // Check for overlap with already selected spans
            let overlaps = selected.contains { existing in
                !(span.charEnd <= existing.charStart || span.charStart >= existing.charEnd)
            }

            if !overlaps {
                selected.append(span)
            }
        }

        // Format output based on flags
        if includeSpans && includeConfidence {
            return selected.map { span in
                [
                    "text": span.text,
                    "confidence": span.confidence,
                    "start": span.charStart,
                    "end": span.charEnd
                ] as [String: Any]
            }
        } else if includeSpans {
            return selected.map { span in
                [
                    "text": span.text,
                    "start": span.charStart,
                    "end": span.charEnd
                ] as [String: Any]
            }
        } else if includeConfidence {
            return selected.map { span in
                [
                    "text": span.text,
                    "confidence": span.confidence
                ] as [String: Any]
            }
        } else {
            return selected.map { $0.text }
        }
    }

    // MARK: - Score Slicing

    /// Get text span scores (last textLen positions).
    ///
    /// - Parameters:
    ///   - scores: Full span scores [count, fields, totalLen, maxWidth]
    ///   - textLen: Number of text tokens
    /// - Returns: Text span scores [count, fields, textLen, maxWidth]
    public func getTextSpanScores(_ scores: MLXArray, textLen: Int) -> MLXArray {
        // scores[:, :, -textLen:]
        let totalLen = scores.dim(2)
        let startIdx = totalLen - textLen
        return scores[0..., 0..., startIdx...]
    }

    /// Get choice field scores (first positions, before text).
    ///
    /// - Parameters:
    ///   - scores: Full span scores [count, fields, totalLen, maxWidth]
    ///   - textLen: Number of text tokens
    /// - Returns: Choice field scores [count, fields, prefixLen, maxWidth]
    public func getChoiceFieldScores(_ scores: MLXArray, textLen: Int) -> MLXArray {
        // scores[:, :, :-textLen]
        let totalLen = scores.dim(2)
        let prefixLen = totalLen - textLen
        guard prefixLen > 0 else {
            return MLXArray.zeros([scores.dim(0), scores.dim(1), 0, scores.dim(3)])
        }
        return scores[0..., 0..., 0..<prefixLen]
    }
}

// MARK: - Choice Field Decoding

extension SpanDecoder {
    /// Find choice field value from prefix scores.
    ///
    /// - Parameters:
    ///   - prefixScores: Scores for prefix tokens [prefixLen, maxWidth]
    ///   - choices: Available choices
    ///   - textTokens: Prefix text tokens
    ///   - threshold: Confidence threshold
    ///   - dtype: "str" for single value, "list" for multiple
    /// - Returns: Selected choice(s) or nil
    public func decodeChoiceField(
        prefixScores: MLXArray,
        choices: [String],
        textTokens: [String],
        threshold: Float,
        dtype: String
    ) -> Any? {
        if dtype == "list" {
            var selected: [(String, Float)] = []
            var seen: Set<String> = []

            for choice in choices {
                if seen.contains(choice) { continue }

                if let idx = findChoiceIndex(choice, in: textTokens) {
                    guard idx < prefixScores.dim(0) else { continue }

                    let score = Float(prefixScores[idx, 0].item(Float32.self))
                    if score >= threshold {
                        selected.append((choice, score))
                        seen.insert(choice)
                    }
                }
            }

            return selected.isEmpty ? nil : selected.map { $0.0 }
        } else {
            // dtype == "str": return best match
            var best: String? = nil
            var bestScore: Float = -1.0

            for choice in choices {
                if let idx = findChoiceIndex(choice, in: textTokens) {
                    guard idx < prefixScores.dim(0) else { continue }

                    let score = Float(prefixScores[idx, 0].item(Float32.self))
                    if score > bestScore {
                        bestScore = score
                        best = choice
                    }
                }
            }

            if let best = best, bestScore >= threshold {
                return best
            }
            return nil
        }
    }

    /// Find index of choice in tokens (case-insensitive).
    private func findChoiceIndex(_ choice: String, in tokens: [String]) -> Int? {
        let choiceLower = choice.lowercased()
        for (i, token) in tokens.enumerated() {
            if token.lowercased() == choiceLower || token.lowercased().contains(choiceLower) {
                return i
            }
        }
        return nil
    }
}
