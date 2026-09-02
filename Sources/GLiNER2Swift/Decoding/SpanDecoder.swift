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

// MARK: - Span Score Buffer

/// One schema's sigmoid span scores, `[instances, fields, rows, width]`, held on the CPU.
///
/// The whole tensor is read back once per schema. Slicing it on the GPU per field and per
/// count-instance instead meant one `asArray` — and so one blocking round trip — for every
/// (instance, field) pair, to fetch bytes that a single contiguous copy already brings over.
public struct SpanScoreBuffer {
    /// Row-major contents, `[instances][fields][rows][width]`.
    public let values: [Float]

    public let instances: Int
    public let fields: Int
    public let rows: Int
    public let width: Int

    /// Reads `scores` back in a single eval plus contiguous copy.
    ///
    /// The `contiguous` call is load-bearing, not tidiness: the einsum that produces these
    /// scores returns a strided view, and `asArray` on a strided array falls out of its
    /// `copyBytes` fast path into a Swift loop that walks the source chunk by chunk,
    /// recomputing the source offset per chunk. Measured at 6 ms per call on a one-sentence
    /// NER schema — more than the readback itself. One GPU copy kernel avoids all of it.
    ///
    /// - Parameter scores: `[instances, fields, rows, width]`, post-sigmoid.
    public init(_ scores: MLXArray) {
        precondition(scores.ndim == 4, "Span scores must be [instances, fields, rows, width]")
        self.instances = scores.dim(0)
        self.fields = scores.dim(1)
        self.rows = scores.dim(2)
        self.width = scores.dim(3)
        self.values = MLX.contiguous(scores).asArray(Float32.self)
    }

    /// Flat offset of `[instance][field][row][0]`.
    @inline(__always)
    func rowStart(instance: Int, field: Int, row: Int) -> Int {
        (((instance * fields) + field) * rows + row) * width
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
    ///   - scores: The schema's score buffer
    ///   - instance: Count instance to read
    ///   - field: Field to read
    ///   - rowOffset: First row of the text region (the prefix rows precede it)
    ///   - threshold: Confidence threshold
    ///   - textLen: Number of text tokens
    ///   - text: Original text
    ///   - startMap: Start character positions for each token
    ///   - endMap: End character positions for each token
    /// - Returns: List of extracted spans (text, confidence, charStart, charEnd)
    public func findSpans(
        scores: SpanScoreBuffer,
        instance: Int,
        field: Int,
        rowOffset: Int,
        threshold: Float,
        textLen: Int,
        text: String,
        startMap: [Int],
        endMap: [Int]
    ) -> [ExtractedSpan] {
        var spans: [ExtractedSpan] = []

        let widthDim = scores.width
        let rowCount = min(textLen, scores.rows - rowOffset)
        guard rowCount > 0 else { return [] }

        // Hoisted out of the candidate loop: `text.count` and index walks are O(n) each,
        // and findSpans runs once per field per count-instance over the same text.
        let scalars = text.unicodeScalars
        let scalarCount = scalars.count

        // Iterate over all positions
        let base = scores.rowStart(instance: instance, field: field, row: rowOffset)
        for start in 0..<rowCount {
            let rowBase = base + start * widthDim
            for width in 0..<widthDim {
                let score = scores.values[rowBase + width]

                guard score >= threshold else { continue }

                let end = start + width + 1  // end is exclusive

                // Validate bounds
                guard start >= 0 && start < textLen && end <= textLen else { continue }
                guard start < startMap.count && (end - 1) < endMap.count else { continue }

                // Get character positions
                let charStart = startMap[start]
                let charEnd = endMap[end - 1]

                // Slice in UNICODE SCALARS, matching the offsets the splitter produced and
                // Python's codepoint-based character offsets. Indexing by Character would
                // land in the wrong place whenever the text contains multi-scalar grapheme
                // clusters (decomposed accents, emoji sequences).
                guard charStart < scalarCount && charEnd <= scalarCount else { continue }
                let startIdx = scalars.index(scalars.startIndex, offsetBy: charStart)
                let endIdx = scalars.index(scalars.startIndex, offsetBy: charEnd)
                let spanText = String(String.UnicodeScalarView(scalars[startIdx..<endIdx]))
                    .trimmingCharacters(in: .whitespaces)

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

}

// MARK: - Choice Field Decoding

extension SpanDecoder {
    /// Find choice field value from prefix scores.
    ///
    /// - Parameters:
    ///   - scores: The schema's score buffer
    ///   - instance: Count instance to read
    ///   - field: Field to read
    ///   - prefixLength: Number of classification-prefix rows, which precede the text rows
    ///   - choices: Available choices
    ///   - textTokens: Prefix text tokens
    ///   - threshold: Confidence threshold
    ///   - dtype: "str" for single value, "list" for multiple
    /// - Returns: Selected choice(s) or nil
    public func decodeChoiceField(
        scores: SpanScoreBuffer,
        instance: Int,
        field: Int,
        prefixLength: Int,
        choices: [String],
        textTokens: [String],
        threshold: Float,
        dtype: String,
        includeConfidence: Bool = false
    ) -> Any? {
        guard prefixLength > 0 else { return nil }

        // Python reads `prefix_scores[idx, 0]` — the width-0 column only.
        func score(at index: Int) -> Float {
            scores.values[scores.rowStart(instance: instance, field: field, row: index)]
        }

        // Choice values carry no character span, so `includeSpans` adds nothing here —
        // Python emits {"text", "confidence"} for them, never start/end.
        func format(_ choice: String, _ confidence: Float) -> Any {
            includeConfidence ? ["text": choice, "confidence": confidence] : choice
        }

        if dtype == "list" {
            var selected: [Any] = []
            var seen: Set<String> = []

            for choice in choices {
                if seen.contains(choice) { continue }

                if let idx = findChoiceIndex(choice, in: textTokens) {
                    guard idx < prefixLength else { continue }

                    let value = score(at: idx)
                    if value >= threshold {
                        selected.append(format(choice, value))
                        seen.insert(choice)
                    }
                }
            }

            // Python returns the (possibly empty) list rather than dropping the key.
            return selected
        } else {
            // dtype == "str": return best match
            var best: String? = nil
            var bestScore: Float = -1.0

            for choice in choices {
                if let idx = findChoiceIndex(choice, in: textTokens) {
                    guard idx < prefixLength else { continue }

                    let value = score(at: idx)
                    if value > bestScore {
                        bestScore = value
                        best = choice
                    }
                }
            }

            if let best = best, bestScore >= threshold {
                return format(best, bestScore)
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
