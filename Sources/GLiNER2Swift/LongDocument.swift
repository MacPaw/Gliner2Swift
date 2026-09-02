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
// LongDocument.swift
// Phase 6.2 — chunked extraction for documents longer than the model's context. Split into
// overlapping word-chunks, extract each (capped at the chunk size), remap character spans
// back to the original text, and merge. Ports Python chunking.py + engine.py:938-1108.

import Foundation

// MARK: - Chunk

/// A window of the original text plus the character (unicode-scalar) offset where it starts,
/// so spans found inside the chunk can be shifted back onto the original document.
public struct TextChunk: Equatable, Sendable {
    public let text: String
    public let charOffset: Int
}

public enum TextChunker {

    /// Split `text` into overlapping windows of at most `chunkSize` whitespace-split words,
    /// advancing by `chunkSize - overlap` words each step. Each chunk's `text` is an exact
    /// substring of the original (case and spacing preserved), so span offsets shift by a
    /// simple add.
    public static func split(
        _ text: String, chunkSize: Int, overlap: Int, splitter: WhitespaceTokenSplitter
    ) -> [TextChunk] {
        precondition(chunkSize > 0 && overlap >= 0 && overlap < chunkSize,
                     "need chunkSize > 0 and 0 <= overlap < chunkSize")

        // Preserve case for the chunk text; offsets are unicode-scalar positions.
        let words = splitter.tokenize(text, lower: false)
        guard !words.isEmpty else { return [] }
        if words.count <= chunkSize { return [TextChunk(text: text, charOffset: 0)] }

        let scalars = Array(text.unicodeScalars)
        let step = chunkSize - overlap
        var chunks: [TextChunk] = []
        var start = 0
        while start < words.count {
            let end = min(start + chunkSize, words.count)
            let from = words[start].start
            let to = words[end - 1].end
            guard from < to, to <= scalars.count else { break }
            let piece = String(String.UnicodeScalarView(scalars[from ..< to]))
            chunks.append(TextChunk(text: piece, charOffset: from))
            if end == words.count { break }
            start += step
        }
        return chunks
    }
}

// MARK: - Long-document extraction

extension GLiNER2 {

    /// Extract from a document too long for one pass, by chunking. Character spans in the
    /// result are relative to the ORIGINAL text. (Python engine.py:938-1108)
    ///
    /// - Parameters:
    ///   - chunkSize: words per chunk (default 384)
    ///   - overlap: overlapping words between adjacent chunks (default 64)
    public func extractLong(
        text: String,
        schema: Schema,
        chunkSize: Int = 384,
        overlap: Int = 64,
        threshold: Float = 0.5,
        includeConfidence: Bool = false,
        includeSpans: Bool = false
    ) -> [String: Any] {
        let chunks = TextChunker.split(text, chunkSize: chunkSize, overlap: overlap,
                                       splitter: processor.wordSplitter)
        guard !chunks.isEmpty else { return [:] }

        // Extract each chunk WITH spans + confidence (needed to remap and to merge by
        // confidence), then shift spans onto the original document.
        let perChunk: [[String: Any]] = chunks.map { chunk in
            let raw = extract(text: chunk.text, schema: schema, threshold: threshold,
                              includeConfidence: true, includeSpans: true, maxLen: chunkSize)
            return Self.shiftSpans(raw, by: chunk.charOffset)
        }

        var merged = Self.mergeChunkResults(perChunk)
        // Strip the span/confidence metadata the caller did not ask for (we always compute
        // it internally for remapping and merging).
        merged = Self.applyOutputFlags(merged, includeConfidence: includeConfidence,
                                       includeSpans: includeSpans)
        return merged
    }

    /// `extractLong` over many documents.
    public func batchExtractLong(
        texts: [String],
        schema: Schema,
        chunkSize: Int = 384,
        overlap: Int = 64,
        threshold: Float = 0.5,
        includeConfidence: Bool = false,
        includeSpans: Bool = false
    ) -> [[String: Any]] {
        texts.map {
            extractLong(text: $0, schema: schema, chunkSize: chunkSize, overlap: overlap,
                        threshold: threshold, includeConfidence: includeConfidence,
                        includeSpans: includeSpans)
        }
    }

    /// Convenience: long-document entity extraction.
    public func extractEntitiesLong(
        text: String,
        entityTypes: [String],
        chunkSize: Int = 384,
        overlap: Int = 64,
        threshold: Float = 0.5,
        includeConfidence: Bool = false,
        includeSpans: Bool = false
    ) -> [String: Any] {
        extractLong(text: text, schema: createSchema().entities(entityTypes),
                    chunkSize: chunkSize, overlap: overlap, threshold: threshold,
                    includeConfidence: includeConfidence, includeSpans: includeSpans)
    }

    // MARK: - Span remapping

    /// Add `offset` to every `start`/`end` in the result so chunk-local spans become
    /// document-global. Walks entities, structures and relations uniformly.
    static func shiftSpans(_ value: Any, by offset: Int) -> [String: Any] {
        (shiftAny(value, by: offset) as? [String: Any]) ?? [:]
    }

    private static func shiftAny(_ value: Any, by offset: Int) -> Any {
        switch value {
        case var dict as [String: Any]:
            for (key, inner) in dict { dict[key] = shiftAny(inner, by: offset) }
            if let start = dict["start"] as? Int { dict["start"] = start + offset }
            if let end = dict["end"] as? Int { dict["end"] = end + offset }
            return dict
        case let array as [Any]:
            return array.map { shiftAny($0, by: offset) }
        case let pair as (String, String):   // relation tuple form has no spans; pass through
            return pair
        default:
            return value
        }
    }

    // MARK: - Merge

    /// Combine per-chunk results (Python engine.py:938-1108): entity/structure lists are
    /// concatenated then de-duplicated, relations exact-deduped, classifications resolved
    /// to the highest-confidence label across chunks.
    static func mergeChunkResults(_ chunks: [[String: Any]]) -> [String: Any] {
        guard let first = chunks.first else { return [:] }
        if chunks.count == 1 { return first }

        var out: [String: Any] = [:]
        // Union of all top-level keys, in first-seen order.
        var keyOrder: [String] = []
        for chunk in chunks { for key in chunk.keys where !keyOrder.contains(key) { keyOrder.append(key) } }

        for key in keyOrder {
            let values = chunks.compactMap { $0[key] }
            if key == "relation_extraction" {
                out[key] = mergeRelationGroups(values)
            } else if values.allSatisfy({ isClassification($0) }) {
                out[key] = mergeClassification(values)
            } else if values.contains(where: { $0 is [String: Any] }) {
                // Entity dict: label -> [spans]. Merge per label with text dedup.
                out[key] = mergeEntityDicts(values)
            } else if values.contains(where: { $0 is [Any] }) {
                // Structure instances (list of dicts): concat + dedup by content.
                out[key] = dedupList(values.flatMap { ($0 as? [Any]) ?? [] })
            } else {
                out[key] = values.first
            }
        }
        return out
    }

    /// A classification value is either `String`/`[String]` or `{label,confidence}` /
    /// `[{label,confidence}]`.
    private static func isClassification(_ value: Any) -> Bool {
        if value is String { return true }
        if let dict = value as? [String: Any] { return dict["label"] != nil }
        if let arr = value as? [Any] {
            return arr.allSatisfy { $0 is String || ($0 as? [String: Any])?["label"] != nil }
        }
        return false
    }

    /// Pick the highest-confidence label across chunks (falling back to most-frequent when
    /// confidences are absent).
    private static func mergeClassification(_ values: [Any]) -> Any {
        var best: (label: String, confidence: Float)?
        var counts: [String: Int] = [:]
        var sawConfidence = false

        func consider(_ v: Any) {
            if let s = v as? String {
                counts[s, default: 0] += 1
            } else if let d = v as? [String: Any], let label = d["label"] as? String {
                counts[label, default: 0] += 1
                if let c = d["confidence"] as? Float {
                    sawConfidence = true
                    if best == nil || c > best!.confidence { best = (label, c) }
                }
            }
        }
        for value in values {
            if let arr = value as? [Any] { arr.forEach(consider) } else { consider(value) }
        }

        if sawConfidence, let best {
            return ["label": best.label, "confidence": best.confidence] as [String: Any]
        }
        // Most-frequent (ties broken by first-seen via stable max over insertion — good enough).
        return counts.max { $0.value < $1.value }?.key ?? (values.first ?? "")
    }

    /// Merge `label -> [spans]` dicts across chunks, de-duplicating spans within a label by
    /// lowercased surface text (removing the overlap-region duplicates).
    private static func mergeEntityDicts(_ values: [Any]) -> [String: Any] {
        var byLabel: [String: [Any]] = [:]
        var labelOrder: [String] = []
        for value in values {
            guard let dict = value as? [String: Any] else { continue }
            for (label, spans) in dict {
                if !labelOrder.contains(label) { labelOrder.append(label) }
                byLabel[label, default: []].append(contentsOf: (spans as? [Any]) ?? [])
            }
        }
        var out: [String: Any] = [:]
        for label in labelOrder {
            out[label] = dedupByText(byLabel[label] ?? [])
        }
        return out
    }

    private static func mergeRelationGroups(_ values: [Any]) -> [String: Any] {
        var byName: [String: [Any]] = [:]
        var order: [String] = []
        for value in values {
            guard let dict = value as? [String: Any] else { continue }
            for (name, list) in dict {
                if !order.contains(name) { order.append(name) }
                byName[name, default: []].append(contentsOf: (list as? [Any]) ?? [])
            }
        }
        var out: [String: Any] = [:]
        for name in order { out[name] = dedupList(byName[name] ?? []) }
        return out
    }

    // MARK: - Dedup helpers

    /// Drop later values whose lowercased `text` (or plain string) was already seen.
    private static func dedupByText(_ values: [Any]) -> [Any] {
        var seen = Set<String>()
        var out: [Any] = []
        for value in values {
            let text = (value as? String) ?? ((value as? [String: Any])?["text"] as? String)
            guard let text else { out.append(value); continue }
            if seen.insert(text.lowercased()).inserted { out.append(value) }
        }
        return out
    }

    /// Drop later values equal (by stable serialization) to an earlier one.
    private static func dedupList(_ values: [Any]) -> [Any] {
        var seen = Set<String>()
        var out: [Any] = []
        for value in values where seen.insert(stableKey(value)).inserted { out.append(value) }
        return out
    }

    private static func stableKey(_ value: Any) -> String {
        switch value {
        case let d as [String: Any]:
            // Exclude confidence so the same span from two chunks (different confidences)
            // dedups.
            return "{" + d.keys.filter { $0 != "confidence" }.sorted()
                .map { "\($0):\(stableKey(d[$0]!))" }.joined(separator: ",") + "}"
        case let a as [Any]: return "[" + a.map(stableKey).joined(separator: ",") + "]"
        case let s as String: return "s:" + s.lowercased()
        case let n as Int: return "n:\(n)"
        case let f as Float: return "f:\(f)"
        case let t as (String, String): return "t:\(t.0.lowercased())|\(t.1.lowercased())"
        default: return "\(value)"
        }
    }

    // MARK: - Output flags

    /// Remove `start`/`end` (when spans not requested) and `confidence` (when not requested)
    /// from the merged result, and collapse span dicts back to plain strings where the
    /// non-flag output shape is a bare string.
    static func applyOutputFlags(
        _ value: Any, includeConfidence: Bool, includeSpans: Bool
    ) -> [String: Any] {
        (stripAny(value, includeConfidence: includeConfidence, includeSpans: includeSpans)
            as? [String: Any]) ?? [:]
    }

    private static func stripAny(_ value: Any, includeConfidence: Bool, includeSpans: Bool) -> Any {
        switch value {
        case var dict as [String: Any]:
            // A span dict is {text[, start, end][, confidence]}. If neither spans nor
            // confidence are wanted, collapse it to its bare text.
            let isSpan = dict["text"] != nil
                && dict.keys.allSatisfy { ["text", "start", "end", "confidence"].contains($0) }
            if isSpan, !includeSpans, !includeConfidence, let text = dict["text"] as? String {
                return text
            }
            if !includeSpans { dict["start"] = nil; dict["end"] = nil }
            if !includeConfidence { dict["confidence"] = nil }
            for (key, inner) in dict {
                dict[key] = stripAny(inner, includeConfidence: includeConfidence, includeSpans: includeSpans)
            }
            return dict
        case let array as [Any]:
            return array.map { stripAny($0, includeConfidence: includeConfidence, includeSpans: includeSpans) }
        default:
            return value
        }
    }
}
