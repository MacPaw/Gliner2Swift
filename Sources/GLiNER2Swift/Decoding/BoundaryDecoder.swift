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
// BoundaryDecoder.swift
// Host-side decoding of the boundary head's candidates into extraction results.
//
// Matches Python: gliner2/models/boundary/model.py:_group_scored_candidates,
// models/boundary/engine.py:_decode_entities / _decode_legacy_structures /
// _format_structure_field, inference/runtime.py:_format_spans / _format_entity_dict /
// _format_struct, inference/candidate_decoder.py:token_boundaries_to_character_offsets.

import Foundation
import MLX

/// One batch's head outputs, read back to the CPU in a single pass.
public struct BoundaryReadback {
    public let batch: Int
    public let queries: Int
    public let pool: Int
    /// `[B * Q * P]` sigmoid(pair logits / pair temperature).
    public let probabilities: [Float]
    /// `[B * Q * P]` raw pair logits (the record head seeds instances from them).
    public let pairLogits: [Float]
    /// `[B * P * 2]` (start, end) boundary indices.
    public let indices: [Int32]
    /// `[B * P]`.
    public let valid: [Bool]
    /// `[B * Q]` sigmoid(null logits), or empty when abstention is disabled.
    public let nullProbabilities: [Float]
    /// `[B * Q]` count log-rates, or empty when the count head is disabled.
    public let countLogRates: [Float]

    /// Evaluate and copy everything the decoder needs with one synchronisation.
    public init(output: BoundaryHead.Output, settings: BoundaryHeadSettings) {
        let logits = MLX.contiguous(output.pairLogits.asType(.float32))
        let probs = MLX.sigmoid(logits / settings.pairTemperature)
        let indices = MLX.contiguous(output.indices.asType(.int32))
        let valid = MLX.contiguous(output.validMask)
        var pending: [MLXArray] = [logits, probs, indices, valid]
        let nulls = settings.enableAbstention ? MLX.sigmoid(output.nullLogits.asType(.float32)) : nil
        let counts = settings.adaptiveThreshold ? output.countLogRates.asType(.float32) : nil
        if let nulls { pending.append(nulls) }
        if let counts { pending.append(counts) }
        MLX.eval(pending)

        self.batch = output.pairLogits.dim(0)
        self.queries = output.pairLogits.dim(1)
        self.pool = output.pairLogits.dim(2)
        self.probabilities = probs.asArray(Float.self)
        self.pairLogits = logits.asArray(Float.self)
        self.indices = indices.asArray(Int32.self)
        self.valid = valid.asArray(Bool.self)
        self.nullProbabilities = nulls?.asArray(Float.self) ?? []
        self.countLogRates = counts?.asArray(Float.self) ?? []
    }

    public func probability(sample b: Int, query q: Int, candidate c: Int) -> Float {
        probabilities[(b * queries + q) * pool + c]
    }

    public func pairLogit(sample b: Int, query q: Int, candidate c: Int) -> Float {
        pairLogits[(b * queries + q) * pool + c]
    }

    /// `_candidate_span_probability`: the calibrated score of an exact candidate span
    /// for a query, 0 when the span is not in the pool.
    public func spanProbability(sample b: Int, query q: Int, start: Int, end: Int) -> Float {
        guard q >= 0, q < queries else { return 0 }
        var best: Float = 0
        var found = false
        for c in 0..<pool where isValid(sample: b, candidate: c) {
            let s = span(sample: b, candidate: c)
            if s.start == start && s.end == end {
                best = max(best, probability(sample: b, query: q, candidate: c))
                found = true
            }
        }
        return found ? best : 0
    }

    public func span(sample b: Int, candidate c: Int) -> (start: Int, end: Int) {
        let base = (b * pool + c) * 2
        return (Int(indices[base]), Int(indices[base + 1]))
    }

    public func isValid(sample b: Int, candidate c: Int) -> Bool {
        valid[b * pool + c]
    }

    public func nullProbability(sample b: Int, query q: Int) -> Float? {
        nullProbabilities.isEmpty ? nil : nullProbabilities[b * queries + q]
    }

    /// `_group_scored_candidates` for one sample: the candidates of every query that pass
    /// its threshold, in pool order. With `adaptiveThreshold` the top `round(exp(rate))`
    /// by probability are added (never removed).
    public func groupedCandidates(
        sample b: Int, queryCount: Int, thresholds: [Float], adaptiveThreshold: Bool
    ) -> [[ScoredSpan]] {
        var out: [[ScoredSpan]] = []
        out.reserveCapacity(queryCount)
        for q in 0..<queryCount {
            let threshold = q < thresholds.count ? thresholds[q] : 0.5
            var keep = [Bool](repeating: false, count: pool)
            for c in 0..<pool where isValid(sample: b, candidate: c) {
                keep[c] = probability(sample: b, query: q, candidate: c) >= threshold
            }
            if adaptiveThreshold, !countLogRates.isEmpty {
                let rate = countLogRates[b * queries + q]
                let predicted = min(max(Int(exp(rate).rounded()), 0), pool)
                // rank eligible candidates by probability desc (stable)
                let eligible = (0..<pool).filter { isValid(sample: b, candidate: $0) }
                let ranked = eligible.enumerated().sorted { lhs, rhs in
                    let pl = probability(sample: b, query: q, candidate: lhs.element)
                    let pr = probability(sample: b, query: q, candidate: rhs.element)
                    return pl != pr ? pl > pr : lhs.offset < rhs.offset
                }
                for (rank, entry) in ranked.enumerated() where rank < predicted {
                    keep[entry.element] = true
                }
            }
            var spans: [ScoredSpan] = []
            for c in 0..<pool where keep[c] {
                let (s, e) = span(sample: b, candidate: c)
                spans.append(ScoredSpan(score: probability(sample: b, query: q, candidate: c), start: s, end: e))
            }
            out.append(spans)
        }
        return out
    }
}

/// Turns grouped candidates into the per-sample result dictionary the shared formatter
/// consumes (the same shape the v1 span path produces).
struct BoundaryDecoder {
    let settings: BoundaryHeadSettings
    let policy: OverlapPolicy

    init(settings: BoundaryHeadSettings) {
        self.settings = settings
        self.policy = settings.canonicalOverlapPolicy
    }

    /// A decoded span in character coordinates.
    struct Item {
        let text: String
        let confidence: Float
        let start: Int
        let end: Int
    }

    /// Resolve overlaps, map to characters, strip, validate.
    func materialize(
        _ scored: [ScoredSpan], offset: Int, textLength: Int, text: String,
        startMap: [Int], endMap: [Int], validators: [RegexValidator]?
    ) -> [Item] {
        var items: [Item] = []
        let unicodeScalars = Array(text.unicodeScalars)
        for span in OverlapResolver.resolve(scored, policy: policy) {
            let tokenStart = span.start - offset
            let tokenEnd = span.end - offset
            guard tokenStart >= 0, tokenStart < tokenEnd, tokenEnd <= textLength,
                  tokenStart < startMap.count, tokenEnd - 1 < endMap.count else { continue }
            let charStart = startMap[tokenStart]
            let charEnd = endMap[tokenEnd - 1]
            guard charStart >= 0, charEnd <= unicodeScalars.count, charStart < charEnd else { continue }
            let raw = String(String.UnicodeScalarView(unicodeScalars[charStart..<charEnd]))
            let surface = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !surface.isEmpty else { continue }
            if let validators, !validators.isEmpty,
               !validators.allSatisfy({ $0.validate(surface) }) { continue }
            items.append(Item(text: surface, confidence: span.score, start: charStart, end: charEnd))
        }
        return items
    }

    /// `_format_spans(..., already_finalized=True)`.
    static func formatItems(_ items: [Item], includeConfidence: Bool, includeSpans: Bool) -> [Any] {
        items.map { formatItem($0, includeConfidence: includeConfidence, includeSpans: includeSpans) }
    }

    static func formatItem(_ item: Item, includeConfidence: Bool, includeSpans: Bool) -> Any {
        if includeSpans && includeConfidence {
            return ["text": item.text, "confidence": item.confidence, "start": item.start, "end": item.end]
        } else if includeSpans {
            return ["text": item.text, "start": item.start, "end": item.end]
        } else if includeConfidence {
            return ["text": item.text, "confidence": item.confidence]
        }
        return item.text
    }

    /// `_format_entity_dict` / `_format_struct` list de-duplication: by
    /// (lowercased text, start, end) when spans are present, else by lowercased text.
    static func dedupeFormatted(_ values: [Any]) -> [Any] {
        var unique: [Any] = []
        var seen: Set<String> = []
        for value in values {
            let key: String
            if let s = value as? String {
                guard !s.isEmpty else { continue }
                key = s.lowercased()
            } else if let dict = value as? [String: Any] {
                let text = dict["text"] as? String ?? ""
                guard !text.isEmpty else { continue }
                if let start = dict["start"], let end = dict["end"] {
                    key = "\(text.lowercased())\u{1}\(start)\u{1}\(end)"
                } else {
                    key = text.lowercased()
                }
            } else {
                unique.append(value)
                continue
            }
            if seen.insert(key).inserted { unique.append(value) }
        }
        return unique
    }

    /// `_decode_entities` + `_format_entity_dict` for one sample.
    ///
    /// - Returns: the `entities` dictionary (label → formatted values), or nil when the
    ///   sample has no entity queries.
    func decodeEntities(
        layout: [QuerySpec], grouped: [[ScoredSpan]], nullProbabilities: (Int) -> Float?,
        metadata: SchemaMetadata, offset: Int, textLength: Int, text: String,
        startMap: [Int], endMap: [Int], includeConfidence: Bool, includeSpans: Bool
    ) -> [String: Any]? {
        var byName: [String: Int] = [:]
        var discovered: [String] = []
        for spec in layout where spec.taskType == "entities" {
            if byName[spec.roleName] == nil {
                byName[spec.roleName] = spec.queryId
                discovered.append(spec.roleName)
            }
        }
        guard !byName.isEmpty else { return nil }
        var names = metadata.entityOrder.filter { byName[$0] != nil }
        names += discovered.filter { !metadata.entityOrder.contains($0) }

        var results: [String: Any] = [:]
        for name in names {
            guard let queryId = byName[name] else { continue }
            let entityMetadata = metadata.entityMetadata[name]
            let dtype = entityMetadata?.dtype ?? "list"
            let abstained = (nullProbabilities(queryId) ?? 0) > settings.abstentionThreshold
            let scored = (abstained || queryId >= grouped.count) ? [] : grouped[queryId]
            let items = materialize(scored, offset: offset, textLength: textLength, text: text,
                                    startMap: startMap, endMap: endMap, validators: nil)
            if dtype == "list" {
                results[name] = Self.dedupeFormatted(
                    Self.formatItems(items, includeConfidence: includeConfidence, includeSpans: includeSpans))
            } else if items.isEmpty {
                // Python: None if spans/confidence requested else "" — and _format_entity_dict
                // turns the empty string into None as well.
                results[name] = NSNull()
            } else {
                results[name] = Self.formatItem(items[0], includeConfidence: includeConfidence,
                                                includeSpans: includeSpans)
            }
        }
        return results
    }

    /// `_decode_legacy_structures` + `_format_struct` for one sample.
    ///
    /// Boundary checkpoints have no count-slot axis, so a structure is one instance with
    /// every list field and the best scalar value per scalar field.
    func decodeLegacyStructures(
        layout: [QuerySpec], grouped: [[ScoredSpan]], metadata: SchemaMetadata,
        choiceDecoder: (_ queryId: Int, _ fieldPath: String, _ choices: [String], _ dtype: String,
                        _ threshold: Float?) -> Any?,
        skipNames: Set<String> = [],
        offset: Int, textLength: Int, text: String, startMap: [Int], endMap: [Int],
        includeConfidence: Bool, includeSpans: Bool
    ) -> [String: Any] {
        // structure name → (field name → query id), in prompt order
        var structures: [String] = []
        var fields: [String: [(name: String, queryId: Int)]] = [:]
        for spec in layout where spec.taskType == "json_structures" {
            if fields[spec.taskName] == nil {
                fields[spec.taskName] = []
                structures.append(spec.taskName)
            }
            fields[spec.taskName]!.append((spec.roleName, spec.queryId))
        }

        var results: [String: Any] = [:]
        for structureName in structures where !skipNames.contains(structureName) {
            let fieldList = fields[structureName] ?? []
            let byName = Dictionary(fieldList.map { ($0.name, $0.queryId) }, uniquingKeysWith: { a, _ in a })
            let ordered = metadata.fieldOrders[structureName] ?? fieldList.map(\.name)
            var instance: [String: Any] = [:]
            var hasContent = false
            for fieldName in ordered {
                guard let queryId = byName[fieldName] else { continue }
                let fieldPath = "\(structureName).\(fieldName)"
                let fieldMetadata = metadata.fieldMetadata[fieldPath]
                let dtype = fieldMetadata?.dtype ?? "list"
                if let choices = fieldMetadata?.choices, !choices.isEmpty {
                    let value = choiceDecoder(queryId, fieldPath, choices, dtype, fieldMetadata?.threshold)
                    instance[fieldName] = value ?? NSNull()
                    if let value, !(value is NSNull), !((value as? [Any])?.isEmpty ?? false) { hasContent = true }
                    continue
                }
                let scored = queryId < grouped.count ? grouped[queryId] : []
                let items = materialize(scored, offset: offset, textLength: textLength, text: text,
                                        startMap: startMap, endMap: endMap,
                                        validators: fieldMetadata?.validators)
                if dtype == "str" {
                    if let first = items.first {
                        instance[fieldName] = Self.formatItem(first, includeConfidence: includeConfidence,
                                                              includeSpans: includeSpans)
                        hasContent = true
                    } else {
                        instance[fieldName] = NSNull()
                    }
                } else {
                    let formatted = Self.dedupeFormatted(
                        Self.formatItems(items, includeConfidence: includeConfidence, includeSpans: includeSpans))
                    instance[fieldName] = formatted
                    if !formatted.isEmpty { hasContent = true }
                }
            }
            if hasContent {
                // `_format_struct` de-duplicates every list-valued field.
                for (field, value) in instance {
                    if let list = value as? [Any] { instance[field] = Self.dedupeFormatted(list) }
                }
                results[structureName] = [instance]
            }
        }
        return results
    }
}
