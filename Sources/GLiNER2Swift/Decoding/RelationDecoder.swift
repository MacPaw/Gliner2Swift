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
// RelationDecoder.swift
// Typed, capped relation pair generation over the entity candidates, and the edge
// de-duplication of the public relation output.
//
// Matches Python: gliner2/models/boundary/relations.py:TypedRelationPairGenerator
// (`generate_batched(compact=True)`) and models/boundary/engine.py:_decode_relations /
// _deduplicate_relation_edges.

import Foundation

/// A relation type with the query ids of its head and tail roles.
struct RelationTypeSpec {
    let relationType: String
    let headQuery: Int
    let tailQuery: Int
    let allowSelf: Bool = false
}

enum RelationPairGenerator {

    /// Generate the capped head × tail cross product of every relation type for one
    /// sample, from the read-back candidates.
    ///
    /// Selection order mirrors the reference's stable sorts exactly: for each side the
    /// candidates of the role query with probability ≥ `argumentThreshold` are ranked by
    /// (probability desc, start asc, end asc, candidate index asc) and the top
    /// `headsPerRelation` / `tailsPerRelation` kept; pairs are ranked by the product of
    /// the two probabilities (ties by head rank, then tail rank), identical spans dropped
    /// unless `allowSelf`, and the top `pairCap` kept.
    static func generate(
        readback: BoundaryReadback, sample b: Int, specs: [RelationTypeSpec],
        settings: BoundaryHeadSettings
    ) -> [RelationPair] {
        guard !specs.isEmpty else { return [] }
        let C = readback.pool
        let threshold = settings.relationArgumentProposalThreshold

        struct Slot { let flat: Int; let start: Int; let end: Int; let prob: Float }

        func select(query q: Int, take: Int) -> [Slot] {
            guard q >= 0, q < readback.queries else { return [] }
            var slots: [Slot] = []
            for c in 0..<C where readback.isValid(sample: b, candidate: c) {
                let prob = readback.probability(sample: b, query: q, candidate: c)
                guard prob >= threshold else { continue }
                let span = readback.span(sample: b, candidate: c)
                slots.append(Slot(flat: q * C + c, start: span.start, end: span.end, prob: prob))
            }
            slots.sort { lhs, rhs in
                if lhs.prob != rhs.prob { return lhs.prob > rhs.prob }
                if lhs.start != rhs.start { return lhs.start < rhs.start }
                if lhs.end != rhs.end { return lhs.end < rhs.end }
                return lhs.flat < rhs.flat
            }
            return Array(slots.prefix(take))
        }

        var pairs: [RelationPair] = []
        for (relationIndex, spec) in specs.enumerated() {
            let heads = select(query: spec.headQuery, take: settings.relationHeadsPerType)
            let tails = select(query: spec.tailQuery, take: settings.relationTailsPerType)
            struct Candidate { let hi: Int; let ti: Int; let score: Float }
            var candidates: [Candidate] = []
            for (hi, h) in heads.enumerated() {
                for (ti, t) in tails.enumerated() {
                    let same = h.start == t.start && h.end == t.end
                    if same && !spec.allowSelf { continue }
                    candidates.append(Candidate(hi: hi, ti: ti, score: h.prob * t.prob))
                }
            }
            candidates.sort { lhs, rhs in
                if lhs.score != rhs.score { return lhs.score > rhs.score }
                if lhs.hi != rhs.hi { return lhs.hi < rhs.hi }
                return lhs.ti < rhs.ti
            }
            for candidate in candidates.prefix(settings.relationPairCap) {
                let h = heads[candidate.hi], t = tails[candidate.ti]
                pairs.append(RelationPair(
                    relationIndex: relationIndex,
                    headStart: h.start, headEnd: h.end, tailStart: t.start, tailEnd: t.end,
                    headProbability: h.prob, tailProbability: t.prob))
            }
        }
        return pairs
    }
}

enum RelationEdgeDecoder {

    /// A scored edge in character coordinates.
    struct Edge {
        var score: Float
        var head: (text: String, start: Int, end: Int)
        var tail: (text: String, start: Int, end: Int)
    }

    /// `_deduplicate_relation_edges`: canonicalise each argument to its longest containing
    /// mention, keep the best score per exact coordinates, keep the closest edge per
    /// casefolded text pair, drop edges whose argument is a strict token subset of another
    /// edge with the same opposite argument, then sort by head start, tail start, −score.
    static func deduplicate(_ edges: [Edge]) -> [Edge] {
        guard edges.count >= 2 else { return edges }

        typealias Mention = (text: String, start: Int, end: Int)
        func canonical(_ side: (Edge) -> Mention) -> [[Int]: Mention] {
            var mentions: [[Int]: Mention] = [:]
            var order: [[Int]] = []
            for edge in edges {
                let m = side(edge)
                if mentions[[m.start, m.end]] == nil { order.append([m.start, m.end]) }
                mentions[[m.start, m.end]] = m
            }
            var out: [[Int]: Mention] = [:]
            for coords in order {
                let (start, end) = (coords[0], coords[1])
                let containing = order.compactMap { mentions[$0] }.filter { $0.start <= start && $0.end >= end }
                // max by (length, -start): longest, ties → smallest start
                var best = containing[0]
                for m in containing.dropFirst() {
                    let lm = m.end - m.start, lb = best.end - best.start
                    if lm > lb || (lm == lb && -m.start > -best.start) { best = m }
                }
                out[coords] = best
            }
            return out
        }
        let headCanonical = canonical { $0.head }
        let tailCanonical = canonical { $0.tail }

        var exact: [[Int]: Edge] = [:]
        var exactOrder: [[Int]] = []
        for edge in edges {
            let head = headCanonical[[edge.head.start, edge.head.end]] ?? edge.head
            let tail = tailCanonical[[edge.tail.start, edge.tail.end]] ?? edge.tail
            let normalized = Edge(score: edge.score, head: head, tail: tail)
            let key = [head.start, head.end, tail.start, tail.end]
            if let previous = exact[key] {
                if edge.score > previous.score { exact[key] = normalized }
            } else {
                exact[key] = normalized
                exactOrder.append(key)
            }
        }

        func semanticText(_ value: String) -> String {
            value.lowercased().split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
        }
        func rank(_ e: Edge) -> (Int, Float, Int, Int) {
            let distance = max(e.head.start - e.tail.end, e.tail.start - e.head.end, 0)
            return (distance, -e.score, e.head.start, e.tail.start)
        }
        func less(_ a: (Int, Float, Int, Int), _ b: (Int, Float, Int, Int)) -> Bool {
            if a.0 != b.0 { return a.0 < b.0 }
            if a.1 != b.1 { return a.1 < b.1 }
            if a.2 != b.2 { return a.2 < b.2 }
            return a.3 < b.3
        }
        var semantic: [String: Edge] = [:]
        var semanticOrder: [String] = []
        for key in exactOrder {
            guard let edge = exact[key] else { continue }
            let sk = semanticText(edge.head.text) + "\u{1}" + semanticText(edge.tail.text)
            if let previous = semantic[sk] {
                if less(rank(edge), rank(previous)) { semantic[sk] = edge }
            } else {
                semantic[sk] = edge
                semanticOrder.append(sk)
            }
        }

        let values = semanticOrder.compactMap { semantic[$0] }
        var kept: [Edge] = []
        for (i, edge) in values.enumerated() {
            let headTokens = Set(semanticText(edge.head.text).split(separator: " "))
            let tailTokens = Set(semanticText(edge.tail.text).split(separator: " "))
            var dominated = false
            for (j, other) in values.enumerated() where j != i {
                let otherHead = Set(semanticText(other.head.text).split(separator: " "))
                let otherTail = Set(semanticText(other.tail.text).split(separator: " "))
                if (headTokens.isStrictSubset(of: otherHead) && tailTokens == otherTail)
                    || (tailTokens.isStrictSubset(of: otherTail) && headTokens == otherHead) {
                    dominated = true
                    break
                }
            }
            if !dominated { kept.append(edge) }
        }

        return kept.sorted { a, b in
            if a.head.start != b.head.start { return a.head.start < b.head.start }
            if a.tail.start != b.tail.start { return a.tail.start < b.tail.start }
            return a.score > b.score
        }
    }

    /// Format one relation type's edges the way the shared formatter expects.
    static func format(_ edges: [Edge], includeConfidence: Bool, includeSpans: Bool) -> [Any] {
        edges.map { edge -> Any in
            if includeSpans {
                var head: [String: Any] = ["text": edge.head.text, "start": edge.head.start, "end": edge.head.end]
                var tail: [String: Any] = ["text": edge.tail.text, "start": edge.tail.start, "end": edge.tail.end]
                if includeConfidence {
                    head["confidence"] = edge.score
                    tail["confidence"] = edge.score
                }
                return ["head": head, "tail": tail]
            } else if includeConfidence {
                return ["head": ["text": edge.head.text, "confidence": edge.score],
                        "tail": ["text": edge.tail.text, "confidence": edge.score]]
            }
            return (edge.head.text, edge.tail.text)
        }
    }
}
