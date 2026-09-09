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
// OverlapResolver.swift
// Deterministic overlap-policy resolution over half-open spans.
//
// Matches Python: gliner2/inference/overlap.py (normalize_overlap_policy,
// resolve_overlaps). Shared by the boundary decoder and chunk merging.

import Foundation

/// Canonical overlap policies.
public enum OverlapPolicy: String, Sendable {
    /// Keep every distinct span.
    case allow
    /// Keep disjoint and containment overlaps, reject crossings.
    case nested
    /// Maximum-total-score non-overlapping set (`flat` / `disallow`).
    case disallow
    /// Drop spans strictly contained by another candidate.
    case longest

    /// Accepts every alias Python does (`flat`, `no_overlap`, `keep_longest`, …).
    public init?(rawName: String) {
        let key = rawName.trimmingCharacters(in: .whitespaces).lowercased()
            .replacingOccurrences(of: "-", with: "_")
        switch key {
        case "allow", "all", "none": self = .allow
        case "nested", "allow_nested": self = .nested
        case "flat", "disallow", "no_overlap", "non_overlapping": self = .disallow
        case "longest", "keep_longest": self = .longest
        default: return nil
        }
    }
}

/// A scored half-open span `[start, end)`.
public struct ScoredSpan: Sendable, Equatable {
    public var score: Float
    public var start: Int
    public var end: Int

    public init(score: Float, start: Int, end: Int) {
        self.score = score
        self.start = start
        self.end = end
    }
}

public enum OverlapResolver {

    /// Resolve half-open spans with the shared semantics.
    ///
    /// Exact-boundary duplicates are first collapsed to their highest-ranked
    /// representative (rank = score desc, start asc, end asc, input index asc). Results
    /// are always ranked by descending score, then ascending start / end.
    public static func resolve(_ items: [ScoredSpan], policy: OverlapPolicy) -> [ScoredSpan] {
        guard !items.isEmpty else { return [] }

        typealias Row = (index: Int, item: ScoredSpan)
        func rankKey(_ row: Row) -> (Double, Int, Int, Int) {
            (-Double(row.item.score), row.item.start, row.item.end, row.index)
        }
        func rankLess(_ a: Row, _ b: Row) -> Bool { rankKey(a) < rankKey(b) }

        let ranked = items.enumerated().map { Row(index: $0.offset, item: $0.element) }
            .sorted(by: rankLess)
        var distinct: [Row] = []
        var seen: Set<[Int]> = []
        for row in ranked {
            let boundaries = [row.item.start, row.item.end]
            if seen.contains(boundaries) { continue }
            seen.insert(boundaries)
            distinct.append(row)
        }

        switch policy {
        case .allow:
            return distinct.map(\.item)

        case .nested:
            var kept: [Row] = []
            for row in distinct {
                let c = row.item
                var crossing = false
                for existing in kept {
                    let e = existing.item
                    let overlaps = c.start < e.end && e.start < c.end
                    let contains = (c.start <= e.start && e.end <= c.end)
                        || (e.start <= c.start && c.end <= e.end)
                    if overlaps && !contains { crossing = true; break }
                }
                if !crossing { kept.append(row) }
            }
            return kept.map(\.item)

        case .longest:
            var kept: [Row] = []
            for row in distinct {
                let c = row.item
                let strictlyContained = distinct.contains { other in
                    let o = other.item
                    return o.start <= c.start && c.end <= o.end
                        && (o.start < c.start || c.end < o.end)
                }
                if !strictlyContained { kept.append(row) }
            }
            return kept.map(\.item)

        case .disallow:
            // Weighted interval scheduling. Dynamic-program ties prefer more spans, then
            // the lexicographically better (score, start, end, index) ranking.
            let byEnd = distinct.sorted { a, b in
                let ka = (a.item.end, a.item.start, -Double(a.item.score), a.index)
                let kb = (b.item.end, b.item.start, -Double(b.item.score), b.index)
                return ka < kb
            }
            let ends = byEnd.map(\.item.end)
            // predecessor[i] = index of the last span (before i) ending <= start_i, else -1
            var predecessors: [Int] = []
            for (index, row) in byEnd.enumerated() {
                // bisect_right(ends, start, 0, index) - 1
                var lo = 0, hi = index
                while lo < hi {
                    let mid = (lo + hi) / 2
                    if row.item.start < ends[mid] { hi = mid } else { lo = mid + 1 }
                }
                predecessors.append(lo - 1)
            }

            func selectionKey(_ selection: [Int]) -> [(Double, Int, Int, Int)] {
                selection.map { byEnd[$0] }.sorted(by: rankLess).map(rankKey)
            }
            func keyLess(_ a: [(Double, Int, Int, Int)], _ b: [(Double, Int, Int, Int)]) -> Bool {
                for (x, y) in zip(a, b) {
                    if x < y { return true }
                    if y < x { return false }
                }
                return a.count < b.count
            }

            var best: [(score: Double, selection: [Int])] = [(0, [])]
            for (index, row) in byEnd.enumerated() {
                let previous = best[predecessors[index] + 1]
                let with = (score: previous.score + Double(row.item.score),
                            selection: previous.selection + [index])
                let without = best[index]
                if with.score > without.score {
                    best.append(with)
                } else if with.score < without.score {
                    best.append(without)
                } else if with.selection.count > without.selection.count {
                    best.append(with)
                } else if with.selection.count < without.selection.count {
                    best.append(without)
                } else {
                    best.append(keyLess(selectionKey(with.selection), selectionKey(without.selection))
                                ? with : without)
                }
            }
            let selected = best[best.count - 1].selection.map { byEnd[$0] }.sorted(by: rankLess)
            return selected.map(\.item)
        }
    }
}
