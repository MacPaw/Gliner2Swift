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
// RecordDecoder.swift
// Record spec compilation and host-side decoding of record groups.
//
// Matches Python: gliner2/processing/records.py (compile_record_specs, defaults) and
// models/boundary/records.py:decode_group (+ scipy.optimize.linear_sum_assignment).

import Foundation

/// The `record_metadata` entry of one structure, as declared through the Schema DSL.
public struct RecordMetadata: Sendable, Equatable {
    public var mode: String
    public var anchor: String?
    public var occurrencePolicy: String?
    /// Per field: declared cardinality / exclusivity.
    public var fields: [String: RecordFieldConfig] = [:]
}

public struct RecordFieldConfig: Sendable, Equatable {
    public var cardinality: String?
    public var exclusive: Bool = false
}

enum RecordSpecCompiler {

    /// `compile_record_specs`: bind every record-mode structure of a sample to its
    /// queries, in prompt order. Structures without a declared mode stay on the legacy
    /// path.
    static func compile(
        layout: [QuerySpec], recordMetadata: [String: RecordMetadata],
        fieldMetadata: [String: FieldMetadata] = [:]
    ) -> [RecordSpec] {
        guard !recordMetadata.isEmpty else { return [] }
        var byTask: [Int: [QuerySpec]] = [:]
        var order: [Int] = []
        for q in layout where q.taskType == "json_structures" {
            if byTask[q.taskIndex] == nil { order.append(q.taskIndex) }
            byTask[q.taskIndex, default: []].append(q)
        }
        var specs: [RecordSpec] = []
        for taskIndex in order {
            let queries = (byTask[taskIndex] ?? []).sorted { $0.roleIndex < $1.roleIndex }
            guard let name = queries.first?.taskName, let cfg = recordMetadata[name] else { continue }
            var fields: [RecordFieldSpec] = []
            var anchorQueryId: Int? = nil
            for q in queries {
                let fcfg = cfg.fields[q.roleName]
                let isAnchor = cfg.mode == "natural" && q.roleName == cfg.anchor
                // Defaults as the reference runtime applies them: the collator compiles
                // record specs without field dtypes, so a non-anchor field is
                // `zero_or_more` even when declared `str` (the `str` only affects
                // formatting, which keeps the first resolved span). Explicit cardinalities
                // (e.g. `extract_json`'s `required_one`) win.
                let cardinality: FieldCardinality
                if let raw = fcfg?.cardinality, let parsed = FieldCardinality(rawValue: raw) {
                    cardinality = parsed
                } else if isAnchor {
                    cardinality = .requiredOne
                } else {
                    cardinality = .zeroOrMore
                }
                fields.append(RecordFieldSpec(
                    queryId: q.queryId, name: q.roleName, roleIndex: q.roleIndex,
                    cardinality: cardinality, isAnchor: isAnchor, exclusive: fcfg?.exclusive ?? false))
                if isAnchor { anchorQueryId = q.queryId }
            }
            if cfg.mode == "natural" && anchorQueryId == nil { continue }
            specs.append(RecordSpec(taskIndex: taskIndex, taskName: name, mode: cfg.mode,
                                    fields: fields, anchorQueryId: anchorQueryId))
        }
        return specs
    }
}

/// One decoded record: field query id → selected half-open token spans (+ scores).
struct DecodedRecord {
    var fields: [Int: [(start: Int, end: Int)]] = [:]
    var fieldScores: [Int: [Float]] = [:]
    var anchorSpan: (start: Int, end: Int)? = nil
    var score: Float = 0
}

enum RecordGroupDecoder {

    static func sigmoid(_ x: Float) -> Float { 1 / (1 + exp(-x)) }

    static func softmax(_ xs: [Float]) -> [Float] {
        guard let m = xs.max() else { return [] }
        let exps = xs.map { exp($0 - m) }
        let sum = exps.reduce(0, +)
        return exps.map { $0 / sum }
    }

    /// `decode_group`.
    static func decode(
        _ group: RecordGroupOutput, anchorThreshold: Float, fieldThreshold: Float,
        objectThreshold: Float, temperature: Float
    ) -> [DecodedRecord] {
        let ni = group.instanceCount
        guard ni > 0 else { return [] }
        let objProb = group.objectLogits.map { sigmoid($0 / temperature) }
        let selectThreshold = group.spec.mode == "anchorless" ? objectThreshold : anchorThreshold
        let order = (0..<ni).sorted { a, b in objProb[a] != objProb[b] ? objProb[a] > objProb[b] : a < b }
        let selected = order.filter { objProb[$0] >= selectThreshold }

        // Exclusive fields: scalar fields are a joint assignment problem; list candidates
        // go to their strongest instance.
        var scalarChoices: [[Int]: (col: Int, probability: Float)?] = [:]
        var listOwners: [[Int]: (instance: Int, probability: Float)] = [:]
        for (f, fspec) in group.spec.fields.enumerated() where fspec.exclusive && !selected.isEmpty {
            let rows = selected.map { inst in group.assignLogits[f][inst].map { $0 / temperature } }
            let candidateCount = max((rows.first?.count ?? 1) - 1, 0)
            if fspec.cardinality.isScalar {
                if candidateCount == 0 {
                    for inst in selected { scalarChoices[[inst, f]] = .some(nil) }
                    continue
                }
                let probs = rows.map(softmax)
                let eps = Float.ulpOfOne
                let candidateCost = probs.map { row in row.dropFirst().map { -log(max($0, eps)) } }
                let rowCount = selected.count
                var diagonal = probs.map { -log(max($0[0], eps)) }
                let maxCandidateCost = candidateCost.flatMap { $0 }.max() ?? 0
                if !fspec.allowsAbsent {
                    diagonal = Array(repeating: maxCandidateCost + 50, count: rowCount)
                }
                let invalidCost = max(maxCandidateCost, diagonal.max() ?? 0) + 1_000
                // cost [rows, candidates + rows]
                var cost: [[Float]] = []
                for r in 0..<rowCount {
                    var row = candidateCost[r]
                    for c in 0..<rowCount { row.append(c == r ? diagonal[r] : invalidCost) }
                    cost.append(row)
                }
                let assignment = HungarianAssignment.solve(cost)
                for (r, inst) in selected.enumerated() {
                    let col = assignment[r] ?? (candidateCount + r)
                    if col >= candidateCount {
                        scalarChoices[[inst, f]] = .some(nil)
                        continue
                    }
                    let probability = probs[r][col + 1]
                    if probability < fieldThreshold && fspec.allowsAbsent {
                        scalarChoices[[inst, f]] = .some(nil)
                        continue
                    }
                    scalarChoices[[inst, f]] = (col, probability)
                }
            } else {
                guard candidateCount > 0 else { continue }
                for c in 0..<candidateCount {
                    var bestRow = 0
                    var best = sigmoid(rows[0][c + 1])
                    for r in 1..<rows.count {
                        let p = sigmoid(rows[r][c + 1])
                        if p > best { best = p; bestRow = r }
                    }
                    if best >= fieldThreshold {
                        listOwners[[f, c]] = (selected[bestRow], best)
                    }
                }
            }
        }

        var records: [DecodedRecord] = []
        for inst in selected {
            var rec = DecodedRecord(score: objProb[inst])
            var anchorField: Int? = nil
            if group.spec.mode == "natural" {
                anchorField = group.spec.fields.firstIndex { $0.queryId == group.spec.anchorQueryId }
                rec.anchorSpan = group.instanceSpans[inst]
            }
            for (f, fspec) in group.spec.fields.enumerated() {
                let qid = fspec.queryId
                let spans = group.fieldSpans[f]
                let row = group.assignLogits[f][inst].map { $0 / temperature }
                if let anchorField, f == anchorField {
                    if let anchor = rec.anchorSpan {
                        rec.fields[qid, default: []].append(anchor)
                        rec.fieldScores[qid, default: []].append(rec.score)
                    }
                    continue
                }
                if fspec.cardinality.isScalar {
                    if fspec.exclusive {
                        guard let choice = scalarChoices[[inst, f]], let (col, probability) = choice else { continue }
                        rec.fields[qid, default: []].append(spans[col])
                        rec.fieldScores[qid, default: []].append(probability)
                        continue
                    }
                    let probs = softmax(row)
                    // argsort descending (ties: lower index first, as torch.argsort)
                    let ranked = (0..<probs.count).sorted { a, b in probs[a] != probs[b] ? probs[a] > probs[b] : a < b }
                    var chosen: Int? = nil
                    for col in ranked {
                        if col == 0 {
                            if fspec.allowsAbsent { chosen = 0; break }
                            continue
                        }
                        chosen = col
                        break
                    }
                    guard let picked = chosen, picked != 0 else { continue }
                    if probs[picked] < fieldThreshold && fspec.allowsAbsent { continue }
                    rec.fields[qid, default: []].append(spans[picked - 1])
                    rec.fieldScores[qid, default: []].append(probs[picked])
                } else {
                    let candLogits = Array(row.dropFirst())
                    guard !candLogits.isEmpty else { continue }
                    var selectedSpans: [(start: Int, end: Int)] = []
                    var selectedScores: [Float] = []
                    for c in 0..<candLogits.count {
                        let probability: Float
                        if fspec.exclusive {
                            guard let owner = listOwners[[f, c]], owner.instance == inst else { continue }
                            probability = owner.probability
                        } else {
                            probability = sigmoid(candLogits[c])
                            if probability < fieldThreshold { continue }
                        }
                        selectedSpans.append(spans[c])
                        selectedScores.append(probability)
                    }
                    if !selectedSpans.isEmpty {
                        rec.fields[qid, default: []] += selectedSpans
                        rec.fieldScores[qid, default: []] += selectedScores
                    }
                }
            }
            if !rec.fields.isEmpty { records.append(rec) }
        }

        if group.spec.mode == "latent" || group.spec.mode == "anchorless" {
            var best: [String: DecodedRecord] = [:]
            var keys: [String] = []
            for rec in records {
                let key = rec.fields.keys.sorted().map { qid in
                    "\(qid):" + rec.fields[qid]!.map { "\($0.start)-\($0.end)" }.sorted().joined(separator: ",")
                }.joined(separator: ";")
                if let existing = best[key] {
                    if rec.score > existing.score { best[key] = rec }
                } else {
                    best[key] = rec
                    keys.append(key)
                }
            }
            records = keys.compactMap { best[$0] }
        } else if group.spec.mode == "natural" {
            records.sort { a, b in
                switch (a.anchorSpan, b.anchorSpan) {
                case (nil, nil): return false
                case (nil, _): return false
                case (_, nil): return true
                case let (x?, y?): return x.start != y.start ? x.start < y.start : x.end < y.end
                }
            }
        }
        return records
    }
}

/// Minimum-cost assignment of `n` rows to `m ≥ n` columns (`scipy.optimize.linear_sum_assignment`).
enum HungarianAssignment {
    /// - Returns: column for each row.
    static func solve(_ cost: [[Float]]) -> [Int: Int] {
        let n = cost.count
        guard n > 0, let m = cost.first?.count, m >= n else { return [:] }
        // Classic O(n²m) Hungarian algorithm with potentials (rows 1...n, cols 1...m).
        var u = [Double](repeating: 0, count: n + 1)
        var v = [Double](repeating: 0, count: m + 1)
        var p = [Int](repeating: 0, count: m + 1)      // p[j] = row assigned to column j
        var way = [Int](repeating: 0, count: m + 1)
        for i in 1...n {
            p[0] = i
            var j0 = 0
            var minv = [Double](repeating: .infinity, count: m + 1)
            var used = [Bool](repeating: false, count: m + 1)
            repeat {
                used[j0] = true
                let i0 = p[j0]
                var delta = Double.infinity
                var j1 = 0
                for j in 1...m where !used[j] {
                    let cur = Double(cost[i0 - 1][j - 1]) - u[i0] - v[j]
                    if cur < minv[j] { minv[j] = cur; way[j] = j0 }
                    if minv[j] < delta { delta = minv[j]; j1 = j }
                }
                for j in 0...m {
                    if used[j] { u[p[j]] += delta; v[j] -= delta } else { minv[j] -= delta }
                }
                j0 = j1
            } while p[j0] != 0
            repeat {
                let j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
            } while j0 != 0
        }
        var result: [Int: Int] = [:]
        for j in 1...m where p[j] != 0 { result[p[j] - 1] = j - 1 }
        return result
    }
}
