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
// GLiNER2+Boundary.swift
// The GLiNER2.5 (boundary architecture) decode path behind the shared public API.
//
// Matches Python: gliner2/models/boundary/engine.py:BoundaryExtractor._extract_from_batch
// followed by inference/runtime.py:format_results.

import Foundation
import MLX

extension GLiNER2 {

    /// The model as a GLiNER2.5 extractor, when the loaded checkpoint is one.
    public var boundaryModel: BoundaryExtractor? { model as? BoundaryExtractor }

    /// The model as a v1 span extractor, when the loaded checkpoint is one.
    public var spanModel: SpanExtractor? { model as? SpanExtractor }

    /// Decode one collated batch through the boundary head.
    func decodeBoundaryBatch(
        model: BoundaryExtractor,
        batch: PreprocessedBatch,
        hiddenStates: MLXArray,
        threshold: Float,
        metadata: SchemaMetadata,
        includeConfidence: Bool,
        includeSpans: Bool
    ) -> [[String: Any]] {
        guard !batch.isEmpty else { return [] }
        let settings = model.settings
        let inputs = BoundaryBatchInputs.build(from: batch, metadata: metadata, threshold: threshold)
        let hasQueries = inputs.queryLayouts.contains { !$0.isEmpty }

        var readback: BoundaryReadback? = nil
        var core: BoundaryExtractor.EncodedCore? = nil
        var output: BoundaryHead.Output? = nil
        if hasQueries {
            let encoded = model.encodeCore(hiddenStates: hiddenStates, inputs: inputs)
            core = encoded
            let scored = model.score(encoded)
            output = scored
            readback = BoundaryReadback(output: scored, settings: settings)
        }
        let decoder = BoundaryDecoder(settings: settings)

        var results: [[String: Any]] = []
        for i in 0..<batch.count {
            var sample: [String: Any] = [:]
            let layout = inputs.queryLayouts[i]
            let offset = inputs.wordOffsets[i]
            let startMap = batch.startMappings[i]
            let endMap = batch.endMappings[i]
            let text = batch.originalTexts[i]
            let textLength = startMap.count

            if let readback, let core, !layout.isEmpty {
                let grouped = readback.groupedCandidates(
                    sample: i, queryCount: layout.count, thresholds: inputs.queryThresholds[i],
                    adaptiveThreshold: settings.adaptiveThreshold)
                let prefixTokens = Array(batch.textTokens[i].prefix(offset))

                // Record-mode structures through the instance-formation head.
                var recordNames: Set<String> = []
                if settings.enableRecords {
                    let specs = RecordSpecCompiler.compile(
                        layout: layout, recordMetadata: metadata.recordMetadata,
                        fieldMetadata: metadata.fieldMetadata)
                    recordNames = Set(specs.map(\.taskName))
                    if !specs.isEmpty {
                        let records = decodeRecords(
                            model: model, core: core, readback: readback, output: output,
                            sample: i, specs: specs, metadata: metadata, threshold: threshold,
                            offset: offset, textLength: textLength, text: text,
                            startMap: startMap, endMap: endMap, prefixTokens: prefixTokens,
                            includeConfidence: includeConfidence, includeSpans: includeSpans)
                        for (name, instances) in records where !instances.isEmpty { sample[name] = instances }
                    }
                }

                // Legacy structures (one instance; choice fields are scored at their
                // classification-prefix positions through the per-query reranker).
                let structures = decoder.decodeLegacyStructures(
                    layout: layout, grouped: grouped, metadata: metadata,
                    choiceDecoder: { queryId, _, choices, dtype, configured in
                        self.decodeChoiceField(
                            model: model, core: core, sample: i, queryId: queryId,
                            choices: choices, dtype: dtype, configuredThreshold: configured,
                            defaultThreshold: threshold, prefixTokens: prefixTokens,
                            includeConfidence: includeConfidence)
                    },
                    skipNames: recordNames,
                    offset: offset, textLength: textLength, text: text,
                    startMap: startMap, endMap: endMap,
                    includeConfidence: includeConfidence, includeSpans: includeSpans)
                for (name, value) in structures { sample[name] = value }

                // Entities.
                if let entities = decoder.decodeEntities(
                    layout: layout, grouped: grouped,
                    nullProbabilities: { readback.nullProbability(sample: i, query: $0) },
                    metadata: metadata, offset: offset, textLength: textLength, text: text,
                    startMap: startMap, endMap: endMap,
                    includeConfidence: includeConfidence, includeSpans: includeSpans) {
                    sample["entities"] = entities
                }
            }

            // Relations: typed pairs over the entity candidates, scored by the sparse
            // relation head. Python lists every requested relation under
            // `relation_extraction`, empty when nothing matched.
            if !metadata.relationOrder.isEmpty {
                var grouped: [String: Any] = [:]
                for name in metadata.relationOrder { grouped[name] = [] as [Any] }
                if settings.enableRelations, let readback, let core, !layout.isEmpty {
                    let decoded = decodeRelations(
                        model: model, core: core, readback: readback, sample: i, layout: layout,
                        metadata: metadata, threshold: threshold, offset: offset,
                        textLength: textLength, text: text, startMap: startMap, endMap: endMap,
                        includeConfidence: includeConfidence, includeSpans: includeSpans)
                    for (name, edges) in decoded { grouped[name] = edges }
                }
                sample["relation_extraction"] = grouped
            }

            // Classification groups: the shared classifier on the [L] marker states.
            let sampleHidden = hiddenStates[i]
            let taskTypes = batch.taskTypes[i]
            let schemaTokensList = batch.schemaTokensList[i]
            let markerPositions = batch.schemaMarkerPositions[i]
            for (g, schemaTokens) in schemaTokensList.enumerated()
            where g < taskTypes.count && taskTypes[g] == "classifications" {
                guard schemaTokens.count >= 4, g < markerPositions.count,
                      !markerPositions[g].isEmpty else { continue }
                let embs = MLX.take(sampleHidden,
                                    MLXArray(markerPositions[g].map { Int32($0) }), axis: 0)
                let schemaName = schemaTokens[2].components(separatedBy: " [DESCRIPTION] ")[0]
                extractClassificationResult(
                    results: &sample, schemaName: schemaName, schema: batch.originalSchemas[i],
                    embs: embs, schemaTokens: schemaTokens, includeConfidence: includeConfidence,
                    temperature: settings.classificationTemperature)
            }

            results.append(sample)
        }
        return results
    }

    /// `_decode_relations` for one sample: pair generation on the host, scoring on the
    /// device, one readback, then thresholding and edge de-duplication.
    func decodeRelations(
        model: BoundaryExtractor, core: BoundaryExtractor.EncodedCore, readback: BoundaryReadback,
        sample i: Int, layout: [QuerySpec], metadata: SchemaMetadata, threshold: Float,
        offset: Int, textLength: Int, text: String, startMap: [Int], endMap: [Int],
        includeConfidence: Bool, includeSpans: Bool
    ) -> [String: [Any]] {
        // One spec per relation group: its first two queries are the head and tail roles.
        var groups: [Int: [QuerySpec]] = [:]
        var groupOrder: [Int] = []
        for spec in layout where spec.taskType == "relations" {
            if groups[spec.taskIndex] == nil { groupOrder.append(spec.taskIndex) }
            groups[spec.taskIndex, default: []].append(spec)
        }
        var specs: [RelationTypeSpec] = []
        for g in groupOrder {
            let roles = groups[g] ?? []
            guard roles.count >= 2 else { continue }
            // A described relation is prompted as "name: description"; map it back.
            let taskName = roles[0].taskName
            let relationType = metadata.relationOrder.first { $0 == taskName || taskName.hasPrefix($0 + ": ") } ?? taskName
            specs.append(RelationTypeSpec(relationType: relationType,
                                          headQuery: roles[0].queryId, tailQuery: roles[1].queryId))
        }
        guard !specs.isEmpty else { return [:] }

        let settings = model.settings
        let pairs = RelationPairGenerator.generate(readback: readback, sample: i, specs: specs, settings: settings)
        guard !pairs.isEmpty else { return [:] }

        // Relation query state = head-role marker state ⧺ tail-role marker state.
        let queryStates = core.queryStates[i]                                          // [Q, H]
        let headIdx = MLXArray(specs.map { Int32($0.headQuery) })
        let tailIdx = MLXArray(specs.map { Int32($0.tailQuery) })
        let relationStates = settings.directionalRelationStates
            ? MLX.concatenated([MLX.take(queryStates, headIdx, axis: 0),
                                MLX.take(queryStates, tailIdx, axis: 0)], axis: -1)      // [R, 2H]
            : (MLX.take(queryStates, headIdx, axis: 0) + MLX.take(queryStates, tailIdx, axis: 0)) / 2
        let words = core.textStates[i][0..<max(textLength + offset, 1)]                // [L, H] unpadded
        let logits = model.relationScorer(wordStates: words, relationQueryStates: relationStates, pairs: pairs)
        let probabilities = MLX.sigmoid(logits.asType(.float32) / settings.relationTemperature).asArray(Float.self)

        let scalars = Array(text.unicodeScalars)
        func surface(_ tokenStart: Int, _ tokenEnd: Int) -> (String, Int, Int)? {
            guard tokenStart >= 0, tokenStart < tokenEnd, tokenEnd <= textLength,
                  tokenStart < startMap.count, tokenEnd - 1 < endMap.count else { return nil }
            let c0 = startMap[tokenStart], c1 = endMap[tokenEnd - 1]
            guard c0 >= 0, c1 <= scalars.count, c0 < c1 else { return nil }
            let value = String(String.UnicodeScalarView(scalars[c0..<c1])).trimmingCharacters(in: .whitespacesAndNewlines)
            return value.isEmpty ? nil : (value, c0, c1)
        }

        var edges: [String: [RelationEdgeDecoder.Edge]] = [:]
        var order: [String] = []
        for (k, pair) in pairs.enumerated() {
            let relationType = specs[pair.relationIndex].relationType
            let relationThreshold = metadata.relationMetadata[relationType]?.threshold ?? threshold
            let score = probabilities[k]
            guard score >= relationThreshold else { continue }
            guard let head = surface(pair.headStart - offset, pair.headEnd - offset),
                  let tail = surface(pair.tailStart - offset, pair.tailEnd - offset) else { continue }
            if edges[relationType] == nil { order.append(relationType) }
            edges[relationType, default: []].append(RelationEdgeDecoder.Edge(
                score: score, head: (head.0, head.1, head.2), tail: (tail.0, tail.1, tail.2)))
        }

        var out: [String: [Any]] = [:]
        for relationType in order {
            let deduped = RelationEdgeDecoder.deduplicate(edges[relationType] ?? [])
            out[relationType] = RelationEdgeDecoder.format(
                deduped, includeConfidence: includeConfidence, includeSpans: includeSpans)
        }
        return out
    }

    /// `_decode_records` for one sample: instance formation through the record head,
    /// `decode_group`, then per-field formatting with the candidate-probability gate,
    /// validators, overlap resolution and choice handling.
    func decodeRecords(
        model: BoundaryExtractor, core: BoundaryExtractor.EncodedCore, readback: BoundaryReadback,
        output: BoundaryHead.Output?, sample i: Int, specs: [RecordSpec], metadata: SchemaMetadata,
        threshold: Float, offset: Int, textLength: Int, text: String,
        startMap: [Int], endMap: [Int], prefixTokens: [String],
        includeConfidence: Bool, includeSpans: Bool
    ) -> [String: [[String: Any]]] {
        guard let output else { return [:] }
        let settings = model.settings
        let recordThreshold = threshold
        let policy = settings.canonicalOverlapPolicy
        let queryStates = core.queryStates[i]                     // [Q, H]
        let candidateStates = output.candidateStates[i]           // [P, H]
        let scalars = Array(text.unicodeScalars)

        func charSpan(_ tokenStart: Int, _ tokenEnd: Int) -> (Int, Int)? {
            guard tokenStart >= 0, tokenStart < tokenEnd, tokenEnd <= textLength,
                  tokenStart < startMap.count, tokenEnd - 1 < endMap.count else { return nil }
            return (startMap[tokenStart], endMap[tokenEnd - 1])
        }
        func surface(_ cs: Int, _ ce: Int) -> String {
            guard cs >= 0, ce <= scalars.count, cs < ce else { return "" }
            return String(String.UnicodeScalarView(scalars[cs..<ce])).trimmingCharacters(in: .whitespacesAndNewlines)
        }

        typealias Formatted = (text: String, probability: Float, start: Int, end: Int)

        func formatStructureField(_ spans: [Formatted], isScalar: Bool) -> Any? {
            if !isScalar {
                return spans.map { span -> Any in
                    BoundaryDecoder.formatItem(
                        BoundaryDecoder.Item(text: span.text, confidence: span.probability, start: span.start, end: span.end),
                        includeConfidence: includeConfidence, includeSpans: includeSpans)
                }
            }
            guard let first = spans.first else { return nil }
            return BoundaryDecoder.formatItem(
                BoundaryDecoder.Item(text: first.text, confidence: first.probability, start: first.start, end: first.end),
                includeConfidence: includeConfidence, includeSpans: includeSpans)
        }

        /// `_format_field`.
        func formatField(
            spans: [(start: Int, end: Int)], fspec: RecordFieldSpec, structureName: String,
            assignmentScores: [Float]?, recordIndex: Int, recordAnchors: [(start: Int, end: Int)?]
        ) -> Any? {
            let fieldMeta = metadata.fieldMetadata["\(structureName).\(fspec.name)"]
            let isScalar = fspec.cardinality.isScalar || fieldMeta?.dtype == "str"
            let choices = fieldMeta?.choices
            var formatted: [Formatted] = []
            for (k, span) in spans.enumerated() {
                let ts = span.start - offset, te = span.end - offset
                guard ts >= 0, te <= textLength, te > ts, let (cs, ce) = charSpan(ts, te) else { continue }
                let value = surface(cs, ce)
                let candidateProbability = readback.spanProbability(sample: i, query: fspec.queryId,
                                                                    start: span.start, end: span.end)
                let assignment = (assignmentScores != nil && k < assignmentScores!.count) ? assignmentScores![k] : nil
                let probability = assignment.map { min(candidateProbability, $0) } ?? candidateProbability
                guard !value.isEmpty else { continue }
                guard !fspec.cardinality.allowsAbsent || candidateProbability >= recordThreshold else { continue }
                if let configured = fieldMeta?.threshold, probability < configured { continue }
                if let validators = fieldMeta?.validators, !validators.isEmpty,
                   !validators.allSatisfy({ $0.validate(value) }) { continue }
                formatted.append((value, probability, cs, ce))
            }
            // finalize_spans: resolve overlaps in character coordinates; scalar keeps the first.
            let resolved = OverlapResolver.resolve(
                formatted.map { ScoredSpan(score: $0.probability, start: $0.start, end: $0.end) }, policy: policy)
            var finalized: [Formatted] = []
            for span in resolved {
                if let match = formatted.first(where: { $0.start == span.start && $0.end == span.end && $0.probability == span.score }) {
                    finalized.append(match)
                }
            }
            if isScalar { finalized = Array(finalized.prefix(1)) }

            if let choices, !choices.isEmpty {
                let anchorCharSpans: [(Int, Int)?] = recordAnchors.map { anchor in
                    guard let anchor else { return nil }
                    return charSpan(anchor.start - offset, anchor.end - offset)
                }
                if recordIndex < anchorCharSpans.count, let anchorSpan = anchorCharSpans[recordIndex] {
                    let (hasLiteral, local) = Self.recordLocalChoiceMentions(
                        text: text, scalars: scalars, choices: choices, anchorCharSpans: anchorCharSpans)
                    var preferred = local[recordIndex] ?? []
                    if isScalar, !preferred.isEmpty {
                        func distance(_ m: (choice: String, start: Int, end: Int)) -> Int {
                            max(anchorSpan.0 - m.end, m.start - anchorSpan.1, 0)
                        }
                        var best = preferred[0]
                        for m in preferred.dropFirst() where distance(m) < distance(best) { best = m }
                        preferred = [best]
                    }
                    if !preferred.isEmpty {
                        return decodeChoiceField(
                            model: model, core: core, sample: i, queryId: fspec.queryId, choices: choices,
                            dtype: isScalar ? "str" : "list", configuredThreshold: fieldMeta?.threshold,
                            defaultThreshold: recordThreshold, prefixTokens: prefixTokens,
                            includeConfidence: includeConfidence, preferredChoices: preferred,
                            includeSpans: includeSpans)
                    }
                    if hasLiteral { return isScalar ? nil : [] as [Any] }
                }
                var lookup: [String: String] = [:]
                for choice in choices { lookup[choice.lowercased()] = choice }
                let matched: [Formatted] = finalized.compactMap { span in
                    guard let canonical = lookup[span.text.lowercased()] else { return nil }
                    return (canonical, span.probability, span.start, span.end)
                }
                if !matched.isEmpty { return formatStructureField(matched, isScalar: isScalar) }
                return decodeChoiceField(
                    model: model, core: core, sample: i, queryId: fspec.queryId, choices: choices,
                    dtype: isScalar ? "str" : "list", configuredThreshold: fieldMeta?.threshold,
                    defaultThreshold: recordThreshold, prefixTokens: prefixTokens,
                    includeConfidence: includeConfidence, includeSpans: includeSpans)
            }
            return formatStructureField(finalized, isScalar: isScalar)
        }

        var out: [String: [[String: Any]]] = [:]
        for spec in specs {
            let group = model.recordDecoder.forwardGroup(
                spec: spec, queryStates: queryStates, candidateStates: candidateStates,
                readback: readback, sample: i)
            let decoded = RecordGroupDecoder.decode(
                group, anchorThreshold: recordThreshold, fieldThreshold: recordThreshold,
                objectThreshold: recordThreshold, temperature: settings.recordTemperature)
            let anchors = decoded.map(\.anchorSpan)
            var instances: [[String: Any]] = []
            for (recordIndex, rec) in decoded.enumerated() {
                var instance: [String: Any] = [:]
                var hasContent = false
                for fspec in spec.fields {
                    let value = formatField(
                        spans: rec.fields[fspec.queryId] ?? [], fspec: fspec, structureName: spec.taskName,
                        assignmentScores: rec.fieldScores[fspec.queryId], recordIndex: recordIndex,
                        recordAnchors: anchors)
                    if let value {
                        if let list = value as? [Any] {
                            instance[fspec.name] = BoundaryDecoder.dedupeFormatted(list)
                            if !list.isEmpty { hasContent = true }
                        } else {
                            instance[fspec.name] = value
                            hasContent = true
                        }
                    } else {
                        instance[fspec.name] = NSNull()
                    }
                }
                if hasContent { instances.append(instance) }
            }
            out[spec.taskName] = instances
        }
        return out
    }

    /// `_record_local_choice_mentions`: literal enum mentions in the text, each bound to
    /// the nearest preceding record anchor (mentions before the first anchor bind to it).
    /// Offsets are unicode-scalar positions.
    static func recordLocalChoiceMentions(
        text: String, scalars: [Unicode.Scalar], choices: [String], anchorCharSpans: [(Int, Int)?]
    ) -> (hasLiteral: Bool, byOwner: [Int: [(choice: String, start: Int, end: Int)]]) {
        var mentions: [(choice: String, start: Int, end: Int)] = []
        let nsText = text as NSString
        // NSString (UTF-16) offset → unicode-scalar offset
        var utf16ToScalar: [Int] = []
        utf16ToScalar.reserveCapacity(nsText.length + 1)
        var scalarIndex = 0
        for scalar in scalars {
            for _ in 0..<scalar.utf16.count { utf16ToScalar.append(scalarIndex) }
            scalarIndex += 1
        }
        utf16ToScalar.append(scalarIndex)
        for choice in choices {
            let pattern = "(?<!\\w)" + NSRegularExpression.escapedPattern(for: choice) + "(?!\\w)"
            guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { continue }
            for match in regex.matches(in: text, range: NSRange(location: 0, length: nsText.length)) {
                let start = utf16ToScalar[match.range.location]
                let end = utf16ToScalar[match.range.location + match.range.length]
                mentions.append((choice, start, end))
            }
        }
        guard !mentions.isEmpty else { return (false, [:]) }
        let validAnchors = anchorCharSpans.enumerated().compactMap { index, anchor -> (index: Int, start: Int, end: Int)? in
            guard let anchor else { return nil }
            return (index, anchor.0, anchor.1)
        }.sorted { a, b in a.start != b.start ? a.start < b.start : a.index < b.index }
        guard !validAnchors.isEmpty else { return (true, [:]) }

        var assigned: [Int: [(choice: String, start: Int, end: Int)]] = [:]
        for mention in mentions.sorted(by: { $0.start < $1.start }) {
            let preceding = validAnchors.filter { $0.start <= mention.start }
            let owner: Int
            if let best = preceding.max(by: { a, b in a.start != b.start ? a.start < b.start : a.index < b.index }) {
                owner = best.index
            } else {
                owner = validAnchors[0].index
            }
            assigned[owner, default: []].append(mention)
        }
        for (owner, owned) in assigned {
            var unique: [String: (choice: String, start: Int, end: Int)] = [:]
            var order: [String] = []
            for m in owned where unique[m.choice] == nil { unique[m.choice] = m; order.append(m.choice) }
            assigned[owner] = order.compactMap { unique[$0] }.sorted { $0.start < $1.start }
        }
        return (true, assigned)
    }

    /// `_decode_choice_field`: score every enum value of a choices field at its
    /// classification-prefix position `(idx, idx+1)` through `scoreExplicitSpans`.
    ///
    /// - Returns: for `dtype == "list"` every choice with probability ≥ threshold (an empty
    ///   list when none); otherwise the best choice or nil when below threshold.
    func decodeChoiceField(
        model: BoundaryExtractor, core: BoundaryExtractor.EncodedCore,
        sample i: Int, queryId: Int, choices: [String], dtype: String,
        configuredThreshold: Float?, defaultThreshold: Float,
        prefixTokens: [String], includeConfidence: Bool,
        preferredChoices: [(choice: String, start: Int, end: Int)]? = nil,
        includeSpans: Bool = false
    ) -> Any? {
        // `_find_choice_idx`: exact (case-insensitive) match against the prefix words.
        var present: [(choice: String, index: Int)] = []
        var seen: Set<String> = []
        for choice in choices where !seen.contains(choice) {
            seen.insert(choice)
            let lowered = choice.lowercased()
            if let index = prefixTokens.firstIndex(where: { $0.lowercased() == lowered }) {
                present.append((choice, index))
            }
        }
        guard !present.isEmpty else { return dtype == "list" ? [] as [Any] : nil }

        var pairs = [Int32](repeating: 0, count: present.count * 2)
        for (k, item) in present.enumerated() {
            pairs[2 * k] = Int32(item.index)
            pairs[2 * k + 1] = Int32(item.index + 1)
        }
        let indices = MLXArray(pairs).reshaped([1, 1, present.count, 2])
        let sliced = BoundaryExtractor.EncodedCore(
            textStates: core.textStates[i..<(i + 1)],
            textMask: core.textMask[i..<(i + 1)],
            queryStates: core.queryStates[i..<(i + 1), queryId..<(queryId + 1)],
            queryMask: core.queryMask[i..<(i + 1), queryId..<(queryId + 1)])
        let logits = model.scoreExplicitSpans(sliced, indices: indices)
        let probabilities = MLX.sigmoid(logits.asType(.float32) / model.settings.pairTemperature)
            .reshaped([present.count]).asArray(Float.self)

        // Record decoding hands over literal mentions bound to the record's anchor; they
        // are reported with their scored probability and no threshold (Python
        // `preferred_choices`).
        if let preferredChoices, !preferredChoices.isEmpty {
            var probabilityByChoice: [String: Float] = [:]
            for (k, item) in present.enumerated() { probabilityByChoice[item.choice] = probabilities[k] }
            let preferred = preferredChoices.filter { probabilityByChoice[$0.choice] != nil }
            func formatPreferred(_ item: (choice: String, start: Int, end: Int)) -> Any {
                let probability = probabilityByChoice[item.choice] ?? 0
                if includeConfidence || includeSpans {
                    var value: [String: Any] = ["text": item.choice]
                    if includeConfidence { value["confidence"] = probability }
                    if includeSpans { value["start"] = item.start; value["end"] = item.end }
                    return value
                }
                return item.choice
            }
            if dtype == "list", !preferred.isEmpty { return preferred.map(formatPreferred) }
            if let first = preferred.first { return formatPreferred(first) }
        }

        let fieldThreshold = configuredThreshold ?? defaultThreshold
        if dtype == "list" {
            var out: [Any] = []
            for (k, item) in present.enumerated() where probabilities[k] >= fieldThreshold {
                out.append(includeConfidence ? ["text": item.choice, "confidence": probabilities[k]] as [String: Any]
                                             : item.choice)
            }
            return out
        }
        var best = 0
        for k in 1..<present.count where probabilities[k] > probabilities[best] { best = k }
        guard probabilities[best] >= fieldThreshold else { return nil }
        return includeConfidence ? ["text": present[best].choice, "confidence": probabilities[best]] as [String: Any]
                                 : present[best].choice
    }
}
