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
// BoundaryExtractor.swift
// GLiNER2.5 extractor: shared encoder + classifier from `Extractor`, plus the boundary
// head.
//
// Matches Python: gliner2/models/boundary/model.py:BoundaryExtractorModel (`_encode_core`
// fast path, `token_pooling == "first"`).

import Foundation
import MLX
import MLXNN

/// GLiNER2.5 model: DeBERTa encoder, classification MLP and the boundary head.
public final class BoundaryExtractor: Extractor {
    /// The boundary decoder head.
    public let boundaryHead: BoundaryHead

    /// The sparse relation scorer (`relation_scorer.*`). Built for every checkpoint; only
    /// used when `settings.enableRelations`.
    public let relationScorer: SparseRelationScorer

    /// The record (instance formation) head (`record_decoder.*`). Only used when
    /// `settings.enableRecords` and a structure declares a record mode.
    public let recordDecoder: RecordHead

    /// The checkpoint's `boundary_head` settings.
    public let settings: BoundaryHeadSettings

    public override init(config: ExtractorConfig) {
        let settings = config.boundaryHead ?? BoundaryHeadSettings()
        self.settings = settings
        self.boundaryHead = BoundaryHead(hiddenSize: config.hiddenSize, settings: settings)
        let relationQueryDim = settings.directionalRelationStates ? 2 * config.hiddenSize : config.hiddenSize
        self.relationScorer = SparseRelationScorer(hiddenSize: config.hiddenSize, relationQueryDim: relationQueryDim)
        self.recordDecoder = RecordHead(hiddenSize: config.hiddenSize, recordDim: settings.recordDim,
                                        instanceQueries: settings.recordInstanceQueries)
        super.init(config: config)
    }

    /// The word and query states the head consumes (`_encode_core`).
    public struct EncodedCore {
        /// `[B, L, H]` first-sub-token state of every word, zero where masked.
        public let textStates: MLXArray
        /// `[B, L]` bool.
        public let textMask: MLXArray
        /// `[B, Q, H]` state of every `[E]`/`[C]`/`[R]` marker, zero where masked.
        public let queryStates: MLXArray
        /// `[B, Q]` bool.
        public let queryMask: MLXArray
    }

    /// Gather word and query states out of the encoder output.
    ///
    /// Pure gathers, no projections: each query embedding is the contextual state of its
    /// own marker token, each word state that of its first sub-token.
    ///
    /// The states are promoted to fp32: the head runs at full precision whatever the
    /// encoder's dtype (see `loadModelWeights`). It is ~5 % of the parameters and its
    /// score sums are what decide near-threshold spans, so the fp16 saving is not worth
    /// the parity cost there.
    public func encodeCore(hiddenStates: MLXArray, inputs: BoundaryBatchInputs) -> EncodedCore {
        let hiddenStates = hiddenStates.asType(.float32)
        let dtype = hiddenStates.dtype
        let textStates = BoundaryOps.gatherRows(hiddenStates, inputs.textWordIndices)
            * inputs.textWordMask.asType(dtype).expandedDimensions(axis: -1)
        let queryStates = BoundaryOps.gatherRows(hiddenStates, inputs.queryMarkerIndices)
            * inputs.queryMask.asType(dtype).expandedDimensions(axis: -1)
        return EncodedCore(
            textStates: textStates, textMask: inputs.textWordMask,
            queryStates: queryStates, queryMask: inputs.queryMask)
    }

    /// Run the boundary head on an encoded core.
    public func score(_ core: EncodedCore) -> BoundaryHead.Output {
        boundaryHead(tokenStates: core.textStates, textMask: core.textMask,
                     queryStates: core.queryStates, queryMask: core.queryMask)
    }

    /// Score explicit `[B, Q, C, 2]` spans on an encoded core (`score_explicit_spans`).
    public func scoreExplicitSpans(_ core: EncodedCore, indices: MLXArray, validMask: MLXArray? = nil) -> MLXArray {
        boundaryHead.scoreExplicitSpans(
            tokenStates: core.textStates, textMask: core.textMask,
            queryStates: core.queryStates, queryMask: core.queryMask,
            indices: indices, validMask: validMask)
    }

    // MARK: - Weights

    public override func loadModelWeights(_ weights: [String: MLXArray]) {
        super.loadModelWeights(weights)

        // The head's Swift property names are the converted key names, so the whole
        // `boundaryHead.*` namespace maps onto parameter paths directly. Goes through
        // `update(parameters:)`, which swaps the arrays in place (reflection-safe).
        // Head parameters are kept in fp32 whatever the checkpoint stores (fp16 directories
        // included): the head is small and the encoder is where the dtype policy pays.
        func promote(_ value: MLXArray) -> MLXArray {
            value.dtype.isFloatingPoint ? value.asType(.float32) : value
        }
        let prefix = "boundaryHead."
        var params: [String: MLXArray] = [:]
        for (key, value) in weights where key.hasPrefix(prefix) {
            params[String(key.dropFirst(prefix.count))] = promote(value)
        }
        guard !params.isEmpty else {
            print("⚠️ WARNING: no boundaryHead.* weights found, keeping random initialization!")
            return
        }
        boundaryHead.update(parameters: ModuleParameters.unflattened(params))

        let relationPrefix = "relationScorer."
        var relationParams: [String: MLXArray] = [:]
        for (key, value) in weights where key.hasPrefix(relationPrefix) {
            relationParams[String(key.dropFirst(relationPrefix.count))] = promote(value)
        }
        if !relationParams.isEmpty {
            relationScorer.update(parameters: ModuleParameters.unflattened(relationParams))
        } else if settings.enableRelations {
            print("⚠️ WARNING: no relationScorer.* weights found; relation extraction would use random weights")
        }

        let recordPrefix = "recordDecoder."
        var recordParams: [String: MLXArray] = [:]
        for (key, value) in weights where key.hasPrefix(recordPrefix) {
            recordParams[String(key.dropFirst(recordPrefix.count))] = promote(value)
        }
        if !recordParams.isEmpty {
            recordDecoder.update(parameters: ModuleParameters.unflattened(recordParams))
        } else if settings.enableRecords {
            print("⚠️ WARNING: no recordDecoder.* weights found; record-mode structures would use random weights")
        }
    }

    override func requiredWeightKeys() -> (exact: [String], families: [String]) {
        var required = super.requiredWeightKeys()
        required.exact += [
            "boundaryHead.boundaryEncoder.bosState",
            "boundaryHead.boundaryEncoder.eosState",
            "boundaryHead.nullProjection.weight",
        ]
        required.families += [
            "boundaryHead.boundaryEncoder.attentionBlocks.",
            "boundaryHead.boundaryQueryHead.",
            "boundaryHead.sharedPoolBuilder.",
            "boundaryHead.sharedPoolScorer.",
        ]
        if settings.enableRelations {
            required.families.append("relationScorer.mlp.")
        }
        if settings.enableRecords {
            required.families.append("recordDecoder.")
        }
        return required
    }
}
