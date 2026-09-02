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
// TestModelLocator.swift
// Shared model/weights discovery for the test suite.
//
// Every weight-dependent test resolves its model through this type and SKIPS
// (never fails, never falls through to a network fetch of a bogus repo id) when
// the model is absent. See IMPLEMENTATION_PLAN.md §0.2.
//
// Environment variables, by dtype — do NOT cross the streams:
//
//   GLINER2_FP16_MODEL   canonical fp16 MLX snapshot (macpaw-research/gliner2_mlx).
//                        Used by benchmarks and prediction-parity tests.
//   GLINER2_WEIGHTS_PATH converted **fp32** weights (camelCase keys), produced by
//                        `scripts/convert_weights.py` WITHOUT --dtype. The
//                        hardcoded L1-sum / component-tensor gates assume fp32;
//                        pointing this at an fp16 model makes them fail.
//   GLINER2_MODEL_PATH   raw PyTorch weights (fastino/gliner2-base-v1 snapshot).
//   GLINER2_ADAPTER_PATH LoRA adapter directory.

import XCTest
import Foundation
import Metal

enum TestModel {

    // MARK: - Repo-relative fallbacks

    /// Repository root, derived from this file's location.
    static let repoRoot: URL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // GLiNER2SwiftTests
        .deletingLastPathComponent()   // Tests
        .deletingLastPathComponent()   // <repo>

    // MARK: - fp32 converted weights (camelCase keys)

    /// Converted **fp32** weights directory. Component/L1 parity gates require fp32.
    static let fp32WeightsPath: String = {
        if let env = ProcessInfo.processInfo.environment["GLINER2_WEIGHTS_PATH"] {
            return env
        }
        return repoRoot.appendingPathComponent("weights").path
    }()

    /// Returns the fp32 weights directory, or skips the test if it is not usable.
    static func requireFP32Weights(
        file: StaticString = #filePath, line: UInt = #line
    ) throws -> String {
        try require(directory: fp32WeightsPath,
                    describedAs: "Converted fp32 weights",
                    envVar: "GLINER2_WEIGHTS_PATH")
    }

    // MARK: - fp16 canonical snapshot

    /// Canonical fp16 MLX model (`macpaw-research/gliner2_mlx`), if present.
    ///
    /// Resolution order: `GLINER2_FP16_MODEL`, then the HuggingFace hub cache, then
    /// the location `swift-transformers` downloads into (`~/Documents/huggingface`).
    static let fp16ModelPath: String? = {
        if let env = ProcessInfo.processInfo.environment["GLINER2_FP16_MODEL"] {
            return env
        }
        let fm = FileManager.default
        let home = fm.homeDirectoryForCurrentUser

        // ~/.cache/huggingface/hub/models--macpaw-research--gliner2_mlx/snapshots/<sha>
        let hubSnapshots = home
            .appendingPathComponent(".cache/huggingface/hub")
            .appendingPathComponent("models--macpaw-research--gliner2_mlx")
            .appendingPathComponent("snapshots")
        if let entries = try? fm.contentsOfDirectory(
            at: hubSnapshots, includingPropertiesForKeys: nil
        ) {
            for snapshot in entries.sorted(by: { $0.path < $1.path })
            where fm.fileExists(atPath: snapshot.appendingPathComponent("model.safetensors").path) {
                return snapshot.path
            }
        }

        // ~/Documents/huggingface/models/macpaw-research/gliner2_mlx (HubApi default)
        let hubApiDir = home
            .appendingPathComponent("Documents/huggingface/models")
            .appendingPathComponent("macpaw-research/gliner2_mlx")
        if fm.fileExists(atPath: hubApiDir.appendingPathComponent("model.safetensors").path) {
            return hubApiDir.path
        }
        return nil
    }()

    /// Returns the canonical fp16 model directory, or skips the test.
    static func requireFP16Model(
        file: StaticString = #filePath, line: UInt = #line
    ) throws -> String {
        guard let path = fp16ModelPath else {
            throw XCTSkip("""
                Canonical fp16 model not found. Set GLINER2_FP16_MODEL to a directory \
                containing model.safetensors + config.json + tokenizer.json, or download \
                macpaw-research/gliner2_mlx into the HuggingFace cache.
                """)
        }
        return try require(directory: path,
                           describedAs: "Canonical fp16 model",
                           envVar: "GLINER2_FP16_MODEL")
    }

    // MARK: - Raw PyTorch weights

    /// Raw PyTorch weights (snake_case keys), e.g. a `fastino/gliner2-base-v1` snapshot.
    static let rawWeightsPath: String = {
        if let env = ProcessInfo.processInfo.environment["GLINER2_MODEL_PATH"] {
            return env
        }
        return repoRoot.appendingPathComponent("gliner2-base-v1").path
    }()

    /// Returns the raw-weights directory, or skips the test.
    static func requireRawWeights(
        file: StaticString = #filePath, line: UInt = #line
    ) throws -> String {
        try require(directory: rawWeightsPath,
                    describedAs: "Raw PyTorch weights",
                    envVar: "GLINER2_MODEL_PATH")
    }

    // MARK: - Tokenizer-only

    /// First available directory containing a `tokenizer.json`.
    ///
    /// Tests that only need tokenization (schema serialization, description ordering)
    /// should use this rather than hardcoding a model directory — any of the three
    /// checkpoints ships the same tokenizer.
    static let tokenizerDirectory: String? = {
        let fm = FileManager.default
        let candidates = [fp32WeightsPath, rawWeightsPath, fp16ModelPath].compactMap { $0 }
        return candidates.first {
            fm.fileExists(atPath: $0 + "/tokenizer.json")
        }
    }()

    /// Returns a directory containing `tokenizer.json`, or skips the test.
    static func requireTokenizerDirectory() throws -> URL {
        guard let path = tokenizerDirectory else {
            throw XCTSkip("""
                No tokenizer.json found. Set GLINER2_WEIGHTS_PATH, GLINER2_MODEL_PATH, or \
                GLINER2_FP16_MODEL to a model directory.
                """)
        }
        return URL(fileURLWithPath: path)
    }

    // MARK: - Shared guards

    /// Skips the test when no Metal device is available (CI, headless runners).
    static func requireGPU() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal GPU not available")
        }
    }

    /// Skips unless `path` looks like a loadable model directory.
    ///
    /// Critically this also prevents `GLiNER2.fromPretrained` from treating a
    /// nonexistent *path* as a HuggingFace *repo id* and attempting a network fetch.
    private static func require(
        directory path: String, describedAs description: String, envVar: String
    ) throws -> String {
        let fm = FileManager.default
        guard fm.fileExists(atPath: path) else {
            throw XCTSkip("\(description) not found at \(path). Set \(envVar) to override.")
        }
        let hasCombined = fm.fileExists(atPath: path + "/model.safetensors")
        let hasSharded = ((try? fm.contentsOfDirectory(atPath: path)) ?? [])
            .contains { $0.hasSuffix(".safetensors") }
        guard hasCombined || hasSharded else {
            throw XCTSkip("\(description) at \(path) contains no .safetensors. Set \(envVar) to override.")
        }
        return path
    }

    /// True when every floating-point tensor in the directory's weights is fp16.
    ///
    /// Used by fp32-only numeric gates so they skip rather than fail on an fp16 model.
    static func isFloat16Checkpoint(at path: String) -> Bool {
        let url = URL(fileURLWithPath: path).appendingPathComponent("model.safetensors")
        guard let handle = try? FileHandle(forReadingFrom: url) else { return false }
        defer { try? handle.close() }
        guard let lengthData = try? handle.read(upToCount: 8), lengthData.count == 8 else {
            return false
        }
        let headerLength = lengthData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        guard headerLength > 0, headerLength < 100_000_000,
              let headerData = try? handle.read(upToCount: Int(headerLength)),
              let header = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any]
        else { return false }

        var sawFloat = false
        for (name, value) in header where name != "__metadata__" {
            guard let entry = value as? [String: Any],
                  let dtype = entry["dtype"] as? String else { continue }
            if dtype.hasPrefix("F") || dtype.hasPrefix("BF") {
                sawFloat = true
                if dtype != "F16" { return false }
            }
        }
        return sawFloat
    }
}
