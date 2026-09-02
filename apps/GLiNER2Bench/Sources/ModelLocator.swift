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
// ModelLocator.swift
// Finds the model directory on the device. Two supported layouts, checked in order:
//
//  1. Bundled: a "Model" folder reference added to the app target (drag the converted
//     fp16 directory into Xcode as a *folder reference* — the blue folder). Fastest, no
//     network, but adds ~400 MB to the .ipa.
//  2. Downloaded: pulled from the Hub on first launch into Application Support, so the app
//     binary stays small. Set `hubRepoId` to your converted repo.
//
// Point this at an **fp16** directory. The app runs int8 by quantizing it at load, so one
// download serves both precisions and the int8 path needs no separate on-disk model.

import Foundation
import Hub

enum ModelLocator {

    /// Set this to your converted fp16 repo to enable on-device download.
    /// e.g. "your-org/gliner2_mlx_fp16". Leave nil to use only a bundled model.
    static let hubRepoId: String? = nil

    /// Name of the bundled folder reference, if you add one.
    static let bundledFolderName = "Model"

    enum LocatorError: LocalizedError {
        case notFound
        var errorDescription: String? {
            "No model found. Bundle a 'Model' folder reference, or set ModelLocator.hubRepoId "
            + "to a converted fp16 repo."
        }
    }

    /// Returns a local directory containing model.safetensors + config.json + tokenizer.json.
    static func resolve(progress: @Sendable @escaping (Double, String) -> Void) async throws -> String {
        if let bundled = bundledModelPath() {
            progress(1.0, "Using bundled model")
            return bundled
        }
        guard let repo = hubRepoId else { throw LocatorError.notFound }

        progress(0.0, "Downloading \(repo)…")
        let hub = HubApi()
        let url = try await hub.snapshot(
            from: Hub.Repo(id: repo),
            matching: ["*.safetensors", "*.json", "spm.model"]
        ) { p in
            progress(p.fractionCompleted, "Downloading \(repo)… \(Int(p.fractionCompleted * 100))%")
        }
        return url.path
    }

    private static func bundledModelPath() -> String? {
        guard let url = Bundle.main.url(forResource: bundledFolderName, withExtension: nil),
              FileManager.default.fileExists(atPath: url.appendingPathComponent("model.safetensors").path)
        else { return nil }
        return url.path
    }
}
