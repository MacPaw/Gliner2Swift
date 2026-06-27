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
// WeightLoader.swift
// Utilities for loading model weights and downloading from HuggingFace

import Foundation
@preconcurrency import Hub
import MLX
import MLXNN

// MARK: - Errors

public enum SafeTensorsError: Error, LocalizedError {
    case invalidFormat(String)
    case unsupportedDtype(String)
    case weightNotFound(String)
    case shapeMismatch(expected: [Int], got: [Int])

    public var errorDescription: String? {
        switch self {
        case .invalidFormat(let reason):
            return "Invalid SafeTensors format: \(reason)"
        case .unsupportedDtype(let dtype):
            return "Unsupported dtype: \(dtype)"
        case .weightNotFound(let name):
            return "Weight not found: \(name)"
        case .shapeMismatch(let expected, let got):
            return "Shape mismatch: expected \(expected), got \(got)"
        }
    }
}

// MARK: - HuggingFace Hub Loading

/// Download a model directory from HuggingFace Hub.
///
/// Downloads safetensors weights, config, and tokenizer files.
/// Falls back to local cache if the network is unavailable.
///
/// - Parameters:
///   - repoId: HuggingFace repository ID (e.g., "fastino/gliner2-base-v1")
///   - hub: HubApi instance (defaults to standard HubApi)
///   - progressHandler: Optional progress callback
/// - Returns: Local URL of the downloaded model directory
public func downloadModelDirectory(
    repoId: String,
    hub: HubApi = HubApi(),
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> URL {
    let repo = Hub.Repo(id: repoId)
    let filePatterns = [
        "*.safetensors", "config.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        // The encoder's true vocab_size lives here (top-level config omits it);
        // needed to size the word-embedding for larger-vocab backbones (mdeberta).
        "encoder_config/config.json"
    ]

    do {
        return try await hub.snapshot(
            from: repo, matching: filePatterns, progressHandler: progressHandler)
    } catch Hub.HubClientError.authorizationRequired {
        return hub.localRepoLocation(repo)
    } catch Hub.HubClientError.networkError(_) {
        return hub.localRepoLocation(repo)
    } catch Hub.HubClientError.downloadError(_) {
        return hub.localRepoLocation(repo)
    } catch is HubApi.EnvironmentError {
        return hub.localRepoLocation(repo)
    } catch {
        let nserror = error as NSError
        if nserror.domain == NSURLErrorDomain {
            let offlineCodes: Set<Int> = [
                NSURLErrorNotConnectedToInternet,  // -1009
                NSURLErrorTimedOut,                 // -1001
                NSURLErrorCannotFindHost,           // -1003
                NSURLErrorCannotConnectToHost,      // -1004
                NSURLErrorNetworkConnectionLost,    // -1005
                NSURLErrorDNSLookupFailed,          // -1006
            ]
            if offlineCodes.contains(nserror.code) {
                return hub.localRepoLocation(repo)
            }
        }
        throw error
    }
}