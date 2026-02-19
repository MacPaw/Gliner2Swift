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
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"
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