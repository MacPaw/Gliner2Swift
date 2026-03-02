// LoRAConfig.swift
// LoRA adapter configuration
//
// Matches Python: gliner2/training/lora.py:LoRAAdapterConfig

import Foundation

/// Configuration for a saved LoRA adapter.
///
/// All parameters are read from `adapter_config.json` — no hardcoded values.
/// Supports any rank, alpha, dropout, and target module combination.
public struct LoRAAdapterConfig: Codable {
    public let adapterType: String
    public let adapterVersion: String
    public let loraR: Int
    public let loraAlpha: Float
    public let loraDropout: Float
    public let targetModules: [String]

    /// Computed scaling factor: alpha / r
    public var scaling: Float { loraAlpha / Float(loraR) }

    enum CodingKeys: String, CodingKey {
        case adapterType = "adapter_type"
        case adapterVersion = "adapter_version"
        case loraR = "lora_r"
        case loraAlpha = "lora_alpha"
        case loraDropout = "lora_dropout"
        case targetModules = "target_modules"
    }

    /// Detect if a directory contains a LoRA adapter
    public static func isAdapterPath(_ path: URL) -> Bool {
        FileManager.default.fileExists(
            atPath: path.appendingPathComponent("adapter_config.json").path
        )
    }

    /// Load adapter config from a directory containing adapter_config.json
    public static func load(from directory: URL) throws -> LoRAAdapterConfig {
        let configUrl = directory.appendingPathComponent("adapter_config.json")
        let data = try Data(contentsOf: configUrl)
        return try JSONDecoder().decode(LoRAAdapterConfig.self, from: data)
    }
}
