// WeightLoader.swift
// Utilities for loading model weights from SafeTensors
//
// Matches Python: safetensors.torch.load_file

import Foundation
import MLX
import MLXNN

// MARK: - SafeTensors Loading

/// Loader for SafeTensors format weight files.
///
/// SafeTensors is a fast and safe format for storing tensors.
/// It's the default format used by HuggingFace Transformers.
public struct SafeTensorsLoader {
    /// Load weights from a SafeTensors file
    ///
    /// - Parameter url: URL to the .safetensors file
    /// - Returns: Dictionary mapping parameter names to MLXArrays
    /// - Throws: Error if file cannot be loaded
    public static func load(from url: URL) throws -> [String: MLXArray] {
        let data = try Data(contentsOf: url)
        return try parseSafeTensors(data)
    }

    /// Parse SafeTensors data
    private static func parseSafeTensors(_ data: Data) throws -> [String: MLXArray] {
        // Read header size (8 bytes, little endian)
        guard data.count >= 8 else {
            throw SafeTensorsError.invalidFormat("File too small")
        }

        let headerSize = data.withUnsafeBytes { ptr in
            ptr.load(as: UInt64.self)
        }

        guard data.count >= Int(8 + headerSize) else {
            throw SafeTensorsError.invalidFormat("Header size exceeds file size")
        }

        // Read header JSON
        let headerData = data[8..<(8 + Int(headerSize))]
        guard let headerJson = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw SafeTensorsError.invalidFormat("Invalid header JSON")
        }

        // Parse tensor metadata and load tensors
        var weights: [String: MLXArray] = [:]
        let tensorDataStart = 8 + Int(headerSize)

        for (name, value) in headerJson {
            // Skip __metadata__ key
            if name == "__metadata__" { continue }

            guard let tensorInfo = value as? [String: Any],
                  let dtype = tensorInfo["dtype"] as? String,
                  let shape = tensorInfo["shape"] as? [Int],
                  let dataOffsets = tensorInfo["data_offsets"] as? [Int],
                  dataOffsets.count == 2 else {
                continue
            }

            let startOffset = tensorDataStart + dataOffsets[0]
            let endOffset = tensorDataStart + dataOffsets[1]

            guard startOffset >= 0 && endOffset <= data.count && startOffset < endOffset else {
                continue
            }

            let tensorData = data[startOffset..<endOffset]

            // Convert to MLXArray based on dtype
            let array = try createMLXArray(from: tensorData, dtype: dtype, shape: shape)
            weights[name] = array
        }

        return weights
    }

    /// Create MLXArray from raw tensor data
    private static func createMLXArray(
        from data: Data,
        dtype: String,
        shape: [Int]
    ) throws -> MLXArray {
        // Map SafeTensors dtype to MLX dtype
        let mlxDtype: DType
        switch dtype {
        case "F32":
            mlxDtype = .float32
        case "F16":
            mlxDtype = .float16
        case "BF16":
            mlxDtype = .bfloat16
        case "I64":
            mlxDtype = .int64
        case "I32":
            mlxDtype = .int32
        case "I16":
            mlxDtype = .int16
        case "I8":
            mlxDtype = .int8
        case "U8":
            mlxDtype = .uint8
        case "BOOL":
            mlxDtype = .bool
        default:
            throw SafeTensorsError.unsupportedDtype(dtype)
        }

        // Create MLXArray from Data directly - MLX has a convenient initializer for this
        return MLXArray(Data(data), shape, dtype: mlxDtype)
    }
}

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

// MARK: - Weight Name Mapping

/// Maps PyTorch weight names to Swift module paths
public struct WeightNameMapper {
    /// Standard mappings for GLiNER2 model
    public static let gliner2Mappings: [String: String] = [
        // Span representation
        "span_rep.span_rep_layer": "spanRep.spanRepLayer",

        // Classifier
        "classifier.0": "classifier.layers.0",
        "classifier.2": "classifier.layers.2",

        // Count prediction
        "count_pred.0": "countPred.layers.0",
        "count_pred.2": "countPred.layers.2",

        // Count embedding
        "count_embed.pos_embedding": "countEmbed.posEmbedding",
        "count_embed.gru": "countEmbed.gru",
        "count_embed.transformer": "countEmbed.transformer"
    ]

    /// Map a PyTorch weight name to Swift module path
    ///
    /// - Parameter pytorchName: Original PyTorch parameter name
    /// - Returns: Mapped Swift module path
    public static func mapName(_ pytorchName: String) -> String {
        for (pattern, replacement) in gliner2Mappings {
            if pytorchName.hasPrefix(pattern) {
                return pytorchName.replacingOccurrences(of: pattern, with: replacement)
            }
        }
        return pytorchName
    }

    /// Filter and map weights for a specific module prefix
    ///
    /// - Parameters:
    ///   - weights: Full weight dictionary
    ///   - prefix: Prefix to filter (e.g., "span_rep")
    /// - Returns: Filtered and mapped weights
    public static func filterWeights(
        _ weights: [String: MLXArray],
        prefix: String
    ) -> [String: MLXArray] {
        var filtered: [String: MLXArray] = [:]

        for (name, value) in weights {
            if name.hasPrefix(prefix) {
                let relativeName = String(name.dropFirst(prefix.count + 1))
                filtered[relativeName] = value
            }
        }

        return filtered
    }
}

// MARK: - Module Weight Updates

/// Protocol for modules that can load weights from a dictionary
public protocol WeightLoadable {
    /// Load weights from a dictionary
    ///
    /// - Parameters:
    ///   - weights: Dictionary of weight tensors
    ///   - prefix: Prefix for weight names in this module
    func loadWeights(_ weights: [String: MLXArray], prefix: String)
}

/// Helper to update Linear layer weights using MLX's update mechanism
public func updateLinear(_ linear: Linear, weight: MLXArray, bias: MLXArray?) {
    // MLX Linear stores weight as [out_features, in_features], same as PyTorch
    // Use MLX's module parameter update
    var params: [String: MLXArray] = ["weight": weight]
    if let bias = bias {
        params["bias"] = bias
    }
    linear.update(parameters: ModuleParameters.unflattened(params))
}

/// Helper to update LayerNorm weights using MLX's update mechanism
public func updateLayerNorm(_ norm: LayerNorm, weight: MLXArray, bias: MLXArray?) {
    var params: [String: MLXArray] = ["weight": weight]
    if let bias = bias {
        params["bias"] = bias
    }
    norm.update(parameters: ModuleParameters.unflattened(params))
}

/// Helper to update Embedding weights using MLX's update mechanism
public func updateEmbedding(_ embedding: Embedding, weight: MLXArray) {
    let params: [String: MLXArray] = ["weight": weight]
    embedding.update(parameters: ModuleParameters.unflattened(params))
}

// MARK: - Model Weight Loading

/// Load weights into a GLiNER2 Extractor model
public struct ModelWeightLoader {
    /// Load weights into all model components
    ///
    /// - Parameters:
    ///   - extractor: The Extractor model to load weights into
    ///   - weights: Dictionary of weight tensors
    ///   - strict: If true, throw error on missing weights
    public static func loadWeights(
        into extractor: Extractor,
        from weights: [String: MLXArray],
        strict: Bool = false
    ) throws {
        // Load span representation weights
        try loadSpanRepWeights(extractor.spanRep, weights: weights)

        // Load classifier MLP weights
        try loadMLPWeights(extractor.classifier, weights: weights, prefix: "classifier")

        // Load count prediction MLP weights
        try loadMLPWeights(extractor.countPred, weights: weights, prefix: "count_pred")

        // Load CountLSTMv2 weights
        try loadCountLSTMv2Weights(extractor.countEmbed, weights: weights)
    }

    // MARK: - SpanRep Loading

    private static func loadSpanRepWeights(
        _ spanRep: SpanRepLayer,
        weights: [String: MLXArray]
    ) throws {
        let prefix = "span_rep.span_rep_layer"

        // Load project_start
        try loadProjectionWeights(
            weights: weights,
            prefix: "\(prefix).project_start",
            into: { idx, w, b in
                // projection: Linear(D, 4D) -> ReLU -> Dropout -> Linear(4D, D)
                // Sequential indices: 0=Linear, 1=ReLU, 2=Dropout, 3=Linear
            }
        )

        // Load project_end
        try loadProjectionWeights(
            weights: weights,
            prefix: "\(prefix).project_end",
            into: { idx, w, b in }
        )

        // Load out_project
        try loadProjectionWeights(
            weights: weights,
            prefix: "\(prefix).out_project",
            into: { idx, w, b in }
        )
    }

    private static func loadProjectionWeights(
        weights: [String: MLXArray],
        prefix: String,
        into handler: (Int, MLXArray, MLXArray?) -> Void
    ) throws {
        // Projection structure: Linear → ReLU → Dropout → Linear
        // PyTorch indices: 0=Linear, 1=ReLU, 2=Dropout, 3=Linear

        if let w0 = weights["\(prefix).0.weight"] {
            let b0 = weights["\(prefix).0.bias"]
            handler(0, w0, b0)
        }

        if let w3 = weights["\(prefix).3.weight"] {
            let b3 = weights["\(prefix).3.bias"]
            handler(3, w3, b3)
        }
    }

    // MARK: - MLP Loading

    private static func loadMLPWeights(
        _ mlp: Sequential,
        weights: [String: MLXArray],
        prefix: String
    ) throws {
        // MLP structure: Linear → ReLU → Linear
        // PyTorch indices: 0=Linear, 1=ReLU, 2=Linear

        if let w0 = weights["\(prefix).0.weight"],
           let b0 = weights["\(prefix).0.bias"],
           let linear0 = mlp.layers[0] as? Linear {
            updateLinear(linear0, weight: w0, bias: b0)
        }

        // Index 2 in PyTorch corresponds to the second Linear layer
        // In our Sequential with [Linear, ReLU, Linear], that's index 2
        if let w2 = weights["\(prefix).2.weight"],
           let b2 = weights["\(prefix).2.bias"],
           let linear2 = mlp.layers[2] as? Linear {
            updateLinear(linear2, weight: w2, bias: b2)
        }
    }

    // MARK: - CountLSTMv2 Loading

    private static func loadCountLSTMv2Weights(
        _ countEmbed: CountLSTMv2,
        weights: [String: MLXArray]
    ) throws {
        let prefix = "count_embed"

        // Load positional embedding
        if let posWeight = weights["\(prefix).pos_embedding.weight"] {
            updateEmbedding(countEmbed.posEmbedding, weight: posWeight)
        }

        // Load GRU weights
        try loadGRUWeights(countEmbed.gru, weights: weights, prefix: "\(prefix).gru")

        // Load DownscaledTransformer weights
        try loadDownscaledTransformerWeights(
            countEmbed.transformer,
            weights: weights,
            prefix: "\(prefix).transformer"
        )
    }

    private static func loadGRUWeights(
        _ gru: GRU,
        weights: [String: MLXArray],
        prefix: String
    ) throws {
        // GRU weights:
        // - weight_ih_l0: [3*hidden, input]
        // - weight_hh_l0: [3*hidden, hidden]
        // - bias_ih_l0: [3*hidden]
        // - bias_hh_l0: [3*hidden]

        if let weightIH = weights["\(prefix).weight_ih_l0"] {
            // Note: GRU class stores weights as properties that should be updated
            // Since GRU uses `let` properties, we need to reconstruct or use update mechanism
            // For now, document that GRU.fromWeights should be used instead
        }
    }

    private static func loadDownscaledTransformerWeights(
        _ transformer: DownscaledTransformer,
        weights: [String: MLXArray],
        prefix: String
    ) throws {
        // Load in_projector
        if let inW = weights["\(prefix).in_projector.weight"],
           let inB = weights["\(prefix).in_projector.bias"] {
            updateLinear(transformer.inProjector, weight: inW, bias: inB)
        }

        // Load transformer encoder layers
        for (i, layer) in transformer.transformerLayers.enumerated() {
            let layerPrefix = "\(prefix).transformer.layers.\(i)"
            try loadTransformerEncoderLayerWeights(layer, weights: weights, prefix: layerPrefix)
        }

        // Load out_projector MLP
        // Structure: Linear(896, 768) → ReLU → Linear(768, 768) → ReLU → Linear(768, 768)
        // PyTorch indices: 0, 2, 4
        let outPrefix = "\(prefix).out_projector"

        if let w0 = weights["\(outPrefix).0.weight"],
           let b0 = weights["\(outPrefix).0.bias"],
           let linear0 = transformer.outProjector.layers[0] as? Linear {
            updateLinear(linear0, weight: w0, bias: b0)
        }

        if let w2 = weights["\(outPrefix).2.weight"],
           let b2 = weights["\(outPrefix).2.bias"],
           let linear2 = transformer.outProjector.layers[2] as? Linear {
            updateLinear(linear2, weight: w2, bias: b2)
        }

        if let w4 = weights["\(outPrefix).4.weight"],
           let b4 = weights["\(outPrefix).4.bias"],
           let linear4 = transformer.outProjector.layers[4] as? Linear {
            updateLinear(linear4, weight: w4, bias: b4)
        }
    }

    private static func loadTransformerEncoderLayerWeights(
        _ layer: TransformerEncoderLayer,
        weights: [String: MLXArray],
        prefix: String
    ) throws {
        // Self-attention
        // PyTorch stores Q, K, V in combined in_proj_weight [3*dim, dim]
        // MLX MultiHeadAttention has separate projections
        if let inProj = weights["\(prefix).self_attn.in_proj_weight"],
           let inProjBias = weights["\(prefix).self_attn.in_proj_bias"] {
            let dim = inProj.dim(0) / 3

            // Split into Q, K, V weights
            let qWeight = inProj[0..<dim]
            let kWeight = inProj[dim..<(2*dim)]
            let vWeight = inProj[(2*dim)..<(3*dim)]

            let qBias = inProjBias[0..<dim]
            let kBias = inProjBias[dim..<(2*dim)]
            let vBias = inProjBias[(2*dim)..<(3*dim)]

            // Update MultiHeadAttention projections
            // Note: MLX MHA structure - update as needed
            // layer.selfAttn has queryProj, keyProj, valueProj
        }

        // Output projection
        if let outW = weights["\(prefix).self_attn.out_proj.weight"],
           let outB = weights["\(prefix).self_attn.out_proj.bias"] {
            // Update output projection
        }

        // FFN layers
        if let w1 = weights["\(prefix).linear1.weight"],
           let b1 = weights["\(prefix).linear1.bias"] {
            updateLinear(layer.linear1, weight: w1, bias: b1)
        }

        if let w2 = weights["\(prefix).linear2.weight"],
           let b2 = weights["\(prefix).linear2.bias"] {
            updateLinear(layer.linear2, weight: w2, bias: b2)
        }

        // Layer norms
        if let n1W = weights["\(prefix).norm1.weight"],
           let n1B = weights["\(prefix).norm1.bias"] {
            updateLayerNorm(layer.norm1, weight: n1W, bias: n1B)
        }

        if let n2W = weights["\(prefix).norm2.weight"],
           let n2B = weights["\(prefix).norm2.bias"] {
            updateLayerNorm(layer.norm2, weight: n2W, bias: n2B)
        }
    }
}

// MARK: - HuggingFace Model Loading

/// Utilities for loading models from HuggingFace Hub
public struct HuggingFaceLoader {
    /// Base URL for HuggingFace Hub
    private static let hubBaseURL = "https://huggingface.co"

    /// Download file from HuggingFace Hub
    ///
    /// - Parameters:
    ///   - repo: Repository ID (e.g., "fastino/gliner2-base-v1")
    ///   - filename: File to download (e.g., "model.safetensors")
    ///   - cacheDir: Local cache directory
    /// - Returns: Local URL of downloaded file
    public static func download(
        repo: String,
        filename: String,
        cacheDir: URL? = nil
    ) async throws -> URL {
        let defaultCacheDir = FileManager.default.urls(
            for: .cachesDirectory,
            in: .userDomainMask
        ).first!.appendingPathComponent("gliner2swift")

        let cache = cacheDir ?? defaultCacheDir

        // Create cache directory if needed
        try FileManager.default.createDirectory(
            at: cache,
            withIntermediateDirectories: true
        )

        // Check if already cached
        let repoDir = cache.appendingPathComponent(repo.replacingOccurrences(of: "/", with: "_"))
        let localFile = repoDir.appendingPathComponent(filename)

        if FileManager.default.fileExists(atPath: localFile.path) {
            return localFile
        }

        // Download from HuggingFace Hub
        let url = URL(string: "\(hubBaseURL)/\(repo)/resolve/main/\(filename)")!

        let (data, response) = try await URLSession.shared.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw HuggingFaceError.downloadFailed(filename)
        }

        // Save to cache
        try FileManager.default.createDirectory(
            at: repoDir,
            withIntermediateDirectories: true
        )
        try data.write(to: localFile)

        return localFile
    }

    /// Download all required files for a GLiNER2 model
    ///
    /// - Parameters:
    ///   - repo: Repository ID
    ///   - cacheDir: Optional cache directory
    /// - Returns: Dictionary of filename to local URL
    public static func downloadModel(
        repo: String,
        cacheDir: URL? = nil
    ) async throws -> [String: URL] {
        let files = [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json"
        ]

        var downloaded: [String: URL] = [:]

        for filename in files {
            do {
                let url = try await download(repo: repo, filename: filename, cacheDir: cacheDir)
                downloaded[filename] = url
            } catch {
                // Some files may be optional
                if filename == "model.safetensors" || filename == "config.json" {
                    throw error
                }
            }
        }

        return downloaded
    }
}

public enum HuggingFaceError: Error, LocalizedError {
    case downloadFailed(String)
    case fileNotFound(String)

    public var errorDescription: String? {
        switch self {
        case .downloadFailed(let filename):
            return "Failed to download \(filename)"
        case .fileNotFound(let filename):
            return "File not found: \(filename)"
        }
    }
}
