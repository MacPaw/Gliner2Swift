// SpanMarker.swift
// Span representation using marker-based approach
//
// Matches Python: gliner/modeling/span_rep.py:SpanMarkerV0

import MLX
import MLXNN

/// Extract elements from a sequence using provided indices.
///
/// - Parameters:
///   - sequence: Input sequence [B, L, D]
///   - indices: Indices to extract [B, K]
/// - Returns: Extracted elements [B, K, D]
private func extractElements(sequence: MLXArray, indices: MLXArray) -> MLXArray {
    let B = sequence.dim(0)
    let K = indices.dim(1)
    let D = sequence.dim(2)

    // Expand indices to [B, K, D]
    let expandedIndices = indices.expandedDimensions(axis: 2)
    let broadcastIndices = MLX.broadcast(expandedIndices, to: [B, K, D])

    // Gather elements along dimension 1
    return MLX.takeAlong(sequence, broadcastIndices, axis: 1)
}

/// Span representation using marker-based approach (V0).
///
/// Projects start and end positions separately and combines them
/// to form span representations.
///
/// Architecture:
///   start_rep = project_start(h)  # Linear → ReLU → Dropout → Linear
///   end_rep = project_end(h)
///   start_span = gather(start_rep, span_idx[:, :, 0])
///   end_span = gather(end_rep, span_idx[:, :, 1])
///   cat = concat([start_span, end_span], -1).relu()
///   return out_project(cat).view(B, L, max_width, D)
public class SpanMarkerV0: Module {
    public let maxWidth: Int
    public let hiddenSize: Int

    /// Start position projection: D → 4D → D
    public let projectStart: Sequential

    /// End position projection: D → 4D → D
    public let projectEnd: Sequential

    /// Output projection: 2D → 4*2D → D
    public let outProject: Sequential

    /// Initialize SpanMarkerV0
    ///
    /// - Parameters:
    ///   - hiddenSize: Hidden dimension (e.g., 768)
    ///   - maxWidth: Maximum span width (e.g., 8)
    ///   - dropout: Dropout rate (default: 0.4)
    public init(hiddenSize: Int, maxWidth: Int, dropout: Float = 0.4) {
        self.hiddenSize = hiddenSize
        self.maxWidth = maxWidth

        // Using create_projection_layer which expands by 4x
        self.projectStart = createProjectionLayer(
            hiddenSize: hiddenSize,
            dropout: dropout,
            outDim: hiddenSize
        )

        self.projectEnd = createProjectionLayer(
            hiddenSize: hiddenSize,
            dropout: dropout,
            outDim: hiddenSize
        )

        self.outProject = createProjectionLayer(
            hiddenSize: hiddenSize * 2,
            dropout: dropout,
            outDim: hiddenSize
        )
    }

    /// Forward pass
    ///
    /// - Parameters:
    ///   - h: Token representations [B, L, D]
    ///   - spanIdx: Span indices [B, num_spans, 2] where
    ///     spanIdx[..., 0] are start indices and spanIdx[..., 1] are end indices
    ///   - debug: If true, print intermediate L1 sums for debugging
    /// - Returns: Span representations [B, L, maxWidth, D]
    public func callAsFunction(_ h: MLXArray, spanIdx: MLXArray, debug: Bool = false) -> MLXArray {
        let B = h.dim(0)
        let L = h.dim(1)
        let D = h.dim(2)

        if debug {
            let hL1 = MLX.sum(MLX.abs(h)).item(Float.self)
            print("SpanMarkerV0 DEBUG:")
            print("  h (input) L1: \(hL1)")
        }

        // Project start and end positions
        let startRep = projectStart(h)  // [B, L, D]
        let endRep = projectEnd(h)      // [B, L, D]

        if debug {
            let startRepL1 = MLX.sum(MLX.abs(startRep)).item(Float.self)
            let endRepL1 = MLX.sum(MLX.abs(endRep)).item(Float.self)
            print("  start_rep L1: \(startRepL1)")
            print("  end_rep L1: \(endRepL1)")
        }

        // Extract span representations using indices
        let startSpanRep = extractElements(sequence: startRep, indices: spanIdx[0..., 0..., 0])
        let endSpanRep = extractElements(sequence: endRep, indices: spanIdx[0..., 0..., 1])

        if debug {
            let startSpanRepL1 = MLX.sum(MLX.abs(startSpanRep)).item(Float.self)
            let endSpanRepL1 = MLX.sum(MLX.abs(endSpanRep)).item(Float.self)
            print("  start_span_rep L1: \(startSpanRepL1)")
            print("  end_span_rep L1: \(endSpanRepL1)")
        }

        // Concatenate and apply ReLU
        var cat = MLX.concatenated([startSpanRep, endSpanRep], axis: -1)

        if debug {
            let catBeforeReluL1 = MLX.sum(MLX.abs(cat)).item(Float.self)
            print("  cat (before relu) L1: \(catBeforeReluL1)")
        }

        cat = MLX.maximum(cat, MLXArray(0))  // ReLU

        if debug {
            let catAfterReluL1 = MLX.sum(MLX.abs(cat)).item(Float.self)
            print("  cat (after relu) L1: \(catAfterReluL1)")

            // Trace outProject step by step
            print("  --- outProject layer-by-layer trace ---")
            var debugX = cat
            for (idx, layer) in outProject.layers.enumerated() {
                if let unaryLayer = layer as? any UnaryLayer {
                    debugX = unaryLayer(debugX)
                    let layerL1 = MLX.sum(MLX.abs(debugX)).item(Float.self)
                    let layerType = String(describing: type(of: layer))
                    print("    [\(idx)] \(layerType): L1=\(layerL1)")

                    // If it's Linear, also print weight L1 sum
                    if let linear = layer as? Linear {
                        let weightL1 = MLX.sum(MLX.abs(linear.weight)).item(Float.self)
                        let biasL1 = linear.bias != nil ? MLX.sum(MLX.abs(linear.bias!)).item(Float.self) : 0
                        print("        weight L1=\(weightL1), bias L1=\(biasL1)")
                    }
                }
            }
            print("  --- end trace ---")
        }

        // Project to output dimension
        let projected = outProject(cat)

        if debug {
            let projectedL1 = MLX.sum(MLX.abs(projected)).item(Float.self)
            print("  out_project output L1: \(projectedL1)")
        }

        // Reshape to [B, L, maxWidth, D]
        return projected.reshaped([B, L, maxWidth, D])
    }
}

// MARK: - Span Representation Factory

/// Factory for creating span representation layers.
public enum SpanMode: String {
    case markerV0 = "markerV0"
    // Could add other modes like query, mlp, etc. if needed
}

/// Factory class for various span representation approaches.
public class SpanRepLayer: Module {
    private let spanRepLayer: SpanMarkerV0

    public init(
        hiddenSize: Int,
        maxWidth: Int,
        spanMode: SpanMode = .markerV0,
        dropout: Float = 0.1
    ) {
        switch spanMode {
        case .markerV0:
            self.spanRepLayer = SpanMarkerV0(
                hiddenSize: hiddenSize,
                maxWidth: maxWidth,
                dropout: dropout
            )
        }
    }

    public func callAsFunction(_ x: MLXArray, spanIdx: MLXArray, debug: Bool = false) -> MLXArray {
        return spanRepLayer(x, spanIdx: spanIdx, debug: debug)
    }

    /// Access the underlying SpanMarkerV0 layer (for testing/debugging)
    public func getSpanRepLayer() -> SpanMarkerV0 {
        return spanRepLayer
    }
}

// MARK: - Weight Loading

extension SpanMarkerV0 {
    /// Load weights from a dictionary (SafeTensors format)
    ///
    /// Converted weights use camelCase (from convert_weights.py):
    /// - spanRep.spanRepLayer.projectStart.{0,3}.{weight,bias}
    /// - spanRep.spanRepLayer.projectEnd.{0,3}.{weight,bias}
    /// - spanRep.spanRepLayer.outProject.{0,3}.{weight,bias}
    ///
    /// Each projection layer has structure:
    ///   [0] Linear(D, 4D)
    ///   [1] ReLU
    ///   [2] Dropout
    ///   [3] Linear(4D, D)
    public func loadWeights(_ weights: [String: MLXArray], prefix: String) {
        // Load projectStart (camelCase)
        loadProjectionWeights(
            projection: projectStart,
            weights: weights,
            prefix: "\(prefix).projectStart"
        )

        // Load projectEnd (camelCase)
        loadProjectionWeights(
            projection: projectEnd,
            weights: weights,
            prefix: "\(prefix).projectEnd"
        )

        // Load outProject (camelCase)
        loadProjectionWeights(
            projection: outProject,
            weights: weights,
            prefix: "\(prefix).outProject"
        )
    }

    private func loadProjectionWeights(
        projection: Sequential,
        weights: [String: MLXArray],
        prefix: String
    ) {
        // Projection structure: Linear → ReLU → Dropout → Linear
        // Indices: 0=Linear, 1=ReLU, 2=Dropout, 3=Linear

        // Debug: Print key availability
        let w0Key = "\(prefix).0.weight"
        let b0Key = "\(prefix).0.bias"
        let w3Key = "\(prefix).3.weight"
        let b3Key = "\(prefix).3.bias"

        print("SpanMarker loading from prefix: \(prefix)")
        print("  \(w0Key): \(weights[w0Key] != nil ? "FOUND \(weights[w0Key]!.shape)" : "⚠️ MISSING")")
        print("  \(b0Key): \(weights[b0Key] != nil ? "FOUND \(weights[b0Key]!.shape)" : "⚠️ MISSING")")
        print("  \(w3Key): \(weights[w3Key] != nil ? "FOUND \(weights[w3Key]!.shape)" : "⚠️ MISSING")")
        print("  \(b3Key): \(weights[b3Key] != nil ? "FOUND \(weights[b3Key]!.shape)" : "⚠️ MISSING")")

        if let linear0 = projection.layers[0] as? Linear {
            updateLinearWeights(
                linear0,
                weight: weights[w0Key],
                bias: weights[b0Key],
                debugKey: "\(prefix).0"
            )
        } else {
            print("❌ ERROR: projection.layers[0] is not a Linear layer!")
        }

        if let linear3 = projection.layers[3] as? Linear {
            updateLinearWeights(
                linear3,
                weight: weights[w3Key],
                bias: weights[b3Key],
                debugKey: "\(prefix).3"
            )
        } else {
            print("❌ ERROR: projection.layers[3] is not a Linear layer!")
        }
    }
}

extension SpanRepLayer {
    /// Load weights from a dictionary
    /// Converted weights use camelCase: spanRep.spanRepLayer.*
    public func loadWeights(_ weights: [String: MLXArray], prefix: String) {
        spanRepLayer.loadWeights(weights, prefix: "\(prefix).spanRepLayer")
    }
}
