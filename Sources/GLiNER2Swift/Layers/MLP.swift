// MLP.swift
// Multi-layer perceptron implementations
//
// Matches Python: gliner2/layers.py:create_mlp

import MLX
import MLXNN
import MLXRandom

/// Creates a multi-layer perceptron (MLP) with specified dimensions and activation functions.
///
/// Architecture: Input → [Linear → LayerNorm? → Activation → Dropout?]* → Linear → Output
///
/// - Parameters:
///   - inputDim: Input dimension
///   - intermediateDims: List of hidden layer dimensions
///   - outputDim: Output dimension
///   - dropout: Dropout probability (0 = no dropout)
///   - activation: Activation function type
///   - addLayerNorm: Whether to add LayerNorm after each linear layer
/// - Returns: Sequential module implementing the MLP
public func createMLP(
    inputDim: Int,
    intermediateDims: [Int],
    outputDim: Int,
    dropout: Float = 0.1,
    activation: ActivationType = .gelu,
    addLayerNorm: Bool = false
) -> Sequential {
    var layers: [Module] = []
    var inDim = inputDim

    for dim in intermediateDims {
        layers.append(Linear(inDim, dim))
        if addLayerNorm {
            layers.append(LayerNorm(dimensions: dim))
        }
        layers.append(activation.module)
        if dropout > 0 {
            layers.append(Dropout(p: dropout))
        }
        inDim = dim
    }

    layers.append(Linear(inDim, outputDim))

    return Sequential(layers: layers)
}

/// Supported activation function types
public enum ActivationType: String, Sendable {
    case relu
    case tanh
    case sigmoid
    case leakyRelu = "leaky_relu"
    case gelu

    var module: Module {
        switch self {
        case .relu:
            return ReLU()
        case .tanh:
            return Tanh()
        case .sigmoid:
            return Sigmoid()
        case .leakyRelu:
            return LeakyReLU()
        case .gelu:
            return GELU()
        }
    }
}

// MARK: - Basic Activation Modules

/// ReLU activation: max(0, x)
public class ReLU: Module, UnaryLayer {
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLX.maximum(x, MLXArray(0))
    }
}

/// Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
public class Tanh: Module, UnaryLayer {
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLX.tanh(x)
    }
}

/// Sigmoid activation: 1 / (1 + exp(-x))
public class Sigmoid: Module, UnaryLayer {
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLX.sigmoid(x)
    }
}

/// LeakyReLU activation: x if x > 0 else negative_slope * x
public class LeakyReLU: Module, UnaryLayer {
    let negativeSlope: Float

    public init(negativeSlope: Float = 0.01) {
        self.negativeSlope = negativeSlope
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLX.maximum(x, x * negativeSlope)
    }
}

// MARK: - Sequential Container

/// Sequential module that runs layers in order
public class Sequential: Module, UnaryLayer {
    public let layers: [Module]

    public init(layers: [Module]) {
        self.layers = layers
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var output = x
        for layer in layers {
            if let unaryLayer = layer as? any UnaryLayer {
                output = unaryLayer(output)
            } else {
                fatalError("Layer \(type(of: layer)) does not conform to UnaryLayer")
            }
        }
        return output
    }
}

// MARK: - Projection Layer (matches gliner/modeling/layers.py:create_projection_layer)

/// Creates a two-layer projection network with ReLU activation and dropout.
/// The projection layer expands the input by 4x in the hidden layer before
/// projecting to the output dimension.
///
/// Architecture: Linear(in, out*4) → ReLU → Dropout → Linear(out*4, out)
///
/// - Parameters:
///   - hiddenSize: Size of the input hidden dimension
///   - dropout: Dropout probability
///   - outDim: Output dimension size. If nil, uses hiddenSize
/// - Returns: Sequential module containing the projection layers
public func createProjectionLayer(
    hiddenSize: Int,
    dropout: Float,
    outDim: Int? = nil
) -> Sequential {
    let outputDim = outDim ?? hiddenSize

    return Sequential(layers: [
        Linear(hiddenSize, outputDim * 4),
        ReLU(),
        Dropout(p: dropout),
        Linear(outputDim * 4, outputDim)
    ])
}
