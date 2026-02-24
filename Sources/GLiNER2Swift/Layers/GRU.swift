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
// GRU.swift
// Gated Recurrent Unit implementation for MLX
//
// CRITICAL: PyTorch GRU uses BOTH bias_ih AND bias_hh
// This differs from some implementations that combine biases.
//
// PyTorch GRU equations:
//   r_t = sigmoid(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)
//   z_t = sigmoid(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)
//   n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_hn))
//   h_t = (1 - z_t) * n_t + z_t * h_{t-1}

import MLX
import MLXNN

/// GRU layer compatible with PyTorch weight format.
///
/// Weight shapes from PyTorch:
/// - weight_ih_l0: [3*hidden_size, input_size] - gates ordered as [reset, update, new]
/// - weight_hh_l0: [3*hidden_size, hidden_size]
/// - bias_ih_l0: [3*hidden_size]
/// - bias_hh_l0: [3*hidden_size]
public class GRU: Module {
    /// Input-to-hidden weights [3*hidden, input]
    public var weightIH: MLXArray
    /// Hidden-to-hidden weights [3*hidden, hidden]
    public var weightHH: MLXArray
    /// Input-to-hidden bias [3*hidden]
    public var biasIH: MLXArray
    /// Hidden-to-hidden bias [3*hidden]
    public var biasHH: MLXArray

    public var inputSize: Int
    public var hiddenSize: Int

    /// Initialize GRU with dimensions
    ///
    /// - Parameters:
    ///   - inputSize: Size of input features
    ///   - hiddenSize: Size of hidden state
    public init(inputSize: Int, hiddenSize: Int) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize

        // Initialize weights using Xavier uniform
        let k = Float(1.0 / Float(hiddenSize)).squareRoot()

        self.weightIH = MLXRandom.uniform(
            low: -k,
            high: k,
            [3 * hiddenSize, inputSize]
        )
        self.weightHH = MLXRandom.uniform(
            low: -k,
            high: k,
            [3 * hiddenSize, hiddenSize]
        )
        self.biasIH = MLXArray.zeros([3 * hiddenSize])
        self.biasHH = MLXArray.zeros([3 * hiddenSize])
    }

    /// Initialize GRU from pre-loaded weights
    ///
    /// - Parameters:
    ///   - weightIH: Input-to-hidden weights [3*hidden, input]
    ///   - weightHH: Hidden-to-hidden weights [3*hidden, hidden]
    ///   - biasIH: Input-to-hidden bias [3*hidden]
    ///   - biasHH: Hidden-to-hidden bias [3*hidden]
    public init(weightIH: MLXArray, weightHH: MLXArray, biasIH: MLXArray, biasHH: MLXArray) {
        self.weightIH = weightIH
        self.weightHH = weightHH
        self.biasIH = biasIH
        self.biasHH = biasHH

        self.inputSize = weightIH.dim(1)
        self.hiddenSize = weightHH.dim(1)
    }

    /// Forward pass through GRU
    ///
    /// - Parameters:
    ///   - input: Input tensor [seq_len, batch, input_size]
    ///   - h0: Initial hidden state [1, batch, hidden_size], or nil for zeros
    /// - Returns: Tuple of (output, h_n) where:
    ///   - output: [seq_len, batch, hidden_size]
    ///   - h_n: [1, batch, hidden_size]
    public func callAsFunction(_ input: MLXArray, h0: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let seqLen = input.dim(0)
        let batch = input.dim(1)

        // Initialize hidden state
        var h: MLXArray
        if let h0 = h0 {
            h = h0.squeezed(axis: 0)  // [batch, hidden]
        } else {
            h = MLXArray.zeros([batch, hiddenSize])
        }

        var outputs: [MLXArray] = []

        // Process each timestep
        for t in 0..<seqLen {
            let x = input[t]  // [batch, input_size]
            h = gruCell(x: x, h: h)
            outputs.append(h.expandedDimensions(axis: 0))
        }

        // Stack outputs: [seq_len, batch, hidden]
        let output = MLX.concatenated(outputs, axis: 0)
        let hN = h.expandedDimensions(axis: 0)  // [1, batch, hidden]

        return (output, hN)
    }

    /// Single GRU cell computation
    ///
    /// PyTorch GRU equations:
    /// - r = sigmoid(x @ W_ir.T + b_ir + h @ W_hr.T + b_hr)
    /// - z = sigmoid(x @ W_iz.T + b_iz + h @ W_hz.T + b_hz)
    /// - n = tanh(x @ W_in.T + b_in + r * (h @ W_hn.T + b_hn))
    /// - h' = (1 - z) * n + z * h
    private func gruCell(x: MLXArray, h: MLXArray) -> MLXArray {
        // Split weights into gates: [reset, update, new]
        let wIR = weightIH[0..<hiddenSize]
        let wIZ = weightIH[hiddenSize..<(2*hiddenSize)]
        let wIN = weightIH[(2*hiddenSize)..<(3*hiddenSize)]

        let wHR = weightHH[0..<hiddenSize]
        let wHZ = weightHH[hiddenSize..<(2*hiddenSize)]
        let wHN = weightHH[(2*hiddenSize)..<(3*hiddenSize)]

        let bIR = biasIH[0..<hiddenSize]
        let bIZ = biasIH[hiddenSize..<(2*hiddenSize)]
        let bIN = biasIH[(2*hiddenSize)..<(3*hiddenSize)]

        let bHR = biasHH[0..<hiddenSize]
        let bHZ = biasHH[hiddenSize..<(2*hiddenSize)]
        let bHN = biasHH[(2*hiddenSize)..<(3*hiddenSize)]

        // Reset gate: r = sigmoid(x @ W_ir.T + b_ir + h @ W_hr.T + b_hr)
        let r = MLX.sigmoid(
            MLX.matmul(x, wIR.transposed()) + bIR +
            MLX.matmul(h, wHR.transposed()) + bHR
        )

        // Update gate: z = sigmoid(x @ W_iz.T + b_iz + h @ W_hz.T + b_hz)
        let z = MLX.sigmoid(
            MLX.matmul(x, wIZ.transposed()) + bIZ +
            MLX.matmul(h, wHZ.transposed()) + bHZ
        )

        // New gate: n = tanh(x @ W_in.T + b_in + r * (h @ W_hn.T + b_hn))
        let hHidden = MLX.matmul(h, wHN.transposed()) + bHN
        let n = MLX.tanh(
            MLX.matmul(x, wIN.transposed()) + bIN + r * hHidden
        )

        // Hidden state: h' = (1 - z) * n + z * h
        let hNew = (1 - z) * n + z * h

        return hNew
    }
}

// MARK: - Weight Loading

extension GRU {
    /// Load weights from a dictionary (SafeTensors format)
    ///
    /// Supports both PyTorch snake_case and converted camelCase keys:
    /// - weightIH / weight_ih_l0: [3*hidden, input]
    /// - weightHH / weight_hh_l0: [3*hidden, hidden]
    /// - biasIH / bias_ih_l0: [3*hidden]
    /// - biasHH / bias_hh_l0: [3*hidden]
    ///
    /// The convert_weights.py script outputs camelCase keys (weightIH, etc.)
    public static func fromWeights(_ weights: [String: MLXArray], prefix: String = "") -> GRU {
        let p = prefix.isEmpty ? "" : "\(prefix)."

        // Try camelCase (from convert_weights.py) first, then snake_case (raw PyTorch)
        let weightIH = weights["\(p)weightIH"] ?? weights["\(p)weight_ih_l0"]
        let weightHH = weights["\(p)weightHH"] ?? weights["\(p)weight_hh_l0"]
        let biasIH = weights["\(p)biasIH"] ?? weights["\(p)bias_ih_l0"]
        let biasHH = weights["\(p)biasHH"] ?? weights["\(p)bias_hh_l0"]

        guard let wIH = weightIH, let wHH = weightHH, let bIH = biasIH, let bHH = biasHH else {
            fatalError("Missing GRU weights with prefix: \(prefix). Tried keys: \(p)weightIH, \(p)weight_ih_l0")
        }

        return GRU(weightIH: wIH, weightHH: wHH, biasIH: bIH, biasHH: bHH)
    }

    /// Load weights into existing GRU instance
    ///
    /// Supports both PyTorch snake_case and converted camelCase keys.
    ///
    /// - Parameters:
    ///   - weights: Dictionary of weight tensors
    ///   - prefix: Prefix for weight names
    public func loadWeights(_ weights: [String: MLXArray], prefix: String = "") {
        let p = prefix.isEmpty ? "" : "\(prefix)."

        // Try camelCase (from convert_weights.py) first, then snake_case (raw PyTorch)
        if let wIH = weights["\(p)weightIH"] ?? weights["\(p)weight_ih_l0"] {
            self.weightIH = wIH
        }
        if let wHH = weights["\(p)weightHH"] ?? weights["\(p)weight_hh_l0"] {
            self.weightHH = wHH
        }
        if let bIH = weights["\(p)biasIH"] ?? weights["\(p)bias_ih_l0"] {
            self.biasIH = bIH
        }
        if let bHH = weights["\(p)biasHH"] ?? weights["\(p)bias_hh_l0"] {
            self.biasHH = bHH
        }

        // Update sizes based on loaded weights
        self.inputSize = self.weightIH.dim(1)
        self.hiddenSize = self.weightHH.dim(1)
    }
}
