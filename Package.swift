// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.
//
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

import PackageDescription

let package = Package(
    name: "GLiNER2Swift",
    platforms: [
        .macOS(.v14)  // Requires macOS 14+ for MLX
    ],
    products: [
        .library(
            name: "GLiNER2Swift",
            targets: ["GLiNER2Swift"]
        ),
    ],
    dependencies: [
        // MLX Swift - Apple's ML framework for Apple Silicon
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.18.0"),
        // For Hub download and tokenizer support
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.0"),
    ],
    targets: [
        .target(
            name: "GLiNER2Swift",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/GLiNER2Swift"
        ),
        .testTarget(
            name: "GLiNER2SwiftTests",
            dependencies: ["GLiNER2Swift"],
            path: "Tests/GLiNER2SwiftTests",
            resources: [
                .copy("Fixtures")
            ]
        ),
    ]
)
