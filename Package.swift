// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

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
        // For tokenizer support
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
