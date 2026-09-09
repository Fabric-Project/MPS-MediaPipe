// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MPS-MediaPipe",
    platforms: [.macOS(.v15), .iOS(.v18), .visionOS(.v2)],
    products: [
        .library(
            name: "MPSMediaPipe",
            targets: ["MPSMediaPipe"]
        ),
    ],
    targets: [
        .target(
            name: "MPSMediaPipe",
            path: "Sources/MPSMediaPipe",
            resources: [
                .copy("Compute"),
                .copy("Models"),
            ]
        ),
        .testTarget(
            name: "MPSMediaPipeTests",
            dependencies: ["MPSMediaPipe"],
            path: "Tests/MPSMediaPipeTests"
        ),
    ],
    swiftLanguageModes: [.v5]
)
