# MPS-MediaPipe

A Swift package that runs [MediaPipe](https://github.com/google-ai-edge/mediapipe)'s "Blaze"-family detection, landmark, and segmentation models directly on the GPU via `MPSGraph` (Metal Performance Shaders Graph) — no CoreML, no Python, no TensorFlow Lite runtime. Models are from-scratch MPSGraph reimplementations of MediaPipe's published TFLite graphs, with their pretrained weights bundled in. This package is not affiliated with or endorsed by Google.

## What's included

| Task | Type |
|---|---|
| Face detection (short-range + full-range) | `MediaPipeFaceDetector` |
| Hand detection | `MediaPipeHandDetector` |
| Pose/body detection | `MediaPipePoseDetector` |
| Face landmarks (468-point FaceMesh) | `MediaPipeFaceLandmarkProjection` |
| Hand landmarks (21-point) | `MediaPipeHandLandmarkProjection` |
| Pose landmarks (33-point, lite/full/heavy tiers) + segmentation mask | `MediaPipePoseLandmarkProjection` |
| Selfie segmentation (general + landscape) | `MediaPipeSelfieSegmentation` |
| Face geometry (head pose + expression-normalized mesh from FaceMesh landmarks) | `FaceGeometrySolver` |

Shared machinery (crop/warp compute kernels, SSD anchor/decode/NMS, the generic TFLite-op-to-MPSGraph interpreter, One Euro Filter landmark smoothing) lives under `Utils/` and isn't tied to any one model.

## Requirements

Swift 5.9+ (`swift-tools-version: 6.0`, `swiftLanguageModes: [.v5]`), macOS 15+ / iOS 18+ / visionOS 2+. Needs a real Metal device — there's no CPU fallback.

## Installation

This repository uses [Git LFS](https://git-lfs.com) for the bundled model weights (`Sources/MPSMediaPipe/Models/`, ~215MB). **Install Git LFS before cloning or adding this as a dependency** (`git lfs install`), or you'll get small text pointer files instead of real weights and every model load will fail.

```swift
dependencies: [
    .package(url: "https://github.com/Fabric-Project/MPS-MediaPipe", from: "1.0.0"),
],
targets: [
    .target(name: "YourTarget", dependencies: [
        .product(name: "MPSMediaPipe", package: "MPS-MediaPipe"),
    ]),
]
```

## Usage

Every model follows the same shape: crop/normalize your input texture with `MediaPipeCropPreprocessor`, run it through a `MediaPipeMPSGraph.loadBundled(named:)` instance, then decode the raw output with that model's own type. Hand detection, end to end:

```swift
import MPSMediaPipe
import Metal

let device: MTLDevice = ...
let commandQueue: MTLCommandQueue = ...
let inputTexture: MTLTexture = ... // your camera frame or image

let preprocessor = try MediaPipeCropPreprocessor(
    device: device,
    outputWidth: MediaPipeHandDetector.detectSize,
    outputHeight: MediaPipeHandDetector.detectSize
)
let model = try MediaPipeMPSGraph.loadBundled(
    named: MediaPipeHandDetector.resourcePrefix,
    inputWidth: MediaPipeHandDetector.detectSize,
    inputHeight: MediaPipeHandDetector.detectSize,
    commandQueue: commandQueue
)

// Stretch the whole frame (center, full extent, no rotation) into the model's
// input size. For non-square input, letterbox instead by computing
// sizeNormalized from `max(width, height) / width` and `/ height`, matching
// what a detector node typically does -- see the type's own doc comments.
let inputBuffer = try preprocessor.encode(
    texture: inputTexture,
    textureTransform: matrix_identity_float4x4,
    presentationSize: simd_float2(Float(inputTexture.width), Float(inputTexture.height)),
    centerNormalizedBottomLeft: simd_float2(0.5, 0.5),
    sizeNormalized: simd_float2(1, 1),
    rotationRadians: 0,
    commandQueue: commandQueue
)

let outputs = model.run(inputBuffer: inputBuffer) // [[Float]], one array per model output tensor
let detections = MediaPipeHandDetector.decodeDetections(
    rawBoxes: outputs[0], rawScores: outputs[1], maxDetections: 2,
    imageWidth: Float(inputTexture.width), imageHeight: Float(inputTexture.height)
)
// detections: [MediaPipeDetection] -- each has .region (cx, cy, width, height), .rotation, .score, .keypoints
```

`model.run(inputBuffer:)` blocks until the GPU finishes; `model.submit(inputBuffer:commandBuffer:completion:)` is the non-blocking counterpart for encoding crop + inference onto one command buffer without waiting. Landmark models (`MediaPipeFaceLandmarkProjection.project(...)`, etc.) take a detector's output region/rotation plus the landmark model's own raw output and return decoded points in full-image normalized coordinates — see each type's own doc comments for its exact contract (coordinate convention, presence-threshold behavior, and any model-specific quirks are documented per type, not repeated here).

## Coordinate conventions

Everything in this package works in MediaPipe's own coordinate space: normalized `[0,1]`, top-left origin. Nothing here assumes a particular host app's rendering convention (e.g. a bottom-left-origin or Metal-clip-space convention) — flip on your own side if your app needs one.

## Scope

The TFLite-op interpreter (`MediaPipeMPSGraph`) handles the specific op set the bundled model families need (standard convolution/pooling/activation ops plus MediaPipe's `Convolution2DTransposeBias` custom op) — it is not a general-purpose TFLite runtime. Bringing your own MediaPipe-style model may require adding support for additional ops; an unrecognized op throws a clear error rather than producing a silently wrong result.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
