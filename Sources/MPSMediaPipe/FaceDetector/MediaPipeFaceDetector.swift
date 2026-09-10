//
//  MediaPipeFaceDetector.swift
//  MPSMediaPipe
//

import Foundation

/// BlazeFace detector — short_range or full_range — owns every geometry
/// fact specific to these two pretrained models and the full decode
/// pipeline that consumes them: anchor generation (cached per variant),
/// SSD decode, weighted NMS, letterbox projection, and rotated-rect
/// derivation. Both variants share num_keypoints/rotation-keypoints/
/// target-angle/rect-scale — confirmed by there being exactly one
/// face_detection_front_detection_to_roi.pbtxt in mediapipe's repo, used
/// regardless of which detector variant feeds it — only detector geometry
/// and the bundled model itself differ per variant.
///
/// short_range config (128x128 input, 4-layer anchor grid, min_score 0.5)
/// confirmed against mediapipe/modules/face_detection/
/// face_detection_short_range.pbtxt; full_range config (192x192 input,
/// single-layer stride-4 anchor grid, min_score 0.6, and a different --
/// and differently structured, see MediaPipeMPSGraph's
/// DEPTH_TO_SPACE case -- underlying model, face_detection_full_range_
/// sparse.tflite) confirmed against face_detection_full_range.pbtxt.
///
/// `decodeDetections`' returned region/keypoints are MediaPipe's native
/// top-left-origin normalized full-image space — callers in a
/// bottom-left-origin coordinate system (e.g. Fabric's own port
/// convention) flip on their own side, since that's a caller convention,
/// not a fact about this model.
public enum MediaPipeFaceDetector
{
    public enum Variant: String, CaseIterable
    {
        case shortRange = "Short Range"
        case fullRange = "Full Range"

        public var detectSize: Int { self == .shortRange ? 128 : 192 }
        var strides: [Int] { self == .shortRange ? [8, 16, 16, 16] : [4] }
        var interpolatedScaleAspectRatio: Float { self == .shortRange ? 1.0 : 0.0 }
        var minScoreThreshold: Float { self == .shortRange ? 0.5 : 0.6 }
        public var resourcePrefix: String { self == .shortRange ? "MediaPipeFaceDetector" : "MediaPipeFaceDetectorFullRange" }

        public static func from(_ rawValue: String?) -> Variant
        {
            rawValue.flatMap(Variant.init(rawValue:)) ?? .shortRange
        }
    }

    public static let detectorPixelRange: (min: Float, max: Float) = (-1, 1)

    // Shared across both detector variants (see this type's own header).
    private static let numKeypoints = 6
    private static let rotationKeypoints = (start: 0, end: 1) // left eye -> right eye
    private static let targetAngleRadians: Float = 0.0
    private static let rectScale: Float = 1.5

    private static let anchorsLock = NSLock()
    private static var anchorsCache: [Variant: [(cx: Float, cy: Float, w: Float, h: Float)]] = [:]

    private static func anchors(for variant: Variant) -> [(cx: Float, cy: Float, w: Float, h: Float)]
    {
        Self.anchorsLock.lock()
        defer { Self.anchorsLock.unlock() }
        if let existing = Self.anchorsCache[variant] { return existing }
        let generated = MediaPipeSSDAnchors.generate(detectSize: variant.detectSize, strides: variant.strides, interpolatedScaleAspectRatio: variant.interpolatedScaleAspectRatio)
        Self.anchorsCache[variant] = generated
        return generated
    }

    public static func decodeDetections(
        rawBoxes: [Float], rawScores: [Float], variant: Variant, maxDetections: Int,
        imageWidth: Float, imageHeight: Float
    ) -> [(region: (cx: Float, cy: Float, width: Float, height: Float), rotation: Float, score: Float, keypoints: [(x: Float, y: Float)])]
    {
        let decoded = MediaPipeSSDDetectorDecoder.decode(rawBoxes: rawBoxes, rawScores: rawScores, anchors: Self.anchors(for: variant), numKeypoints: Self.numKeypoints, detectSize: variant.detectSize, minScore: variant.minScoreThreshold)
        let merged = MediaPipeSSDDetectorDecoder.weightedNonMaximumSuppression(decoded)
        let topDetections = merged.sorted { $0.score > $1.score }.prefix(maxDetections)

        return topDetections.map { detection in
            let projected = MediaPipeSSDRectTransform.project(detection, imageWidth: imageWidth, imageHeight: imageHeight)
            let rect = MediaPipeSSDRectTransform.rect(
                from: projected, imageWidth: imageWidth, imageHeight: imageHeight,
                rotationKeypoints: Self.rotationKeypoints, targetAngleRadians: Self.targetAngleRadians,
                rectScale: Self.rectScale
            )
            return (region: (cx: rect.cx, cy: rect.cy, width: rect.width, height: rect.height), rotation: rect.rotation, score: detection.score, keypoints: projected.keypoints)
        }
    }
}
