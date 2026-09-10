//
//  MediaPipeFaceDetector.swift
//  MPSMediaPipe
//

import Foundation

/// BlazeFace detector (short_range/full_range). Owns anchor generation,
/// SSD decode, weighted NMS, letterbox projection, and rotated-rect
/// derivation for both variants.
///
/// short_range: 128x128 input, 4-layer anchor grid, min_score 0.5.
/// full_range: 192x192 input, single-layer stride-4 anchor grid,
/// min_score 0.6, DEPTH_TO_SPACE model structure (see MediaPipeMPSGraph).
///
/// `decodeDetections` returns MediaPipe's native top-left-origin
/// normalized full-image space; callers in a bottom-left-origin
/// convention flip on their own side.
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
