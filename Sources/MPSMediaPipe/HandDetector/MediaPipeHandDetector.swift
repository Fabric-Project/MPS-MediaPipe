//
//  MediaPipeHandDetector.swift
//  MPSMediaPipe
//

import Foundation

/// BlazePalm hand detector: owns every geometry fact specific to this
/// pretrained model (anchor grid, keypoint count/rotation pair, target
/// angle, ROI scale/shift) and the full decode pipeline that consumes
/// them — anchor generation, SSD decode, weighted NMS, letterbox
/// projection, and rotated-rect derivation (see MediaPipeSSDAnchors/
/// MediaPipeSSDDetectorDecoder/MediaPipeSSDRectTransform's own doc
/// comments for the shared algorithms this composes). A caller only
/// needs the model's raw two-tensor output plus the image it was run
/// against; it never needs to know this model's own magic numbers.
///
/// `decodeDetections`' returned region/keypoints are MediaPipe's native
/// top-left-origin normalized full-image space — callers in a
/// bottom-left-origin coordinate system (e.g. Fabric's own port
/// convention) flip on their own side, since that's a caller convention,
/// not a fact about this model.
public enum MediaPipeHandDetector
{
    public static let detectSize = 192
    public static let resourcePrefix = "MediaPipeHandDetector"

    // BlazePalm-specific constants (see MediaPipeSSDDetectorDecoder/
    // MediaPipeSSDRectTransform's doc comments for BlazeFace's own values).
    private static let numKeypoints = 7
    private static let rotationKeypoints = (start: 0, end: 2) // wrist -> middle finger MCP
    private static let targetAngleRadians: Float = 90.0 // a real MediaPipe proto quirk -- see MediaPipeSSDDetectorDecoder.computeRotation's doc comment
    private static let rectScale: Float = 2.6
    private static let rectShiftY: Float = -0.5

    private static let anchors = MediaPipeSSDAnchors.generate(detectSize: detectSize)

    public static func decodeDetections(
        rawBoxes: [Float], rawScores: [Float], maxDetections: Int,
        imageWidth: Float, imageHeight: Float
    ) -> [(region: (cx: Float, cy: Float, width: Float, height: Float), rotation: Float, score: Float, keypoints: [(x: Float, y: Float)])]
    {
        let decoded = MediaPipeSSDDetectorDecoder.decode(rawBoxes: rawBoxes, rawScores: rawScores, anchors: Self.anchors, numKeypoints: Self.numKeypoints, detectSize: Self.detectSize)
        let merged = MediaPipeSSDDetectorDecoder.weightedNonMaximumSuppression(decoded)
        let topDetections = merged.sorted { $0.score > $1.score }.prefix(maxDetections)

        return topDetections.map { detection in
            let projected = MediaPipeSSDRectTransform.project(detection, imageWidth: imageWidth, imageHeight: imageHeight)
            let rect = MediaPipeSSDRectTransform.rect(
                from: projected, imageWidth: imageWidth, imageHeight: imageHeight,
                rotationKeypoints: Self.rotationKeypoints, targetAngleRadians: Self.targetAngleRadians,
                rectScale: Self.rectScale, rectShiftY: Self.rectShiftY
            )
            return (region: (cx: rect.cx, cy: rect.cy, width: rect.width, height: rect.height), rotation: rect.rotation, score: detection.score, keypoints: projected.keypoints)
        }
    }
}
