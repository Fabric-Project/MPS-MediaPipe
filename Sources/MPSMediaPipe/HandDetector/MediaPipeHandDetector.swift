//
//  MediaPipeHandDetector.swift
//  MPSMediaPipe
//

import Foundation

/// BlazePalm hand detector. Owns anchor generation, SSD decode, weighted
/// NMS, letterbox projection, and rotated-rect derivation.
///
/// `decodeDetections` returns MediaPipe's native top-left-origin
/// normalized full-image space; callers in a bottom-left-origin
/// convention flip on their own side.
public enum MediaPipeHandDetector
{
    public static let detectSize = 192
    public static let resourcePrefix = "MediaPipeHandDetector"

    private static let numKeypoints = 7
    private static let rotationKeypoints = (start: 0, end: 2) // wrist -> middle finger MCP
    private static let targetAngleRadians: Float = 90.0 // raw radians, not degrees -- a MediaPipe proto quirk
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
