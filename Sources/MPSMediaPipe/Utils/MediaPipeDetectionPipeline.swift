//
//  MediaPipeDetectionPipeline.swift
//  MPSMediaPipe
//

import Foundation

public extension MediaPipeSSDDetectorDecoder
{
    /// Shared decode -> weighted NMS -> sort/prefix -> project -> rect-
    /// derivation pipeline used by all three "Blaze"-family detector
    /// facades. Only the final rect-derivation step differs per model
    /// (box-based `MediaPipeSSDRectTransform.rect()` for BlazePalm/
    /// BlazeFace, `.alignmentPointsRect()` for BlazePose) -- pass it as
    /// `deriveRect`.
    static func decodeAndProject(
        rawBoxes: [Float], rawScores: [Float],
        anchors: [(cx: Float, cy: Float, w: Float, h: Float)], numKeypoints: Int, detectSize: Int, minScore: Float = minDetectionConfidence,
        maxDetections: Int, imageWidth: Float, imageHeight: Float,
        deriveRect: (MediaPipeSSDRectTransform.ProjectedDetection) -> (cx: Float, cy: Float, width: Float, height: Float, rotation: Float)
    ) -> [MediaPipeDetection]
    {
        guard maxDetections > 0 else { return [] }

        let decoded = decode(rawBoxes: rawBoxes, rawScores: rawScores, anchors: anchors, numKeypoints: numKeypoints, detectSize: detectSize, minScore: minScore)
        let merged = weightedNonMaximumSuppression(decoded)
        let topDetections = merged.sorted { $0.score > $1.score }.prefix(maxDetections)

        return topDetections.map { detection in
            let projected = MediaPipeSSDRectTransform.project(detection, imageWidth: imageWidth, imageHeight: imageHeight)
            let rect = deriveRect(projected)
            return MediaPipeDetection(
                region: (cx: rect.cx, cy: rect.cy, width: rect.width, height: rect.height),
                rotation: rect.rotation, score: detection.score, keypoints: projected.keypoints
            )
        }
    }
}
