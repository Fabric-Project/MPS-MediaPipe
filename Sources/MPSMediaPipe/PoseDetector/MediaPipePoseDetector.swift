//
//  MediaPipePoseDetector.swift
//  MPSMediaPipe
//

import Foundation

/// BlazePose's detector: owns every geometry fact specific to this
/// pretrained model and the full decode pipeline that consumes them —
/// anchor generation, SSD decode, weighted NMS, letterbox projection,
/// and rotated-rect derivation. Unlike MediaPipeHandDetector/
/// MediaPipeFaceDetector (box-based ROI), BlazePose's own detector
/// derives its ROI from two alignment keypoints, not the SSD box itself
/// — see MediaPipeSSDRectTransform.alignmentPointsRect's own doc comment.
///
/// `decodeDetections`' returned region/keypoints are MediaPipe's native
/// top-left-origin normalized full-image space — callers in a
/// bottom-left-origin coordinate system (e.g. Fabric's own port
/// convention) flip on their own side, since that's a caller convention,
/// not a fact about this model.
public enum MediaPipePoseDetector
{
    public static let detectSize = 224
    public static let resourcePrefix = "MediaPipePoseDetector"
    public static let detectorPixelRange: (min: Float, max: Float) = (-1, 1)

    private static let numKeypoints = 4
    private static let rotationKeypoints = (start: 0, end: 1) // mid-hip -> full-body size/rotation point
    /// mediapipe/modules/pose_landmark/pose_detection_to_roi.pbtxt sets
    /// `rotation_vector_target_angle_degrees: 90` -- the plain,
    /// properly-degrees-converted field (confirmed against
    /// detections_to_rects_calculator.cc's Open(): this field goes through
    /// `M_PI * degrees / 180`, unlike the separate, unitless
    /// `rotation_vector_target_angle` field BlazePalm's own config uses --
    /// so unlike BlazePalm's proto quirk, this is NOT "90 raw radians", it
    /// is 90 degrees.
    private static let targetAngleRadians: Float = .pi / 2
    private static let rectScale: Float = 1.25

    private static let anchors = MediaPipeSSDAnchors.generate(detectSize: detectSize, strides: [8, 16, 32, 32, 32])

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
            let rect = MediaPipeSSDRectTransform.alignmentPointsRect(
                from: projected, imageWidth: imageWidth, imageHeight: imageHeight,
                rotationKeypoints: Self.rotationKeypoints, targetAngleRadians: Self.targetAngleRadians,
                rectScale: Self.rectScale
            )
            return (region: (cx: rect.cx, cy: rect.cy, width: rect.width, height: rect.height), rotation: rect.rotation, score: detection.score, keypoints: projected.keypoints)
        }
    }
}
