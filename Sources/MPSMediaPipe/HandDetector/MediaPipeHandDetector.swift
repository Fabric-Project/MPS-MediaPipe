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

    // No detectorPixelRange override here, unlike MediaPipeFaceDetector/
    // MediaPipePoseDetector -- this model expects MediaPipeCropPreprocessor's
    // default [0,1] pixel range, intentionally, not an oversight.

    private static let numKeypoints = 7
    private static let rotationKeypoints = (start: 0, end: 2) // wrist -> middle finger MCP
    private static let targetAngleRadians: Float = 90.0 // raw radians, not degrees -- a MediaPipe proto quirk
    private static let rectScale: Float = 2.6
    private static let rectShiftY: Float = -0.5

    private static let anchors = MediaPipeSSDAnchors.generate(detectSize: detectSize)

    public static func decodeDetections(
        rawBoxes: [Float], rawScores: [Float], maxDetections: Int,
        imageWidth: Float, imageHeight: Float
    ) -> [MediaPipeDetection]
    {
        MediaPipeSSDDetectorDecoder.decodeAndProject(
            rawBoxes: rawBoxes, rawScores: rawScores,
            anchors: Self.anchors, numKeypoints: Self.numKeypoints, detectSize: Self.detectSize,
            maxDetections: maxDetections, imageWidth: imageWidth, imageHeight: imageHeight
        ) { projected in
            MediaPipeSSDRectTransform.rect(
                from: projected, imageWidth: imageWidth, imageHeight: imageHeight,
                rotationKeypoints: Self.rotationKeypoints, targetAngleRadians: Self.targetAngleRadians,
                rectScale: Self.rectScale, rectShiftY: Self.rectShiftY
            )
        }
    }
}
