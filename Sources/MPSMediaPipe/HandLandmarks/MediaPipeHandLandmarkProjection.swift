//
//  MediaPipeHandLandmarkProjection.swift
//  Fabric
//

import Foundation
import simd

/// Decodes the hand landmark model's raw outputs (63 floats = 21 x,y,z in
/// crop pixels; presence; handedness; 63 world-landmark floats) and projects
/// screen-space landmarks back through the rotated crop rect into full-image
/// normalized coordinates. Ported from fasthands.pipeline.HandLandmarker.
/// _landmarks (TensorsToLandmarksCalculator + LandmarkProjectionCalculator +
/// WorldLandmarkProjectionCalculator), numerically validated against that
/// Python reference (and the real bundled MediaPipeHandLandmarks.mlpackage's
/// output on a real image) to float32 precision.
public enum MediaPipeHandLandmarkProjection
{
    public static let landmarkSize: Float = 224
    public static let resourcePrefix = "MediaPipeHandLandmarks"
    static let normalizeZ: Float = 0.4
    static let minHandPresenceConfidence: Float = 0.5

    // MediaPipe's own 21-point HAND_CONNECTIONS ordering: 0=wrist,
    // 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=little.
    public static let thumbIndices = [1, 2, 3, 4]
    public static let indexIndices = [5, 6, 7, 8]
    public static let middleIndices = [9, 10, 11, 12]
    public static let ringIndices = [13, 14, 15, 16]
    public static let littleIndices = [17, 18, 19, 20]
    public static let wristIndex = 0

    /// hand_landmarks_to_rect_calculator.cc's own GetPartialLandmarks index
    /// set -- wrist, thumb CMC/MCP/IP (not thumb tip), and each other
    /// finger's MCP/PIP (not DIP/tip) -- deliberately excludes fingertips so
    /// extended fingers don't blow out the tracked box. Order matters: it's
    /// the order MediaPipeSSDRectTransform.handLandmarksRect expects.
    private static let trackingPartialIndices = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]
    private static let trackingTargetAngleRadians: Float = .pi / 2
    private static let trackingRectScale: Float = 2.0
    private static let trackingRectShiftY: Float = -0.1

    public struct Hand
    {
        /// x, y normalized full-image [0,1], top-left origin; z is relative
        /// (same units as x, scaled by the crop rect's width — not a
        /// separate normalized space).
        public var landmarks: [simd_float3]
        /// Same layout, but hand-centered, real-world-scale (meters) — not
        /// projected through the crop rect (WorldLandmarkProjectionCalculator
        /// only derotates, since world landmarks are already metric).
        public var worldLandmarks: [simd_float3]
        public var handedness: String
        public var handednessScore: Float
    }

    /// `landmarksRaw`/`worldLandmarksRaw` are the flat 63-float (21x3)
    /// outputs, in the model's own output order (x,y,z per landmark).
    /// Returns nil when presence is below threshold (ThresholdingCalculator)
    /// — matches the Python reference dropping the hand entirely rather
    /// than emitting a low-confidence result.
    public static func project(
        landmarksRaw: [Float], worldLandmarksRaw: [Float], presence: Float, handednessRaw: Float,
        rect: (cx: Float, cy: Float, width: Float, height: Float, rotation: Float)
    ) -> Hand?
    {
        guard presence > minHandPresenceConfidence else { return nil }

        // TensorsToClassificationCalculator binary_classification:
        // label_items[0] = Right (score s), label_items[1] = Left (score 1-s).
        let label: String
        let handednessScore: Float
        if handednessRaw >= 0.5 { label = "Right"; handednessScore = handednessRaw }
        else { label = "Left"; handednessScore = 1 - handednessRaw }

        let sinA = sin(rect.rotation)
        let cosA = cos(rect.rotation)

        var landmarks: [simd_float3] = []
        landmarks.reserveCapacity(21)
        for index in 0..<21
        {
            let x = landmarksRaw[index * 3 + 0] / landmarkSize - 0.5
            let y = landmarksRaw[index * 3 + 1] / landmarkSize - 0.5
            let z = landmarksRaw[index * 3 + 2] / landmarkSize / normalizeZ

            let rotatedX = cosA * x - sinA * y
            let rotatedY = sinA * x + cosA * y

            landmarks.append(simd_float3(
                rotatedX * rect.width + rect.cx,
                rotatedY * rect.height + rect.cy,
                z * rect.width
            ))
        }

        var worldLandmarks: [simd_float3] = []
        worldLandmarks.reserveCapacity(21)
        for index in 0..<21
        {
            let x = worldLandmarksRaw[index * 3 + 0]
            let y = worldLandmarksRaw[index * 3 + 1]
            let z = worldLandmarksRaw[index * 3 + 2]
            worldLandmarks.append(simd_float3(cosA * x - sinA * y, sinA * x + cosA * y, z))
        }

        return Hand(landmarks: landmarks, worldLandmarks: worldLandmarks, handedness: label, handednessScore: handednessScore)
    }

    /// Re-derives a tracking ROI from this frame's own (unsmoothed)
    /// landmarks, the same way MediaPipe's HandLandmarksToRectCalculator
    /// does -- see MediaPipeSSDRectTransform.handLandmarksRect's own header
    /// for the full algorithm. `landmarks` are bottom-left-origin
    /// normalized (matching a caller's own decoded-landmark output space,
    /// e.g. MediaPipeHandLandmarkNode's); this function flips internally to
    /// top-left for handLandmarksRect, then flips the result back, so both
    /// `landmarks` in and `region` out are bottom-left-origin.
    public static func trackedRegion(from landmarks: [simd_float3], presentationSize: CGSize) -> (region: simd_float4, rotation: Float)?
    {
        guard landmarks.count > (Self.trackingPartialIndices.max() ?? -1) else { return nil }

        let imageWidth = Float(presentationSize.width)
        let imageHeight = Float(presentationSize.height)

        let partialPoints = Self.trackingPartialIndices.map { (x: landmarks[$0].x, y: 1 - landmarks[$0].y) }

        guard let rect = MediaPipeSSDRectTransform.handLandmarksRect(
            points: partialPoints, imageWidth: imageWidth, imageHeight: imageHeight,
            targetAngleRadians: Self.trackingTargetAngleRadians, rectScale: Self.trackingRectScale,
            rectShiftY: Self.trackingRectShiftY
        ) else { return nil }

        let regionBottomLeft = simd_float4(
            rect.cx - rect.width / 2,
            1 - (rect.cy - rect.height / 2) - rect.height,
            rect.width,
            rect.height
        )
        return (region: regionBottomLeft, rotation: rect.rotation)
    }
}
