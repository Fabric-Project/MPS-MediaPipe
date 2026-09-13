//
//  MediaPipeHandLandmarkProjection.swift
//  MPSMediaPipe
//

import Foundation
import simd

/// Decodes the hand landmark model's raw outputs (63 floats = 21 x,y,z in
/// crop pixels; presence; handedness; 63 world-landmark floats) and
/// projects them back through the rotated crop rect into full-image
/// normalized coordinates.
public enum MediaPipeHandLandmarkProjection
{
    public static let landmarkSize: Float = 224
    public static let resourcePrefix = "MediaPipeHandLandmarks"
    static let normalizeZ: Float = 0.4
    static let minHandPresenceConfidence: Float = 0.5

    // 21-point HAND_CONNECTIONS order: 0=wrist, 1-4=thumb, 5-8=index,
    // 9-12=middle, 13-16=ring, 17-20=little.
    public static let thumbIndices = [1, 2, 3, 4]
    public static let indexIndices = [5, 6, 7, 8]
    public static let middleIndices = [9, 10, 11, 12]
    public static let ringIndices = [13, 14, 15, 16]
    public static let littleIndices = [17, 18, 19, 20]
    public static let wristIndex = 0

    /// GetPartialLandmarks index set: wrist, thumb CMC/MCP/IP, and each
    /// other finger's MCP/PIP -- excludes fingertips so extended fingers
    /// don't blow out the tracked box. Order matches handLandmarksRect's
    /// expectation.
    private static let trackingPartialIndices = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]
    private static let trackingTargetAngleRadians: Float = .pi / 2
    private static let trackingRectScale: Float = 2.0
    private static let trackingRectShiftY: Float = -0.1

    public struct Hand
    {
        /// x, y normalized full-image [0,1], top-left origin; z is relative
        /// (same units as x, scaled by the crop rect's width).
        public var landmarks: [simd_float3]
        /// Hand-centered, real-world-scale (meters); not projected through
        /// the crop rect (only derotated, since already metric).
        public var worldLandmarks: [simd_float3]
        public var handedness: String
        public var handednessScore: Float
    }

    /// `landmarksRaw`/`worldLandmarksRaw` are the flat 63-float (21x3)
    /// outputs, in the model's own output order -- either shorter than that
    /// returns nil. `presence` is already sigmoid-activated by the model's
    /// own graph (unlike MediaPipeFaceLandmarkProjection's `presenceRaw`,
    /// which is a pre-sigmoid logit this type activates itself) -- compared
    /// directly against the threshold with no activation applied here.
    /// Returns nil when presence is below threshold.
    public static func project(
        landmarksRaw: [Float], worldLandmarksRaw: [Float], presence: Float, handednessRaw: Float,
        rect: (cx: Float, cy: Float, width: Float, height: Float, rotation: Float)
    ) -> Hand?
    {
        guard landmarksRaw.count >= 21 * 3, worldLandmarksRaw.count >= 21 * 3 else { return nil }
        guard presence > minHandPresenceConfidence else { return nil }

        // Binary classification: label_items[0]=Right (score s),
        // label_items[1]=Left (score 1-s).
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
            landmarks.append(MediaPipeLandmarkProjectionMath.rotateAndProject(
                x: landmarksRaw[index * 3 + 0], y: landmarksRaw[index * 3 + 1], z: landmarksRaw[index * 3 + 2],
                landmarkSize: landmarkSize, normalizeZ: normalizeZ,
                sinRotation: sinA, cosRotation: cosA,
                rect: (cx: rect.cx, cy: rect.cy, width: rect.width, height: rect.height)
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

    /// Re-derives a tracking ROI from this frame's landmarks: an oriented
    /// bounding box from a 12-point partial subset, rotated using wrist +
    /// weighted MCP average. `landmarks` and the returned `region` are both
    /// bottom-left-origin normalized.
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

        return MediaPipeLandmarkProjectionMath.regionBottomLeft(from: rect)
    }
}
