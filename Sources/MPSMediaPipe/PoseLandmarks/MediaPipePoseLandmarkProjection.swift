//
//  MediaPipePoseLandmarkProjection.swift
//  MPSMediaPipe
//

import Foundation
import simd

/// Decodes BlazePose's raw landmark tensor (39 points x 5 values: x, y, z,
/// visibility, presence) and projects them through the rotated crop rect
/// into full-image normalized coordinates.
///
/// `landmarksRaw` must already have its x,y refined by
/// MediaPipePoseHeatmapRefinement before calling `project` -- this type has
/// no heatmap dependency itself.
///
/// The first 33 of the 39 points are the real pose landmarks; indices
/// 33-34 are a separate 2-point "auxiliary_landmarks" stream used only to
/// derive the next frame's tracking ROI -- NOT nose/left-eye, despite
/// matching the indices of other, unrelated arrays.
public enum MediaPipePoseLandmarkProjection
{
    /// Speed/accuracy tier for the bundled BlazePose landmark model --
    /// switching tier never changes port shape, only accuracy/latency.
    public enum ModelTier: String, CaseIterable
    {
        case lite = "Lite"
        case full = "Full"
        case heavy = "Heavy"

        public var resourcePrefix: String
        {
            switch self
            {
            case .lite: return "MediaPipePoseLandmarkLite"
            case .full: return "MediaPipePoseLandmarkFull"
            case .heavy: return "MediaPipePoseLandmarkHeavy"
            }
        }

        public static func from(_ rawValue: String?) -> ModelTier
        {
            rawValue.flatMap(ModelTier.init(rawValue:)) ?? .lite
        }
    }

    public static let landmarkSize: Float = 256
    public static let heatmapSize = 64
    public static let maskSize = 256
    static let normalizeZ: Float = 1.0
    static let minPosePresenceConfidence: Float = 0.5
    public static let decodedLandmarkCount = 39
    static let poseLandmarkCount = 33
    static let auxiliaryLandmarkCount = 2

    /// Indices 0/1 *of the 2-point auxiliary_landmarks stream* (not the 33
    /// pose landmarks, and not the detector's own keypoint 0/1 -- three
    /// different arrays, same indices by coincidence).
    private static let trackingRotationKeypoints = (start: 0, end: 1)
    private static let trackingTargetAngleRadians: Float = .pi / 2
    private static let trackingRectScale: Float = 1.25

    public struct Pose
    {
        /// x, y normalized full-image [0,1], top-left origin; z is relative
        /// (same units as x, scaled by the crop rect's width).
        public var landmarks: [simd_float3]
        /// The 2 dedicated ROI-derivation points (decoded indices 33-34).
        public var auxiliaryLandmarks: [simd_float3]
    }

    /// `landmarksRaw` is the flat 195-float (39x5) output (x,y already
    /// heatmap-refined by the caller) -- shorter than that returns nil.
    /// `presenceRaw` is the raw (pre-sigmoid) pose-presence logit. Returns
    /// nil when presence is below threshold.
    public static func project(
        landmarksRaw: [Float], presenceRaw: Float,
        rect: (cx: Float, cy: Float, width: Float, height: Float, rotation: Float)
    ) -> Pose?
    {
        guard landmarksRaw.count >= decodedLandmarkCount * 5 else { return nil }

        let presence = Float(1.0 / (1.0 + exp(-Double(presenceRaw))))
        guard presence > minPosePresenceConfidence else { return nil }

        let sinA = sin(rect.rotation)
        let cosA = cos(rect.rotation)

        func projectPoint(_ index: Int) -> simd_float3
        {
            let base = index * 5
            return MediaPipeLandmarkProjectionMath.rotateAndProject(
                x: landmarksRaw[base + 0], y: landmarksRaw[base + 1], z: landmarksRaw[base + 2],
                landmarkSize: landmarkSize, normalizeZ: normalizeZ,
                sinRotation: sinA, cosRotation: cosA,
                rect: (cx: rect.cx, cy: rect.cy, width: rect.width, height: rect.height)
            )
        }

        var landmarks: [simd_float3] = []
        landmarks.reserveCapacity(poseLandmarkCount)
        for index in 0..<poseLandmarkCount
        {
            landmarks.append(projectPoint(index))
        }

        var auxiliaryLandmarks: [simd_float3] = []
        auxiliaryLandmarks.reserveCapacity(auxiliaryLandmarkCount)
        for index in poseLandmarkCount..<(poseLandmarkCount + auxiliaryLandmarkCount)
        {
            auxiliaryLandmarks.append(projectPoint(index))
        }

        return Pose(landmarks: landmarks, auxiliaryLandmarks: auxiliaryLandmarks)
    }

    /// Re-derives a tracking ROI from this frame's auxiliary landmarks. nil
    /// when fewer than 2 points are available. `auxiliaryLandmarks` and the
    /// returned `region` are both bottom-left-origin normalized.
    public static func trackedRegion(from auxiliaryLandmarks: [simd_float3], presentationSize: CGSize) -> (region: simd_float4, rotation: Float)?
    {
        guard auxiliaryLandmarks.count >= 2 else { return nil }

        let imageWidth = Float(presentationSize.width)
        let imageHeight = Float(presentationSize.height)

        let projected = MediaPipeSSDRectTransform.ProjectedDetection(
            xmin: 0, ymin: 0, width: 0, height: 0,
            keypoints: [
                (x: auxiliaryLandmarks[0].x, y: 1 - auxiliaryLandmarks[0].y),
                (x: auxiliaryLandmarks[1].x, y: 1 - auxiliaryLandmarks[1].y),
            ],
            score: 1
        )

        let rect = MediaPipeSSDRectTransform.alignmentPointsRect(
            from: projected, imageWidth: imageWidth, imageHeight: imageHeight,
            rotationKeypoints: Self.trackingRotationKeypoints, targetAngleRadians: Self.trackingTargetAngleRadians,
            rectScale: Self.trackingRectScale
        )

        return MediaPipeLandmarkProjectionMath.regionBottomLeft(from: rect)
    }
}
