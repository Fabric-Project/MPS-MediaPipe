//
//  MediaPipeFaceLandmarkProjection.swift
//  Fabric
//

import Foundation
import simd

/// Decodes FaceMesh's raw outputs (1404 floats = 468 x,y,z; 1 face-presence
/// flag) and projects screen-space landmarks back through the rotated crop
/// rect into full-image normalized coordinates. Ported from mediapipe's
/// actual graph config (mediapipe/modules/face_landmark/
/// face_landmark_cpu.pbtxt, tensors_to_face_landmarks.pbtxt) — no local
/// third-party reference exists for this model (unlike
/// MediaPipeHandLandmarkProjection, which fasthands.pipeline already
/// validated), so this was derived directly from the calculator source and
/// sanity-checked with hand-computed cases, not validated end-to-end
/// against a real detected face.
///
/// Differs from the hand landmark model's decode in three confirmed ways:
/// `normalize_z` defaults to 1.0 here (TensorsToLandmarksCalculatorOptions'
/// own proto default — the hand model's config explicitly overrides it to
/// 0.4, FaceMesh's does not), presence needs an explicit sigmoid
/// (TensorsToFloatsCalculatorOptions.activation: SIGMOID in the graph,
/// applied outside the model itself — the hand port's third-party reference
/// reads hand presence raw, unsigmoided, which this deliberately does not
/// copy since there's no equivalent validated reference for face to check
/// that assumption against), and there is no handedness or world-landmark
/// equivalent at all for the base (non-attention) FaceMesh variant.
public enum MediaPipeFaceLandmarkProjection
{
    public static let landmarkSize: Float = 192
    public static let resourcePrefix = "MediaPipeFaceLandmarks"
    static let normalizeZ: Float = 1.0
    public static let minFacePresenceConfidence: Float = 0.5
    static let landmarkCount = 468

    /// mediapipe/modules/face_landmark/face_landmark_landmarks_to_roi.pbtxt's
    /// own DetectionsToRectsCalculator + RectTransformationCalculator config
    /// -- mesh indices 33 (left eye, inner corner) and 263 (right eye, outer
    /// corner), target angle 0, scale 1.5x1.5, no shift. Box-based (like the
    /// detector's own rect()), not alignment-point-based -- confirmed
    /// directly against the real pbtxt, not assumed equal to the detector's
    /// own config (which uses different rotation keypoints: raw detector
    /// keypoints 0/1, the two eyes, not mesh indices 33/263).
    private static let trackingRotationKeypoints = (start: 33, end: 263)
    private static let trackingTargetAngleRadians: Float = 0.0
    private static let trackingRectScale: Float = 1.5

    public struct Face
    {
        /// x, y normalized full-image [0,1], top-left origin; z is relative
        /// (same units as x, scaled by the crop rect's width).
        public var landmarks: [simd_float3]
    }

    /// `landmarksRaw` is the flat 1404-float (468x3) output, `presenceRaw`
    /// the single raw (pre-sigmoid) face-presence logit. Returns nil when
    /// presence is below threshold (ThresholdingCalculator) — matches the
    /// official graph dropping the face entirely rather than emitting a
    /// low-confidence result.
    public static func project(
        landmarksRaw: [Float], presenceRaw: Float,
        rect: (cx: Float, cy: Float, width: Float, height: Float, rotation: Float)
    ) -> Face?
    {
        let presence = Float(1.0 / (1.0 + exp(-Double(presenceRaw))))
        guard presence > minFacePresenceConfidence else { return nil }

        let sinA = sin(rect.rotation)
        let cosA = cos(rect.rotation)

        var landmarks: [simd_float3] = []
        landmarks.reserveCapacity(landmarkCount)
        for index in 0..<landmarkCount
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

        return Face(landmarks: landmarks)
    }

    /// Re-derives a tracking ROI from this frame's own (unsmoothed)
    /// landmarks, the same way MediaPipe's face_landmark_landmarks_to_roi.pbtxt
    /// does: a box enclosing all 468 landmarks (LandmarksToDetectionCalculator),
    /// rotated using mesh indices 33/263 (DetectionsToRectsCalculator).
    /// `landmarks` are bottom-left-origin normalized (matching this type's
    /// own `project(...)` output once a caller has flipped it to that
    /// convention, e.g. MediaPipeFaceLandmarkNode's own landmark space) --
    /// this function flips internally to top-left for the rect math, then
    /// flips the result back, so both `landmarks` in and `region` out are
    /// bottom-left-origin.
    public static func trackedRegion(from landmarks: [simd_float3], presentationSize: CGSize) -> (region: simd_float4, rotation: Float)?
    {
        guard landmarks.count > max(Self.trackingRotationKeypoints.start, Self.trackingRotationKeypoints.end) else { return nil }

        let imageWidth = Float(presentationSize.width)
        let imageHeight = Float(presentationSize.height)

        let xs = landmarks.map(\.x)
        let topLeftYs = landmarks.map { 1 - $0.y }
        let xmin = xs.min()!, xmax = xs.max()!
        let ymin = topLeftYs.min()!, ymax = topLeftYs.max()!

        let startLandmark = landmarks[Self.trackingRotationKeypoints.start]
        let endLandmark = landmarks[Self.trackingRotationKeypoints.end]

        let projected = MediaPipeSSDRectTransform.ProjectedDetection(
            xmin: xmin, ymin: ymin, width: xmax - xmin, height: ymax - ymin,
            keypoints: [
                (x: startLandmark.x, y: 1 - startLandmark.y),
                (x: endLandmark.x, y: 1 - endLandmark.y),
            ],
            score: 1
        )

        let rect = MediaPipeSSDRectTransform.rect(
            from: projected, imageWidth: imageWidth, imageHeight: imageHeight,
            rotationKeypoints: (start: 0, end: 1), targetAngleRadians: Self.trackingTargetAngleRadians,
            rectScale: Self.trackingRectScale
        )

        let regionBottomLeft = simd_float4(
            rect.cx - rect.width / 2,
            1 - (rect.cy - rect.height / 2) - rect.height,
            rect.width,
            rect.height
        )
        return (region: regionBottomLeft, rotation: rect.rotation)
    }
}
