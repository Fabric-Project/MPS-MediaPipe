//
//  MediaPipePoseLandmarkProjection.swift
//  Fabric
//

import Foundation
import simd

/// Decodes BlazePose's raw landmark tensor (39 points x 5 values: x, y, z,
/// visibility, presence — confirmed against mediapipe/calculators/tensor/
/// tensors_to_landmarks_calculator.cc's own per-point channel order, and
/// against the real bundled model's own output shape, [1,195] = 39*5) and
/// projects screen-space landmarks back through the rotated crop rect into
/// full-image normalized coordinates — same rotation/rect math as
/// MediaPipeFaceLandmarkProjection/MediaPipeHandLandmarkProjection, ported
/// from mediapipe/modules/pose_landmark/
/// tensors_to_pose_landmarks_and_segmentation.pbtxt's
/// TensorsToLandmarksCalculatorOptions (num_landmarks: 39,
/// input_image_width/height: 256, visibility/presence_activation: SIGMOID,
/// normalize_z left at its proto default of 1.0, matching the base FaceMesh
/// model's own default).
///
/// `landmarksRaw` is expected to already have its x,y refined by
/// MediaPipePoseHeatmapRefinement — that step runs on all 39 raw points
/// BEFORE this centering/rotation math, matching
/// tensors_to_pose_landmarks_and_segmentation.pbtxt's own calculator order
/// (TensorsToLandmarksCalculator -> RefineLandmarksFromHeatmapCalculator ->
/// SplitNormalizedLandmarkListCalculator). This type has no heatmap
/// dependency itself and doesn't call that step — the caller (
/// MediaPipePoseLandmarkNode) owns that ordering, keeping the two concerns
/// (heatmap refinement, rotation/rect projection) independently testable.
///
/// The first 33 of the 39 decoded points are the real pose landmarks
/// (SplitNormalizedLandmarkListCalculator's `ranges: {0,33}`); indices 33-34
/// are a separate, dedicated 2-point "auxiliary_landmarks" stream
/// (`ranges: {33,35}`) MediaPipe's own pose_landmarks_to_roi.pbtxt uses --
/// via AlignmentPointsRectsCalculator's rotation_vector_start/end_keypoint_index
/// 0/1 -- to derive next frame's tracking ROI. Those indices are 0/1 *of
/// the 2-point auxiliary_landmarks stream*, not of the 33-point pose
/// landmarks -- confirmed against the real pbtxt's own stream wiring
/// (`LandmarksToDetectionCalculator` takes `NORM_LANDMARKS:landmarks` fed
/// from the graph's own `LANDMARKS` input, which pose_landmark_gpu.pbtxt
/// binds to `auxiliary_landmarks`, not the main `pose_landmarks`) --
/// indices 33/34 are NOT nose/eye. Both are returned here (auxiliaryLandmarks
/// separate from the main 33 in `landmarks`), decoded through the identical
/// rotation/rect math, so a caller deriving a tracking ROI uses the correct
/// points rather than substituting two of the 33 main landmarks.
public enum MediaPipePoseLandmarkProjection
{
    /// Speed/accuracy tier for the bundled BlazePose landmark model —
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

    /// mediapipe/modules/pose_landmark/pose_landmarks_to_roi.pbtxt's own
    /// AlignmentPointsRectsCalculator + RectTransformationCalculator config
    /// -- indices 0 and 1 *of the 2-point auxiliary_landmarks stream*
    /// (this type's own Pose.auxiliaryLandmarks, decoded indices 33-34 of
    /// the raw 39-point tensor), confirmed against the real pbtxt's own
    /// stream wiring (pose_landmark_gpu.pbtxt binds this subgraph's
    /// LANDMARKS input to auxiliary_landmarks, not the main pose_landmarks)
    /// -- NOT indices 0/1 of the 33-point pose landmarks (nose/left-eye),
    /// which an earlier version of the caller used by mistake and which
    /// produces a tiny, wrong ROI. Also NOT the detector's own keypoint 0/1
    /// (mid-hip / full-body size point) -- three different arrays, same
    /// numeric indices by coincidence. Same target angle/scale as
    /// MediaPipePoseDetector's own ROI derivation, confirmed independently
    /// rather than assumed equal.
    private static let trackingRotationKeypoints = (start: 0, end: 1)
    private static let trackingTargetAngleRadians: Float = .pi / 2
    private static let trackingRectScale: Float = 1.25

    public struct Pose
    {
        /// x, y normalized full-image [0,1], top-left origin; z is relative
        /// (same units as x, scaled by the crop rect's width).
        public var landmarks: [simd_float3]
        /// The 2 dedicated ROI-derivation points (decoded indices 33-34) --
        /// see this enum's own header for why these, not landmarks[0]/[1],
        /// are what a tracking ROI must be derived from.
        public var auxiliaryLandmarks: [simd_float3]
    }

    /// `landmarksRaw` is the flat 195-float (39x5) output (x,y already
    /// heatmap-refined by the caller), `presenceRaw` the single raw
    /// (pre-sigmoid) pose-presence logit (mediapipe's "pose flag" tensor).
    /// Returns nil when presence is below threshold, matching the official
    /// graph's ThresholdingCalculator dropping the pose entirely.
    public static func project(
        landmarksRaw: [Float], presenceRaw: Float,
        rect: (cx: Float, cy: Float, width: Float, height: Float, rotation: Float)
    ) -> Pose?
    {
        let presence = Float(1.0 / (1.0 + exp(-Double(presenceRaw))))
        guard presence > minPosePresenceConfidence else { return nil }

        let sinA = sin(rect.rotation)
        let cosA = cos(rect.rotation)

        func projectPoint(_ index: Int) -> simd_float3
        {
            let base = index * 5
            let x = landmarksRaw[base + 0] / landmarkSize - 0.5
            let y = landmarksRaw[base + 1] / landmarkSize - 0.5
            let z = landmarksRaw[base + 2] / landmarkSize / normalizeZ

            let rotatedX = cosA * x - sinA * y
            let rotatedY = sinA * x + cosA * y

            return simd_float3(
                rotatedX * rect.width + rect.cx,
                rotatedY * rect.height + rect.cy,
                z * rect.width
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

    /// Re-derives a tracking ROI from this frame's own (unsmoothed)
    /// auxiliary landmarks, matching mediapipe's own PreviousLoopbackCalculator-
    /// fed tracking path -- nil whenever there aren't at least 2 points to
    /// derive a rotation from, so callers can send nil downstream rather
    /// than propagate a stale region. `auxiliaryLandmarks` are bottom-left-
    /// origin normalized (matching a caller's own decoded-landmark output
    /// space, e.g. MediaPipePoseLandmarkNode's); this function flips
    /// internally to top-left for alignmentPointsRect, then flips the
    /// result back, so both `auxiliaryLandmarks` in and `region` out are
    /// bottom-left-origin.
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

        let regionBottomLeft = simd_float4(
            rect.cx - rect.width / 2,
            1 - (rect.cy - rect.height / 2) - rect.height,
            rect.width,
            rect.height
        )
        return (region: regionBottomLeft, rotation: rect.rotation)
    }
}
