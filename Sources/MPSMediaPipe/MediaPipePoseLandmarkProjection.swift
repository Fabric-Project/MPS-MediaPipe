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
    public static let landmarkSize: Float = 256
    static let normalizeZ: Float = 1.0
    static let minPosePresenceConfidence: Float = 0.5
    public static let decodedLandmarkCount = 39
    static let poseLandmarkCount = 33
    static let auxiliaryLandmarkCount = 2

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
}
