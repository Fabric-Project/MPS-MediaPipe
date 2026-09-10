//
//  MediaPipePoseHeatmapRefinement.swift
//  Fabric
//

import Foundation
import simd

/// Ports mediapipe/calculators/util/refine_landmarks_from_heatmap_calculator.cc
/// exactly: for each landmark, replaces its x,y with the sigmoid-confidence-
/// weighted centroid of a `kernelSize`×`kernelSize` window (clamped at the
/// heatmap edges) centered on the landmark's own already-decoded position in
/// the heatmap's grid, but only when that window's max confidence clears
/// `minConfidenceToRefine` — otherwise the landmark is left untouched.
///
/// `landmarks` and the heatmap are expected in the same normalized-to-crop
/// space (BlazePose: landmarks normalized by the 256x256 crop, heatmap a
/// 64x64 downsample of the same crop — the ratio between the two is exactly
/// what makes a single shared `landmarks[i]*heatmapSize` lookup correct).
///
/// The heatmap array is expected in CHW order (channel-major), NOT the
/// model's own native NHWC -- this is deliberate, not an oversight: every
/// op in MediaPipeMPSGraph.swift operates in NCHW internally, and
/// only outputs that pass through a final RESHAPE get converted back to
/// NHWC before being flattened (RESHAPE's own layout handling does this).
/// The heatmap is a raw CONV_2D result with no trailing reshape, so what
/// MediaPipeMPSGraph.run()/submit() actually hands back for it is
/// genuinely CHW -- confirmed by numeric validation against the PyTorch
/// reference (which itself permutes back to NHWC before returning, so
/// comparing against it directly requires accounting for this). Indexing
/// here matches that reality rather than "fixing" it upstream in the
/// shared interpreter, which every other MediaPipe model also depends on
/// and where this same layout question doesn't otherwise arise (every
/// other model's raw outputs are 1D/2D by the time they're returned).
///
/// `refine_presence`/`refine_visibility` are intentionally not implemented —
/// mediapipe/modules/pose_landmark/tensors_to_pose_landmarks_and_segmentation.pbtxt
/// leaves both at RefineLandmarksFromHeatmapCalculatorOptions' own proto
/// default of `false`; only `kernel_size` is overridden there (to 7, from a
/// default of 9), and `min_confidence_to_refine` is left at its default 0.5
/// (confirmed directly against that calculator's .proto).
///
/// Pure Swift, no CoreML/Metal dependency — independently unit-testable,
/// matching MediaPipeSSDDetectorDecoder.swift's pattern.
public enum MediaPipePoseHeatmapRefinement
{
    public static let defaultKernelSize = 7
    public static let defaultMinConfidenceToRefine: Float = 0.5

    /// `heatmap` is the flattened `[channelCount, heatmapHeight, heatmapWidth]`
    /// (CHW) tensor, row-major — channel `i` is landmark `i`'s confidence map
    /// (see this type's own header for why CHW, not the model's native HWC).
    /// `landmarks.count` must equal `channelCount`.
    public static func refine(
        landmarks: [simd_float2], heatmap: [Float],
        heatmapWidth: Int, heatmapHeight: Int, channelCount: Int,
        kernelSize: Int = defaultKernelSize, minConfidenceToRefine: Float = defaultMinConfidenceToRefine
    ) -> [simd_float2]
    {
        guard landmarks.count == channelCount, heatmap.count == heatmapWidth * heatmapHeight * channelCount else { return landmarks }

        var refined = landmarks
        let offset = (kernelSize - 1) / 2
        let channelStride = heatmapWidth * heatmapHeight

        for index in landmarks.indices
        {
            let centerCol = Int(landmarks[index].x * Float(heatmapWidth))
            let centerRow = Int(landmarks[index].y * Float(heatmapHeight))
            guard centerCol >= 0, centerCol < heatmapWidth, centerRow >= 0, centerRow < heatmapHeight else { continue }

            let beginCol = max(0, centerCol - offset)
            let endCol = min(heatmapWidth, centerCol + offset + 1)
            let beginRow = max(0, centerRow - offset)
            let endRow = min(heatmapHeight, centerRow + offset + 1)

            var sum: Float = 0
            var weightedCol: Float = 0
            var weightedRow: Float = 0
            var maxConfidence: Float = 0

            let channelOffset = index * channelStride
            for row in beginRow..<endRow
            {
                for col in beginCol..<endCol
                {
                    let heatmapIndex = channelOffset + row * heatmapWidth + col
                    let confidence = Float(1.0 / (1.0 + exp(-Double(heatmap[heatmapIndex]))))
                    sum += confidence
                    maxConfidence = max(maxConfidence, confidence)
                    weightedCol += Float(col) * confidence
                    weightedRow += Float(row) * confidence
                }
            }

            if maxConfidence >= minConfidenceToRefine, sum > 0
            {
                refined[index] = simd_float2(weightedCol / Float(heatmapWidth) / sum, weightedRow / Float(heatmapHeight) / sum)
            }
        }

        return refined
    }
}
