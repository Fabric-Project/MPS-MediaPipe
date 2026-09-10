//
//  MediaPipePoseHeatmapRefinement.swift
//  MPSMediaPipe
//

import Foundation
import simd

/// Ports refine_landmarks_from_heatmap_calculator.cc: for each landmark,
/// replaces its x,y with the sigmoid-confidence-weighted centroid of a
/// kernelSize x kernelSize window centered on its heatmap position, but
/// only when the window's max confidence clears `minConfidenceToRefine`.
///
/// `landmarks` and the heatmap must share the same normalized-to-crop
/// space (BlazePose: landmarks normalized by the 256x256 crop, heatmap a
/// 64x64 downsample of it).
///
/// `heatmap` is expected in CHW order, not the model's native NHWC --
/// MediaPipeMPSGraph's raw CONV_2D output (no trailing reshape) is
/// genuinely CHW; indexing here matches that.
///
/// `refine_presence`/`refine_visibility` are not implemented (both default
/// false upstream); only `kernel_size`/`min_confidence_to_refine` apply.
public enum MediaPipePoseHeatmapRefinement
{
    public static let defaultKernelSize = 7
    public static let defaultMinConfidenceToRefine: Float = 0.5

    /// `heatmap` is the flattened [channelCount, heatmapHeight, heatmapWidth]
    /// (CHW) tensor, row-major -- channel i is landmark i's confidence map.
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
