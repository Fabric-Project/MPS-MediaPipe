//
//  MediaPipePoseHeatmapRefinementTests.swift
//  FabricTests
//

import Foundation
import Testing
import simd
@testable import MPSMediaPipe

@Suite("MediaPipe Pose Heatmap Refinement")
struct MediaPipePoseHeatmapRefinementTests
{
    /// Builds a single-channel (channelCount=1) heatmap with one confident
    /// logit spike at (spikeCol, spikeRow) and near-zero everywhere else,
    /// pre-sigmoid (large negative -> ~0 confidence, large positive -> ~1).
    private static func singleChannelHeatmap(width: Int, height: Int, spikeCol: Int, spikeRow: Int, spikeLogit: Float, backgroundLogit: Float) -> [Float]
    {
        var heatmap = [Float](repeating: backgroundLogit, count: width * height)
        heatmap[spikeRow * width + spikeCol] = spikeLogit
        return heatmap
    }

    @Test("A confident nearby peak pulls the landmark toward it")
    func confidentPeakMovesLandmark() throws
    {
        let width = 16, height = 16
        // Landmark initially decoded at grid (4,4); true peak one cell over at (6,4).
        let landmark = simd_float2(4.0 / Float(width), 4.0 / Float(height))
        let heatmap = Self.singleChannelHeatmap(width: width, height: height, spikeCol: 6, spikeRow: 4, spikeLogit: 10, backgroundLogit: -10)

        let refined = MediaPipePoseHeatmapRefinement.refine(
            landmarks: [landmark], heatmap: heatmap,
            heatmapWidth: width, heatmapHeight: height, channelCount: 1,
            kernelSize: 7, minConfidenceToRefine: 0.5
        )

        let refinedCol = refined[0].x * Float(width)
        #expect(refinedCol > 4.5) // moved toward the peak at col 6
        #expect(abs(refined[0].y * Float(height) - 4.0) < 0.5) // row barely moves, peak is on the same row
    }

    @Test("A low-confidence window leaves the landmark unchanged")
    func lowConfidenceLeavesLandmarkUnchanged() throws
    {
        let width = 16, height = 16
        let landmark = simd_float2(4.0 / Float(width), 4.0 / Float(height))
        // Every logit well below sigmoid^-1(0.5)=0, so max confidence < 0.5.
        let heatmap = [Float](repeating: -5, count: width * height)

        let refined = MediaPipePoseHeatmapRefinement.refine(
            landmarks: [landmark], heatmap: heatmap,
            heatmapWidth: width, heatmapHeight: height, channelCount: 1,
            kernelSize: 7, minConfidenceToRefine: 0.5
        )

        #expect(refined[0] == landmark)
    }

    @Test("A landmark outside the heatmap bounds is left unchanged")
    func outOfBoundsLandmarkUnchanged() throws
    {
        let width = 16, height = 16
        let landmark = simd_float2(1.5, -0.2) // outside [0,1)
        let heatmap = [Float](repeating: 10, count: width * height)

        let refined = MediaPipePoseHeatmapRefinement.refine(
            landmarks: [landmark], heatmap: heatmap,
            heatmapWidth: width, heatmapHeight: height, channelCount: 1,
            kernelSize: 7, minConfidenceToRefine: 0.5
        )

        #expect(refined[0] == landmark)
    }

    @Test("Mismatched channel count returns the input unchanged")
    func mismatchedChannelCountReturnsInput() throws
    {
        let landmarks = [simd_float2(0.5, 0.5), simd_float2(0.25, 0.25)]
        let heatmap = [Float](repeating: 1, count: 16 * 16 * 1) // channelCount=1, but 2 landmarks
        let refined = MediaPipePoseHeatmapRefinement.refine(landmarks: landmarks, heatmap: heatmap, heatmapWidth: 16, heatmapHeight: 16, channelCount: 1)
        #expect(refined == landmarks)
    }
}
