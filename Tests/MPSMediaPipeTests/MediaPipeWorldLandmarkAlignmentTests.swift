import Foundation
import simd
import Testing
@testable import MPSMediaPipe

/// A known placement must be recovered: world landmarks are put at metric
/// size at a known position and viewed through a camera to make the image
/// points; aligning the same world landmarks to those image points must put
/// them back.
@Test func worldLandmarkAlignmentRecoversPositionAndDepth() throws
{
    let cameraDistance: Float = 2.4142137
    let sceneUnitsPerMeter: Float = 2.4142137
    // A non-planar shape in MediaPipe axes (y down, z negative toward camera), meters.
    let worldLandmarks: [simd_float3] = [
        simd_float3(0.00, 0.00, 0.00),
        simd_float3(0.03, -0.02, -0.01),
        simd_float3(0.05, -0.06, -0.03),
        simd_float3(-0.02, -0.07, 0.02),
        simd_float3(-0.04, 0.03, 0.04),
        simd_float3(0.01, 0.05, -0.02),
    ]
    // Nearer than the image plane (positive z), off-center.
    for shift in [simd_float3(0.35, -0.2, 1.1), simd_float3(-0.5, 0.3, -0.8)]
    {
        let placed = worldLandmarks.map { simd_float3($0.x, -$0.y, -$0.z) * sceneUnitsPerMeter + shift }
        let imagePlanePoints = placed.map { simd_float2($0.x, $0.y) * cameraDistance / (cameraDistance - $0.z) }

        let aligned = try #require(MediaPipeWorldLandmarkAlignment.align(
            worldLandmarks: worldLandmarks,
            imagePlanePoints: imagePlanePoints,
            cameraDistance: cameraDistance,
            sceneUnitsPerMeter: sceneUnitsPerMeter
        ))
        #expect(aligned.count == worldLandmarks.count)
        for index in aligned.indices
        {
            let reprojected = simd_float2(aligned[index].x, aligned[index].y) * cameraDistance / (cameraDistance - aligned[index].z)
            #expect(simd_distance(reprojected, imagePlanePoints[index]) < 1e-3)
            #expect(simd_distance(aligned[index], placed[index]) < 1e-2)
        }
    }
}

/// The same shape seen larger in the image must be placed nearer the camera.
@Test func worldLandmarkAlignmentPlacesLargerSubjectsNearer() throws
{
    let cameraDistance: Float = 2.4142137
    let worldLandmarks: [simd_float3] = [
        simd_float3(0.00, 0.00, 0.00),
        simd_float3(0.05, -0.02, -0.01),
        simd_float3(-0.04, -0.06, 0.02),
        simd_float3(0.02, 0.07, -0.02),
    ]
    func meanDepth(imageScale: Float) throws -> Float
    {
        let imagePlanePoints = worldLandmarks.map { simd_float2($0.x, -$0.y) * imageScale }
        let aligned = try #require(MediaPipeWorldLandmarkAlignment.align(
            worldLandmarks: worldLandmarks,
            imagePlanePoints: imagePlanePoints,
            cameraDistance: cameraDistance,
            sceneUnitsPerMeter: cameraDistance
        ))
        return aligned.map(\.z).reduce(0, +) / Float(aligned.count)
    }
    #expect(try meanDepth(imageScale: 6) > meanDepth(imageScale: 3))
}

@Test func worldLandmarkAlignmentRejectsUnusableInput()
{
    let points = [simd_float3(0, 0, 0), simd_float3(0.1, 0, 0)]
    // Mismatched counts.
    #expect(MediaPipeWorldLandmarkAlignment.align(worldLandmarks: points, imagePlanePoints: [simd_float2(0, 0)], cameraDistance: 2, sceneUnitsPerMeter: 2) == nil)
    // Identical image points fix no depth.
    #expect(MediaPipeWorldLandmarkAlignment.align(worldLandmarks: points, imagePlanePoints: [simd_float2(0, 0), simd_float2(0, 0)], cameraDistance: 2, sceneUnitsPerMeter: 2) == nil)
    // All-zero weights.
    #expect(MediaPipeWorldLandmarkAlignment.align(worldLandmarks: points, imagePlanePoints: [simd_float2(0, 0), simd_float2(1, 0)], weights: [0, 0], cameraDistance: 2, sceneUnitsPerMeter: 2) == nil)
}
