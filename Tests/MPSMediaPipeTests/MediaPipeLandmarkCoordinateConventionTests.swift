//
//  MediaPipeLandmarkCoordinateConventionTests.swift
//  MPSMediaPipeTests
//

import Foundation
import Testing
import simd
@testable import MPSMediaPipe

/// Face/Hand/Pose landmark projection each decode a differently-shaped raw
/// tensor, but all three route through the same
/// MediaPipeLandmarkProjectionMath.rotateAndProject -- these tests exercise
/// each concrete `project(...)` entry point (not just the shared helper in
/// isolation) to confirm the three models actually agree, end to end, on:
/// origin/scale of the output space, how rotation is applied (x/y only),
/// and how z is scaled (by the rect's width, via each model's own
/// mediapipe-matching `normalizeZ` divisor -- see each type's own file for
/// the citation against mediapipe's real per-model calculator configs).
@Suite("MediaPipe Landmark Coordinate Convention")
struct MediaPipeLandmarkCoordinateConventionTests
{
    private static let identityRect = (cx: Float(0.5), cy: Float(0.5), width: Float(1), height: Float(1), rotation: Float(0))

    // MARK: - A raw point at the model's own center, zero depth, through an
    // identity rect must land at the rect's center with zero depth -- for
    // all three models alike. project()'s own doc comments say the output
    // is top-left-origin; this only checks that boundary, not the
    // additional bottom-left flip each Fabric node applies afterward.

    @Test("Face: model-center point with zero depth lands at rect center")
    func faceCenterMapsToRectCenter() throws
    {
        var raw = [Float](repeating: 0, count: MediaPipeFaceLandmarkProjection.landmarkCount * 3)
        raw[0] = MediaPipeFaceLandmarkProjection.landmarkSize / 2
        raw[1] = MediaPipeFaceLandmarkProjection.landmarkSize / 2
        raw[2] = 0

        let face = try #require(MediaPipeFaceLandmarkProjection.project(landmarksRaw: raw, presenceRaw: 10, rect: Self.identityRect))
        #expect(abs(face.landmarks[0].x - 0.5) < 1e-5)
        #expect(abs(face.landmarks[0].y - 0.5) < 1e-5)
        #expect(abs(face.landmarks[0].z - 0) < 1e-5)
    }

    @Test("Hand: model-center point with zero depth lands at rect center")
    func handCenterMapsToRectCenter() throws
    {
        var raw = [Float](repeating: 0, count: 21 * 3)
        raw[0] = MediaPipeHandLandmarkProjection.landmarkSize / 2
        raw[1] = MediaPipeHandLandmarkProjection.landmarkSize / 2
        raw[2] = 0
        let world = [Float](repeating: 0, count: 21 * 3)

        let hand = try #require(MediaPipeHandLandmarkProjection.project(landmarksRaw: raw, worldLandmarksRaw: world, presence: 1, handednessRaw: 1, rect: Self.identityRect))
        #expect(abs(hand.landmarks[0].x - 0.5) < 1e-5)
        #expect(abs(hand.landmarks[0].y - 0.5) < 1e-5)
        #expect(abs(hand.landmarks[0].z - 0) < 1e-5)
    }

    @Test("Pose: model-center point with zero depth lands at rect center")
    func poseCenterMapsToRectCenter() throws
    {
        var raw = [Float](repeating: 0, count: 39 * 5)
        raw[0] = MediaPipePoseLandmarkProjection.landmarkSize / 2
        raw[1] = MediaPipePoseLandmarkProjection.landmarkSize / 2
        raw[2] = 0

        let pose = try #require(MediaPipePoseLandmarkProjection.project(landmarksRaw: raw, presenceRaw: 10, rect: Self.identityRect))
        #expect(abs(pose.landmarks[0].x - 0.5) < 1e-5)
        #expect(abs(pose.landmarks[0].y - 0.5) < 1e-5)
        #expect(abs(pose.landmarks[0].z - 0) < 1e-5)
    }

    // MARK: - Z is scaled by the rect's width only (matching mediapipe's
    // real LandmarkProjectionCalculator: `new_z = landmark.z() * rect.width()`,
    // never by height or by any rotation), through each model's own
    // normalizeZ divisor. Raw z is chosen per model so every model targets
    // the same pre-scale normalized depth (0.1) -- proving the three models
    // agree on the *formula* even though their raw-to-normalized divisors
    // (1.0 for Face/Pose, 0.4 for Hand, both matching mediapipe's real
    // per-model calculator configs) intentionally differ.

    private static let nonSquareRect = (cx: Float(0.5), cy: Float(0.5), width: Float(0.4), height: Float(0.8), rotation: Float(0))
    private static let targetNormalizedZ: Float = 0.1
    private static let expectedProjectedZ: Float = Self.targetNormalizedZ * Self.nonSquareRect.width // 0.04, width-only

    @Test("Face: z scales by rect width only, via Face's own normalizeZ")
    func faceZScalesByRectWidth() throws
    {
        var raw = [Float](repeating: 0, count: MediaPipeFaceLandmarkProjection.landmarkCount * 3)
        raw[2] = Self.targetNormalizedZ * MediaPipeFaceLandmarkProjection.landmarkSize

        let face = try #require(MediaPipeFaceLandmarkProjection.project(landmarksRaw: raw, presenceRaw: 10, rect: Self.nonSquareRect))
        #expect(abs(face.landmarks[0].z - Self.expectedProjectedZ) < 1e-4)
    }

    @Test("Hand: z scales by rect width only, via Hand's own normalizeZ")
    func handZScalesByRectWidth() throws
    {
        var raw = [Float](repeating: 0, count: 21 * 3)
        raw[2] = Self.targetNormalizedZ * MediaPipeHandLandmarkProjection.landmarkSize * MediaPipeHandLandmarkProjection.normalizeZ
        let world = [Float](repeating: 0, count: 21 * 3)

        let hand = try #require(MediaPipeHandLandmarkProjection.project(landmarksRaw: raw, worldLandmarksRaw: world, presence: 1, handednessRaw: 1, rect: Self.nonSquareRect))
        #expect(abs(hand.landmarks[0].z - Self.expectedProjectedZ) < 1e-4)
    }

    @Test("Pose: z scales by rect width only, via Pose's own normalizeZ")
    func poseZScalesByRectWidth() throws
    {
        var raw = [Float](repeating: 0, count: 39 * 5)
        raw[2] = Self.targetNormalizedZ * MediaPipePoseLandmarkProjection.landmarkSize

        let pose = try #require(MediaPipePoseLandmarkProjection.project(landmarksRaw: raw, presenceRaw: 10, rect: Self.nonSquareRect))
        #expect(abs(pose.landmarks[0].z - Self.expectedProjectedZ) < 1e-4)
    }

    // MARK: - In-plane rotation must rotate x/y only, never z -- depth is
    // orthogonal to a 2D crop rotation. Checked directly against the shared
    // helper all three models route through, since it's the one place this
    // is decided.

    @Test("rotateAndProject: rotation changes x/y but leaves z untouched")
    func rotationLeavesZUntouched() throws
    {
        let unrotated = MediaPipeLandmarkProjectionMath.rotateAndProject(
            x: 96, y: 64, z: 19.2, landmarkSize: 192, normalizeZ: 1.0,
            sinRotation: 0, cosRotation: 1,
            rect: (cx: 0.5, cy: 0.5, width: 0.4, height: 0.8)
        )
        let rotated = MediaPipeLandmarkProjectionMath.rotateAndProject(
            x: 96, y: 64, z: 19.2, landmarkSize: 192, normalizeZ: 1.0,
            sinRotation: sin(Float.pi / 2), cosRotation: cos(Float.pi / 2),
            rect: (cx: 0.5, cy: 0.5, width: 0.4, height: 0.8)
        )

        #expect(abs(unrotated.z - rotated.z) < 1e-5)
        #expect(abs(unrotated.x - rotated.x) > 1e-3 || abs(unrotated.y - rotated.y) > 1e-3)
    }
}
