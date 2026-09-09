//
//  FaceGeometrySolverTests.swift
//  FabricTests
//

import Foundation
import Testing
import simd
@testable import MPSMediaPipe

@Suite("Face Geometry Solver")
struct FaceGeometrySolverTests
{
    private static let frameWidth: Float = 1280
    private static let frameHeight: Float = 720

    @Test("Canonical model resource loads with the expected mediapipe-confirmed sizes")
    func canonicalModelSizesMatchMediapipe() throws
    {
        #expect(FaceGeometrySolver.canonicalPositions.count == 468)
        #expect(FaceGeometrySolver.canonicalUVs.count == 468)
        #expect(FaceGeometrySolver.canonicalTriangles.count == 898)
    }

    @Test("Weighted orthogonal Procrustes recovers a known rotation + uniform scale + translation")
    func procrustesRecoversKnownTransform() throws
    {
        let axis = simd_normalize(simd_float3(0.2, 1.0, 0.3))
        let knownRotation = simd_float3x3(simd_quatf(angle: 0.7, axis: axis))
        let knownScale: Float = 2.3
        let knownTranslation = simd_float3(4, -2.5, 10)

        let source = FaceGeometrySolver.canonicalPositions
        let weights = source.indices.map { $0 % 7 == 0 ? Float(1) : Float(0) } // sparse weights, mirroring the real 33-of-468 basis shape
        let target = source.map { knownScale * (knownRotation * $0) + knownTranslation }

        let transform = try #require(FaceGeometrySolver.solveWeightedOrthogonalProblem(source: source, target: target, weights: weights))

        let recoveredTranslation = simd_float3(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
        let recoveredScale = simd_length(simd_float3(transform.columns.0.x, transform.columns.0.y, transform.columns.0.z))

        #expect(simd_distance(recoveredTranslation, knownTranslation) < 1e-2)
        #expect(abs(recoveredScale - knownScale) < 1e-2)

        // Reconstructed rotation should be a proper rotation (det ~= +1, no reflection).
        let rotationAndScale = simd_float3x3(
            simd_float3(transform.columns.0.x, transform.columns.0.y, transform.columns.0.z),
            simd_float3(transform.columns.1.x, transform.columns.1.y, transform.columns.1.z),
            simd_float3(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z)
        )
        #expect(rotationAndScale.determinant > 0)

        for index in source.indices where weights[index] > 0
        {
            let mapped = transform * simd_float4(source[index], 1)
            #expect(simd_distance(simd_float3(mapped.x, mapped.y, mapped.z), target[index]) < 1e-1)
        }
    }

    @Test("Mismatched array counts are rejected")
    func mismatchedCountsReturnNil() throws
    {
        let transform = FaceGeometrySolver.solveWeightedOrthogonalProblem(
            source: [simd_float3(0, 0, 0)], target: [simd_float3(0, 0, 0), simd_float3(1, 1, 1)], weights: [1]
        )
        #expect(transform == nil)
    }

    @Test("DIAGNOSTIC: full solve() recovers a chosen ground-truth camera distance")
    func diagnosticFullSolveDistance() throws
    {
        let near: Float = 1.0
        let canonical = FaceGeometrySolver.canonicalPositions
        let meanCanonicalZ = canonical.reduce(Float(0)) { $0 + $1.z } / Float(canonical.count)
        let groundTruthDistance: Float = 50.0 // cm, plausible webcam distance

        // Self-consistent forward construction: place the canonical model
        // (identity rotation, scale 1) at exactly `groundTruthDistance` in
        // front of the camera, then invert solve()'s own projectXY ->
        // moveAndRescaleZ -> unprojectXY -> changeHandedness chain
        // algebraically to recover the synthetic screen-space input that
        // would produce it. `scale` here is NOT free -- self-consistency
        // (mean(p1.z) reproducing whatever depthOffset solve() will itself
        // compute) forces scale = near / (groundTruthDistance - meanCanonicalZ).
        let scale = near / (groundTruthDistance - meanCanonicalZ)
        let depthOffset: Float = 0

        let degreesToRadians = Float.pi / 180
        let heightAtNear = 2 * near * tan(0.5 * degreesToRadians * FaceGeometrySolver.verticalFieldOfViewDegrees)
        let widthAtNear = Self.frameWidth * heightAtNear / Self.frameHeight
        let left = -0.5 * widthAtNear, right = 0.5 * widthAtNear
        let bottom = -0.5 * heightAtNear, top = 0.5 * heightAtNear
        let xScale = right - left
        let yScale = top - bottom

        let screenLandmarks = canonical.map { point -> simd_float3 in
            let metricZ = point.z - groundTruthDistance // identity rotation, scale 1, translation (0,0,-D)
            let p2z = -metricZ // inverse of changeHandedness
            let p2x = point.x * near / p2z // inverse of unprojectXY
            let p2y = point.y * near / p2z
            let p1z = depthOffset - near + scale * p2z // inverse of moveAndRescaleZ

            let screenX = (p2x - left) / xScale // inverse of projectXY
            let screenY = (p2y - bottom) / yScale
            let screenZ = p1z / xScale
            return simd_float3(screenX, screenY, screenZ)
        }

        let result = try #require(FaceGeometrySolver.solve(screenLandmarks: screenLandmarks, frameWidth: Self.frameWidth, frameHeight: Self.frameHeight))
        let recoveredDistance = -result.poseTransform.columns.3.z
        let recoveredScale = simd_length(simd_float3(result.poseTransform.columns.0.x, result.poseTransform.columns.0.y, result.poseTransform.columns.0.z))
        print("DIAGNOSTIC ground truth distance:", groundTruthDistance, "recovered:", recoveredDistance, "recovered dimensionless scale:", recoveredScale)
        #expect(abs(recoveredDistance - groundTruthDistance) < 1.0)
        #expect(abs(recoveredScale - 1.0) < 0.05)
    }

    @Test("solve() rejects a too-compact (degenerate) screen landmark spread")
    func solveRejectsCompactSpread() throws
    {
        // All 468 points crammed into a tiny screen region -- mirrors
        // IsScreenLandmarkListTooCompact's own rejection case.
        let compact = (0..<FaceGeometrySolver.landmarkCount).map { _ in simd_float3(0.5, 0.5, 0) }
        let result = FaceGeometrySolver.solve(screenLandmarks: compact, frameWidth: Self.frameWidth, frameHeight: Self.frameHeight)
        #expect(result == nil)
    }

    @Test("solve() rejects a landmark count that doesn't match the canonical model")
    func solveRejectsWrongLandmarkCount() throws
    {
        let result = FaceGeometrySolver.solve(screenLandmarks: [simd_float3(0.5, 0.5, 0)], frameWidth: Self.frameWidth, frameHeight: Self.frameHeight)
        #expect(result == nil)
    }
}
