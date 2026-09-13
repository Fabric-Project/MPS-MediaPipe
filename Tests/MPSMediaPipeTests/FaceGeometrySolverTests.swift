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

    /// Same self-consistent forward construction as
    /// `diagnosticFullSolveDistance`, but the canonical model is rotated
    /// before translation. `normalizedLandmarks` (the tuple's
    /// `metricLandmarks`) is supposed to be pose-normalized -- head
    /// rotation/translation fully removed, expression only -- so it should
    /// reproduce `canonical` regardless of `rotation`, up to the same
    /// tolerance the identity-rotation case achieves. A reported bug is that
    /// this reconstruction visibly distorts under head yaw/pitch and
    /// distance changes; this test isolates whether that's true of
    /// `solve()` itself, independent of anything Fabric-side.
    @Test("DIAGNOSTIC: rotated head pose leaks into normalized (pose-removed) landmarks", arguments: [
        ("yaw 25deg", simd_quatf(angle: 25 * Float.pi / 180, axis: simd_float3(0, 1, 0))),
        ("pitch 20deg", simd_quatf(angle: 20 * Float.pi / 180, axis: simd_float3(1, 0, 0))),
        ("yaw+pitch", simd_quatf(angle: 25 * Float.pi / 180, axis: simd_normalize(simd_float3(1, 1, 0)))),
    ])
    func diagnosticRotatedPoseLeaksIntoNormalizedLandmarks(label: String, quaternion: simd_quatf) throws
    {
        let near: Float = 1.0
        let canonical = FaceGeometrySolver.canonicalPositions
        let meanCanonicalZ = canonical.reduce(Float(0)) { $0 + $1.z } / Float(canonical.count)
        let groundTruthDistance: Float = 50.0 // cm, same baseline as the identity-rotation diagnostic
        let rotation = simd_float3x3(quaternion)

        // Same scale/depthOffset formula as the identity-rotation diagnostic
        // (an approximation once rotation is introduced -- solve()'s own
        // two-pass refinement is specifically meant to correct for exactly
        // this kind of imperfect initial scale, so a reasonable but inexact
        // guess here should still converge if solve() itself is correct).
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
            let worldPoint = rotation * point // known head rotation, applied before translation
            let metricZ = worldPoint.z - groundTruthDistance
            let p2z = -metricZ
            let p2x = worldPoint.x * near / p2z
            let p2y = worldPoint.y * near / p2z
            let p1z = depthOffset - near + scale * p2z

            let screenX = (p2x - left) / xScale
            let screenY = (p2y - bottom) / yScale
            let screenZ = p1z / xScale
            return simd_float3(screenX, screenY, screenZ)
        }

        let result = try #require(FaceGeometrySolver.solve(screenLandmarks: screenLandmarks, frameWidth: Self.frameWidth, frameHeight: Self.frameHeight))

        var maxNormalizedError: Float = 0
        var sumNormalizedError: Float = 0
        for (canonicalPoint, normalizedPoint) in zip(canonical, result.metricLandmarks)
        {
            let error = simd_distance(canonicalPoint, normalizedPoint)
            maxNormalizedError = max(maxNormalizedError, error)
            sumNormalizedError += error
        }
        let meanNormalizedError = sumNormalizedError / Float(canonical.count)

        let recoveredRotation = simd_float3x3(
            simd_normalize(simd_float3(result.poseTransform.columns.0.x, result.poseTransform.columns.0.y, result.poseTransform.columns.0.z)),
            simd_normalize(simd_float3(result.poseTransform.columns.1.x, result.poseTransform.columns.1.y, result.poseTransform.columns.1.z)),
            simd_normalize(simd_float3(result.poseTransform.columns.2.x, result.poseTransform.columns.2.y, result.poseTransform.columns.2.z))
        )
        var maxRotationColumnError: Float = 0
        for column in 0..<3
        {
            let recoveredColumn = recoveredRotation[column]
            let trueColumn = rotation[column]
            maxRotationColumnError = max(maxRotationColumnError, simd_distance(recoveredColumn, trueColumn))
        }

        print("DIAGNOSTIC [\(label)] mean normalized-landmark error (cm):", meanNormalizedError, "max:", maxNormalizedError, "max rotation column error:", maxRotationColumnError)

        #expect(meanNormalizedError < 0.5, "[\(label)] mean pose-normalized landmark error \(meanNormalizedError)cm -- should be near-zero if head pose is fully removed")
        #expect(maxRotationColumnError < 0.05, "[\(label)] recovered rotation deviates from the injected rotation by \(maxRotationColumnError) per column")
    }

    /// Every diagnostic above placed the face dead-center on the camera's
    /// optical axis (x=0) before translating in z only -- never off to the
    /// side. The user reports apparent rotation flipping direction with
    /// screen position (left of frame vs. right of frame), with head
    /// rotation held constant -- classic "off-axis viewing ray not
    /// accounted for" symptom. This isolates exactly that: the head is NOT
    /// rotated at all (identity), only placed off-center laterally, at a
    /// magnitude representative of a face well off-center in a 16:9 frame.
    /// If `solve()` correctly reconstructs identity rotation regardless of
    /// lateral position, recoveredRotation should stay near-identity. If it
    /// doesn't, that's the bug: off-axis position is leaking into the
    /// recovered rotation.
    @Test("DIAGNOSTIC: lateral (off-axis) position alone, no real head rotation", arguments: [-20.0, -10.0, 0.0, 10.0, 20.0])
    func diagnosticLateralOffsetLeaksIntoRotation(lateralOffsetCm: Float) throws
    {
        let near: Float = 1.0
        let canonical = FaceGeometrySolver.canonicalPositions
        let meanCanonicalZ = canonical.reduce(Float(0)) { $0 + $1.z } / Float(canonical.count)
        let groundTruthDistance: Float = 50.0
        // Identity rotation -- the head itself does not turn at all.
        let rotation = matrix_identity_float3x3

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
            var worldPoint = rotation * point
            worldPoint.x += lateralOffsetCm // face shifted sideways, camera axis unchanged
            let metricZ = worldPoint.z - groundTruthDistance
            let p2z = -metricZ
            let p2x = worldPoint.x * near / p2z
            let p2y = worldPoint.y * near / p2z
            let p1z = depthOffset - near + scale * p2z

            let screenX = (p2x - left) / xScale
            let screenY = (p2y - bottom) / yScale
            let screenZ = p1z / xScale
            return simd_float3(screenX, screenY, screenZ)
        }

        let result = try #require(FaceGeometrySolver.solve(screenLandmarks: screenLandmarks, frameWidth: Self.frameWidth, frameHeight: Self.frameHeight))

        let recoveredRotation = simd_float3x3(
            simd_normalize(simd_float3(result.poseTransform.columns.0.x, result.poseTransform.columns.0.y, result.poseTransform.columns.0.z)),
            simd_normalize(simd_float3(result.poseTransform.columns.1.x, result.poseTransform.columns.1.y, result.poseTransform.columns.1.z)),
            simd_normalize(simd_float3(result.poseTransform.columns.2.x, result.poseTransform.columns.2.y, result.poseTransform.columns.2.z))
        )
        // Extract an implied yaw angle from the recovered rotation matrix
        // (rotation about y: forward column's x/z components) to report a
        // human-readable "apparent turn" even though truth is zero.
        let impliedYawDegrees = atan2(recoveredRotation.columns.2.x, recoveredRotation.columns.2.z) * 180 / Float.pi

        var maxRotationColumnError: Float = 0
        for column in 0..<3
        {
            maxRotationColumnError = max(maxRotationColumnError, simd_distance(recoveredRotation[column], rotation[column]))
        }

        print("DIAGNOSTIC [lateral offset \(lateralOffsetCm)cm, true rotation = identity] implied yaw (deg):", impliedYawDegrees, "max rotation column error:", maxRotationColumnError)
    }

    /// `scenePoint()` (the port confirmed to visually line up) scales z by a
    /// flat, position-independent constant -- z noise stays z noise, never
    /// touching x/y. `FaceGeometrySolver`'s `unprojectXY` divides x/y by z
    /// (`x*z/near`), so any inaccuracy in the network's actual per-point z
    /// estimate gets coupled directly into x/y shape error. This measures
    /// how much: same rotated ground truth as the diagnostic above, with
    /// growing *proportional* per-point z noise layered on top of an
    /// otherwise-perfect z, to see whether realistic-magnitude network
    /// noise alone is sufficient to explain visible shape distortion --
    /// no code changes implied either way, this is purely measurement.
    @Test("DIAGNOSTIC: sensitivity of normalized landmarks to per-point z noise", arguments: [0.0, 0.01, 0.02, 0.05, 0.10, 0.20])
    func diagnosticZNoiseSensitivity(zNoiseFraction: Float) throws
    {
        let near: Float = 1.0
        let canonical = FaceGeometrySolver.canonicalPositions
        let meanCanonicalZ = canonical.reduce(Float(0)) { $0 + $1.z } / Float(canonical.count)
        let groundTruthDistance: Float = 50.0
        let rotation = simd_float3x3(simd_quatf(angle: 25 * Float.pi / 180, axis: simd_float3(0, 1, 0)))

        let scale = near / (groundTruthDistance - meanCanonicalZ)
        let depthOffset: Float = 0

        let degreesToRadians = Float.pi / 180
        let heightAtNear = 2 * near * tan(0.5 * degreesToRadians * FaceGeometrySolver.verticalFieldOfViewDegrees)
        let widthAtNear = Self.frameWidth * heightAtNear / Self.frameHeight
        let left = -0.5 * widthAtNear, right = 0.5 * widthAtNear
        let bottom = -0.5 * heightAtNear, top = 0.5 * heightAtNear
        let xScale = right - left
        let yScale = top - bottom

        // Deterministic per-point "noise" (alternating sign by index, not
        // random -- reproducible across runs), proportional to each point's
        // own z magnitude so zNoiseFraction reads as a plausible relative-
        // error percentage rather than an arbitrary absolute unit.
        let screenLandmarks = canonical.enumerated().map { index, point -> simd_float3 in
            let noiseSign: Float = index % 2 == 0 ? 1 : -1
            let worldPoint = rotation * point
            let metricZ = worldPoint.z - groundTruthDistance
            let p2z = -metricZ
            let p2x = worldPoint.x * near / p2z
            let p2y = worldPoint.y * near / p2z
            let p1zTrue = depthOffset - near + scale * p2z
            let p1z = p1zTrue * (1 + zNoiseFraction * noiseSign)

            let screenX = (p2x - left) / xScale
            let screenY = (p2y - bottom) / yScale
            let screenZ = p1z / xScale
            return simd_float3(screenX, screenY, screenZ)
        }

        let result = try #require(FaceGeometrySolver.solve(screenLandmarks: screenLandmarks, frameWidth: Self.frameWidth, frameHeight: Self.frameHeight))

        var maxNormalizedError: Float = 0
        var sumNormalizedError: Float = 0
        for (canonicalPoint, normalizedPoint) in zip(canonical, result.metricLandmarks)
        {
            let error = simd_distance(canonicalPoint, normalizedPoint)
            maxNormalizedError = max(maxNormalizedError, error)
            sumNormalizedError += error
        }
        let meanNormalizedError = sumNormalizedError / Float(canonical.count)

        print("DIAGNOSTIC [z noise \(Int(zNoiseFraction * 100))%] mean normalized-landmark error (cm):", meanNormalizedError, "max:", maxNormalizedError)
    }

    /// Same z-error idea, but *systematic* (every point biased the same
    /// direction, e.g. a wrong constant somewhere) rather than random, and
    /// with rotation angle as the swept variable at a fixed bias -- this is
    /// what "fine when neutral, distorts more the more the head turns"
    /// would look like if a coherent z scale error is the cause, versus the
    /// random-noise case above which held rotation fixed.
    @Test("DIAGNOSTIC: fixed systematic z bias, error growth vs rotation angle", arguments: [0.0, 10.0, 20.0, 30.0, 40.0])
    func diagnosticSystematicZBiasVsRotationAngle(yawDegrees: Float) throws
    {
        let near: Float = 1.0
        let canonical = FaceGeometrySolver.canonicalPositions
        let meanCanonicalZ = canonical.reduce(Float(0)) { $0 + $1.z } / Float(canonical.count)
        let groundTruthDistance: Float = 50.0
        let rotation = simd_float3x3(simd_quatf(angle: yawDegrees * Float.pi / 180, axis: simd_float3(0, 1, 0)))
        let systematicZBias: Float = 0.10 // every point's z scaled the same direction, not alternating

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
            let worldPoint = rotation * point
            let metricZ = worldPoint.z - groundTruthDistance
            let p2z = -metricZ
            let p2x = worldPoint.x * near / p2z
            let p2y = worldPoint.y * near / p2z
            let p1zTrue = depthOffset - near + scale * p2z
            let p1z = p1zTrue * (1 + systematicZBias)

            let screenX = (p2x - left) / xScale
            let screenY = (p2y - bottom) / yScale
            let screenZ = p1z / xScale
            return simd_float3(screenX, screenY, screenZ)
        }

        let result = try #require(FaceGeometrySolver.solve(screenLandmarks: screenLandmarks, frameWidth: Self.frameWidth, frameHeight: Self.frameHeight))

        var maxNormalizedError: Float = 0
        var sumNormalizedError: Float = 0
        for (canonicalPoint, normalizedPoint) in zip(canonical, result.metricLandmarks)
        {
            let error = simd_distance(canonicalPoint, normalizedPoint)
            maxNormalizedError = max(maxNormalizedError, error)
            sumNormalizedError += error
        }
        let meanNormalizedError = sumNormalizedError / Float(canonical.count)

        print("DIAGNOSTIC [10% systematic z bias, yaw \(yawDegrees)deg] mean normalized-landmark error (cm):", meanNormalizedError, "max:", maxNormalizedError)
    }

    /// Wider sweep than the 10%-fixed test above: a lightweight depth-
    /// estimation model could plausibly *compress or exaggerate the real
    /// depth range* by a large amount (not a small wrong constant, not
    /// random noise) -- e.g. systematically underestimating how far the
    /// nose actually protrudes relative to the cheeks/ears when the face is
    /// off-axis, a real characteristic of lightweight networks. This checks
    /// whether that magnitude of structural z error, at a fixed rotation,
    /// is what it'd take to produce the severity of distortion reported
    /// (confirmed static/held, not motion) -- still no code change implied,
    /// this narrows whether the remaining gap is "z accuracy" or something
    /// else entirely.
    @Test("DIAGNOSTIC: severity of distortion across a wide range of z scale error, fixed rotation", arguments: [-0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 1.0, 1.5])
    func diagnosticWideZBiasSweepAtFixedRotation(zBias: Float) throws
    {
        let near: Float = 1.0
        let canonical = FaceGeometrySolver.canonicalPositions
        let meanCanonicalZ = canonical.reduce(Float(0)) { $0 + $1.z } / Float(canonical.count)
        let groundTruthDistance: Float = 50.0
        let rotation = simd_float3x3(simd_quatf(angle: 30 * Float.pi / 180, axis: simd_float3(0, 1, 0)))

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
            let worldPoint = rotation * point
            let metricZ = worldPoint.z - groundTruthDistance
            let p2z = -metricZ
            let p2x = worldPoint.x * near / p2z
            let p2y = worldPoint.y * near / p2z
            let p1zTrue = depthOffset - near + scale * p2z
            let p1z = p1zTrue * (1 + zBias)

            let screenX = (p2x - left) / xScale
            let screenY = (p2y - bottom) / yScale
            let screenZ = p1z / xScale
            return simd_float3(screenX, screenY, screenZ)
        }

        let result = try #require(FaceGeometrySolver.solve(screenLandmarks: screenLandmarks, frameWidth: Self.frameWidth, frameHeight: Self.frameHeight))

        var maxNormalizedError: Float = 0
        var sumNormalizedError: Float = 0
        for (canonicalPoint, normalizedPoint) in zip(canonical, result.metricLandmarks)
        {
            let error = simd_distance(canonicalPoint, normalizedPoint)
            maxNormalizedError = max(maxNormalizedError, error)
            sumNormalizedError += error
        }
        let meanNormalizedError = sumNormalizedError / Float(canonical.count)
        let faceHeightCm: Float = 17.665 // measured bounding-box height of canonicalPositions

        print("DIAGNOSTIC [z bias \(Int(zBias * 100))%, yaw 30deg] mean error (cm):", meanNormalizedError, "max:", maxNormalizedError, "max as % of face height:", maxNormalizedError / faceHeightCm * 100)
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

    /// Decisive comparison, not another synthetic guess: reconstructs the
    /// SAME real captured landmarks two ways -- (a) scenePoint()'s formula
    /// (confirmed by the user to visually track the real face correctly,
    /// including under rotation), and (b) FaceGeometrySolver's own
    /// unprojectXY/moveAndRescaleZ chain -- then fits *both* to
    /// canonicalPositions with the identical weighted-Procrustes solver. If
    /// scenePoint's reconstruction fits cleanly (symmetric bbox, low
    /// deviation) while FaceGeometrySolver's doesn't, the bug is inside
    /// FaceGeometrySolver's specific reconstruction math, not "the model's
    /// z is unreliable" -- the raw data supports a good fit, just not via
    /// this path.
    @Test("DIAGNOSTIC: real data, scenePoint reconstruction vs FaceGeometrySolver reconstruction")
    func diagnosticRealDataScenePointVsFaceGeometrySolver() throws
    {
        let realFrameWidth: Float = 1920.0
        let realFrameHeight: Float = 1080.0
        let aspect = realFrameHeight / realFrameWidth
        let cameraDistance: Float = 1.0 / tan((30.0 * Float.pi / 180) / 2.0) // PerspectiveCameraNode default

        func unitPoint(_ landmark: simd_float3) -> simd_float2
        {
            simd_float2(landmark.x * 2 - 1, landmark.y * 2 * aspect - aspect)
        }
        func scenePoint(_ landmark: simd_float3) -> simd_float3
        {
            let flat = unitPoint(landmark)
            let depth = -landmark.z * 2
            let depthScale = (cameraDistance - depth) / cameraDistance
            return simd_float3(flat.x * depthScale, flat.y * depthScale, depth)
        }

        let scenePointReconstruction = Self.realCapturedScreenLandmarks.map(scenePoint)

        let originalWeights: [(Int, Float)] = [
            (4, 0.070909939706326), (6, 0.032100144773722), (10, 0.008446550928056),
            (33, 0.058724168688059), (54, 0.007667080033571), (67, 0.009078059345484),
            (117, 0.009791937656701), (119, 0.014565368182957), (121, 0.018591361120343),
            (127, 0.005197994410992), (129, 0.120625205338001), (132, 0.005560018587857),
            (133, 0.05328618362546), (136, 0.066890455782413), (143, 0.014816547743976),
            (147, 0.014262833632529), (198, 0.025462191551924), (205, 0.047252278774977),
            (263, 0.058724168688059), (284, 0.007667080033571), (297, 0.009078059345484),
            (346, 0.009791937656701), (348, 0.014565368182957), (350, 0.018591361120343),
            (356, 0.005197994410992), (358, 0.120625205338001), (361, 0.005560018587857),
            (362, 0.05328618362546), (365, 0.066890455782413), (372, 0.014816547743976),
            (376, 0.014262833632529), (420, 0.025462191551924), (425, 0.047252278774977),
        ]
        var weights = [Float](repeating: 0, count: FaceGeometrySolver.landmarkCount)
        for (index, weight) in originalWeights { weights[index] = weight }

        let scenePointFit = try #require(FaceGeometrySolver.solveWeightedOrthogonalProblem(source: FaceGeometrySolver.canonicalPositions, target: scenePointReconstruction, weights: weights))
        let scenePointInverse = scenePointFit.inverse
        let scenePointNormalized = scenePointReconstruction.map { p -> simd_float3 in
            let h = scenePointInverse * simd_float4(p, 1)
            return simd_float3(h.x, h.y, h.z)
        }

        let solverResult = try #require(FaceGeometrySolver.solve(screenLandmarks: Self.realCapturedScreenLandmarks, frameWidth: realFrameWidth, frameHeight: realFrameHeight))

        func report(_ label: String, _ normalized: [simd_float3])
        {
            let canonical = FaceGeometrySolver.canonicalPositions
            let range = normalized.reduce((min: simd_float3(repeating: .greatestFiniteMagnitude), max: simd_float3(repeating: -.greatestFiniteMagnitude))) {
                (simd_min($0.min, $1), simd_max($0.max, $1))
            }
            var meanDeviation: Float = 0
            for (c, n) in zip(canonical, normalized)
            {
                // Normalize both to the same scale for a fair shape comparison:
                // compare bounding-box aspect/symmetry, not absolute units.
                meanDeviation += simd_distance(c, n)
            }
            meanDeviation /= Float(normalized.count)
            let xExtentNegative = -range.min.x, xExtentPositive = range.max.x
            print("\(label): bbox=\(range.min) to \(range.max), x-symmetry (neg vs pos extent)=\(xExtentNegative) vs \(xExtentPositive), meanDeviationFromCanonical(raw units)=\(meanDeviation)")
        }

        report("scenePoint reconstruction, Procrustes-fit to canonical", scenePointNormalized)
        report("FaceGeometrySolver's own normalizedLandmarks           ", solverResult.metricLandmarks)

        // The actual shipped API (fitCanonicalModel) should reproduce the
        // manual scenePointNormalized computation above, and should clearly
        // beat solve()'s own reconstruction on this same real data.
        let fitResult = try #require(FaceGeometrySolver.fitCanonicalModel(to: scenePointReconstruction))
        report("fitCanonicalModel(to: scenePointReconstruction)        ", fitResult.metricLandmarks)

        let canonical = FaceGeometrySolver.canonicalPositions
        var fitMeanDeviation: Float = 0
        for (c, n) in zip(canonical, fitResult.metricLandmarks) { fitMeanDeviation += simd_distance(c, n) }
        fitMeanDeviation /= Float(fitResult.metricLandmarks.count)

        var solverMeanDeviation: Float = 0
        for (c, n) in zip(canonical, solverResult.metricLandmarks) { solverMeanDeviation += simd_distance(c, n) }
        solverMeanDeviation /= Float(solverResult.metricLandmarks.count)

        #expect(fitMeanDeviation < solverMeanDeviation, "fitCanonicalModel(to: scenePoint) [\(fitMeanDeviation)] should beat solve()'s own reconstruction [\(solverMeanDeviation)] on this real off-axis capture")
        #expect(fitMeanDeviation < 0.6, "fitCanonicalModel deviation \(fitMeanDeviation)cm should be plausible for real expression variance, not a skewed fit")
    }

    /// Real captured screenLandmarks from a live, held, off-axis pose where
    /// the rendered Geometry/Transform visibly doesn't align (user-provided
    /// console dump, FABRIC_DUMP_FACE_LANDMARKS). Runs solve() on the actual
    /// data instead of a synthetic construction, and reports what comes out.
    @Test("DIAGNOSTIC: real captured off-axis pose")
    func diagnosticRealCapturedOffAxisPose() throws
    {
        let realFrameWidth: Float = 1920.0
        let realFrameHeight: Float = 1080.0
        print("REAL captured landmark count:", Self.realCapturedScreenLandmarks.count)

        let result = try #require(FaceGeometrySolver.solve(screenLandmarks: Self.realCapturedScreenLandmarks, frameWidth: realFrameWidth, frameHeight: realFrameHeight), "solve() returned nil for the real captured data")

        let distanceCm = abs(result.poseTransform.columns.3.z)
        let recoveredScale = simd_length(simd_float3(result.poseTransform.columns.0.x, result.poseTransform.columns.0.y, result.poseTransform.columns.0.z))
        let rotationAndScale = simd_float3x3(
            simd_float3(result.poseTransform.columns.0.x, result.poseTransform.columns.0.y, result.poseTransform.columns.0.z),
            simd_float3(result.poseTransform.columns.1.x, result.poseTransform.columns.1.y, result.poseTransform.columns.1.z),
            simd_float3(result.poseTransform.columns.2.x, result.poseTransform.columns.2.y, result.poseTransform.columns.2.z)
        )
        let rotation = rotationAndScale * (1 / recoveredScale)
        let impliedYawDegrees = atan2(rotation.columns.2.x, rotation.columns.2.z) * 180 / Float.pi
        let impliedPitchDegrees = asin(-rotation.columns.2.y) * 180 / Float.pi

        let canonical = FaceGeometrySolver.canonicalPositions
        var meanNormalizedDeviation: Float = 0
        var maxNormalizedDeviation: Float = 0
        var maxDeviationIndex = -1
        for (index, (canonicalPoint, normalizedPoint)) in zip(canonical, result.metricLandmarks).enumerated()
        {
            let deviation = simd_distance(canonicalPoint, normalizedPoint)
            meanNormalizedDeviation += deviation
            if deviation > maxNormalizedDeviation { maxNormalizedDeviation = deviation; maxDeviationIndex = index }
        }
        meanNormalizedDeviation /= Float(canonical.count)

        let normalizedRange = result.metricLandmarks.reduce((min: simd_float3(repeating: .greatestFiniteMagnitude), max: simd_float3(repeating: -.greatestFiniteMagnitude))) {
            (simd_min($0.min, $1), simd_max($0.max, $1))
        }
        let canonicalRange = canonical.reduce((min: simd_float3(repeating: .greatestFiniteMagnitude), max: simd_float3(repeating: -.greatestFiniteMagnitude))) {
            (simd_min($0.min, $1), simd_max($0.max, $1))
        }

        print("REAL distanceCm:", distanceCm, "recoveredScale (dimensionless):", recoveredScale)
        print("REAL implied yaw (deg):", impliedYawDegrees, "implied pitch (deg):", impliedPitchDegrees)
        print("REAL mean deviation from canonical (cm):", meanNormalizedDeviation, "max:", maxNormalizedDeviation, "at index:", maxDeviationIndex)
        print("REAL metricLandmarks bounding box:", normalizedRange.min, "to", normalizedRange.max)
        print("REAL canonicalPositions bounding box:", canonicalRange.min, "to", canonicalRange.max)
    }

    /// Reimplements solve()'s exact pipeline (its own helpers are private,
    /// so this mirrors makeFrustum/projectXY/moveAndRescaleZ/unprojectXY/
    /// changeHandedness verbatim -- same formulas verified against the
    /// bundled source above) but parameterized on the weight vector, so the
    /// same real captured data can be run through the identical algorithm
    /// with a different Procrustes weighting. This is not testing a
    /// hypothetical -- it's asking whether *down-weighting the specific
    /// points already shown to carry implausible z* changes the outcome,
    /// using the exact same math the shipped code runs.
    private static func solveWithCustomWeights(screenLandmarks: [simd_float3], frameWidth: Float, frameHeight: Float, weights: [Float], useSecondIteration: Bool = true) -> (metricLandmarks: [simd_float3], poseTransform: simd_float4x4)?
    {
        let near: Float = 1.0
        let degreesToRadians = Float.pi / 180
        let heightAtNear = 2 * near * tan(0.5 * degreesToRadians * FaceGeometrySolver.verticalFieldOfViewDegrees)
        let widthAtNear = frameWidth * heightAtNear / frameHeight
        let left = -0.5 * widthAtNear, right = 0.5 * widthAtNear
        let bottom = -0.5 * heightAtNear, top = 0.5 * heightAtNear
        let xScale = right - left, yScale = top - bottom

        func projectXY(_ p: simd_float3) -> simd_float3 { simd_float3(p.x * xScale + left, p.y * yScale + bottom, p.z * xScale) }
        func moveAndRescaleZ(_ p: simd_float3, depthOffset: Float, scale: Float) -> simd_float3 { simd_float3(p.x, p.y, (p.z - depthOffset + near) / scale) }
        func unprojectXY(_ p: simd_float3) -> simd_float3 { simd_float3(p.x * p.z / near, p.y * p.z / near, p.z) }
        func changeHandedness(_ p: simd_float3) -> simd_float3 { simd_float3(p.x, p.y, -p.z) }
        func estimateScale(_ landmarks: [simd_float3]) -> Float? {
            guard let transform = FaceGeometrySolver.solveWeightedOrthogonalProblem(source: FaceGeometrySolver.canonicalPositions, target: landmarks, weights: weights) else { return nil }
            return simd_length(simd_float3(transform.columns.0.x, transform.columns.0.y, transform.columns.0.z))
        }

        let projected = screenLandmarks.map(projectXY)
        let depthOffset = projected.reduce(Float(0)) { $0 + $1.z } / Float(projected.count)

        let firstPass = projected.map(changeHandedness)
        guard let scale1 = estimateScale(firstPass) else { return nil }

        var totalScale = scale1
        if useSecondIteration
        {
            var secondPass = projected.map { moveAndRescaleZ($0, depthOffset: depthOffset, scale: scale1) }
            secondPass = secondPass.map(unprojectXY).map(changeHandedness)
            guard let scale2 = estimateScale(secondPass) else { return nil }
            totalScale = scale1 * scale2
        }
        var metricLandmarks = projected.map { moveAndRescaleZ($0, depthOffset: depthOffset, scale: totalScale) }
        metricLandmarks = metricLandmarks.map(unprojectXY).map(changeHandedness)

        guard let poseTransform = FaceGeometrySolver.solveWeightedOrthogonalProblem(source: FaceGeometrySolver.canonicalPositions, target: metricLandmarks, weights: weights) else { return nil }
        let inverse = poseTransform.inverse
        let normalized = metricLandmarks.map { p -> simd_float3 in
            let h = inverse * simd_float4(p, 1)
            return simd_float3(h.x, h.y, h.z)
        }
        return (normalized, poseTransform)
    }

    /// scenePoint's reconstruction (confirmed correct) fit the real captured
    /// data nearly symmetrically; FaceGeometrySolver's own two-pass
    /// unprojectXY/moveAndRescaleZ reconstruction, same data, came out
    /// badly asymmetric. Isolates which of the two passes causes the
    /// degradation by running the real data through the second iteration
    /// disabled (scale = first-pass estimate only, no refinement).
    @Test("DIAGNOSTIC: real data, one-pass vs two-pass scale iteration")
    func diagnosticRealDataOnePassVsTwoPass() throws
    {
        let realFrameWidth: Float = 1920.0
        let realFrameHeight: Float = 1080.0
        let originalWeights: [(Int, Float)] = [
            (4, 0.070909939706326), (6, 0.032100144773722), (10, 0.008446550928056),
            (33, 0.058724168688059), (54, 0.007667080033571), (67, 0.009078059345484),
            (117, 0.009791937656701), (119, 0.014565368182957), (121, 0.018591361120343),
            (127, 0.005197994410992), (129, 0.120625205338001), (132, 0.005560018587857),
            (133, 0.05328618362546), (136, 0.066890455782413), (143, 0.014816547743976),
            (147, 0.014262833632529), (198, 0.025462191551924), (205, 0.047252278774977),
            (263, 0.058724168688059), (284, 0.007667080033571), (297, 0.009078059345484),
            (346, 0.009791937656701), (348, 0.014565368182957), (350, 0.018591361120343),
            (356, 0.005197994410992), (358, 0.120625205338001), (361, 0.005560018587857),
            (362, 0.05328618362546), (365, 0.066890455782413), (372, 0.014816547743976),
            (376, 0.014262833632529), (420, 0.025462191551924), (425, 0.047252278774977),
        ]
        var weights = [Float](repeating: 0, count: FaceGeometrySolver.landmarkCount)
        for (index, weight) in originalWeights { weights[index] = weight }

        let twoPass = try #require(Self.solveWithCustomWeights(screenLandmarks: Self.realCapturedScreenLandmarks, frameWidth: realFrameWidth, frameHeight: realFrameHeight, weights: weights, useSecondIteration: true))
        let onePass = try #require(Self.solveWithCustomWeights(screenLandmarks: Self.realCapturedScreenLandmarks, frameWidth: realFrameWidth, frameHeight: realFrameHeight, weights: weights, useSecondIteration: false))

        func report(_ label: String, _ result: (metricLandmarks: [simd_float3], poseTransform: simd_float4x4))
        {
            let scale = simd_length(simd_float3(result.poseTransform.columns.0.x, result.poseTransform.columns.0.y, result.poseTransform.columns.0.z))
            let range = result.metricLandmarks.reduce((min: simd_float3(repeating: .greatestFiniteMagnitude), max: simd_float3(repeating: -.greatestFiniteMagnitude))) {
                (simd_min($0.min, $1), simd_max($0.max, $1))
            }
            var meanDeviation: Float = 0
            for (c, n) in zip(FaceGeometrySolver.canonicalPositions, result.metricLandmarks) { meanDeviation += simd_distance(c, n) }
            meanDeviation /= Float(result.metricLandmarks.count)
            print("\(label): scale=\(scale), bbox=\(range.min) to \(range.max), meanDeviation=\(meanDeviation)")
        }
        report("TWO-PASS (current, shipped)", twoPass)
        report("ONE-PASS (no refinement)   ", onePass)
    }

    @Test("DIAGNOSTIC: real data, down-weighting the points already shown to carry implausible z")
    func diagnosticRealDataRobustWeighting() throws
    {
        let realFrameWidth: Float = 1920.0
        let realFrameHeight: Float = 1080.0

        let originalWeights: [(Int, Float)] = [
            (4, 0.070909939706326), (6, 0.032100144773722), (10, 0.008446550928056),
            (33, 0.058724168688059), (54, 0.007667080033571), (67, 0.009078059345484),
            (117, 0.009791937656701), (119, 0.014565368182957), (121, 0.018591361120343),
            (127, 0.005197994410992), (129, 0.120625205338001), (132, 0.005560018587857),
            (133, 0.05328618362546), (136, 0.066890455782413), (143, 0.014816547743976),
            (147, 0.014262833632529), (198, 0.025462191551924), (205, 0.047252278774977),
            (263, 0.058724168688059), (284, 0.007667080033571), (297, 0.009078059345484),
            (346, 0.009791937656701), (348, 0.014565368182957), (350, 0.018591361120343),
            (356, 0.005197994410992), (358, 0.120625205338001), (361, 0.005560018587857),
            (362, 0.05328618362546), (365, 0.066890455782413), (372, 0.014816547743976),
            (376, 0.014262833632529), (420, 0.025462191551924), (425, 0.047252278774977),
        ]
        // The 15 right-side indices already shown (from the real capture) to
        // carry z 3-5x larger than their mirror-symmetric left-side partners.
        let implausibleZIndices: Set<Int> = [263, 284, 297, 346, 348, 350, 356, 358, 361, 362, 365, 372, 376, 420, 425]

        var fullWeights = [Float](repeating: 0, count: FaceGeometrySolver.landmarkCount)
        var robustWeights = [Float](repeating: 0, count: FaceGeometrySolver.landmarkCount)
        for (index, weight) in originalWeights
        {
            fullWeights[index] = weight
            robustWeights[index] = implausibleZIndices.contains(index) ? 0 : weight
        }

        let fullFitResult = try #require(Self.solveWithCustomWeights(screenLandmarks: Self.realCapturedScreenLandmarks, frameWidth: realFrameWidth, frameHeight: realFrameHeight, weights: fullWeights))
        let robustFitResult = try #require(Self.solveWithCustomWeights(screenLandmarks: Self.realCapturedScreenLandmarks, frameWidth: realFrameWidth, frameHeight: realFrameHeight, weights: robustWeights))

        func report(_ label: String, _ result: (metricLandmarks: [simd_float3], poseTransform: simd_float4x4))
        {
            let scale = simd_length(simd_float3(result.poseTransform.columns.0.x, result.poseTransform.columns.0.y, result.poseTransform.columns.0.z))
            let distanceCm = abs(result.poseTransform.columns.3.z)
            let range = result.metricLandmarks.reduce((min: simd_float3(repeating: .greatestFiniteMagnitude), max: simd_float3(repeating: -.greatestFiniteMagnitude))) {
                (simd_min($0.min, $1), simd_max($0.max, $1))
            }
            var meanDeviation: Float = 0
            for (c, n) in zip(FaceGeometrySolver.canonicalPositions, result.metricLandmarks) { meanDeviation += simd_distance(c, n) }
            meanDeviation /= Float(result.metricLandmarks.count)
            print("\(label): scale=\(scale), distanceCm=\(distanceCm), bbox=\(range.min) to \(range.max), meanDeviationFromCanonical=\(meanDeviation)")
        }

        report("FULL 33-weight fit  ", fullFitResult)
        report("ROBUST 18-weight fit", robustFitResult)
    }

    // Real captured screenLandmarks, frameWidth: 1920.0, frameHeight: 1080.0
    private static let realCapturedScreenLandmarks: [simd_float3] = [
        simd_float3(0.7846275, 0.23804748, -0.034318116),
        simd_float3(0.80043596, 0.32163128, -0.061623316),
        simd_float3(0.7844948, 0.29441017, -0.033084802),
        simd_float3(0.7809929, 0.39749599, -0.05164109),
        simd_float3(0.80207974, 0.3455112, -0.06524162),
        simd_float3(0.79761994, 0.37552518, -0.06013411),
        simd_float3(0.7818449, 0.44947296, -0.028601915),
        simd_float3(0.6708071, 0.4646093, -0.028725915),
        simd_float3(0.77843124, 0.502961, -0.020267991),
        simd_float3(0.7804884, 0.53422076, -0.02239499),
        simd_float3(0.7757994, 0.6557693, -0.008372745),
        simd_float3(0.7836586, 0.22627097, -0.032821998),
        simd_float3(0.78116286, 0.21780877, -0.029006802),
        simd_float3(0.7779468, 0.21471561, -0.023804702),
        simd_float3(0.77813333, 0.21397959, -0.022730792),
        simd_float3(0.77928776, 0.20537582, -0.024664795),
        simd_float3(0.77943873, 0.19399194, -0.027329247),
        simd_float3(0.7777191, 0.18072942, -0.026106913),
        simd_float3(0.7704763, 0.15540732, -0.015353062),
        simd_float3(0.79563355, 0.30865398, -0.05586601),
        simd_float3(0.7777677, 0.30903456, -0.045425396),
        simd_float3(0.58740044, 0.5681353, 0.002459291),
        simd_float3(0.7111916, 0.44042665, -0.0240067),
        simd_float3(0.6980153, 0.43908203, -0.029503295),
        simd_float3(0.68425256, 0.44103843, -0.03195872),
        simd_float3(0.66256267, 0.45657593, -0.029793013),
        simd_float3(0.72209734, 0.44573963, -0.01675418),
        simd_float3(0.69711596, 0.5046834, -0.036325),
        simd_float3(0.71105206, 0.5014959, -0.028190278),
        simd_float3(0.6824701, 0.5029974, -0.03833453),
        simd_float3(0.67100596, 0.495771, -0.037043013),
        simd_float3(0.6491115, 0.44151902, -0.027615326),
        simd_float3(0.7153032, 0.10851085, -0.024262518),
        simd_float3(0.664251, 0.471904, -0.027020114),
        simd_float3(0.5819143, 0.45827264, 0.0045075873),
        simd_float3(0.6282281, 0.4601048, -0.022638414),
        simd_float3(0.7086225, 0.34176573, -0.03669808),
        simd_float3(0.7695731, 0.24151693, -0.040241834),
        simd_float3(0.7676451, 0.21778284, -0.034005187),
        simd_float3(0.75012267, 0.23401865, -0.041719243),
        simd_float3(0.7350916, 0.22535749, -0.039312128),
        simd_float3(0.75297886, 0.21656735, -0.036251683),
        simd_float3(0.7403503, 0.21487668, -0.033127002),
        simd_float3(0.7085782, 0.19084248, -0.028885724),
        simd_float3(0.78963155, 0.32263732, -0.06480826),
        simd_float3(0.7893571, 0.3453304, -0.06880785),
        simd_float3(0.64853734, 0.52093154, -0.042470388),
        simd_float3(0.7337764, 0.400774, -0.02737631),
        simd_float3(0.744587, 0.32853734, -0.050410602),
        simd_float3(0.7413136, 0.34139457, -0.047255054),
        simd_float3(0.66012615, 0.34625375, -0.041414697),
        simd_float3(0.78571165, 0.37362885, -0.062856756),
        simd_float3(0.6930611, 0.54281926, -0.045645095),
        simd_float3(0.6690414, 0.5374046, -0.046398856),
        simd_float3(0.6158063, 0.6110752, -0.013077591),
        simd_float3(0.75282013, 0.5124085, -0.029480625),
        simd_float3(0.7222486, 0.49247843, -0.018973919),
        simd_float3(0.6988064, 0.21184026, -0.031505827),
        simd_float3(0.5718846, 0.22006905, 0.037381783),
        simd_float3(0.752763, 0.3144679, -0.0413121),
        simd_float3(0.7638208, 0.30526647, -0.036247257),
        simd_float3(0.71527314, 0.21257044, -0.02823952),
        simd_float3(0.7198898, 0.21365511, -0.027701676),
        simd_float3(0.658735, 0.55503577, -0.0436823),
        simd_float3(0.740318, 0.31758347, -0.043745022),
        simd_float3(0.72114617, 0.53554, -0.040218808),
        simd_float3(0.7201163, 0.55674756, -0.043208186),
        simd_float3(0.69607264, 0.6588255, -0.030441102),
        simd_float3(0.63897055, 0.5836427, -0.030941652),
        simd_float3(0.7111102, 0.6097058, -0.037431717),
        simd_float3(0.63496435, 0.5330199, -0.03599064),
        simd_float3(0.612481, 0.54903334, -0.01773574),
        simd_float3(0.76905626, 0.22711898, -0.03880749),
        simd_float3(0.752723, 0.22316848, -0.03964326),
        simd_float3(0.7386494, 0.21965963, -0.037189964),
        simd_float3(0.7550991, 0.30930007, -0.036848646),
        simd_float3(0.71815896, 0.21297038, -0.027671693),
        simd_float3(0.7260824, 0.21057299, -0.031010246),
        simd_float3(0.72009385, 0.21445826, -0.027525755),
        simd_float3(0.7667286, 0.3226672, -0.05623038),
        simd_float3(0.73981124, 0.21643923, -0.030384388),
        simd_float3(0.7522571, 0.21605484, -0.030791225),
        simd_float3(0.76541436, 0.21544906, -0.029246861),
        simd_float3(0.7543951, 0.1558228, -0.021414109),
        simd_float3(0.76279885, 0.18244587, -0.032618508),
        simd_float3(0.7652775, 0.19606675, -0.03374409),
        simd_float3(0.7659284, 0.20737463, -0.03086252),
        simd_float3(0.7655179, 0.21447286, -0.028474774),
        simd_float3(0.7387913, 0.21474299, -0.029872676),
        simd_float3(0.73801917, 0.21234947, -0.032060843),
        simd_float3(0.736195, 0.2066999, -0.034335334),
        simd_float3(0.733243, 0.1987518, -0.0339218),
        simd_float3(0.7225559, 0.25259402, -0.03872085),
        simd_float3(0.55499595, 0.34723222, 0.048786543),
        simd_float3(0.7900667, 0.3024664, -0.04110306),
        simd_float3(0.7295465, 0.21413258, -0.028660271),
        simd_float3(0.7285791, 0.21332894, -0.030018827),
        simd_float3(0.7683513, 0.29400522, -0.03668127),
        simd_float3(0.74223363, 0.30194443, -0.031869628),
        simd_float3(0.7662335, 0.29936317, -0.036595717),
        simd_float3(0.71881, 0.38795182, -0.03000528),
        simd_float3(0.69710195, 0.37301716, -0.036023255),
        simd_float3(0.7362499, 0.3307359, -0.04146271),
        simd_float3(0.6515078, 0.64236486, -0.025323363),
        simd_float3(0.6708676, 0.6053284, -0.03697806),
        simd_float3(0.68783814, 0.56397986, -0.045644883),
        simd_float3(0.7214132, 0.17642011, -0.029720875),
        simd_float3(0.7516627, 0.54517394, -0.03509695),
        simd_float3(0.7455314, 0.60483575, -0.030585507),
        simd_float3(0.73605245, 0.6620757, -0.02422608),
        simd_float3(0.67077, 0.44668368, -0.03179838),
        simd_float3(0.63122684, 0.4252463, -0.027171995),
        simd_float3(0.7293561, 0.45131093, -0.012653223),
        simd_float3(0.6511715, 0.48856354, -0.033899236),
        simd_float3(0.7450691, 0.4153843, -0.027633809),
        simd_float3(0.75836825, 0.33743584, -0.058253147),
        simd_float3(0.60902274, 0.40957397, -0.022501102),
        simd_float3(0.6460379, 0.4070496, -0.034444552),
        simd_float3(0.6689638, 0.3968619, -0.036893394),
        simd_float3(0.69709045, 0.39932245, -0.031127611),
        simd_float3(0.71569306, 0.40813473, -0.024937995),
        simd_float3(0.72999316, 0.41837272, -0.021446284),
        simd_float3(0.76892155, 0.44456506, -0.03081912),
        simd_float3(0.6118235, 0.35665053, -0.026113791),
        simd_float3(0.63635325, 0.493062, -0.03364778),
        simd_float3(0.789352, 0.30969894, -0.058611076),
        simd_float3(0.7380369, 0.37731394, -0.03185075),
        simd_float3(0.55739886, 0.4613808, 0.039735265),
        simd_float3(0.74170476, 0.42997304, -0.01859503),
        simd_float3(0.73103464, 0.32905552, -0.02947691),
        simd_float3(0.65733314, 0.4718189, -0.027261253),
        simd_float3(0.75429875, 0.35258675, -0.05387603),
        simd_float3(0.55917966, 0.28530207, 0.04539157),
        simd_float3(0.72806174, 0.45963734, -0.010553501),
        simd_float3(0.7706877, 0.3656272, -0.06084315),
        simd_float3(0.63412946, 0.15827724, -0.00831684),
        simd_float3(0.6212123, 0.12867184, 0.011023272),
        simd_float3(0.5785261, 0.35086167, 0.009143545),
        simd_float3(0.609178, 0.19339675, -0.0017262467),
        simd_float3(0.59524554, 0.50781864, -0.003988996),
        simd_float3(0.70783806, 0.08041286, -0.019885072),
        simd_float3(0.78528506, 0.30376792, -0.042336266),
        simd_float3(0.724416, 0.36035082, -0.032050196),
        simd_float3(0.61018807, 0.45655158, -0.017004387),
        simd_float3(0.6870195, 0.4551423, -0.029908173),
        simd_float3(0.6991432, 0.45233998, -0.028042186),
        simd_float3(0.72234666, 0.20608935, -0.030679427),
        simd_float3(0.61084586, 0.3057985, -0.020757627),
        simd_float3(0.7289547, 0.04782484, -0.006540761),
        simd_float3(0.67714393, 0.07458196, -0.0028127618),
        simd_float3(0.65184826, 0.09569318, 0.002505218),
        simd_float3(0.7794896, 0.59687185, -0.016192818),
        simd_float3(0.7563756, 0.046869624, 0.0029454087),
        simd_float3(0.7096109, 0.4533354, -0.023175992),
        simd_float3(0.7191385, 0.45655885, -0.017370211),
        simd_float3(0.7250987, 0.45812252, -0.012477218),
        simd_float3(0.62012154, 0.49940333, -0.02590913),
        simd_float3(0.71786195, 0.47685444, -0.019783644),
        simd_float3(0.7069317, 0.48434415, -0.02606052),
        simd_float3(0.696138, 0.48645866, -0.030873088),
        simd_float3(0.6840484, 0.48413554, -0.032921378),
        simd_float3(0.6754973, 0.47920448, -0.032130446),
        simd_float3(0.56893814, 0.5212835, 0.023028443),
        simd_float3(0.67784315, 0.45949242, -0.029725036),
        simd_float3(0.78354913, 0.27595744, -0.030762805),
        simd_float3(0.7368341, 0.26679355, -0.03852087),
        simd_float3(0.7546429, 0.31865385, -0.044959288),
        simd_float3(0.7667666, 0.273621, -0.037652574),
        simd_float3(0.7770769, 0.47688338, -0.019970901),
        simd_float3(0.6587078, 0.12606011, -0.011172092),
        simd_float3(0.6820908, 0.1015794, -0.014903182),
        simd_float3(0.7359684, 0.06745549, -0.020621754),
        simd_float3(0.59424907, 0.16744958, 0.024288291),
        simd_float3(0.72528535, 0.46626768, -0.014540548),
        simd_float3(0.76263374, 0.4081757, -0.038513437),
        simd_float3(0.7640198, 0.066175126, -0.011038503),
        simd_float3(0.7034327, 0.058123134, -0.008549848),
        simd_float3(0.5807694, 0.29438657, 0.009310333),
        simd_float3(0.75206393, 0.21507986, -0.03029218),
        simd_float3(0.75129855, 0.20990941, -0.033034094),
        simd_float3(0.74999756, 0.20143431, -0.03572373),
        simd_float3(0.74774575, 0.18896416, -0.03521001),
        simd_float3(0.73679906, 0.16436087, -0.028631728),
        simd_float3(0.7280437, 0.21397622, -0.030995239),
        simd_float3(0.72562104, 0.21587673, -0.033385478),
        simd_float3(0.72216624, 0.21934819, -0.03496055),
        simd_float3(0.7088246, 0.23382126, -0.037092354),
        simd_float3(0.6407774, 0.29247147, -0.03427499),
        simd_float3(0.75724274, 0.43145183, -0.028220136),
        simd_float3(0.7427364, 0.47549897, -0.011167673),
        simd_float3(0.73231906, 0.47368336, -0.0127988905),
        simd_float3(0.7287745, 0.21645044, -0.028853886),
        simd_float3(0.629, 0.23096752, -0.019357251),
        simd_float3(0.75888073, 0.47524828, -0.020424878),
        simd_float3(0.72475654, 0.13654853, -0.025178634),
        simd_float3(0.7919967, 0.40046504, -0.04910086),
        simd_float3(0.7750139, 0.4197922, -0.041406333),
        simd_float3(0.787092, 0.42395264, -0.0383959),
        simd_float3(0.7531914, 0.3720701, -0.042052705),
        simd_float3(0.7681462, 0.09379818, -0.01722579),
        simd_float3(0.7692945, 0.12672053, -0.01627281),
        simd_float3(0.74806887, 0.1266359, -0.022706455),
        simd_float3(0.69045997, 0.17755994, -0.026998777),
        simd_float3(0.7187948, 0.309406, -0.03218153),
        simd_float3(0.70561534, 0.15300083, -0.026030578),
        simd_float3(0.6847278, 0.3137508, -0.041095134),
        simd_float3(0.7056447, 0.2849125, -0.036971595),
        simd_float3(0.66674334, 0.2734132, -0.037931357),
        simd_float3(0.74206316, 0.094770595, -0.02550656),
        simd_float3(0.74098897, 0.35926935, -0.037344094),
        simd_float3(0.6718324, 0.15633313, -0.021537274),
        simd_float3(0.69306254, 0.12960915, -0.020958744),
        simd_float3(0.6804092, 0.20752689, -0.031657353),
        simd_float3(0.6106884, 0.26201493, -0.0147950705),
        simd_float3(0.6543663, 0.20277676, -0.026479658),
        simd_float3(0.58849406, 0.24200654, 0.0067237522),
        simd_float3(0.6913919, 0.25211176, -0.037435498),
        simd_float3(0.7502124, 0.3931538, -0.034860667),
        simd_float3(0.7633753, 0.32678112, -0.05948987),
        simd_float3(0.7491411, 0.3205561, -0.048990443),
        simd_float3(0.7739397, 0.34254336, -0.06518151),
        simd_float3(0.73510873, 0.4964707, -0.020091232),
        simd_float3(0.7155589, 0.5123268, -0.031449486),
        simd_float3(0.6968736, 0.5181909, -0.039311476),
        simd_float3(0.67865205, 0.5166752, -0.04247434),
        simd_float3(0.6626607, 0.50714916, -0.040612172),
        simd_float3(0.6450149, 0.46575135, -0.02608302),
        simd_float3(0.57853943, 0.40531784, 0.006004323),
        simd_float3(0.65957385, 0.43022114, -0.030471487),
        simd_float3(0.67680395, 0.42215198, -0.031417295),
        simd_float3(0.6962713, 0.42102593, -0.028568525),
        simd_float3(0.7131258, 0.42561263, -0.022747159),
        simd_float3(0.72594446, 0.43267956, -0.016549952),
        simd_float3(0.73551977, 0.4394925, -0.013341356),
        simd_float3(0.5545385, 0.40457383, 0.047724828),
        simd_float3(0.7467492, 0.31381255, -0.04297487),
        simd_float3(0.76749134, 0.38715744, -0.048131518),
        simd_float3(0.7777853, 0.32566524, -0.063664965),
        simd_float3(0.78056705, 0.3128505, -0.055732027),
        simd_float3(0.7743642, 0.32127997, -0.058305435),
        simd_float3(0.748464, 0.30574042, -0.036662236),
        simd_float3(0.78452003, 0.31116757, -0.05779977),
        simd_float3(0.7809237, 0.3052623, -0.043920256),
        simd_float3(0.7341687, 0.45809442, -0.008685091),
        simd_float3(0.7433283, 0.45170382, -0.011209762),
        simd_float3(0.75083333, 0.447553, -0.017282188),
        simd_float3(0.669841, 0.47534806, -0.030153342),
        simd_float3(0.66089743, 0.48647138, -0.033011943),
        simd_float3(0.7997955, 0.39591077, -0.03866219),
        simd_float3(0.8327655, 0.44483495, 0.06504007),
        simd_float3(0.79829156, 0.3086236, -0.035420977),
        simd_float3(0.8423435, 0.52498734, 0.14606827),
        simd_float3(0.80493677, 0.42798492, 0.036055032),
        simd_float3(0.81526953, 0.4246564, 0.042923935),
        simd_float3(0.8243584, 0.42513716, 0.051714662),
        simd_float3(0.8325882, 0.43631828, 0.06930967),
        simd_float3(0.79724586, 0.43530646, 0.030698659),
        simd_float3(0.8237633, 0.4820392, 0.039566826),
        simd_float3(0.8125054, 0.48201063, 0.03154424),
        simd_float3(0.8325115, 0.47796902, 0.049375232),
        simd_float3(0.83663493, 0.469759, 0.057853162),
        simd_float3(0.83760136, 0.42075947, 0.078728184),
        simd_float3(0.7995604, 0.115899555, 0.018827057),
        simd_float3(0.8344308, 0.44961208, 0.07105117),
        simd_float3(0.83906, 0.4284905, 0.15147895),
        simd_float3(0.8417608, 0.43543804, 0.09740205),
        simd_float3(0.82119185, 0.33473724, 0.023112832),
        simd_float3(0.79640543, 0.2407432, -0.025197206),
        simd_float3(0.7904279, 0.21768366, -0.020581754),
        simd_float3(0.8052289, 0.23423274, -0.010499035),
        simd_float3(0.8087323, 0.22462797, 0.003481917),
        simd_float3(0.7965431, 0.21678786, -0.009491691),
        simd_float3(0.8006757, 0.21448955, 0.0024652348),
        simd_float3(0.81050897, 0.19129531, 0.026224224),
        simd_float3(0.8080695, 0.32154256, -0.05633266),
        simd_float3(0.81076515, 0.34394377, -0.05884876),
        simd_float3(0.8495705, 0.48950428, 0.07037984),
        simd_float3(0.8012037, 0.3949337, 0.011804828),
        simd_float3(0.816497, 0.3256729, -0.01317856),
        simd_float3(0.81425667, 0.33813393, -0.008840925),
        simd_float3(0.84279853, 0.33400986, 0.057932716),
        simd_float3(0.80603844, 0.37221807, -0.05024501),
        simd_float3(0.8385206, 0.51457775, 0.032268587),
        simd_float3(0.8473624, 0.50616014, 0.051364973),
        simd_float3(0.84469694, 0.5676831, 0.11574083),
        simd_float3(0.799278, 0.50273293, -0.0038354308),
        simd_float3(0.80192477, 0.47687942, 0.02937738),
        simd_float3(0.8187667, 0.2103798, 0.035387244),
        simd_float3(0.8115086, 0.21850842, 0.17209816),
        simd_float3(0.8078639, 0.31302536, -0.013596955),
        simd_float3(0.79952276, 0.30483633, -0.022488426),
        simd_float3(0.80938584, 0.2106562, 0.029359218),
        simd_float3(0.8044427, 0.21223408, 0.022635225),
        simd_float3(0.8501977, 0.5208233, 0.060827713),
        simd_float3(0.8127525, 0.31538716, -0.0074002766),
        simd_float3(0.82335526, 0.5148747, 0.013427423),
        simd_float3(0.82679313, 0.5345145, 0.0122424085),
        simd_float3(0.82742345, 0.62952995, 0.04399513),
        simd_float3(0.8481684, 0.54552317, 0.084666766),
        simd_float3(0.82763296, 0.58475244, 0.027042303),
        simd_float3(0.851301, 0.49883452, 0.08485992),
        simd_float3(0.84762704, 0.5106051, 0.113377646),
        simd_float3(0.7941839, 0.22696835, -0.023958312),
        simd_float3(0.8011705, 0.22356388, -0.010512844),
        simd_float3(0.80521655, 0.21915373, 0.0017394156),
        simd_float3(0.8040647, 0.3086192, -0.014155554),
        simd_float3(0.8070735, 0.21123049, 0.02535663),
        simd_float3(0.8049194, 0.20963481, 0.013603212),
        simd_float3(0.80251795, 0.21349713, 0.021360321),
        simd_float3(0.8100946, 0.32160342, -0.030688707),
        simd_float3(0.7962425, 0.21590355, 0.0034552354),
        simd_float3(0.79170424, 0.21593522, -0.0060024676),
        simd_float3(0.78583825, 0.21507834, -0.01564747),
        simd_float3(0.7828642, 0.15726921, -0.008352652),
        simd_float3(0.7889438, 0.18292353, -0.018320475),
        simd_float3(0.78996634, 0.19645946, -0.019129),
        simd_float3(0.78906524, 0.20713393, -0.016197618),
        simd_float3(0.7870754, 0.21420029, -0.014416267),
        simd_float3(0.7977413, 0.2140807, 0.0039577074),
        simd_float3(0.7997517, 0.21175212, 0.0019444011),
        simd_float3(0.8020233, 0.20651722, 0.0005452166),
        simd_float3(0.80340475, 0.19878474, 0.0039690076),
        simd_float3(0.81646234, 0.25045308, 0.0111060245),
        simd_float3(0.81749886, 0.3286482, 0.19926122),
        simd_float3(0.7998028, 0.21320325, 0.013975938),
        simd_float3(0.8024713, 0.21247654, 0.012817621),
        simd_float3(0.7961384, 0.2939576, -0.023284337),
        simd_float3(0.8062744, 0.30088434, -0.0031479406),
        simd_float3(0.79790056, 0.29898155, -0.023413654),
        simd_float3(0.8093823, 0.38031372, 0.02223004),
        simd_float3(0.82453054, 0.36288083, 0.03448863),
        simd_float3(0.81255025, 0.327664, -0.001939804),
        simd_float3(0.84054255, 0.60296315, 0.08007466),
        simd_float3(0.8418378, 0.57115626, 0.056400478),
        simd_float3(0.8418791, 0.5338323, 0.03731428),
        simd_float3(0.80340695, 0.1787666, 0.015276958),
        simd_float3(0.8056136, 0.5325287, -0.006966655),
        simd_float3(0.80721754, 0.5914132, 0.0031849223),
        simd_float3(0.80683535, 0.64572424, 0.013372823),
        simd_float3(0.83020633, 0.4285112, 0.061548118),
        simd_float3(0.84388405, 0.40374374, 0.09122394),
        simd_float3(0.7927924, 0.44230944, 0.02963079),
        simd_float3(0.84150577, 0.46250507, 0.07505401),
        simd_float3(0.79648185, 0.4103785, 0.0037573958),
        simd_float3(0.81619126, 0.3346782, -0.027875807),
        simd_float3(0.8475427, 0.38697678, 0.111906864),
        simd_float3(0.8433304, 0.38851807, 0.076611295),
        simd_float3(0.8361909, 0.38174474, 0.056992345),
        simd_float3(0.81983757, 0.38764107, 0.04027064),
        simd_float3(0.806773, 0.3992094, 0.029223738),
        simd_float3(0.7979582, 0.41128328, 0.019297758),
        simd_float3(0.7899793, 0.44194725, -0.017297596),
        simd_float3(0.849621, 0.3398644, 0.10751788),
        simd_float3(0.8457594, 0.46505854, 0.08583711),
        simd_float3(0.8003239, 0.30917373, -0.051875602),
        simd_float3(0.8036757, 0.3726277, 0.006165544),
        simd_float3(0.8253491, 0.42855185, 0.1942063),
        simd_float3(0.79223245, 0.42424384, 0.013316772),
        simd_float3(0.80883014, 0.32607082, 0.009551106),
        simd_float3(0.83545846, 0.44832513, 0.07578031),
        simd_float3(0.8129784, 0.34967494, -0.023051409),
        simd_float3(0.8152082, 0.2748826, 0.18970306),
        simd_float3(0.79195696, 0.4506476, 0.033666443),
        simd_float3(0.8104071, 0.3632083, -0.03840147),
        simd_float3(0.82137275, 0.16545247, 0.091396704),
        simd_float3(0.80832535, 0.13890564, 0.11308408),
        simd_float3(0.8383516, 0.33233693, 0.15576248),
        simd_float3(0.82662714, 0.19591887, 0.1157506),
        simd_float3(0.84317213, 0.47277126, 0.13635986),
        simd_float3(0.7981898, 0.09006696, 0.02727625),
        simd_float3(0.7940038, 0.30322415, -0.03821474),
        simd_float3(0.80939907, 0.35466728, 0.014635747),
        simd_float3(0.843357, 0.4302684, 0.115382105),
        simd_float3(0.8253454, 0.4379046, 0.052265335),
        simd_float3(0.8165244, 0.43669334, 0.04457276),
        simd_float3(0.8065625, 0.2051079, 0.017460952),
        simd_float3(0.84576976, 0.29423913, 0.107626945),
        simd_float3(0.7779082, 0.05396333, 0.01926337),
        simd_float3(0.79707843, 0.087427534, 0.06306075),
        simd_float3(0.8026163, 0.10876925, 0.08539332),
        simd_float3(0.8073928, 0.43958253, 0.038639557),
        simd_float3(0.7993506, 0.44494504, 0.035745356),
        simd_float3(0.794334, 0.44816506, 0.03534385),
        simd_float3(0.84680265, 0.46840632, 0.10241126),
        simd_float3(0.80071884, 0.4640905, 0.034810927),
        simd_float3(0.8095769, 0.46797964, 0.03710426),
        simd_float3(0.8193651, 0.46835998, 0.042969644),
        simd_float3(0.8278498, 0.4643082, 0.050865497),
        simd_float3(0.83200806, 0.45803592, 0.05785793),
        simd_float3(0.8356397, 0.48150498, 0.17402142),
        simd_float3(0.83022535, 0.44117546, 0.059597578),
        simd_float3(0.81056315, 0.26568168, -0.00053710653),
        simd_float3(0.81046826, 0.31681228, -0.01865847),
        simd_float3(0.7969195, 0.273308, -0.02206062),
        simd_float3(0.8134869, 0.13634971, 0.069512814),
        simd_float3(0.8058728, 0.113064446, 0.04878317),
        simd_float3(0.78526604, 0.073049515, 0.0048277993),
        simd_float3(0.80964464, 0.1727985, 0.14471236),
        simd_float3(0.79494053, 0.45672145, 0.034185156),
        simd_float3(0.79874474, 0.4049835, -0.01703058),
        simd_float3(0.79017, 0.068664595, 0.040094174),
        simd_float3(0.83472574, 0.28313795, 0.14993659),
        simd_float3(0.7937071, 0.21448556, -0.0050530606),
        simd_float3(0.79575425, 0.20975421, -0.006992821),
        simd_float3(0.79731905, 0.20159957, -0.008569208),
        simd_float3(0.7976692, 0.18965411, -0.006577702),
        simd_float3(0.7945403, 0.1666977, 0.003486322),
        simd_float3(0.80325705, 0.21275087, 0.013759689),
        simd_float3(0.80722654, 0.21431065, 0.015912408),
        simd_float3(0.8102704, 0.21737371, 0.01764912),
        simd_float3(0.81996536, 0.2314542, 0.024941647),
        simd_float3(0.8436075, 0.28359514, 0.07307432),
        simd_float3(0.79297566, 0.42748484, -0.006044206),
        simd_float3(0.7889193, 0.46800363, 0.017411113),
        simd_float3(0.7922285, 0.4638099, 0.028089453),
        simd_float3(0.79947704, 0.21550502, 0.013881508),
        simd_float3(0.8354567, 0.22858876, 0.09028993),
        simd_float3(0.7868379, 0.4706302, -0.00364999),
        simd_float3(0.79635894, 0.14171109, 0.011336279),
        simd_float3(0.7953796, 0.41771117, -0.029984143),
        simd_float3(0.8053665, 0.36878964, -0.013823236),
        simd_float3(0.7844284, 0.1296883, -0.0050353277),
        simd_float3(0.8162028, 0.18045384, 0.03763757),
        simd_float3(0.81521076, 0.30566975, 0.016540833),
        simd_float3(0.80629057, 0.15830798, 0.024914412),
        simd_float3(0.83438754, 0.30599585, 0.035515346),
        simd_float3(0.82360655, 0.2805607, 0.023818221),
        simd_float3(0.83765227, 0.2675008, 0.049582496),
        simd_float3(0.7871316, 0.099042825, -0.0019165425),
        simd_float3(0.8066213, 0.3555975, -0.0021468836),
        simd_float3(0.81857574, 0.1626066, 0.05392198),
        simd_float3(0.8082514, 0.1379351, 0.036980964),
        simd_float3(0.8253469, 0.2070764, 0.046284225),
        simd_float3(0.8400237, 0.25560158, 0.1079736),
        simd_float3(0.8301211, 0.2038611, 0.064906605),
        simd_float3(0.830411, 0.23786873, 0.14007118),
        simd_float3(0.82803226, 0.24814738, 0.033990335),
        simd_float3(0.8008658, 0.3892896, -0.004568964),
        simd_float3(0.81398267, 0.3251601, -0.03185597),
        simd_float3(0.814396, 0.31827003, -0.017518645),
        simd_float3(0.8146944, 0.34053895, -0.042914044),
        simd_float3(0.7990757, 0.4849011, 0.019373572),
        simd_float3(0.81649214, 0.49370998, 0.026018692),
        simd_float3(0.8302843, 0.49393323, 0.035762217),
        simd_float3(0.8396714, 0.4891155, 0.048729282),
        simd_float3(0.8432905, 0.47874185, 0.061706226),
        simd_float3(0.83821017, 0.4421493, 0.08400231),
        simd_float3(0.83963376, 0.38061985, 0.15614255),
        simd_float3(0.835584, 0.41160423, 0.06931252),
        simd_float3(0.82878476, 0.40640122, 0.0555226),
        simd_float3(0.8171561, 0.40797344, 0.042864803),
        simd_float3(0.80486906, 0.4150169, 0.033464894),
        simd_float3(0.7962006, 0.42410904, 0.02581971),
        simd_float3(0.79078263, 0.43246725, 0.021142075),
        simd_float3(0.81910527, 0.3787585, 0.20151785),
        simd_float3(0.81082755, 0.31212118, -0.012244407),
        simd_float3(0.80384904, 0.3846232, -0.025929635),
        simd_float3(0.8122022, 0.3247187, -0.046015393),
        simd_float3(0.8023376, 0.31269914, -0.04539446),
        simd_float3(0.8086759, 0.32074594, -0.0402805),
        simd_float3(0.8058801, 0.30471998, -0.01112276),
        simd_float3(0.8023927, 0.31041503, -0.04947624),
        simd_float3(0.7966189, 0.30470413, -0.03678394),
        simd_float3(0.78967065, 0.44987786, 0.028102186),
        simd_float3(0.7866798, 0.44549018, 0.016076969),
        simd_float3(0.7878087, 0.44242412, 0.0053954697),
        simd_float3(0.8338106, 0.45355415, 0.063876666),
        simd_float3(0.8380097, 0.46105838, 0.067609325),
    ]
}
