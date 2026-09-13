//
//  FaceGeometrySolver.swift
//  MPSMediaPipe
//

import Foundation
import simd

/// Ports MediaPipe's Face Geometry pipeline (face_geometry/libs/
/// {geometry_pipeline.cc,procrustes_solver.cc}) -- fits a static canonical
/// 3D face model to a frame's 468 FaceMesh landmarks via weighted-
/// orthogonal Procrustes, producing (a) a deformed mesh with head pose
/// normalized out, and (b) a 4x4 rigid transform (scale + rotation +
/// translation) separating head pose from expression.
///
/// Only the FACE_LANDMARK_PIPELINE branch is ported (not
/// FACE_DETECTION_PIPELINE). Environment constants: vertical FOV 63
/// degrees, near 1cm, far 10000cm.
///
/// `screenLandmarks` are expected bottom-left-origin normalized, with
/// MediaPipe's own relative Z -- already the state MediaPipe's own
/// ProjectXY produces internally, so `projectXY` here omits the
/// corresponding flip.
public enum FaceGeometrySolver
{
    public static let landmarkCount = 468

    // MARK: - Bundled canonical face model (see Models/Face/CanonicalFaceModel.json)

    public static let canonicalPositions: [simd_float3] = CanonicalModel.shared.positions
    public static let canonicalUVs: [simd_float2] = CanonicalModel.shared.uvs
    public static let canonicalTriangles: [(UInt32, UInt32, UInt32)] = CanonicalModel.shared.triangles
    private static let procrustesWeights: [Float] = CanonicalModel.shared.procrustesWeights

    // MARK: - Environment

    /// Public: callers doing their own cm-to-real-world-unit conversion need this.
    public static let verticalFieldOfViewDegrees: Float = 63.0
    private static let nearPlane: Float = 1.0
    private static let farPlane: Float = 10000.0

    /// Mirrors `IsScreenLandmarkListTooCompact`'s threshold.
    private static let compactnessThreshold: Float = 1e-3

    private struct Frustum
    {
        var left: Float
        var right: Float
        var bottom: Float
        var top: Float
        var near: Float
        var far: Float
    }

    /// `screenLandmarks` are normalized [0,1], bottom-left origin, with
    /// MediaPipe's own relative Z. Returns nil for a degenerate
    /// (too-compact) or malformed input.
    public static func solve(screenLandmarks: [simd_float3], frameWidth: Float, frameHeight: Float) -> (metricLandmarks: [simd_float3], poseTransform: simd_float4x4)?
    {
        guard screenLandmarks.count == landmarkCount, frameWidth > 0, frameHeight > 0 else { return nil }
        guard isSpreadSufficient(screenLandmarks) else { return nil }

        let frustum = makeFrustum(frameWidth: frameWidth, frameHeight: frameHeight)

        let projected = screenLandmarks.map { projectXY($0, frustum: frustum) }
        let depthOffset = projected.reduce(Float(0)) { $0 + $1.z } / Float(projected.count)

        // 1st iteration: don't unproject XY (unsafe given the screen Z's
        // relative nature) -- estimate scale on the projected+handedness-
        // flipped landmarks directly, to unproject on the 2nd iteration.
        let firstPassLandmarks = projected.map(changeHandedness)
        guard let firstIterationScale = estimateScale(firstPassLandmarks) else { return nil }

        // 2nd iteration: unproject XY using the 1st iteration's scale.
        var secondPassLandmarks = projected.map { moveAndRescaleZ($0, frustum: frustum, depthOffset: depthOffset, scale: firstIterationScale) }
        secondPassLandmarks = secondPassLandmarks.map { unprojectXY($0, frustum: frustum) }
        secondPassLandmarks = secondPassLandmarks.map(changeHandedness)

        guard let secondIterationScale = estimateScale(secondPassLandmarks) else { return nil }

        // Use the total scale to unproject the screen landmarks -- the
        // final metric landmarks.
        let totalScale = firstIterationScale * secondIterationScale
        var metricLandmarks = projected.map { moveAndRescaleZ($0, frustum: frustum, depthOffset: depthOffset, scale: totalScale) }
        metricLandmarks = metricLandmarks.map { unprojectXY($0, frustum: frustum) }
        metricLandmarks = metricLandmarks.map(changeHandedness)

        guard let poseTransform = solveWeightedOrthogonalProblem(source: canonicalPositions, target: metricLandmarks, weights: procrustesWeights) else { return nil }

        let inversePoseTransform = poseTransform.inverse
        let normalizedLandmarks = metricLandmarks.map { point -> simd_float3 in
            let homogeneous = inversePoseTransform * simd_float4(point, 1)
            return simd_float3(homogeneous.x, homogeneous.y, homogeneous.z)
        }

        return (normalizedLandmarks, poseTransform)
    }

    /// Fits the canonical face model to an already-reconstructed 3D point
    /// cloud, via the same weighted-landmark Procrustes basis `solve()`
    /// uses internally -- for callers with their own real-3D landmark
    /// reconstruction (already placed correctly relative to whichever
    /// camera is actually rendering them) who want the same canonical-
    /// model fit and expression-normalization `solve()` provides, without
    /// going through `solve()`'s own near-plane unprojectXY. That
    /// unprojection divides by `nearPlane` (1cm) -- a small reference
    /// distance that amplifies real-world per-point z inaccuracy far more
    /// than a caller's own, already-correct reconstruction typically would;
    /// for landmark data whose z accuracy doesn't support that amplification
    /// well, fitting a reconstruction from a gentler unprojection (e.g. one
    /// referenced against actual camera geometry, not an arbitrary 1cm
    /// plane) can produce a visibly more faithful, better-conditioned fit.
    /// `target` must be in the same order as `canonicalPositions`.
    public static func fitCanonicalModel(to target: [simd_float3]) -> (metricLandmarks: [simd_float3], poseTransform: simd_float4x4)?
    {
        guard target.count == landmarkCount else { return nil }
        guard let poseTransform = solveWeightedOrthogonalProblem(source: canonicalPositions, target: target, weights: procrustesWeights) else { return nil }

        let inversePoseTransform = poseTransform.inverse
        let normalizedLandmarks = target.map { point -> simd_float3 in
            let homogeneous = inversePoseTransform * simd_float4(point, 1)
            return simd_float3(homogeneous.x, homogeneous.y, homogeneous.z)
        }

        return (normalizedLandmarks, poseTransform)
    }

    // MARK: - Screen <-> metric space conversion (geometry_pipeline.cc)

    private static func makeFrustum(frameWidth: Float, frameHeight: Float) -> Frustum
    {
        let degreesToRadians = Float.pi / 180
        let heightAtNear = 2 * nearPlane * tan(0.5 * degreesToRadians * verticalFieldOfViewDegrees)
        let widthAtNear = frameWidth * heightAtNear / frameHeight

        return Frustum(
            left: -0.5 * widthAtNear, right: 0.5 * widthAtNear,
            bottom: -0.5 * heightAtNear, top: 0.5 * heightAtNear,
            near: nearPlane, far: farPlane
        )
    }

    private static func projectXY(_ point: simd_float3, frustum: Frustum) -> simd_float3
    {
        let xScale = frustum.right - frustum.left
        let yScale = frustum.top - frustum.bottom
        return simd_float3(point.x * xScale + frustum.left, point.y * yScale + frustum.bottom, point.z * xScale)
    }

    private static func moveAndRescaleZ(_ point: simd_float3, frustum: Frustum, depthOffset: Float, scale: Float) -> simd_float3
    {
        simd_float3(point.x, point.y, (point.z - depthOffset + frustum.near) / scale)
    }

    private static func unprojectXY(_ point: simd_float3, frustum: Frustum) -> simd_float3
    {
        simd_float3(point.x * point.z / frustum.near, point.y * point.z / frustum.near, point.z)
    }

    private static func changeHandedness(_ point: simd_float3) -> simd_float3
    {
        simd_float3(point.x, point.y, -point.z)
    }

    private static func estimateScale(_ landmarks: [simd_float3]) -> Float?
    {
        guard let transform = solveWeightedOrthogonalProblem(source: canonicalPositions, target: landmarks, weights: procrustesWeights) else { return nil }
        return simd_length(simd_float3(transform.columns.0.x, transform.columns.0.y, transform.columns.0.z))
    }

    private static func isSpreadSufficient(_ landmarks: [simd_float3]) -> Bool
    {
        var meanX: Float = 0
        var meanY: Float = 0
        for (index, point) in landmarks.enumerated()
        {
            let count = Float(index + 1)
            meanX += (point.x - meanX) / count
            meanY += (point.y - meanY) / count
        }

        var maxSquaredDistance: Float = 0
        for point in landmarks
        {
            let dx = point.x - meanX
            let dy = point.y - meanY
            maxSquaredDistance = max(maxSquaredDistance, dx * dx + dy * dy)
        }

        return sqrt(maxSquaredDistance) > compactnessThreshold
    }

    // MARK: - Weighted orthogonal Procrustes solve (procrustes_solver.cc)

    private static let absoluteErrorEpsilon: Float = 1e-9

    /// Returns a 4x4 transform (uniform scale + rotation in the top-left
    /// 3x3, translation in the last column) mapping `source` onto `target`
    /// in a weighted least-squares sense. `weights` must be non-negative
    /// with a non-negligible sum.
    static func solveWeightedOrthogonalProblem(source: [simd_float3], target: [simd_float3], weights: [Float]) -> simd_float4x4?
    {
        guard source.isEmpty == false, source.count == target.count, source.count == weights.count else { return nil }

        let totalWeight = weights.reduce(0, +)
        guard totalWeight > absoluteErrorEpsilon, weights.allSatisfy({ $0 >= 0 }) else { return nil }

        let sqrtWeights = weights.map { sqrt(max($0, 0)) }
        let weightedSources = zip(source, sqrtWeights).map { $0 * $1 }
        let weightedTargets = zip(target, sqrtWeights).map { $0 * $1 }

        // Weighted centroid of the source point cloud.
        var sourceCenterOfMass = simd_float3.zero
        for index in source.indices { sourceCenterOfMass += weightedSources[index] * sqrtWeights[index] }
        sourceCenterOfMass /= totalWeight

        let centeredWeightedSources = source.indices.map { weightedSources[$0] - sourceCenterOfMass * sqrtWeights[$0] }

        // designMatrix.columns[k] = sum_i centeredWeightedSources[i][k] * weightedTargets[i]
        var designMatrix = simd_float3x3(0)
        for index in source.indices
        {
            let c = centeredWeightedSources[index]
            let t = weightedTargets[index]
            designMatrix.columns.0 += c.x * t
            designMatrix.columns.1 += c.y * t
            designMatrix.columns.2 += c.z * t
        }

        let designMatrixNorm = simd_length(designMatrix.columns.0) + simd_length(designMatrix.columns.1) + simd_length(designMatrix.columns.2)
        guard designMatrixNorm > absoluteErrorEpsilon else { return nil }

        guard let rotation = computeOptimalRotation(designMatrix) else { return nil }

        var numerator: Float = 0
        var denominator: Float = 0
        for index in source.indices
        {
            numerator += simd_dot(rotation * centeredWeightedSources[index], weightedTargets[index])
            denominator += simd_dot(centeredWeightedSources[index], weightedSources[index])
        }
        guard denominator > absoluteErrorEpsilon, numerator / denominator > absoluteErrorEpsilon else { return nil }
        let scale = numerator / denominator

        let rotationAndScale = scale * rotation

        var translation = simd_float3.zero
        for index in source.indices
        {
            translation += (weightedTargets[index] - rotationAndScale * weightedSources[index]) * sqrtWeights[index]
        }
        translation /= totalWeight

        var transform = simd_float4x4(1)
        transform.columns.0 = simd_float4(rotationAndScale.columns.0, 0)
        transform.columns.1 = simd_float4(rotationAndScale.columns.1, 0)
        transform.columns.2 = simd_float4(rotationAndScale.columns.2, 0)
        transform.columns.3 = simd_float4(translation, 1)
        return transform
    }

    /// SVD-based optimal rotation: `design = U Σ Vᵀ`, `rotation = U Vᵀ`,
    /// with a reflection correction so `det(rotation) = +1`. The SVD is
    /// derived via Jacobi eigendecomposition of `designᵀ * design`: its
    /// eigenvectors are `V`, and `U`'s columns are `design * v_i / σ_i`.
    private static func computeOptimalRotation(_ design: simd_float3x3) -> simd_float3x3?
    {
        let designTransposeDesign = design.transpose * design
        guard let (eigenvalues, v) = Jacobi3x3.symmetricEigendecomposition(designTransposeDesign) else { return nil }

        let singularValues = eigenvalues.map { sqrt(max($0, 0)) }
        let vColumns = [v.columns.0, v.columns.1, v.columns.2]

        var uColumns = [simd_float3](repeating: .zero, count: 3)
        var degenerateIndices: [Int] = []
        for index in 0..<3
        {
            if singularValues[index] > 1e-6
            {
                uColumns[index] = (design * vColumns[index]) / singularValues[index]
            }
            else
            {
                degenerateIndices.append(index)
            }
        }

        // Degenerate (near-zero) singular values: complete U into an
        // orthonormal basis so the result is still a valid rotation.
        if degenerateIndices.count == 1
        {
            let index = degenerateIndices[0]
            let others = [0, 1, 2].filter { $0 != index }
            uColumns[index] = simd_normalize(simd_cross(uColumns[others[0]], uColumns[others[1]]))
        }
        else if degenerateIndices.count >= 2
        {
            return nil
        }

        var u = simd_float3x3(columns: (uColumns[0], uColumns[1], uColumns[2]))
        let vTransposed = v.transpose

        if u.determinant * v.determinant < 0
        {
            u.columns.2 = -u.columns.2
        }

        return u * vTransposed
    }

    // MARK: - Bundled canonical model loading

    /// Loads Models/Face/CanonicalFaceModel.json -- MediaPipe's own
    /// canonical_mesh (468 vertices x [X,Y,Z,U,V], centimeters; 898
    /// triangles) and procrustes_landmark_basis (33 weighted landmark IDs)
    /// fields.
    private struct CanonicalModel: Decodable
    {
        let positions: [simd_float3]
        let uvs: [simd_float2]
        let triangles: [(UInt32, UInt32, UInt32)]
        let procrustesWeights: [Float]

        private enum CodingKeys: String, CodingKey { case positions, uvs, triangles, procrustesWeights }

        init(from decoder: any Decoder) throws
        {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.positions = try container.decode([[Float]].self, forKey: .positions).map {
                guard $0.count >= 3 else
                {
                    throw DecodingError.dataCorruptedError(forKey: .positions, in: container, debugDescription: "Expected 3 components per position row, got \($0.count)")
                }
                return simd_float3($0[0], $0[1], $0[2])
            }
            self.uvs = try container.decode([[Float]].self, forKey: .uvs).map {
                guard $0.count >= 2 else
                {
                    throw DecodingError.dataCorruptedError(forKey: .uvs, in: container, debugDescription: "Expected 2 components per uv row, got \($0.count)")
                }
                return simd_float2($0[0], $0[1])
            }
            self.triangles = try container.decode([[UInt32]].self, forKey: .triangles).map {
                guard $0.count >= 3 else
                {
                    throw DecodingError.dataCorruptedError(forKey: .triangles, in: container, debugDescription: "Expected 3 indices per triangle row, got \($0.count)")
                }
                return ($0[0], $0[1], $0[2])
            }
            self.procrustesWeights = try container.decode([Float].self, forKey: .procrustesWeights)
        }

        static let shared: CanonicalModel = {
            guard
                let url = Bundle.module.url(forResource: "CanonicalFaceModel", withExtension: "json", subdirectory: "Models/Face"),
                let data = try? Data(contentsOf: url),
                let model = try? JSONDecoder().decode(CanonicalModel.self, from: data)
            else
            {
                fatalError("Could not load bundled Models/Face/CanonicalFaceModel.json")
            }
            return model
        }()
    }
}
