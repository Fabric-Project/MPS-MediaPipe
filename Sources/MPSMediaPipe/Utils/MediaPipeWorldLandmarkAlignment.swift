//
//  MediaPipeWorldLandmarkAlignment.swift
//  MPSMediaPipe
//

import Foundation
import simd

/// Places a model's world landmarks -- a metric, view-independent 3D shape
/// centered on the subject -- in a scene so that a perspective camera sees
/// them on top of the image.
///
/// World landmarks say what shape and size the hand or body is, in meters;
/// the image landmarks say where each point appears. The shape keeps its
/// metric size (a fixed number of scene units per meter) and only its
/// position is solved, by least squares, so its perspective projection lands
/// on the image landmarks. Distance from the camera therefore comes from
/// apparent size: a subject that looks larger is placed nearer. No rotation
/// is solved: world landmarks are already expressed facing the camera.
///
/// The distance is only as right as the assumption that the scene camera's
/// field of view matches the real camera's; a mismatch scales distances but
/// keeps their order.
public enum MediaPipeWorldLandmarkAlignment
{
    /// - Parameters:
    ///   - worldLandmarks: the model's world landmarks in MediaPipe's own
    ///     axes, in meters: x right, y down, z more negative toward the camera.
    ///   - imagePlanePoints: where each landmark appears on the image plane
    ///     at z = 0, in scene units with y up, index-aligned with
    ///     `worldLandmarks`.
    ///   - weights: per-landmark confidence in its image position, or nil to
    ///     weight every landmark equally.
    ///   - cameraDistance: the perspective camera's distance from the image
    ///     plane along +z, looking at the origin.
    ///   - sceneUnitsPerMeter: the scene's size for one meter. A subject this
    ///     many units from the camera is `distance / sceneUnitsPerMeter`
    ///     meters away, so `cameraDistance` puts a subject 1 m away on the
    ///     image plane.
    /// - Returns: the landmarks in scene space (y up, larger z closer to the
    ///   camera), or nil when the inputs do not determine a placement.
    public static func align(
        worldLandmarks: [simd_float3],
        imagePlanePoints: [simd_float2],
        weights: [Float]? = nil,
        cameraDistance: Float,
        sceneUnitsPerMeter: Float
    ) -> [simd_float3]?
    {
        guard sceneUnitsPerMeter > 0 else { return nil }
        // MediaPipe axes -> scene axes, at metric size.
        let shape = worldLandmarks.map { simd_float3($0.x, -$0.y, -$0.z) * sceneUnitsPerMeter }
        guard let shift = Self.translation(placing: shape, onto: imagePlanePoints, weights: weights, cameraDistance: cameraDistance) else { return nil }
        return shape.map { $0 + shift }
    }

    /// The translation that puts `sceneShape` -- a shape already in scene
    /// axes (y up, larger z closer to the camera), at its final size and
    /// orientation -- where its perspective projection best lands on
    /// `imagePlanePoints`, by weighted least squares. nil when the inputs do
    /// not determine a placement or it would put a point behind the camera.
    public static func translation(
        placing sceneShape: [simd_float3],
        onto imagePlanePoints: [simd_float2],
        weights: [Float]? = nil,
        cameraDistance: Float
    ) -> simd_float3?
    {
        let count = sceneShape.count
        guard count >= 2, imagePlanePoints.count == count, cameraDistance > 0 else { return nil }
        if let weights, weights.count != count { return nil }

        // A scene point (X, Y, Z) appears on the image plane at
        // (X, Y) * cameraDistance / (cameraDistance - Z). With
        // P = shape + (shiftX, shiftY, shiftZ), requiring that to equal the
        // image point (u, v) is linear in the shift (D = cameraDistance):
        //   D * shiftX + u * shiftZ = u * (D - shapeZ) - shapeX * D
        //   D * shiftY + v * shiftZ = v * (D - shapeZ) - shapeY * D
        // Solved as weighted least squares through the 3x3 normal equations.
        var normalMatrix = simd_float3x3(0)
        var normalVector = simd_float3(repeating: 0)
        for index in 0..<count
        {
            let weight = weights?[index] ?? 1
            let imagePoint = imagePlanePoints[index]
            let horizontalRow = simd_float3(cameraDistance, 0, imagePoint.x)
            let verticalRow = simd_float3(0, cameraDistance, imagePoint.y)
            let horizontalTarget = imagePoint.x * (cameraDistance - sceneShape[index].z) - sceneShape[index].x * cameraDistance
            let verticalTarget = imagePoint.y * (cameraDistance - sceneShape[index].z) - sceneShape[index].y * cameraDistance
            for (row, target) in [(horizontalRow, horizontalTarget), (verticalRow, verticalTarget)]
            {
                normalMatrix += weight * simd_float3x3(row * row.x, row * row.y, row * row.z)
                normalVector += weight * row * target
            }
        }
        guard abs(normalMatrix.determinant) > .ulpOfOne else { return nil }
        let shift = normalMatrix.inverse * normalVector
        guard shift.x.isFinite, shift.y.isFinite, shift.z.isFinite else { return nil }

        // Every point must be in front of the camera.
        guard sceneShape.allSatisfy({ $0.z + shift.z < cameraDistance }) else { return nil }
        return shift
    }
}
