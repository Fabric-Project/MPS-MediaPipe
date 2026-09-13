//
//  MediaPipeLandmarkProjectionMath.swift
//  MPSMediaPipe
//

import Foundation
import simd

/// Shared per-point math used by MediaPipeFaceLandmarkProjection,
/// MediaPipeHandLandmarkProjection, and MediaPipePoseLandmarkProjection --
/// each decodes a differently-shaped raw tensor, but every point's actual
/// normalize/rotate/scale-into-rect math, and every tracked-region's
/// rect-to-bottom-left-origin-region conversion, is identical.
public enum MediaPipeLandmarkProjectionMath
{
    /// Normalizes a raw (x, y, z) triple by `landmarkSize` (and z further by
    /// `normalizeZ`), rotates by the given sin/cos, then scales into
    /// `rect`'s width/height and offsets by its center. `x`/`y`/`z` are the
    /// model's raw per-point output values, not yet normalized.
    public static func rotateAndProject(
        x: Float, y: Float, z: Float,
        landmarkSize: Float, normalizeZ: Float,
        sinRotation: Float, cosRotation: Float,
        rect: (cx: Float, cy: Float, width: Float, height: Float)
    ) -> simd_float3
    {
        let normalizedX = x / landmarkSize - 0.5
        let normalizedY = y / landmarkSize - 0.5
        let normalizedZ = z / landmarkSize / normalizeZ

        let rotatedX = cosRotation * normalizedX - sinRotation * normalizedY
        let rotatedY = sinRotation * normalizedX + cosRotation * normalizedY

        return simd_float3(
            rotatedX * rect.width + rect.cx,
            rotatedY * rect.height + rect.cy,
            normalizedZ * rect.width
        )
    }

    /// Converts a top-left-origin rect record (as returned by
    /// MediaPipeSSDRectTransform's rect derivation functions) into a
    /// bottom-left-origin `simd_float4` region, matching the convention a
    /// caller's own decoded-landmark output space already uses.
    public static func regionBottomLeft(from rect: (cx: Float, cy: Float, width: Float, height: Float, rotation: Float)) -> (region: simd_float4, rotation: Float)
    {
        let region = simd_float4(
            rect.cx - rect.width / 2,
            1 - (rect.cy - rect.height / 2) - rect.height,
            rect.width,
            rect.height
        )
        return (region: region, rotation: rect.rotation)
    }
}
