//
//  MediaPipeSegmentationMaskProjector.swift
//  MPSMediaPipe
//

import Foundation
import Metal
import simd

/// Reprojects a segmentation mask, decoded in crop/tensor space, back into
/// full-image space -- the inverse of MediaPipeCropPreprocessor's forward
/// crop, using the same rect. Works for both a rotated crop and a
/// full-frame mask (center (0.5,0.5), size (1,1), rotation 0), since the
/// rect is a plain parameter rather than assumed by the shader.
public final class MediaPipeSegmentationMaskProjector
{
    private struct Uniforms
    {
        var centerPixels: simd_float2
        var rectSizePixels: simd_float2
        var rotationRadians: Float
        var maskSize: simd_uint2
        var outputSize: simd_uint2
        var applySigmoid: UInt32
    }

    private let maskWidth: Int
    private let maskHeight: Int
    private let pipeline: MTLComputePipelineState
    private let maskBuffer: MTLBuffer

    public init(device: MTLDevice, maskWidth: Int, maskHeight: Int) throws
    {
        self.maskWidth = maskWidth
        self.maskHeight = maskHeight

        guard
            let shaderURL = Bundle.module.url(
                forResource: "MediaPipeSegmentationMaskWarp",
                withExtension: "metal",
                subdirectory: "Compute"
            ),
            let source = try? String(contentsOf: shaderURL, encoding: .utf8),
            let library = try? device.makeLibrary(source: source, options: nil),
            let function = library.makeFunction(name: "warpSegmentationMaskInverseNHWC")
        else {
            throw MediaPipeMPSGraphError("Could not load MediaPipe segmentation mask warp kernel")
        }

        self.pipeline = try device.makeComputePipelineState(function: function)

        let byteCount = maskWidth * maskHeight * MemoryLayout<Float>.stride
        guard let maskBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else
        {
            throw MediaPipeMPSGraphError("Could not allocate MediaPipe segmentation mask buffer")
        }
        maskBuffer.label = "MediaPipe segmentation mask logits \(maskWidth)x\(maskHeight)"
        self.maskBuffer = maskBuffer
    }

    /// `centerNormalizedBottomLeft`/`sizeNormalized`/`rotationRadians` must
    /// match the rect passed to MediaPipeCropPreprocessor.encode for this
    /// frame. `maskValues` is the model's raw mask tensor, row-major
    /// top-left origin. `applySigmoid` (default true) must be false if the
    /// model's own graph already ends in a sigmoid -- re-applying it would
    /// double-activate and wash the mask toward uniform grey. Encodes onto
    /// `commandBuffer` without committing.
    public func encode(
        maskValues: [Float],
        applySigmoid: Bool = true,
        centerNormalizedBottomLeft: simd_float2,
        sizeNormalized: simd_float2,
        rotationRadians: Float,
        presentationSize: simd_float2,
        destinationTexture: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) throws
    {
        guard maskValues.count == self.maskWidth * self.maskHeight else
        {
            throw MediaPipeMPSGraphError("MediaPipe segmentation mask values size mismatch")
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else
        {
            throw MediaPipeMPSGraphError("Could not create MediaPipe segmentation mask warp compute pass")
        }

        maskValues.withUnsafeBytes { rawBuffer in
            self.maskBuffer.contents().copyMemory(from: rawBuffer.baseAddress!, byteCount: rawBuffer.count)
        }

        let centerYTopLeft = 1 - centerNormalizedBottomLeft.y
        var uniforms = Uniforms(
            centerPixels: simd_float2(centerNormalizedBottomLeft.x * presentationSize.x, centerYTopLeft * presentationSize.y),
            rectSizePixels: simd_float2(sizeNormalized.x * presentationSize.x, sizeNormalized.y * presentationSize.y),
            rotationRadians: rotationRadians,
            maskSize: simd_uint2(UInt32(self.maskWidth), UInt32(self.maskHeight)),
            outputSize: simd_uint2(UInt32(destinationTexture.width), UInt32(destinationTexture.height)),
            applySigmoid: applySigmoid ? 1 : 0
        )

        encoder.label = "MediaPipe segmentation mask inverse warp"
        encoder.setComputePipelineState(self.pipeline)
        encoder.setBuffer(self.maskBuffer, offset: 0, index: 0)
        encoder.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        encoder.setTexture(destinationTexture, index: 0)

        let threadgroupWidth = self.pipeline.threadExecutionWidth
        let threadgroupHeight = max(1, min(8, self.pipeline.maxTotalThreadsPerThreadgroup / threadgroupWidth))
        encoder.dispatchThreads(
            MTLSize(width: destinationTexture.width, height: destinationTexture.height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupWidth, height: threadgroupHeight, depth: 1)
        )
        encoder.endEncoding()
    }
}
