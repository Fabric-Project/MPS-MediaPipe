//
//  MediaPipeCropPreprocessor.swift
//  MPSMediaPipe
//

import Foundation
import Metal
import simd

/// Encodes a rotated crop + normalize directly from a texture into an NHWC
/// float32 buffer, ready for MPSGraph inference with no CPU-side copy.
///
/// `outputPixelRange` is fixed per instance: most models normalize to
/// [0,1], but some detectors expect [-1,1] (see the model's own
/// ImageToTensorCalculatorOptions.output_tensor_float_range).
public final class MediaPipeCropPreprocessor
{
    private struct Uniforms
    {
        var centerPixels: simd_float2
        var rectSizePixels: simd_float2
        var rotationRadians: Float
        var textureTransform: simd_float4x4
        var presentationSizePixels: simd_float2
        var outputSize: simd_uint2
        var outputPixelRange: simd_float2
    }

    private let outputWidth: Int
    private let outputHeight: Int
    private let outputPixelRange: simd_float2
    private let pipeline: MTLComputePipelineState
    private let outputBuffer: MTLBuffer

    public init(device: MTLDevice, outputWidth: Int, outputHeight: Int, outputPixelRange: (min: Float, max: Float) = (0, 1)) throws
    {
        self.outputWidth = outputWidth
        self.outputHeight = outputHeight
        self.outputPixelRange = simd_float2(outputPixelRange.min, outputPixelRange.max)

        guard
            let shaderURL = Bundle.module.url(
                forResource: "MediaPipeCropPreprocess",
                withExtension: "metal",
                subdirectory: "Compute"
            ),
            let source = try? String(contentsOf: shaderURL, encoding: .utf8),
            let library = try? device.makeLibrary(source: source, options: nil),
            let function = library.makeFunction(name: "cropRotateAndNormalizeNHWC")
        else {
            throw MediaPipeMPSGraphError("Could not load MediaPipe crop preprocessing kernel")
        }

        self.pipeline = try device.makeComputePipelineState(function: function)

        let byteCount = outputWidth * outputHeight * 3 * MemoryLayout<Float>.stride
        guard let outputBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else
        {
            throw MediaPipeMPSGraphError("Could not allocate MediaPipe crop buffer")
        }
        outputBuffer.label = "MediaPipe crop NHWC \(outputWidth)x\(outputHeight)"
        self.outputBuffer = outputBuffer
    }

    /// Synchronous convenience overload -- commits its own command buffer
    /// and waits. Use the command-buffer overload to share one submission
    /// with subsequent MPSGraph inference.
    ///
    /// `textureTransform`/`presentationSize` describe how `texture` maps
    /// onto presentation pixels; pass identity and the texture's own
    /// dimensions if there's no transform.
    @discardableResult
    public func encode(
        texture: MTLTexture,
        textureTransform: simd_float4x4,
        presentationSize: simd_float2,
        centerNormalizedBottomLeft: simd_float2,
        sizeNormalized: simd_float2,
        rotationRadians: Float,
        commandQueue: MTLCommandQueue
    ) throws -> MTLBuffer
    {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else
        {
            throw MediaPipeMPSGraphError("Could not create MediaPipe crop command buffer")
        }

        let outputBuffer = try self.encode(
            texture: texture,
            textureTransform: textureTransform,
            presentationSize: presentationSize,
            centerNormalizedBottomLeft: centerNormalizedBottomLeft,
            sizeNormalized: sizeNormalized,
            rotationRadians: rotationRadians,
            commandBuffer: commandBuffer
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return outputBuffer
    }

    /// `centerNormalizedBottomLeft`/`sizeNormalized` are bottom-left-origin,
    /// normalized [0,1]; `rotationRadians` uses
    /// MediaPipeSSDDetectorDecoder's convention (top-left/Y-down). Encodes
    /// onto `commandBuffer` without committing -- the caller commits (and
    /// waits, if needed).
    @discardableResult
    public func encode(
        texture: MTLTexture,
        textureTransform: simd_float4x4,
        presentationSize: simd_float2,
        centerNormalizedBottomLeft: simd_float2,
        sizeNormalized: simd_float2,
        rotationRadians: Float,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLBuffer
    {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else
        {
            throw MediaPipeMPSGraphError("Could not create MediaPipe crop compute pass")
        }

        // Only the coordinate flips (bottom-left -> top-left); rotation is
        // already in the shader's native convention.
        let centerYTopLeft = 1 - centerNormalizedBottomLeft.y
        var uniforms = Uniforms(
            centerPixels: simd_float2(centerNormalizedBottomLeft.x * presentationSize.x, centerYTopLeft * presentationSize.y),
            rectSizePixels: simd_float2(sizeNormalized.x * presentationSize.x, sizeNormalized.y * presentationSize.y),
            rotationRadians: rotationRadians,
            textureTransform: textureTransform,
            presentationSizePixels: presentationSize,
            outputSize: simd_uint2(UInt32(self.outputWidth), UInt32(self.outputHeight)),
            outputPixelRange: self.outputPixelRange
        )

        encoder.label = "MediaPipe crop, rotate, and normalize"
        encoder.setComputePipelineState(self.pipeline)
        encoder.setTexture(texture, index: 0)
        encoder.setBuffer(self.outputBuffer, offset: 0, index: 0)
        encoder.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)

        let threadgroupWidth = self.pipeline.threadExecutionWidth
        let threadgroupHeight = max(1, min(8, self.pipeline.maxTotalThreadsPerThreadgroup / threadgroupWidth))
        encoder.dispatchThreads(
            MTLSize(width: self.outputWidth, height: self.outputHeight, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupWidth, height: threadgroupHeight, depth: 1)
        )
        encoder.endEncoding()

        return self.outputBuffer
    }
}
