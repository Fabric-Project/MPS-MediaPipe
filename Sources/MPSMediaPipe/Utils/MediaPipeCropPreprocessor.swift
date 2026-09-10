//
//  MediaPipeCropPreprocessor.swift
//  MPSMediaPipe
//

import Foundation
import Metal
import simd

/// Encodes a rotated crop + normalize directly from a texture into an NHWC
/// float32 buffer — no CVPixelBuffer, no Vision, no CPU-side pixel copy.
/// The output buffer is `.storageModeShared` (genuinely unified CPU/GPU
/// memory on Apple Silicon) and fed directly into MediaPipeTFLiteMPSGraph
/// as an MPSGraphTensorData -- "GPU writes directly into the model's input
/// buffer," no intermediate copy.
///
/// `outputPixelRange` is fixed per instance (not per call), since a given
/// preprocessor is always paired with one model: BlazePalm/BlazeFace's
/// landmark models both normalize to [0,1], but BlazeFace's *detector*
/// normalizes to [-1,1] — confirmed against each model's own
/// ImageToTensorCalculatorOptions.output_tensor_float_range, not assumed
/// from BlazePalm's convention.
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

    /// Convenience entry point for callers that want preprocessing as its
    /// own submission (a synchronous inference path). An asynchronous path
    /// should use the command-buffer overload below so preprocessing and
    /// MPSGraph inference share one submission with no intermediate wait.
    ///
    /// `textureTransform`/`presentationSize` describe how `texture`'s own
    /// storage maps onto presentation pixels (an arbitrary rotation/flip a
    /// caller's own image type may apply) -- pass `matrix_identity_float4x4`
    /// and the texture's own pixel dimensions if the texture has no such
    /// transform.
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
    /// normalized [0,1]; `rotationRadians` is
    /// MediaPipeSSDDetectorDecoder.computeRotation's own convention
    /// (top-left/Y-down, independent of the coordinate's origin choice —
    /// see that type's doc comment). Encodes onto `commandBuffer` without
    /// committing, so MPSGraph inference can be appended to the same
    /// command buffer (an asynchronous submission path) — the caller is
    /// responsible for committing (and, for the convenience overload above,
    /// waiting).
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
