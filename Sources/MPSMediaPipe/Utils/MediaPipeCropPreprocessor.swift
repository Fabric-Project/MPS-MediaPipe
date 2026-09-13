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
///
/// Up to `maxFramesInFlight` overlapping `encode()` calls may be in flight
/// on one instance at once (each gets its own scratch buffer). The
/// command-buffer overload throws immediately if that capacity is already
/// used, rather than blocking -- it exists specifically for callers that
/// can't afford to block. The synchronous convenience overload blocks
/// until a slot is free instead, since it already blocks for the GPU work
/// itself. Use a separate instance, or a larger `maxFramesInFlight`, for
/// more concurrency.
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

    /// One scratch buffer per in-flight slot -- see `acquireSlot`.
    private let outputBuffers: [MTLBuffer]

    private let maxFramesInFlight: Int
    private let slotSemaphore: DispatchSemaphore
    private let slotLock = NSLock()
    private var freeSlots: [Int]

    public init(device: MTLDevice, outputWidth: Int, outputHeight: Int, outputPixelRange: (min: Float, max: Float) = (0, 1), maxFramesInFlight: Int = 3) throws
    {
        self.outputWidth = outputWidth
        self.outputHeight = outputHeight
        self.outputPixelRange = simd_float2(outputPixelRange.min, outputPixelRange.max)
        self.maxFramesInFlight = maxFramesInFlight
        self.slotSemaphore = DispatchSemaphore(value: maxFramesInFlight)
        self.freeSlots = Array(0..<maxFramesInFlight)

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
        self.outputBuffers = try (0..<maxFramesInFlight).map { slot in
            guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else
            {
                throw MediaPipeMPSGraphError("Could not allocate MediaPipe crop buffer")
            }
            buffer.label = "MediaPipe crop NHWC \(outputWidth)x\(outputHeight) [slot \(slot)]"
            return buffer
        }
    }

    /// Synchronous convenience overload -- commits its own command buffer
    /// and waits. Use the command-buffer overload to share one submission
    /// with subsequent MPSGraph inference.
    ///
    /// `textureTransform` describes how `texture` maps onto presentation
    /// pixels; pass identity if there's no transform. The presentation size
    /// this needs internally is derived from `texture`'s own dimensions and
    /// `textureTransform` -- not a separate parameter, so it can never
    /// disagree with the texture actually being read.
    @discardableResult
    public func encode(
        texture: MTLTexture,
        textureTransform: simd_float4x4,
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

        let slot = try self.acquireSlot(blocking: true)
        let outputBuffer = try self.encode(
            slot: slot,
            texture: texture,
            textureTransform: textureTransform,
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
    /// waits, if needed). Throws immediately (never blocks) if all
    /// `maxFramesInFlight` slots are already in use.
    @discardableResult
    public func encode(
        texture: MTLTexture,
        textureTransform: simd_float4x4,
        centerNormalizedBottomLeft: simd_float2,
        sizeNormalized: simd_float2,
        rotationRadians: Float,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLBuffer
    {
        let slot = try self.acquireSlot(blocking: false)
        return try self.encode(
            slot: slot,
            texture: texture,
            textureTransform: textureTransform,
            centerNormalizedBottomLeft: centerNormalizedBottomLeft,
            sizeNormalized: sizeNormalized,
            rotationRadians: rotationRadians,
            commandBuffer: commandBuffer
        )
    }

    private func encode(
        slot: Int,
        texture: MTLTexture,
        textureTransform: simd_float4x4,
        centerNormalizedBottomLeft: simd_float2,
        sizeNormalized: simd_float2,
        rotationRadians: Float,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLBuffer
    {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else
        {
            self.releaseSlot(slot)
            throw MediaPipeMPSGraphError("Could not create MediaPipe crop compute pass")
        }
        commandBuffer.addCompletedHandler { [weak self] _ in self?.releaseSlot(slot) }

        // Matches FabricImage.calculatePresentationSize()/the shader-side
        // fabricPresentationSize helper exactly: transform (texture.width,
        // texture.height) as a direction (w=0), not a point, so translation
        // in the transform doesn't leak in -- only rotation/flip/scale do.
        let transformedSize = simd_abs(textureTransform * simd_float4(Float(texture.width), Float(texture.height), 0, 0))
        let presentationSize = simd_float2(transformedSize.x, transformedSize.y)

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

        let outputBuffer = self.outputBuffers[slot]

        encoder.label = "MediaPipe crop, rotate, and normalize [slot \(slot)]"
        encoder.setComputePipelineState(self.pipeline)
        encoder.setTexture(texture, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 0)
        encoder.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)

        let threadgroupWidth = self.pipeline.threadExecutionWidth
        let threadgroupHeight = max(1, min(8, self.pipeline.maxTotalThreadsPerThreadgroup / threadgroupWidth))
        encoder.dispatchThreads(
            MTLSize(width: self.outputWidth, height: self.outputHeight, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupWidth, height: threadgroupHeight, depth: 1)
        )
        encoder.endEncoding()

        return outputBuffer
    }

    /// `blocking: true` (the synchronous overload) waits for a slot to
    /// free up, since that overload already blocks for the GPU work
    /// itself -- it can't deadlock, since every acquired slot is always
    /// released via `commandBuffer.addCompletedHandler` regardless of
    /// which overload acquired it. `blocking: false` (the command-buffer
    /// overload) fails fast instead, preserving that overload's
    /// never-blocks-the-caller contract.
    private func acquireSlot(blocking: Bool) throws -> Int
    {
        if blocking
        {
            self.slotSemaphore.wait()
        }
        else
        {
            guard self.slotSemaphore.wait(timeout: .now()) == .success else
            {
                throw MediaPipeMPSGraphError("MediaPipeCropPreprocessor.encode() called with all \(self.maxFramesInFlight) in-flight slots busy -- increase maxFramesInFlight, or use a separate instance for more concurrent crops.")
            }
        }
        self.slotLock.lock()
        defer { self.slotLock.unlock() }
        return self.freeSlots.removeLast()
    }

    private func releaseSlot(_ slot: Int)
    {
        self.slotLock.lock()
        self.freeSlots.append(slot)
        self.slotLock.unlock()
        self.slotSemaphore.signal()
    }
}
