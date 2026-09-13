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
///
/// Up to `maxFramesInFlight` overlapping `encode()` calls may be in flight
/// on one instance at once (each gets its own scratch buffer); `encode()`
/// throws if that capacity is already used. Use a separate instance, or a
/// larger `maxFramesInFlight`, for more concurrent projections.
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

    /// One scratch buffer per in-flight slot, written from the CPU side at
    /// the start of that slot's `encode()` call -- see `acquireSlot`.
    private let maskBuffers: [MTLBuffer]

    private let maxFramesInFlight: Int
    private let slotSemaphore: DispatchSemaphore
    private let slotLock = NSLock()
    private var freeSlots: [Int]

    public init(device: MTLDevice, maskWidth: Int, maskHeight: Int, maxFramesInFlight: Int = 3) throws
    {
        self.maskWidth = maskWidth
        self.maskHeight = maskHeight
        self.maxFramesInFlight = maxFramesInFlight
        self.slotSemaphore = DispatchSemaphore(value: maxFramesInFlight)
        self.freeSlots = Array(0..<maxFramesInFlight)

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
        self.maskBuffers = try (0..<maxFramesInFlight).map { slot in
            guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else
            {
                throw MediaPipeMPSGraphError("Could not allocate MediaPipe segmentation mask buffer")
            }
            buffer.label = "MediaPipe segmentation mask logits \(maskWidth)x\(maskHeight) [slot \(slot)]"
            return buffer
        }
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
        destinationTexture: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) throws
    {
        guard maskValues.count == self.maskWidth * self.maskHeight else
        {
            throw MediaPipeMPSGraphError("MediaPipe segmentation mask values size mismatch")
        }

        let slot = try self.acquireSlot()

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else
        {
            self.releaseSlot(slot)
            throw MediaPipeMPSGraphError("Could not create MediaPipe segmentation mask warp compute pass")
        }
        commandBuffer.addCompletedHandler { [weak self] _ in self?.releaseSlot(slot) }

        let maskBuffer = self.maskBuffers[slot]
        maskValues.withUnsafeBytes { rawBuffer in
            maskBuffer.contents().copyMemory(from: rawBuffer.baseAddress!, byteCount: rawBuffer.count)
        }

        let presentationSize = simd_float2(Float(destinationTexture.width), Float(destinationTexture.height))
        let centerYTopLeft = 1 - centerNormalizedBottomLeft.y
        var uniforms = Uniforms(
            centerPixels: simd_float2(centerNormalizedBottomLeft.x * presentationSize.x, centerYTopLeft * presentationSize.y),
            rectSizePixels: simd_float2(sizeNormalized.x * presentationSize.x, sizeNormalized.y * presentationSize.y),
            rotationRadians: rotationRadians,
            maskSize: simd_uint2(UInt32(self.maskWidth), UInt32(self.maskHeight)),
            outputSize: simd_uint2(UInt32(destinationTexture.width), UInt32(destinationTexture.height)),
            applySigmoid: applySigmoid ? 1 : 0
        )

        encoder.label = "MediaPipe segmentation mask inverse warp [slot \(slot)]"
        encoder.setComputePipelineState(self.pipeline)
        encoder.setBuffer(maskBuffer, offset: 0, index: 0)
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

    /// Non-blocking: this type has no synchronous/blocking `encode()`
    /// overload, so every caller is on the "never block" async path --
    /// throws immediately if all `maxFramesInFlight` slots are busy, rather
    /// than waiting for the GPU to catch up.
    private func acquireSlot() throws -> Int
    {
        guard self.slotSemaphore.wait(timeout: .now()) == .success else
        {
            throw MediaPipeMPSGraphError("MediaPipeSegmentationMaskProjector.encode() called with all \(self.maxFramesInFlight) in-flight slots busy -- increase maxFramesInFlight, or use a separate instance for more concurrent projections.")
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
