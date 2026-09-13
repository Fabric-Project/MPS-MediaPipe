// MediaPipeModelWeights.swift
//
// Loads the flat binary blob + JSON manifest a MediaPipe TFLite model was
// exported to (name -> {offset, shape, dtype}, keyed by stringified TFLite
// tensor index) and hands out MPSGraph constant tensors by name.

import Foundation
import MetalPerformanceShadersGraph

public final class MediaPipeModelWeights
{
    private struct Entry: Decodable
    {
        let offset: Int
        let shape: [Int]
        let dtype: String
    }

    private let manifest: [String: Entry]
    private let data: Data

    public init(binaryURL: URL, manifestURL: URL) throws
    {
        self.data = try Data(contentsOf: binaryURL)
        let manifestData = try Data(contentsOf: manifestURL)
        self.manifest = try JSONDecoder().decode([String: Entry].self, from: manifestData)
    }

    /// Raw float32 values for a named tensor, in the same flattened
    /// row-major order the export tooling's own .numpy() call produced.
    public func floatArray(named name: String) throws -> [Float]
    {
        guard let entry = manifest[name] else
        {
            throw MediaPipeMPSGraphError("MediaPipeModelWeights: missing tensor '\(name)'")
        }

        let elementCount = entry.shape.reduce(1, *)
        let byteOffset = entry.offset * MemoryLayout<Float>.stride
        let byteCount = elementCount * MemoryLayout<Float>.stride

        guard byteOffset >= 0, byteCount >= 0, byteOffset + byteCount <= self.data.count else
        {
            throw MediaPipeMPSGraphError("MediaPipeModelWeights: tensor '\(name)' (offset \(byteOffset), \(byteCount) bytes) exceeds the loaded binary's \(self.data.count) bytes -- manifest/binary mismatch")
        }

        var values = [Float](repeating: 0, count: elementCount)
        self.data.withUnsafeBytes { rawBuffer in
            let source = rawBuffer.baseAddress!.advanced(by: byteOffset)
            values.withUnsafeMutableBytes { destination in
                destination.copyMemory(from: UnsafeRawBufferPointer(start: source, count: byteCount))
            }
        }
        return values
    }

    public func shape(named name: String) throws -> [Int]
    {
        guard let entry = manifest[name] else
        {
            throw MediaPipeMPSGraphError("MediaPipeModelWeights: missing tensor '\(name)'")
        }
        return entry.shape
    }

    /// Builds an MPSGraph constant tensor from the named weight, in its
    /// native export-time shape (OIHW for conv weights, [out, in] for
    /// linear weights, etc.) -- callers transpose/reshape as needed per op.
    public func constant(_ graph: MPSGraph, named name: String) throws -> MPSGraphTensor
    {
        let values = try self.floatArray(named: name)
        let shape = try self.shape(named: name).map { NSNumber(value: $0) }
        return graph.constant(Data(bytes: values, count: values.count * MemoryLayout<Float>.stride), shape: shape, dataType: .float32)
    }
}
