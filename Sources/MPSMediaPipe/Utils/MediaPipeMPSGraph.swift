// MediaPipeMPSGraph.swift
//
// From-scratch MPSGraph reimplementation of any MediaPipe "Blaze"-family
// TFLite model, run directly on GPU -- no CoreML. Shared by BlazePalm/
// BlazeHand (hand detector + landmark model) and BlazeFace/FaceMesh (face
// detector + landmark model): a single generic interpreter that walks a
// fully-resolved TFLite op graph (padding/groups/layout already resolved
// offline) rather than a per-model port. The op set it handles is the one
// the two bundled model families need, not arbitrary TFLite graphs -- an
// unrecognized op throws rather than producing a silently wrong result.

import Foundation
import Metal
import MetalPerformanceShadersGraph

public final class MediaPipeMPSGraph
{
    private let graph = MPSGraph()
    private let device: MPSGraphDevice
    private let commandQueue: MTLCommandQueue
    private let inputTensor: MPSGraphTensor
    private let outputTensors: [MPSGraphTensor]
    private let executable: MPSGraphExecutable

    private let maxFramesInFlight: Int
    private let slotSemaphore: DispatchSemaphore
    private let slotLock = NSLock()
    private var freeSlots: [Int]

    /// A reference type, not a plain array element: two different slots'
    /// completion handlers can run concurrently on different threads, and
    /// each must mutate genuinely separate storage -- a shared Swift
    /// `Array`'s backing buffer is not safe for unsynchronized concurrent
    /// writes to different indices, even though each thread only ever
    /// touches its own slot. One heap-allocated box per slot sidesteps that
    /// entirely: there is no shared backing store between slots at all.
    private final class OutputBufferSlot
    {
        var buffers: [[Float]] = []
    }

    /// One per in-flight slot -- see `floatArray(from:slot:cacheIndex:)`.
    /// The outer array itself is only ever read after init (never
    /// resized), so no synchronization is needed for that; only each
    /// slot's own `buffers` mutates, and only from that slot's own thread.
    private let outputBufferCache: [OutputBufferSlot]

    public let inputWidth: Int
    public let inputHeight: Int

    /// Loads `<name>_weights.bin`/`<name>_weights.json` (MediaPipeModelWeights'
    /// format) and `<name>_ops.json` (the resolved op list). Up to
    /// `maxFramesInFlight` overlapping `run()`/`submit()` calls may be in
    /// flight on one instance at once.
    public init(weightsBinaryURL: URL, weightsManifestURL: URL, opsJSONURL: URL, inputWidth: Int, inputHeight: Int, commandQueue: MTLCommandQueue, maxFramesInFlight: Int = 3) throws
    {
        let weights = try MediaPipeModelWeights(binaryURL: weightsBinaryURL, manifestURL: weightsManifestURL)
        self.device = MPSGraphDevice(mtlDevice: commandQueue.device)
        self.commandQueue = commandQueue
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        self.maxFramesInFlight = maxFramesInFlight
        self.slotSemaphore = DispatchSemaphore(value: maxFramesInFlight)
        self.freeSlots = Array(0..<maxFramesInFlight)
        self.outputBufferCache = (0..<maxFramesInFlight).map { _ in OutputBufferSlot() }

        let opsData = try Data(contentsOf: opsJSONURL)
        guard let opsJSON = try JSONSerialization.jsonObject(with: opsData) as? [String: Any] else
        {
            throw MediaPipeMPSGraphError("MediaPipeMPSGraph: ops JSON root is not an object")
        }
        guard let opsList = opsJSON["ops"] as? [[String: Any]] else
        {
            throw MediaPipeMPSGraphError("MediaPipeMPSGraph: ops JSON missing 'ops' array")
        }
        guard let inputIdsRaw = opsJSON["inputIds"] as? [Any] else
        {
            throw MediaPipeMPSGraphError("MediaPipeMPSGraph: ops JSON missing 'inputIds'")
        }
        guard let outputIdsRaw = opsJSON["outputIds"] as? [Any] else
        {
            throw MediaPipeMPSGraphError("MediaPipeMPSGraph: ops JSON missing 'outputIds'")
        }
        let inputIds = try inputIdsRaw.map { try Self.intValue($0, context: "ops JSON 'inputIds'") }
        let outputIds = try outputIdsRaw.map { try Self.intValue($0, context: "ops JSON 'outputIds'") }
        guard let firstInputId = inputIds.first else
        {
            throw MediaPipeMPSGraphError("MediaPipeMPSGraph: ops JSON 'inputIds' is empty")
        }

        // Input arrives NHWC (MediaPipeCropPreprocessor's output layout);
        // permute to NCHW immediately -- every op after this point
        // operates in NCHW.
        let inputPlaceholder = self.graph.placeholder(
            shape: [1, NSNumber(value: inputHeight), NSNumber(value: inputWidth), 3],
            dataType: .float32,
            name: "input"
        )
        self.inputTensor = inputPlaceholder

        var env: [Int: MPSGraphTensor] = [:]
        env[firstInputId] = self.graph.transpose(inputPlaceholder, permutation: [0, 3, 1, 2], name: nil)

        for opDict in opsList
        {
            let op = try Op(opDict)
            let result = try Self.build(op: op, graph: self.graph, weights: weights, env: env)
            guard let firstOutput = op.outputs.first else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(op.type)' has no outputs")
            }
            env[firstOutput] = result
        }

        self.outputTensors = try outputIds.map { id in
            guard let tensor = env[id] else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: output tensor id \(id) was never produced by any op")
            }
            return tensor
        }

        let inputType = MPSGraphShapedType(shape: inputPlaceholder.shape!, dataType: .float32)
        let compilationDescriptor = Self.performanceCompilationDescriptor()
        self.executable = self.graph.compile(
            with: self.device,
            feeds: [inputPlaceholder: inputType],
            targetTensors: self.outputTensors,
            targetOperations: nil,
            compilationDescriptor: compilationDescriptor
        )
        self.executable.specialize(with: self.device, inputTypes: [inputType], compilationDescriptor: compilationDescriptor)
    }

    private static func performanceCompilationDescriptor() -> MPSGraphCompilationDescriptor
    {
        let descriptor = MPSGraphCompilationDescriptor()
        descriptor.optimizationLevel = .level1
        descriptor.waitForCompilationCompletion = true
        if #available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        {
            descriptor.reducedPrecisionFastMath = .allowFP16Intermediates
        }
        return descriptor
    }

    /// `inputBuffer` is NHWC float32, matching MediaPipeCropPreprocessor's
    /// output -- fed straight into the compiled executable, no CPU
    /// round-trip. Returns each output tensor's flattened values in the
    /// model's own output order. Blocks for a free slot if
    /// `maxFramesInFlight` calls are already in flight (this call is
    /// already synchronous, so blocking briefly here costs nothing extra).
    public func run(inputBuffer: MTLBuffer) -> [[Float]]
    {
        let slot = self.acquireSlotBlocking()
        defer { self.releaseSlot(slot) }

        let inputData = MPSGraphTensorData(inputBuffer, shape: self.inputTensor.shape!, dataType: .float32)
        let results = self.executable.run(with: self.commandQueue, inputs: [inputData], results: nil, executionDescriptor: nil)
        return results.enumerated().map { index, tensorData in self.floatArray(from: tensorData, slot: slot, cacheIndex: index) }
    }

    /// Async counterpart: encodes onto `commandBuffer` without waiting;
    /// drops the call (returns false, never invokes `completion`) if all
    /// `maxFramesInFlight` slots are already in flight, rather than
    /// blocking the caller. `commandBuffer` must already contain the crop
    /// preprocessor's encode so the GPU sees crop-then-inference in order.
    @discardableResult
    public func submit(inputBuffer: MTLBuffer, commandBuffer: MTLCommandBuffer, completion: @escaping (Result<[[Float]], any Error>) -> Void) -> Bool
    {
        guard let slot = self.acquireSlotNonBlocking() else { return false }

        let inputData = MPSGraphTensorData(inputBuffer, shape: self.inputTensor.shape!, dataType: .float32)
        let executionDescriptor = MPSGraphExecutableExecutionDescriptor()
        executionDescriptor.waitUntilCompleted = false
        executionDescriptor.completionHandler = { [weak self] results, error in
            guard let self else { return }
            defer { self.releaseSlot(slot) }

            if let error
            {
                completion(.failure(error))
            }
            else
            {
                completion(.success(results.enumerated().map { index, tensorData in self.floatArray(from: tensorData, slot: slot, cacheIndex: index) }))
            }
        }

        let mpsCommandBuffer = MPSCommandBuffer(commandBuffer: commandBuffer)
        _ = self.executable.encode(to: mpsCommandBuffer, inputs: [inputData], results: nil, executionDescriptor: executionDescriptor)
        mpsCommandBuffer.commit()
        return true
    }

    /// Waits (if necessary) for a free slot -- used by `run()`, which is
    /// already synchronous, so blocking briefly here costs nothing extra.
    private func acquireSlotBlocking() -> Int
    {
        self.slotSemaphore.wait()
        self.slotLock.lock()
        defer { self.slotLock.unlock() }
        return self.freeSlots.removeLast()
    }

    /// Returns nil immediately instead of waiting if all
    /// `maxFramesInFlight` slots are busy -- used by `submit()`, which
    /// exists specifically so callers never block.
    private func acquireSlotNonBlocking() -> Int?
    {
        guard self.slotSemaphore.wait(timeout: .now()) == .success else { return nil }
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

    /// Reads `tensorData` into slot `slot`'s cached per-output-index
    /// buffer, resizing only when the tensor's element count actually
    /// changes (it never does, in practice, for a given compiled
    /// executable). If a caller retains a previous call's returned array
    /// past the next call using the same slot, Swift's copy-on-write makes
    /// this allocate exactly like before for that one call -- this avoids
    /// the allocation in the common case (each frame's output read and
    /// discarded before the next), not unconditionally.
    private func floatArray(from tensorData: MPSGraphTensorData, slot: Int, cacheIndex: Int) -> [Float]
    {
        let count = tensorData.shape.map(\.intValue).reduce(1, *)
        let cache = self.outputBufferCache[slot]
        while cache.buffers.count <= cacheIndex { cache.buffers.append([]) }
        if cache.buffers[cacheIndex].count != count
        {
            cache.buffers[cacheIndex] = [Float](repeating: 0, count: count)
        }
        cache.buffers[cacheIndex].withUnsafeMutableBufferPointer { buffer in
            tensorData.mpsndarray().readBytes(buffer.baseAddress!, strideBytes: nil)
        }
        return cache.buffers[cacheIndex]
    }

    // MARK: - Op parsing (fully resolved by dump_tflite_graph.py already --
    // padding/groups/dilation/layout, no re-derivation needed here). Malformed
    // input throws rather than crashing: a hand-edited or mismatched ops.json
    // is exactly the kind of input this needs to fail cleanly on.

    private struct Op
    {
        let type: String
        let inputs: [Int]
        let outputs: [Int]
        let options: [String: Any]

        init(_ dict: [String: Any]) throws
        {
            guard let type = dict["type"] as? String else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op missing 'type'")
            }
            guard let inputsRaw = dict["inputs"] as? [Any] else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(type)' missing 'inputs'")
            }
            guard let outputsRaw = dict["outputs"] as? [Any] else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(type)' missing 'outputs'")
            }
            guard let options = dict["options"] as? [String: Any] else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(type)' missing 'options'")
            }

            self.type = type
            self.inputs = try inputsRaw.map { try MediaPipeMPSGraph.intValue($0, context: "op '\(type)' inputs") }
            self.outputs = try outputsRaw.map { try MediaPipeMPSGraph.intValue($0, context: "op '\(type)' outputs") }
            self.options = options
        }

        /// The raw tensor id at `self.inputs[index]` -- as opposed to
        /// `MediaPipeMPSGraph.input(_:in:)`, which additionally resolves
        /// that id against the graph-build environment.
        func inputId(_ index: Int) throws -> Int
        {
            guard self.inputs.indices.contains(index) else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(self.type)' expected at least \(index + 1) input(s), has \(self.inputs.count)")
            }
            return self.inputs[index]
        }

        func intPair(_ key: String) throws -> (Int, Int)
        {
            guard let raw = self.options[key] as? [Any], raw.count == 2 else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(self.type)' option '\(key)' expected a 2-element array")
            }
            let array = try raw.map { try MediaPipeMPSGraph.intValue($0, context: "op '\(self.type)' option '\(key)'") }
            return (array[0], array[1])
        }

        func string(_ key: String) throws -> String
        {
            guard let value = self.options[key] as? String else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(self.type)' option '\(key)' expected a string")
            }
            return value
        }

        func int(_ key: String) throws -> Int
        {
            guard let value = self.options[key] as? NSNumber else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(self.type)' option '\(key)' expected a number")
            }
            return value.intValue
        }

        func bool(_ key: String) throws -> Bool
        {
            guard let value = self.options[key] as? NSNumber else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(self.type)' option '\(key)' expected a bool")
            }
            return value.boolValue
        }

        func intArray(_ key: String) throws -> [Int]
        {
            guard let raw = self.options[key] as? [Any] else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(self.type)' option '\(key)' expected an array")
            }
            return try raw.map { try MediaPipeMPSGraph.intValue($0, context: "op '\(self.type)' option '\(key)'") }
        }

        func fourIntsOrNil(_ key: String) throws -> (Int, Int, Int, Int)?
        {
            guard let value = self.options[key], (value is NSNull) == false else { return nil }
            guard let raw = value as? [Any], raw.count == 4 else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(self.type)' option '\(key)' expected a 4-element array")
            }
            let array = try raw.map { try MediaPipeMPSGraph.intValue($0, context: "op '\(self.type)' option '\(key)'") }
            return (array[0], array[1], array[2], array[3])
        }

        /// [[before,after], [before,after], [before,after], [before,after]] in NCHW dim order.
        func paddingPairs(_ key: String) throws -> [(Int, Int)]
        {
            guard let raw = self.options[key] as? [Any] else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(self.type)' option '\(key)' expected an array")
            }
            return try raw.map { pair in
                guard let pairArray = pair as? [Any], pairArray.count == 2 else
                {
                    throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(self.type)' option '\(key)' expected pairs of 2 numbers")
                }
                let array = try pairArray.map { try MediaPipeMPSGraph.intValue($0, context: "op '\(self.type)' option '\(key)'") }
                return (array[0], array[1])
            }
        }
    }

    private static func intValue(_ any: Any, context: String) throws -> Int
    {
        guard let number = any as? NSNumber else
        {
            throw MediaPipeMPSGraphError("MediaPipeMPSGraph: expected a number in \(context)")
        }
        return number.intValue
    }

    // MARK: - Op building

    private static func build(op: Op, graph: MPSGraph, weights: MediaPipeModelWeights, env: [Int: MPSGraphTensor]) throws -> MPSGraphTensor
    {
        func input(_ index: Int) throws -> MPSGraphTensor
        {
            let id = try op.inputId(index)
            guard let tensor = env[id] else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(op.type)' input tensor id \(id) not found in graph")
            }
            return tensor
        }

        /// Loads the tensor at `op.inputs[index]`, reshapes it to
        /// `[1,C,1,1]` for NCHW broadcasting, and adds it to `x` -- the
        /// shared bias-add tail of CONV_2D and CUSTOM_Convolution2DTransposeBias.
        /// Only applies when that input is actually present (both ops treat
        /// their bias input as optional).
        func addBiasIfPresent(_ x: MPSGraphTensor) throws -> MPSGraphTensor
        {
            guard op.inputs.count > 2 else { return x }
            let biasId = try op.inputId(2)
            let biasShape = try weights.shape(named: String(biasId))
            guard let outChannels = biasShape.first else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(op.type)' bias tensor has an empty shape")
            }
            let bias = try weights.constant(graph, named: String(biasId))
            let biasReshaped = graph.reshape(bias, shape: [1, NSNumber(value: outChannels), 1, 1], name: nil)
            return graph.addition(x, biasReshaped, name: nil)
        }

        switch op.type
        {
        case "CONV_2D", "DEPTHWISE_CONV_2D":
            var x = try input(0)
            if let (pl, pr, pt, pb) = try op.fourIntsOrNil("pre_pad")
            {
                x = graph.padTensor(x, with: .constant,
                                     leftPadding: [0, 0, NSNumber(value: pt), NSNumber(value: pl)],
                                     rightPadding: [0, 0, NSNumber(value: pb), NSNumber(value: pr)],
                                     constantValue: 0, name: nil)
            }
            let weightTensor = try weights.constant(graph, named: String(op.inputId(1)))
            let (strideY, strideX) = try op.intPair("stride")
            let (dilationY, dilationX) = try op.intPair("dilation")
            let (padY, padX) = try op.intPair("conv_pad")
            let groups = try op.int("groups")
            guard let descriptor = MPSGraphConvolution2DOpDescriptor(
                strideInX: strideX, strideInY: strideY,
                dilationRateInX: dilationX, dilationRateInY: dilationY,
                groups: groups,
                paddingLeft: padX, paddingRight: padX, paddingTop: padY, paddingBottom: padY,
                paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW
            ) else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(op.type)' could not build a convolution descriptor")
            }
            let y = graph.convolution2D(x, weights: weightTensor, descriptor: descriptor, name: nil)
            return try Self.activate(try addBiasIfPresent(y), try op.string("activation"), graph: graph)

        case "PRELU":
            let x = try input(0)
            let alphaId = try op.inputId(1)
            let alpha = try weights.constant(graph, named: String(alphaId))
            let alphaShape = try weights.shape(named: String(alphaId))
            guard let alphaChannels = alphaShape.first else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(op.type)' alpha tensor has an empty shape")
            }
            let alphaReshaped = graph.reshape(alpha, shape: [1, NSNumber(value: alphaChannels), 1, 1], name: nil)
            let positive = graph.reLU(with: x, name: nil)
            let negative = graph.subtraction(x, positive, name: nil) // min(x, 0)
            let scaledNegative = graph.multiplication(negative, alphaReshaped, name: nil)
            return graph.addition(positive, scaledNegative, name: nil)

        case "ADD":
            let sum = graph.addition(try input(0), try input(1), name: nil)
            return try Self.activate(sum, try op.string("activation"), graph: graph)

        case "MUL":
            // A squeeze-and-excitation gate: both operands are already
            // NCHW activations by this point ([N,C,H,W] times [N,C,1,1]),
            // so plain broadcasting multiply is correct.
            let product = graph.multiplication(try input(0), try input(1), name: nil)
            return try Self.activate(product, try op.string("activation"), graph: graph)

        case "MAX_POOL_2D":
            var x = try input(0)
            if let (pl, pr, pt, pb) = try op.fourIntsOrNil("pre_pad")
            {
                x = graph.padTensor(x, with: .constant,
                                     leftPadding: [0, 0, NSNumber(value: pt), NSNumber(value: pl)],
                                     rightPadding: [0, 0, NSNumber(value: pb), NSNumber(value: pr)],
                                     constantValue: Double(-Float.greatestFiniteMagnitude), name: nil)
            }
            let (filterH, filterW) = try op.intPair("filter")
            let (strideY, strideX) = try op.intPair("stride")
            guard let descriptor = MPSGraphPooling2DOpDescriptor(
                kernelWidth: filterW, kernelHeight: filterH,
                strideInX: strideX, strideInY: strideY,
                dilationRateInX: 1, dilationRateInY: 1,
                paddingLeft: 0, paddingRight: 0, paddingTop: 0, paddingBottom: 0,
                paddingStyle: .explicit, dataLayout: .NCHW
            ) else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(op.type)' could not build a pooling descriptor")
            }
            let y = graph.maxPooling2D(withSourceTensor: x, descriptor: descriptor, name: nil)
            return try Self.activate(y, try op.string("activation"), graph: graph)

        case "AVERAGE_POOL_2D":
            // TFLite divides by the count of valid (non-padded) elements,
            // not the full kernel area -- explicit asymmetric padding plus
            // includeZeroPadToAverage=false matches this exactly, no
            // separate pad step needed.
            let x = try input(0)
            let (avgFilterH, avgFilterW) = try op.intPair("filter")
            let (avgStrideY, avgStrideX) = try op.intPair("stride")
            let (padLeft, padRight, padTop, padBottom): (Int, Int, Int, Int)
            if let (pl, pr, pt, pb) = try op.fourIntsOrNil("pre_pad")
            {
                (padLeft, padRight, padTop, padBottom) = (pl, pr, pt, pb)
            }
            else
            {
                let (padY, padX) = try op.intPair("conv_pad")
                (padLeft, padRight, padTop, padBottom) = (padX, padX, padY, padY)
            }
            guard let avgDescriptor = MPSGraphPooling2DOpDescriptor(
                kernelWidth: avgFilterW, kernelHeight: avgFilterH,
                strideInX: avgStrideX, strideInY: avgStrideY,
                dilationRateInX: 1, dilationRateInY: 1,
                paddingLeft: padLeft, paddingRight: padRight, paddingTop: padTop, paddingBottom: padBottom,
                paddingStyle: .explicit, dataLayout: .NCHW
            ) else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(op.type)' could not build a pooling descriptor")
            }
            avgDescriptor.includeZeroPadToAverage = false
            let avgY = graph.avgPooling2D(withSourceTensor: x, descriptor: avgDescriptor, name: nil)
            return try Self.activate(avgY, try op.string("activation"), graph: graph)

        case "PAD":
            let pairs = try op.paddingPairs("paddings") // NCHW order already
            let left = pairs.map { NSNumber(value: $0.0) }
            let right = pairs.map { NSNumber(value: $0.1) }
            return graph.padTensor(try input(0), with: .constant, leftPadding: left, rightPadding: right, constantValue: 0, name: nil)

        case "RESIZE_BILINEAR":
            let sizeArray = try op.intArray("size")
            guard sizeArray.count == 2 else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(op.type)' option 'size' expected 2 values")
            }
            let alignCorners = try op.bool("align_corners")
            return graph.resize(
                try input(0), size: [NSNumber(value: sizeArray[0]), NSNumber(value: sizeArray[1])],
                mode: .bilinear, centerResult: !alignCorners, alignCorners: alignCorners,
                layout: .NCHW, name: nil
            )

        case "RESHAPE":
            let fromFourD = try op.bool("from_4d")
            let toFourD = try op.bool("to_4d")
            var x = try input(0)
            if fromFourD
            {
                x = graph.transpose(x, permutation: [0, 2, 3, 1], name: nil) // NCHW -> NHWC semantics
            }
            var shape = try op.intArray("shape").map { NSNumber(value: $0) }
            guard shape.isEmpty == false else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(op.type)' option 'shape' is empty")
            }
            shape[0] = 1 // batch, always 1 here
            var y = graph.reshape(x, shape: shape, name: nil)
            if toFourD
            {
                y = graph.transpose(y, permutation: [0, 3, 1, 2], name: nil) // NHWC -> NCHW
            }
            return y

        case "CONCATENATION":
            let tensors = try op.inputs.indices.map { try input($0) }
            let axis = try op.int("axis")
            let y = graph.concatTensors(tensors, dimension: axis, name: nil)
            return try Self.activate(y, try op.string("activation"), graph: graph)

        case "MEAN":
            let axes = try op.intArray("axes").map { NSNumber(value: $0) }
            let y = graph.mean(of: try input(0), axes: axes, name: nil) // always keeps dims
            if try op.bool("keep_dims") == false
            {
                guard let yShape = y.shape else
                {
                    throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(op.type)' result has no shape")
                }
                let keptShape = yShape.map(\.intValue)
                let reducedAxes = Set(axes.map(\.intValue))
                let squeezedShape = keptShape.enumerated().filter { !reducedAxes.contains($0.offset) }.map { NSNumber(value: $0.element) }
                return graph.reshape(y, shape: squeezedShape, name: nil)
            }
            return y

        case "FULLY_CONNECTED":
            let x = try input(0)
            let weightId = try op.inputId(1)
            let weightTensor = try weights.constant(graph, named: String(weightId)) // [out, in]
            let weightTransposed = graph.transposeTensor(weightTensor, dimension: 0, withDimension: 1, name: nil)
            var y = graph.matrixMultiplication(primary: x, secondary: weightTransposed, name: nil)
            if op.inputs.count > 2
            {
                // Bias here matches the matmul's own 2D [batch, out] output
                // shape directly -- no [1,C,1,1] NCHW-broadcast reshape
                // needed, unlike the convolution ops' bias-add.
                let bias = try weights.constant(graph, named: String(op.inputId(2)))
                y = graph.addition(y, bias, name: nil)
            }
            return try Self.activate(y, try op.string("activation"), graph: graph)

        case "DEPTH_TO_SPACE":
            // TF's channel decomposition is (i*b+j)*C_out+c (block-position-
            // major, channel-minor) -- not the c*b^2+i*b+j order
            // PixelShuffle-style approaches assume -- so this reshapes/
            // transposes by hand.
            let blockSize = try op.int("block_size")
            guard blockSize > 0 else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(op.type)' option 'block_size' must be positive")
            }
            let x = try input(0)
            guard let xShape = x.shape, xShape.count == 4 else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(op.type)' expected a rank-4 input")
            }
            let shape = xShape.map(\.intValue)
            let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3])
            guard c % (blockSize * blockSize) == 0 else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(op.type)' channel count \(c) is not divisible by block_size^2 (\(blockSize * blockSize))")
            }
            let cOut = c / (blockSize * blockSize)
            let reshaped = graph.reshape(x, shape: [
                NSNumber(value: n), NSNumber(value: blockSize), NSNumber(value: blockSize),
                NSNumber(value: cOut), NSNumber(value: h), NSNumber(value: w),
            ], name: nil)
            let permuted = graph.transpose(reshaped, permutation: [0, 3, 4, 1, 5, 2], name: nil)
            return graph.reshape(permuted, shape: [
                NSNumber(value: n), NSNumber(value: cOut), NSNumber(value: h * blockSize), NSNumber(value: w * blockSize),
            ], name: nil)

        case "LOGISTIC":
            return graph.sigmoid(with: try input(0), name: nil)

        case "RELU":
            // Standalone (not fused into a conv/add's activation option).
            return graph.reLU(with: try input(0), name: nil)

        case "CUSTOM_Convolution2DTransposeBias":
            // MediaPipe's own TFLite-GPU custom op: transpose convolution
            // + fused bias add. Weight tensor is stored as [in, out, kh,
            // kw]; MPSGraph's weightsLayout below matches that directly,
            // no further transpose needed.
            let x = try input(0)
            let weightId = try op.inputId(1)
            let weightTensor = try weights.constant(graph, named: String(weightId))
            let weightShape = try weights.shape(named: String(weightId)) // [in, out, kh, kw]
            guard weightShape.count > 1 else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(op.type)' weight tensor has an unexpected shape")
            }
            let (strideY, strideX) = try op.intPair("stride")
            let outChannels = weightShape[1]
            guard let xShape = x.shape, xShape.count == 4 else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(op.type)' expected a rank-4 input")
            }
            let inputShape = xShape.map(\.intValue) // [N, inChannels, H, W]
            let outputHeight = inputShape[2] * strideY
            let outputWidth = inputShape[3] * strideX
            guard let transposeDescriptor = MPSGraphConvolution2DOpDescriptor(
                strideInX: strideX, strideInY: strideY,
                dilationRateInX: 1, dilationRateInY: 1,
                groups: 1,
                paddingLeft: 0, paddingRight: 0, paddingTop: 0, paddingBottom: 0,
                paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW
            ) else
            {
                throw MediaPipeMPSGraphError("MediaPipeMPSGraph: op '\(op.type)' could not build a convolution descriptor")
            }
            let transposed = graph.convolutionTranspose2D(
                x, weights: weightTensor,
                outputShape: [NSNumber(value: inputShape[0]), NSNumber(value: outChannels), NSNumber(value: outputHeight), NSNumber(value: outputWidth)],
                descriptor: transposeDescriptor, name: nil
            )
            return try addBiasIfPresent(transposed)

        case "HARD_SWISH":
            // MobileNetV3's h-swish: x * relu6(x+3) / 6.
            let x = try input(0)
            let shifted = graph.addition(x, graph.constant(3.0, dataType: .float32), name: nil)
            let clamped = try Self.activate(shifted, "relu6", graph: graph)
            let divided = graph.division(clamped, graph.constant(6.0, dataType: .float32), name: nil)
            return graph.multiplication(x, divided, name: nil)

        default:
            throw MediaPipeMPSGraphError("MediaPipeMPSGraph: unhandled TFLite op type '\(op.type)'")
        }
    }

    private static func activate(_ x: MPSGraphTensor, _ activation: String, graph: MPSGraph) throws -> MPSGraphTensor
    {
        switch activation
        {
        case "none": return x
        case "relu": return graph.reLU(with: x, name: nil)
        case "relu6": return graph.minimum(graph.maximum(x, graph.constant(0.0, dataType: .float32), name: nil), graph.constant(6.0, dataType: .float32), name: nil)
        case "tanh": return graph.tanh(with: x, name: nil)
        default: throw MediaPipeMPSGraphError("MediaPipeMPSGraph: unhandled activation '\(activation)'")
        }
    }
}
