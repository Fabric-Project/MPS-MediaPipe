// MediaPipeMPSGraph.swift
//
// From-scratch MPSGraph reimplementation of any MediaPipe "Blaze"-family
// TFLite model, run directly on GPU -- no CoreML. Shared by BlazePalm/
// BlazeHand (hand detector + landmark model) and BlazeFace/FaceMesh (face
// detector + landmark model): a single generic interpreter that walks a
// fully-resolved TFLite op graph (padding/groups/layout already resolved
// offline) rather than a per-model port.

import Foundation
import Metal
import MetalPerformanceShadersGraph

public final class MediaPipeMPSGraph
{
    private let graph = MPSGraph()
    private let weights: MediaPipeModelWeights
    private let device: MPSGraphDevice
    private let commandQueue: MTLCommandQueue
    private let inputTensor: MPSGraphTensor
    private let outputTensors: [MPSGraphTensor]
    private let executable: MPSGraphExecutable

    private let inferenceStateLock = NSLock()
    private var inferenceIsInFlight = false

    public let inputWidth: Int
    public let inputHeight: Int

    /// Loads `<name>_weights.bin`/`<name>_weights.json` (MediaPipeModelWeights'
    /// format) and `<name>_ops.json` (the resolved op list).
    public init(weightsBinaryURL: URL, weightsManifestURL: URL, opsJSONURL: URL, inputWidth: Int, inputHeight: Int, commandQueue: MTLCommandQueue) throws
    {
        self.weights = try MediaPipeModelWeights(binaryURL: weightsBinaryURL, manifestURL: weightsManifestURL)
        self.device = MPSGraphDevice(mtlDevice: commandQueue.device)
        self.commandQueue = commandQueue
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight

        let opsData = try Data(contentsOf: opsJSONURL)
        let opsJSON = try JSONSerialization.jsonObject(with: opsData) as! [String: Any]
        let opsList = opsJSON["ops"] as! [[String: Any]]
        let inputIds = (opsJSON["inputIds"] as! [Any]).map { ($0 as! NSNumber).intValue }
        let outputIds = (opsJSON["outputIds"] as! [Any]).map { ($0 as! NSNumber).intValue }

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
        env[inputIds[0]] = self.graph.transpose(inputPlaceholder, permutation: [0, 3, 1, 2], name: nil)

        for opDict in opsList
        {
            let op = Op(opDict)
            let result = Self.build(op: op, graph: self.graph, weights: self.weights, env: env)
            env[op.outputs[0]] = result
        }

        self.outputTensors = outputIds.map { env[$0]! }

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
    /// model's own output order.
    public func run(inputBuffer: MTLBuffer) -> [[Float]]
    {
        let inputData = MPSGraphTensorData(inputBuffer, shape: self.inputTensor.shape!, dataType: .float32)
        let results = self.executable.run(with: self.commandQueue, inputs: [inputData], results: nil, executionDescriptor: nil)
        return results.map { Self.floatArray(from: $0) }
    }

    /// Async counterpart: encodes onto `commandBuffer` without waiting;
    /// drops the call (returns false, never invokes `completion`) if an
    /// inference is already in flight. `commandBuffer` must already
    /// contain the crop preprocessor's encode so the GPU sees
    /// crop-then-inference in order.
    @discardableResult
    public func submit(inputBuffer: MTLBuffer, commandBuffer: MTLCommandBuffer, completion: @escaping (Result<[[Float]], any Error>) -> Void) -> Bool
    {
        guard self.beginInference() else { return false }

        let inputData = MPSGraphTensorData(inputBuffer, shape: self.inputTensor.shape!, dataType: .float32)
        let executionDescriptor = MPSGraphExecutableExecutionDescriptor()
        executionDescriptor.waitUntilCompleted = false
        executionDescriptor.completionHandler = { [weak self] results, error in
            guard let self else { return }
            defer { self.finishInference() }

            if let error
            {
                completion(.failure(error))
            }
            else
            {
                completion(.success(results.map { Self.floatArray(from: $0) }))
            }
        }

        let mpsCommandBuffer = MPSCommandBuffer(commandBuffer: commandBuffer)
        _ = self.executable.encode(to: mpsCommandBuffer, inputs: [inputData], results: nil, executionDescriptor: executionDescriptor)
        mpsCommandBuffer.commit()
        return true
    }

    private func beginInference() -> Bool
    {
        self.inferenceStateLock.lock()
        defer { self.inferenceStateLock.unlock() }
        guard self.inferenceIsInFlight == false else { return false }
        self.inferenceIsInFlight = true
        return true
    }

    private func finishInference()
    {
        self.inferenceStateLock.lock()
        self.inferenceIsInFlight = false
        self.inferenceStateLock.unlock()
    }

    private static func floatArray(from tensorData: MPSGraphTensorData) -> [Float]
    {
        let count = tensorData.shape.map(\.intValue).reduce(1, *)
        var values = [Float](repeating: 0, count: count)
        tensorData.mpsndarray().readBytes(&values, strideBytes: nil)
        return values
    }

    // MARK: - Op parsing (fully resolved by dump_tflite_graph.py already —
    // padding/groups/dilation/layout, no re-derivation needed here)

    private struct Op
    {
        let type: String
        let inputs: [Int]
        let outputs: [Int]
        let options: [String: Any]

        init(_ dict: [String: Any])
        {
            self.type = dict["type"] as! String
            self.inputs = (dict["inputs"] as! [Any]).map { ($0 as! NSNumber).intValue }
            self.outputs = (dict["outputs"] as! [Any]).map { ($0 as! NSNumber).intValue }
            self.options = dict["options"] as! [String: Any]
        }

        func intPair(_ key: String) -> (Int, Int)
        {
            let array = (self.options[key] as! [Any]).map { ($0 as! NSNumber).intValue }
            return (array[0], array[1])
        }

        func string(_ key: String) -> String { self.options[key] as! String }
        func int(_ key: String) -> Int { (self.options[key] as! NSNumber).intValue }
        func bool(_ key: String) -> Bool { (self.options[key] as! NSNumber).boolValue }
        func intArray(_ key: String) -> [Int] { (self.options[key] as! [Any]).map { ($0 as! NSNumber).intValue } }
        func fourIntsOrNil(_ key: String) -> (Int, Int, Int, Int)?
        {
            guard let value = self.options[key], (value is NSNull) == false else { return nil }
            let array = (value as! [Any]).map { ($0 as! NSNumber).intValue }
            return (array[0], array[1], array[2], array[3])
        }
        /// [[before,after], [before,after], [before,after], [before,after]] in NCHW dim order.
        func paddingPairs(_ key: String) -> [(Int, Int)]
        {
            (self.options[key] as! [Any]).map { pair in
                let array = (pair as! [Any]).map { ($0 as! NSNumber).intValue }
                return (array[0], array[1])
            }
        }
    }

    // MARK: - Op building

    private static func build(op: Op, graph: MPSGraph, weights: MediaPipeModelWeights, env: [Int: MPSGraphTensor]) -> MPSGraphTensor
    {
        func input(_ index: Int) -> MPSGraphTensor { env[op.inputs[index]]! }

        switch op.type
        {
        case "CONV_2D", "DEPTHWISE_CONV_2D":
            var x = input(0)
            if let (pl, pr, pt, pb) = op.fourIntsOrNil("pre_pad")
            {
                x = graph.padTensor(x, with: .constant,
                                     leftPadding: [0, 0, NSNumber(value: pt), NSNumber(value: pl)],
                                     rightPadding: [0, 0, NSNumber(value: pb), NSNumber(value: pr)],
                                     constantValue: 0, name: nil)
            }
            let weightTensor = weights.constant(graph, named: String(op.inputs[1]))
            let (strideY, strideX) = op.intPair("stride")
            let (dilationY, dilationX) = op.intPair("dilation")
            let (padY, padX) = op.intPair("conv_pad")
            let groups = op.int("groups")
            let descriptor = MPSGraphConvolution2DOpDescriptor(
                strideInX: strideX, strideInY: strideY,
                dilationRateInX: dilationX, dilationRateInY: dilationY,
                groups: groups,
                paddingLeft: padX, paddingRight: padX, paddingTop: padY, paddingBottom: padY,
                paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW
            )!
            var y = graph.convolution2D(x, weights: weightTensor, descriptor: descriptor, name: nil)
            if op.inputs.count > 2
            {
                let biasShape = weights.shape(named: String(op.inputs[2]))
                let bias = weights.constant(graph, named: String(op.inputs[2]))
                let biasReshaped = graph.reshape(bias, shape: [1, NSNumber(value: biasShape[0]), 1, 1], name: nil)
                y = graph.addition(y, biasReshaped, name: nil)
            }
            return Self.activate(y, op.string("activation"), graph: graph)

        case "PRELU":
            let x = input(0)
            let alpha = weights.constant(graph, named: String(op.inputs[1]))
            let alphaShape = weights.shape(named: String(op.inputs[1]))
            let alphaReshaped = graph.reshape(alpha, shape: [1, NSNumber(value: alphaShape[0]), 1, 1], name: nil)
            let positive = graph.reLU(with: x, name: nil)
            let negative = graph.subtraction(x, positive, name: nil) // min(x, 0)
            let scaledNegative = graph.multiplication(negative, alphaReshaped, name: nil)
            return graph.addition(positive, scaledNegative, name: nil)

        case "ADD":
            let sum = graph.addition(input(0), input(1), name: nil)
            return Self.activate(sum, op.string("activation"), graph: graph)

        case "MUL":
            // A squeeze-and-excitation gate: both operands are already
            // NCHW activations by this point ([N,C,H,W] times [N,C,1,1]),
            // so plain broadcasting multiply is correct.
            let product = graph.multiplication(input(0), input(1), name: nil)
            return Self.activate(product, op.string("activation"), graph: graph)

        case "MAX_POOL_2D":
            var x = input(0)
            if let (pl, pr, pt, pb) = op.fourIntsOrNil("pre_pad")
            {
                x = graph.padTensor(x, with: .constant,
                                     leftPadding: [0, 0, NSNumber(value: pt), NSNumber(value: pl)],
                                     rightPadding: [0, 0, NSNumber(value: pb), NSNumber(value: pr)],
                                     constantValue: Double(-Float.greatestFiniteMagnitude), name: nil)
            }
            let (filterH, filterW) = op.intPair("filter")
            let (strideY, strideX) = op.intPair("stride")
            let descriptor = MPSGraphPooling2DOpDescriptor(
                kernelWidth: filterW, kernelHeight: filterH,
                strideInX: strideX, strideInY: strideY,
                dilationRateInX: 1, dilationRateInY: 1,
                paddingLeft: 0, paddingRight: 0, paddingTop: 0, paddingBottom: 0,
                paddingStyle: .explicit, dataLayout: .NCHW
            )!
            let y = graph.maxPooling2D(withSourceTensor: x, descriptor: descriptor, name: nil)
            return Self.activate(y, op.string("activation"), graph: graph)

        case "AVERAGE_POOL_2D":
            // TFLite divides by the count of valid (non-padded) elements,
            // not the full kernel area -- explicit asymmetric padding plus
            // includeZeroPadToAverage=false matches this exactly, no
            // separate pad step needed.
            let x = input(0)
            let (avgFilterH, avgFilterW) = op.intPair("filter")
            let (avgStrideY, avgStrideX) = op.intPair("stride")
            let (padLeft, padRight, padTop, padBottom): (Int, Int, Int, Int)
            if let (pl, pr, pt, pb) = op.fourIntsOrNil("pre_pad")
            {
                (padLeft, padRight, padTop, padBottom) = (pl, pr, pt, pb)
            }
            else
            {
                let (padY, padX) = op.intPair("conv_pad")
                (padLeft, padRight, padTop, padBottom) = (padX, padX, padY, padY)
            }
            let avgDescriptor = MPSGraphPooling2DOpDescriptor(
                kernelWidth: avgFilterW, kernelHeight: avgFilterH,
                strideInX: avgStrideX, strideInY: avgStrideY,
                dilationRateInX: 1, dilationRateInY: 1,
                paddingLeft: padLeft, paddingRight: padRight, paddingTop: padTop, paddingBottom: padBottom,
                paddingStyle: .explicit, dataLayout: .NCHW
            )!
            avgDescriptor.includeZeroPadToAverage = false
            let avgY = graph.avgPooling2D(withSourceTensor: x, descriptor: avgDescriptor, name: nil)
            return Self.activate(avgY, op.string("activation"), graph: graph)

        case "PAD":
            let pairs = op.paddingPairs("paddings") // NCHW order already
            let left = pairs.map { NSNumber(value: $0.0) }
            let right = pairs.map { NSNumber(value: $0.1) }
            return graph.padTensor(input(0), with: .constant, leftPadding: left, rightPadding: right, constantValue: 0, name: nil)

        case "RESIZE_BILINEAR":
            let sizeArray = op.intArray("size")
            let alignCorners = op.bool("align_corners")
            return graph.resize(
                input(0), size: [NSNumber(value: sizeArray[0]), NSNumber(value: sizeArray[1])],
                mode: .bilinear, centerResult: !alignCorners, alignCorners: alignCorners,
                layout: .NCHW, name: nil
            )

        case "RESHAPE":
            let fromFourD = op.bool("from_4d")
            let toFourD = op.bool("to_4d")
            var x = input(0)
            if fromFourD
            {
                x = graph.transpose(x, permutation: [0, 2, 3, 1], name: nil) // NCHW -> NHWC semantics
            }
            var shape = op.intArray("shape").map { NSNumber(value: $0) }
            shape[0] = 1 // batch, always 1 here
            var y = graph.reshape(x, shape: shape, name: nil)
            if toFourD
            {
                y = graph.transpose(y, permutation: [0, 3, 1, 2], name: nil) // NHWC -> NCHW
            }
            return y

        case "CONCATENATION":
            let tensors = op.inputs.map { env[$0]! }
            let axis = op.int("axis")
            let y = graph.concatTensors(tensors, dimension: axis, name: nil)
            return Self.activate(y, op.string("activation"), graph: graph)

        case "MEAN":
            let axes = op.intArray("axes").map { NSNumber(value: $0) }
            let y = graph.mean(of: input(0), axes: axes, name: nil) // always keeps dims
            if op.bool("keep_dims") == false
            {
                let keptShape = y.shape!.map(\.intValue)
                let reducedAxes = Set(axes.map(\.intValue))
                let squeezedShape = keptShape.enumerated().filter { !reducedAxes.contains($0.offset) }.map { NSNumber(value: $0.element) }
                return graph.reshape(y, shape: squeezedShape, name: nil)
            }
            return y

        case "FULLY_CONNECTED":
            let x = input(0)
            let weightTensor = weights.constant(graph, named: String(op.inputs[1])) // [out, in]
            let weightTransposed = graph.transposeTensor(weightTensor, dimension: 0, withDimension: 1, name: nil)
            var y = graph.matrixMultiplication(primary: x, secondary: weightTransposed, name: nil)
            if op.inputs.count > 2
            {
                let bias = weights.constant(graph, named: String(op.inputs[2]))
                y = graph.addition(y, bias, name: nil)
            }
            return Self.activate(y, op.string("activation"), graph: graph)

        case "DEPTH_TO_SPACE":
            // TF's channel decomposition is (i*b+j)*C_out+c (block-position-
            // major, channel-minor) -- not the c*b^2+i*b+j order
            // PixelShuffle-style approaches assume -- so this reshapes/
            // transposes by hand.
            let blockSize = op.int("block_size")
            let x = input(0)
            let shape = x.shape!.map(\.intValue)
            let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3])
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
            return graph.sigmoid(with: input(0), name: nil)

        case "RELU":
            // Standalone (not fused into a conv/add's activation option).
            return graph.reLU(with: input(0), name: nil)

        case "CUSTOM_Convolution2DTransposeBias":
            // MediaPipe's own TFLite-GPU custom op: transpose convolution
            // + fused bias add. Weight tensor is stored as [in, out, kh,
            // kw]; MPSGraph's weightsLayout below matches that directly,
            // no further transpose needed.
            let x = input(0)
            let weightTensor = weights.constant(graph, named: String(op.inputs[1]))
            let weightShape = weights.shape(named: String(op.inputs[1])) // [in, out, kh, kw]
            let (strideY, strideX) = op.intPair("stride")
            let outChannels = weightShape[1]
            let inputShape = x.shape!.map(\.intValue) // [N, inChannels, H, W]
            let outputHeight = inputShape[2] * strideY
            let outputWidth = inputShape[3] * strideX
            let transposeDescriptor = MPSGraphConvolution2DOpDescriptor(
                strideInX: strideX, strideInY: strideY,
                dilationRateInX: 1, dilationRateInY: 1,
                groups: 1,
                paddingLeft: 0, paddingRight: 0, paddingTop: 0, paddingBottom: 0,
                paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW
            )!
            var transposed = graph.convolutionTranspose2D(
                x, weights: weightTensor,
                outputShape: [NSNumber(value: inputShape[0]), NSNumber(value: outChannels), NSNumber(value: outputHeight), NSNumber(value: outputWidth)],
                descriptor: transposeDescriptor, name: nil
            )
            if op.inputs.count > 2
            {
                let biasShape = weights.shape(named: String(op.inputs[2]))
                let bias = weights.constant(graph, named: String(op.inputs[2]))
                let biasReshaped = graph.reshape(bias, shape: [1, NSNumber(value: biasShape[0]), 1, 1], name: nil)
                transposed = graph.addition(transposed, biasReshaped, name: nil)
            }
            return transposed

        case "HARD_SWISH":
            // MobileNetV3's h-swish: x * relu6(x+3) / 6.
            let x = input(0)
            let shifted = graph.addition(x, graph.constant(3.0, dataType: .float32), name: nil)
            let clamped = Self.activate(shifted, "relu6", graph: graph)
            let divided = graph.division(clamped, graph.constant(6.0, dataType: .float32), name: nil)
            return graph.multiplication(x, divided, name: nil)

        default:
            fatalError("Unhandled TFLite op type: \(op.type)")
        }
    }

    private static func activate(_ x: MPSGraphTensor, _ activation: String, graph: MPSGraph) -> MPSGraphTensor
    {
        switch activation
        {
        case "none": return x
        case "relu": return graph.reLU(with: x, name: nil)
        case "relu6": return graph.minimum(graph.maximum(x, graph.constant(0.0, dataType: .float32), name: nil), graph.constant(6.0, dataType: .float32), name: nil)
        case "tanh": return graph.tanh(with: x, name: nil)
        default: fatalError("Unhandled activation: \(activation)")
        }
    }
}
