import CoreGraphics
import Foundation
import ImageIO
import Metal
import MetalPerformanceShaders
import Testing
@testable import MPSMediaPipe

/// Every bundled MediaPipe checkpoint at the input size and pixel range
/// Fabric feeds it (the face and pose detectors take -1...1; every other
/// model takes the crop preprocessor's default 0...1).
struct MediaPipeBundledModel: CustomStringConvertible, Sendable
{
    let name: String
    let width: Int
    let height: Int
    var pixelRange: (min: Float, max: Float) = (0, 1)

    var description: String { self.name }

    static let all: [MediaPipeBundledModel] = [
        .init(name: "MediaPipeFaceDetector", width: 128, height: 128, pixelRange: MediaPipeFaceDetector.detectorPixelRange),
        .init(name: "MediaPipeFaceDetectorFullRange", width: 192, height: 192, pixelRange: MediaPipeFaceDetector.detectorPixelRange),
        .init(name: "MediaPipeFaceLandmarks", width: 192, height: 192),
        .init(name: "MediaPipeHandDetector", width: 192, height: 192),
        .init(name: "MediaPipeHandLandmarks", width: 224, height: 224),
        .init(name: "MediaPipePoseDetector", width: 224, height: 224, pixelRange: MediaPipePoseDetector.detectorPixelRange),
        .init(name: "MediaPipePoseLandmarkLite", width: 256, height: 256),
        .init(name: "MediaPipePoseLandmarkFull", width: 256, height: 256),
        .init(name: "MediaPipePoseLandmarkHeavy", width: 256, height: 256),
        .init(name: "MediaPipeSelfieSegmentation", width: 256, height: 256),
        .init(name: "MediaPipeSelfieSegmentationLandscape", width: 256, height: 144),
    ]
}

/// The canonical MediaPipe example photo (people on a horse), resized to
/// `width` x `height` and scaled into `range`, as NHWC RGB floats. A real
/// image keeps activations in the range the models were trained for, which
/// matters when judging the float16 tiers.
private func photoInput(width: Int, height: Int, range: (min: Float, max: Float)) throws -> [Float]
{
    let url = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appending(path: "Fixtures/canonical_people_horse.jpg")
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else
    {
        throw MediaPipeMPSGraphError("Could not load \(url.path)")
    }
    var rgba = [UInt8](repeating: 0, count: width * height * 4)
    let drawn: Bool = rgba.withUnsafeMutableBytes { bytes in
        guard let context = CGContext(
            data: bytes.baseAddress,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else
        {
            return false
        }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return true
    }
    guard drawn else
    {
        throw MediaPipeMPSGraphError("Could not rasterize \(url.path)")
    }
    var values = [Float](repeating: 0, count: width * height * 3)
    for pixel in 0..<(width * height)
    {
        for channel in 0..<3
        {
            let unit = Float(rgba[pixel * 4 + channel]) / 255
            values[pixel * 3 + channel] = range.min + (range.max - range.min) * unit
        }
    }
    return values
}

/// Full-output regression gate for graph rewrites. Runs each model on the
/// canonical example photo and compares every output tensor, element by element,
/// with `Fixtures/<model>.bin`, recorded from the unmodified graph.
/// Record or re-record deliberately with
/// `MEDIAPIPE_WRITE_REFERENCE=1 swift test --filter mediaPipeModelMatchesReference`.
/// `MEDIAPIPE_PRECISION=mixedFloat16` or `float16` compares that tier against
/// the same float32 references at the float16 bar (40 dB) instead of 70 dB.
@Test(arguments: MediaPipeBundledModel.all)
func mediaPipeModelMatchesReference(model: MediaPipeBundledModel) throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let precision = mediaPipeTestPrecision()
    let graph = try MediaPipeMPSGraph.loadBundled(named: model.name, inputWidth: model.width, inputHeight: model.height, commandQueue: commandQueue, precision: precision)
    let input = try photoInput(width: model.width, height: model.height, range: model.pixelRange)
    let inputBuffer = try #require(device.makeBuffer(bytes: input, length: input.count * MemoryLayout<Float>.stride))
    let outputs = try graph.run(inputBuffer: inputBuffer)

    let fixtureURL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appending(path: "Fixtures/\(model.name).bin")
    if ProcessInfo.processInfo.environment["MEDIAPIPE_WRITE_REFERENCE"] != nil
    {
        try #require(precision == .float32, "record references from the float32 graph")
        let flat = outputs.flatMap { $0 }
        try flat.withUnsafeBytes { try Data($0).write(to: fixtureURL) }
        print("Wrote \(model.name) reference (\(outputs.map(\.count)) values)")
        return
    }

    let referenceData = try Data(contentsOf: fixtureURL)
    let reference = referenceData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    try #require(reference.count == outputs.reduce(0) { $0 + $1.count })

    // Per output tensor: outputs mix scales (pixel landmarks, logits, masks),
    // so each is judged against its own peak.
    var offset = 0
    for (index, output) in outputs.enumerated()
    {
        let expected = reference[offset..<(offset + output.count)]
        offset += output.count
        let nonFiniteCount = output.filter { !$0.isFinite }.count
        var maximumError: Float = 0
        var squaredErrorSum: Double = 0
        for (actual, wanted) in zip(output, expected) where actual.isFinite
        {
            let error = abs(actual - wanted)
            maximumError = max(maximumError, error)
            squaredErrorSum += Double(error) * Double(error)
        }
        #expect(nonFiniteCount == 0, "\(model.name) output \(index): \(nonFiniteCount) NaN or infinite values")
        if output.count == 1
        {
            // A lone scalar (presence, handedness) has no meaningful PSNR;
            // judge its absolute error instead. These are sigmoid
            // probabilities the nodes only threshold, so the float16 tiers
            // allow 0.05.
            print("\(model.name) \(precision) output \(index) (scalar): error \(maximumError), non-finite \(nonFiniteCount)")
            #expect(maximumError < (precision == .float32 ? 1e-5 : 0.05), "\(model.name) output \(index) scalar error")
            continue
        }
        let peak = Double(expected.map { abs($0) }.max() ?? 0)
        let meanSquaredError = squaredErrorSum / Double(max(output.count, 1))
        let psnr = meanSquaredError == 0 || peak == 0 ? Double.infinity : 10 * log10(peak * peak / meanSquaredError)
        print("\(model.name) \(precision) output \(index) (\(output.count) values): max error \(maximumError), non-finite \(nonFiniteCount), PSNR \(psnr) dB")
        #expect(psnr > (precision == .float32 ? 70 : 40), "\(model.name) output \(index)")
    }
}

/// Opt-in benchmark over every bundled model: graph construction, CPU encode
/// cost of one frame into an empty queue, and queued wall time per frame.
/// Run twice or discard the first model's numbers: the first launch after a
/// build runs on a cold GPU.
/// `MEDIAPIPE_RUN_BENCHMARK=1 swift test -c release -Xswiftc -enable-testing --filter mediaPipeSteadyStatePerformance`
@Test func mediaPipeSteadyStatePerformance() throws
{
    guard ProcessInfo.processInfo.environment["MEDIAPIPE_RUN_BENCHMARK"] != nil,
          let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let clock = ContinuousClock()
    func milliseconds(_ duration: Duration) -> Double
    {
        Double(duration.components.seconds) * 1_000 + Double(duration.components.attoseconds) / 1e15
    }

    // MEDIAPIPE_GPU_ONLY=1 compiles at .level0 (no ANE placement);
    // MEDIAPIPE_BENCHMARK_MODEL limits the run to one model.
    let computeUnits: MediaPipeComputeUnits = ProcessInfo.processInfo.environment["MEDIAPIPE_GPU_ONLY"] == "1" ? .gpuOnly : .gpuAndNeuralEngine
    let modelFilter = ProcessInfo.processInfo.environment["MEDIAPIPE_BENCHMARK_MODEL"]
    for model in MediaPipeBundledModel.all where modelFilter == nil || model.name == modelFilter
    {
        let constructionStart = clock.now
        let graph = try MediaPipeMPSGraph.loadBundled(named: model.name, inputWidth: model.width, inputHeight: model.height, commandQueue: commandQueue, maxFramesInFlight: 16, precision: mediaPipeTestPrecision(), computeUnits: computeUnits)
        let construction = clock.now - constructionStart

        let inputBuffer = try #require(device.makeBuffer(length: graph.inputBufferLength, options: .storageModePrivate))
        let outputBuffers = try graph.outputBufferLengths.map { try #require(device.makeBuffer(length: $0, options: .storageModePrivate)) }
        func encodeFrame() throws -> (MPSCommandBuffer, Double)
        {
            let rawCommandBuffer = try #require(commandQueue.makeCommandBuffer())
            let commandBuffer = MPSCommandBuffer(commandBuffer: rawCommandBuffer)
            let start = clock.now
            let accepted = try graph.encode(inputBuffer: inputBuffer, outputBuffers: outputBuffers, commandBuffer: commandBuffer)
            let cpu = milliseconds(clock.now - start)
            commandBuffer.commit()
            #expect(accepted)
            return (commandBuffer, cpu)
        }

        for _ in 0..<20 { try encodeFrame().0.waitUntilCompleted() }
        let (emptyQueueBuffer, emptyQueueCPU) = try encodeFrame()
        emptyQueueBuffer.waitUntilCompleted()

        let measuredStart = clock.now
        var last: MPSCommandBuffer?
        for frame in 0..<60
        {
            let (commandBuffer, _) = try encodeFrame()
            last = commandBuffer
            if frame % 8 == 7 { commandBuffer.waitUntilCompleted() }
        }
        last?.waitUntilCompleted()
        let wall = milliseconds(clock.now - measuredStart) / 60

        print("\(model.name) \(model.width)x\(model.height) \(mediaPipeTestPrecision()): construction \(milliseconds(construction)) ms, empty-queue CPU \(emptyQueueCPU) ms, wall \(wall) ms/frame")
    }
}

func mediaPipeTestPrecision() -> MediaPipePrecision
{
    switch ProcessInfo.processInfo.environment["MEDIAPIPE_PRECISION"]
    {
    case "mixedFloat16": return .mixedFloat16
    case "float16": return .float16
    default: return .float32
    }
}
