import Foundation
import Metal
import MetalPerformanceShaders
import Testing
@testable import MPSMediaPipe

/// Every bundled MediaPipe checkpoint at the input size Fabric runs it at.
struct MediaPipeBundledModel: CustomStringConvertible, Sendable
{
    let name: String
    let width: Int
    let height: Int

    var description: String { self.name }

    static let all: [MediaPipeBundledModel] = [
        .init(name: "MediaPipeFaceDetector", width: 128, height: 128),
        .init(name: "MediaPipeFaceDetectorFullRange", width: 192, height: 192),
        .init(name: "MediaPipeFaceLandmarks", width: 192, height: 192),
        .init(name: "MediaPipeHandDetector", width: 192, height: 192),
        .init(name: "MediaPipeHandLandmarks", width: 224, height: 224),
        .init(name: "MediaPipePoseDetector", width: 224, height: 224),
        .init(name: "MediaPipePoseLandmarkLite", width: 256, height: 256),
        .init(name: "MediaPipePoseLandmarkFull", width: 256, height: 256),
        .init(name: "MediaPipePoseLandmarkHeavy", width: 256, height: 256),
        .init(name: "MediaPipeSelfieSegmentation", width: 256, height: 256),
        .init(name: "MediaPipeSelfieSegmentationLandscape", width: 256, height: 144),
    ]
}

/// Deterministic, non-constant NHWC input in [-1, 1].
private func patternedInput(width: Int, height: Int) -> [Float]
{
    var values = [Float](repeating: 0, count: width * height * 3)
    for y in 0..<height
    {
        for x in 0..<width
        {
            let index = (y * width + x) * 3
            let fx = Float(x) / Float(width)
            let fy = Float(y) / Float(height)
            values[index + 0] = sin(fx * 17 + fy * 3)
            values[index + 1] = cos(fy * 23 - fx * 5)
            values[index + 2] = fx * fy * 2 - 1
        }
    }
    return values
}

/// Full-output regression gate for graph rewrites. Runs each model on a
/// patterned input and compares every output tensor, element by element,
/// with `Fixtures/<model>.bin`, recorded from the unmodified graph.
/// Record or re-record deliberately with
/// `MEDIAPIPE_WRITE_REFERENCE=1 swift test --filter mediaPipeModelMatchesReference`.
@Test(arguments: MediaPipeBundledModel.all)
func mediaPipeModelMatchesReference(model: MediaPipeBundledModel) throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let graph = try MediaPipeMPSGraph.loadBundled(named: model.name, inputWidth: model.width, inputHeight: model.height, commandQueue: commandQueue)
    let input = patternedInput(width: model.width, height: model.height)
    let inputBuffer = try #require(device.makeBuffer(bytes: input, length: input.count * MemoryLayout<Float>.stride))
    let outputs = try graph.run(inputBuffer: inputBuffer)

    let fixtureURL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appending(path: "Fixtures/\(model.name).bin")
    if ProcessInfo.processInfo.environment["MEDIAPIPE_WRITE_REFERENCE"] != nil
    {
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
        var maximumError: Float = 0
        var squaredErrorSum: Double = 0
        for (actual, wanted) in zip(output, expected)
        {
            let error = abs(actual - wanted)
            maximumError = max(maximumError, error)
            squaredErrorSum += Double(error) * Double(error)
        }
        let peak = Double(expected.map { abs($0) }.max() ?? 0)
        let meanSquaredError = squaredErrorSum / Double(max(output.count, 1))
        let psnr = meanSquaredError == 0 || peak == 0 ? Double.infinity : 10 * log10(peak * peak / meanSquaredError)
        print("\(model.name) output \(index) (\(output.count) values): max error \(maximumError), PSNR \(psnr) dB")
        #expect(psnr > 70, "\(model.name) output \(index): an exact rewrite should stay above 70 dB")
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

    for model in MediaPipeBundledModel.all
    {
        let constructionStart = clock.now
        let graph = try MediaPipeMPSGraph.loadBundled(named: model.name, inputWidth: model.width, inputHeight: model.height, commandQueue: commandQueue, maxFramesInFlight: 16)
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

        print("\(model.name) \(model.width)x\(model.height): construction \(milliseconds(construction)) ms, empty-queue CPU \(emptyQueueCPU) ms, wall \(wall) ms/frame")
    }
}
