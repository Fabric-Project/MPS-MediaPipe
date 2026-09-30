import Foundation
import Metal
import MetalPerformanceShaders
import Testing
@testable import MPSMediaPipe

/// Opt-in Neural Engine bisect. Construction grows about 1.4 ms per op for a
/// GPU-only graph (the face detector); models that reach the ANE build at
/// 4-9 ms per op. This builds one model (`MEDIAPIPE_ANE_PREFIX_MODEL`,
/// default the pose detector) from growing prefixes of its op list, each
/// ending at that prefix's last op, to find where placement starts.
///
/// `MEDIAPIPE_RUN_ANE_PREFIX_PROBE=1 swift test -c release -Xswiftc -enable-testing --filter mediaPipeANEPrefixProbe`
@Test func mediaPipeANEPrefixProbe() throws
{
    guard ProcessInfo.processInfo.environment["MEDIAPIPE_RUN_ANE_PREFIX_PROBE"] != nil,
          let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let modelName = ProcessInfo.processInfo.environment["MEDIAPIPE_ANE_PREFIX_MODEL"] ?? "MediaPipePoseDetector"
    let model = try #require(MediaPipeBundledModel.all.first { $0.name == modelName })
    let modelsDirectory = URL(filePath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appending(path: "Sources/MPSMediaPipe/Models/Pose")
    let opsURL = modelsDirectory.appending(path: "\(modelName)_ops.json")
    let opsJSON = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: opsURL)) as? [String: Any])
    let ops = try #require(opsJSON["ops"] as? [[String: Any]])
    let temporaryDirectory = URL.temporaryDirectory.appending(path: "mediapipe-ane-prefix-probe")
    try FileManager.default.createDirectory(at: temporaryDirectory, withIntermediateDirectories: true)
    let clock = ContinuousClock()
    // MEDIAPIPE_PRECISION picks the tier; float16 when unset.
    let precision: MediaPipePrecision = ProcessInfo.processInfo.environment["MEDIAPIPE_PRECISION"] == nil ? .float16 : mediaPipeTestPrecision()

    // MEDIAPIPE_ANE_PREFIX_LENGTHS="25,26,27" overrides the default sweep.
    let requestedLengths = ProcessInfo.processInfo.environment["MEDIAPIPE_ANE_PREFIX_LENGTHS"]?
        .split(separator: ",")
        .compactMap { Int($0) }
    for prefixLength in requestedLengths ?? [2, 3, 4, 6, 12, 24, 48, 72, 96, ops.count] where prefixLength <= ops.count
    {
        let prefix = Array(ops.prefix(prefixLength))
        let lastOutputs = try #require(prefix.last?["outputs"] as? [Any])
        var truncated = opsJSON
        truncated["ops"] = prefix
        truncated["outputIds"] = [try #require(lastOutputs.first)]
        let truncatedURL = temporaryDirectory.appending(path: "\(modelName)_\(prefixLength)_ops.json")
        try JSONSerialization.data(withJSONObject: truncated).write(to: truncatedURL)

        let start = clock.now
        let graph = try MediaPipeMPSGraph(
            weightsBinaryURL: modelsDirectory.appending(path: "\(modelName)_weights.bin"),
            weightsManifestURL: modelsDirectory.appending(path: "\(modelName)_weights.json"),
            opsJSONURL: truncatedURL,
            inputWidth: model.width,
            inputHeight: model.height,
            commandQueue: commandQueue,
            precision: precision
        )
        let elapsed = (clock.now - start).components
        let milliseconds = Double(elapsed.seconds) * 1_000 + Double(elapsed.attoseconds) / 1e15
        let types = prefix.compactMap { $0["type"] as? String }.suffix(3).joined(separator: ", ")
        print("ANE prefix \(prefixLength) ops (\(precision)): construction \(milliseconds.formatted(.number.precision(.fractionLength(1)))) ms (last ops: \(types))")

        // MEDIAPIPE_ANE_PREFIX_RUN_SECONDS=3 runs each prefix that long, after
        // an idle gap, so powermetrics' ANE Power can be matched to it.
        guard let runSeconds = ProcessInfo.processInfo.environment["MEDIAPIPE_ANE_PREFIX_RUN_SECONDS"].flatMap({ Int($0) }) else { continue }
        let inputBuffer = try #require(device.makeBuffer(length: graph.inputBufferLength, options: .storageModePrivate))
        let outputBuffers = try graph.outputBufferLengths.map { try #require(device.makeBuffer(length: $0, options: .storageModePrivate)) }
        Thread.sleep(forTimeInterval: 2)
        print("ANE prefix \(prefixLength) ops: running \(Date.now.formatted(date: .omitted, time: .standard))")
        let deadline = clock.now + .seconds(runSeconds)
        var iterations = 0
        while clock.now < deadline
        {
            let commandBuffer = MPSCommandBuffer(commandBuffer: try #require(commandQueue.makeCommandBuffer()))
            _ = try graph.encode(inputBuffer: inputBuffer, outputBuffers: outputBuffers, commandBuffer: commandBuffer)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            iterations += 1
        }
        let millisecondsPerRun = Double(runSeconds) * 1000 / Double(max(iterations, 1))
        print("ANE prefix \(prefixLength) ops: stopped \(Date.now.formatted(date: .omitted, time: .standard)), \(precision) \(millisecondsPerRun.formatted(.number.precision(.fractionLength(3)))) ms/run")
    }
}
