// MediaPipeMPSGraph+Bundled.swift
//
// Bundle.module is scoped to whichever module it's referenced from, so a
// consumer of this package can't do its own Bundle.module lookup to reach
// these bundled model files. This is the entry point to use instead.

import Foundation
import Metal

public extension MediaPipeMPSGraph
{
    /// Loads one of this package's own bundled models by name, matching
    /// the `<name>_weights.bin` / `<name>_weights.json` / `<name>_ops.json`
    /// naming convention every bundled model uses (e.g. "MediaPipePoseDetector",
    /// "MediaPipeSelfieSegmentationLandscape").
    static func loadBundled(named name: String, inputWidth: Int, inputHeight: Int, commandQueue: MTLCommandQueue, maxFramesInFlight: Int = 3) throws -> MediaPipeMPSGraph
    {
        let binaryURL = Bundle.module.url(forResource: "\(name)_weights", withExtension: "bin", subdirectory: "Models/Pose")
        let manifestURL = Bundle.module.url(forResource: "\(name)_weights", withExtension: "json", subdirectory: "Models/Pose")
        let opsURL = Bundle.module.url(forResource: "\(name)_ops", withExtension: "json", subdirectory: "Models/Pose")

        guard let binaryURL, let manifestURL, let opsURL else
        {
            var missing: [String] = []
            if binaryURL == nil { missing.append("\(name)_weights.bin") }
            if manifestURL == nil { missing.append("\(name)_weights.json") }
            if opsURL == nil { missing.append("\(name)_ops.json") }
            throw MediaPipeMPSGraphError("Could not find bundled resource(s) for '\(name)': \(missing.joined(separator: ", ")) (looked under Models/Pose in \(Bundle.module.bundleURL.lastPathComponent))")
        }

        return try MediaPipeMPSGraph(
            weightsBinaryURL: binaryURL, weightsManifestURL: manifestURL, opsJSONURL: opsURL,
            inputWidth: inputWidth, inputHeight: inputHeight, commandQueue: commandQueue,
            maxFramesInFlight: maxFramesInFlight
        )
    }
}
