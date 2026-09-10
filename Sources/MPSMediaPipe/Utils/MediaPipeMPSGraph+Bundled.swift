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
    static func loadBundled(named name: String, inputWidth: Int, inputHeight: Int, commandQueue: MTLCommandQueue) throws -> MediaPipeMPSGraph
    {
        guard
            let binaryURL = Bundle.module.url(forResource: "\(name)_weights", withExtension: "bin", subdirectory: "Models/Pose"),
            let manifestURL = Bundle.module.url(forResource: "\(name)_weights", withExtension: "json", subdirectory: "Models/Pose"),
            let opsURL = Bundle.module.url(forResource: "\(name)_ops", withExtension: "json", subdirectory: "Models/Pose")
        else
        {
            throw MediaPipeMPSGraphError("Could not find bundled '\(name)' graph resources")
        }

        return try MediaPipeMPSGraph(
            weightsBinaryURL: binaryURL, weightsManifestURL: manifestURL, opsJSONURL: opsURL,
            inputWidth: inputWidth, inputHeight: inputHeight, commandQueue: commandQueue
        )
    }
}
