//
//  MediaPipeInferenceTimingLogger.swift
//  Fabric
//

import Foundation

/// Throttled per-node inference latency logging, shared by all six
/// MediaPipe test/comparison nodes (Face/Hand/Pose x Detection/Landmark).
/// Each call site measures wall-clock time from the moment it starts doing
/// work for a newly-changed input frame (crop encode included) to the
/// moment its MPSGraph command buffer actually finishes on the GPU —
/// `model.run(...)` returning for the sync path, or the `model.submit(...)`
/// completion handler firing for the async path — not just CPU submission
/// time. That's the number that actually says how fast a given model runs.
///
/// Logs at most once every `logInterval` seconds per node *type* (keyed by
/// `Node.name`, e.g. "MediaPipe Pose Detection") — not per instance, so two
/// copies of the same node in one graph share a throttle bucket and only
/// one logs per interval. Fine for a debug/profiling aid; revisit with a
/// per-instance key if that ever actually matters.
public enum MediaPipeInferenceTimingLogger
{
    private static let logInterval: TimeInterval = 5.0
    private static let lock = NSLock()
    private static var lastLogTimes: [String: Date] = [:]

    public static func log(nodeName: String, elapsed: TimeInterval)
    {
        Self.lock.lock()
        let now = Date()
        if let last = Self.lastLogTimes[nodeName], now.timeIntervalSince(last) < Self.logInterval
        {
            Self.lock.unlock()
            return
        }
        Self.lastLogTimes[nodeName] = now
        Self.lock.unlock()

        let milliseconds = elapsed * 1000
        let fps = elapsed > 0 ? 1.0 / elapsed : 0
        print(String(format: "%@: %.2f ms (%.1f fps)", nodeName, milliseconds, fps))
    }
}
