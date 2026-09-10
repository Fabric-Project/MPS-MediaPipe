//
//  MediaPipeInferenceTimingLogger.swift
//  MPSMediaPipe
//

import Foundation

/// Throttled inference latency logging, keyed by `nodeName`. Measures
/// wall-clock time to when the GPU actually finishes (not just CPU
/// submission) -- that's the number that says how fast a model really
/// runs.
///
/// Logs at most once every `logInterval` seconds per distinct `nodeName`;
/// callers sharing a name share a throttle bucket.
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
