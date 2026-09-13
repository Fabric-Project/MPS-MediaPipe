//
//  MediaPipeOneEuroFilter.swift
//  MPSMediaPipe
//

import Foundation
import simd

/// Single-pole low-pass filter, ported from low_pass_filter.cc. The first
/// call passes its value straight through; every call after blends toward
/// the new value by `alpha`.
final class MediaPipeLowPassFilter
{
    private var initialized = false
    private var alpha: Float
    private(set) var rawValue: Float = 0
    private(set) var storedValue: Float = 0

    init(alpha: Float) { self.alpha = alpha }

    var hasLastRawValue: Bool { self.initialized }

    @discardableResult
    func apply(_ value: Float) -> Float
    {
        let result: Float
        if self.initialized
        {
            result = self.alpha * value + (1 - self.alpha) * self.storedValue
        }
        else
        {
            result = value
            self.initialized = true
        }
        self.rawValue = value
        self.storedValue = result
        return result
    }

    @discardableResult
    func apply(_ value: Float, alpha: Float) -> Float
    {
        self.alpha = alpha
        return self.apply(value)
    }

    func reset()
    {
        self.initialized = false
        self.rawValue = 0
        self.storedValue = 0
    }
}

/// The "1€ filter" (Casiez et al.) for a single scalar channel, ported
/// from one_euro_filter.cc. Cutoff frequency adapts to the signal's
/// estimated speed: a still value gets heavily smoothed (cutoff near
/// minCutoff), a fast-moving one opens up toward less lag. The derivative
/// is taken against the previous raw value, not the previous filtered
/// one (`use_filtered_derivative=false`).
final class MediaPipeOneEuroFilter
{
    private let minCutoff: Float
    private let beta: Float
    private let derivateCutoff: Float
    private var frequency: Float
    private var lastTimeNanoseconds: Int64 = 0

    private let position = MediaPipeLowPassFilter(alpha: 1)
    private let derivative = MediaPipeLowPassFilter(alpha: 1)

    init(frequency: Float = 30, minCutoff: Float, beta: Float, derivateCutoff: Float = 1)
    {
        self.frequency = frequency
        self.minCutoff = minCutoff
        self.beta = beta
        self.derivateCutoff = derivateCutoff
    }

    /// `valueScale` normalizes cutoff behavior against tracked-subject size
    /// (1/objectScale) -- pass 1 to disable. `timestampNanoseconds` must be
    /// non-zero and increasing to actually filter; a non-increasing
    /// timestamp (a caller re-applying the same still-unchanged sample, or
    /// a genuine clock anomaly) holds the last smoothed output rather than
    /// re-deriving anything -- returning the raw `value` instead would
    /// snap straight past the smoothing on every such call, since `value`
    /// on a repeat call is whatever's cached upstream, not a fresh sample.
    /// Falls back to `value` only before any sample has ever been applied
    /// (`position` has no stored value yet to hold). `debugTag`, when
    /// non-nil, throttled-logs intermediate values via
    /// MediaPipeSmoothingDiagnosticLogger.
    func apply(timestampNanoseconds: Int64, value: Float, valueScale: Float = 1, debugTag: String? = nil) -> Float
    {
        guard timestampNanoseconds > self.lastTimeNanoseconds else
        {
            return self.position.hasLastRawValue ? self.position.storedValue : value
        }

        if self.lastTimeNanoseconds != 0
        {
            let dt = Float(timestampNanoseconds - self.lastTimeNanoseconds) * 1e-9
            self.frequency = 1.0 / dt
        }
        self.lastTimeNanoseconds = timestampNanoseconds

        var deltaValue: Float = 0
        if self.position.hasLastRawValue
        {
            deltaValue = (value - self.position.rawValue) * valueScale * self.frequency
        }
        let filteredDelta = self.derivative.apply(deltaValue, alpha: Self.alpha(cutoff: self.derivateCutoff, frequency: self.frequency))
        let cutoff = self.minCutoff + self.beta * abs(filteredDelta)
        let outputAlpha = Self.alpha(cutoff: cutoff, frequency: self.frequency)
        let result = self.position.apply(value, alpha: outputAlpha)

        if let debugTag
        {
            MediaPipeSmoothingDiagnosticLogger.log(
                tag: debugTag, frequency: self.frequency, valueScale: valueScale,
                rawValue: value, dvalue: deltaValue, edvalue: filteredDelta,
                cutoff: cutoff, alpha: outputAlpha, result: result
            )
        }

        return result
    }

    /// Resets to the "first sample" state: the next apply() call passes
    /// its value straight through instead of blending toward a stale
    /// position.
    func reset()
    {
        self.lastTimeNanoseconds = 0
        self.position.reset()
        self.derivative.reset()
    }

    private static func alpha(cutoff: Float, frequency: Float) -> Float
    {
        let te = 1.0 / frequency
        let tau = 1.0 / (2 * Float.pi * cutoff)
        return 1.0 / (1.0 + tau / te)
    }
}

/// Throttled (once per tag per second) diagnostic print of a
/// MediaPipeOneEuroFilter's internals.
enum MediaPipeSmoothingDiagnosticLogger
{
    private static let lock = NSLock()
    private static var lastLogTimes: [String: Date] = [:]
    private static let logInterval: TimeInterval = 1.0

    static func log(tag: String, frequency: Float, valueScale: Float, rawValue: Float, dvalue: Float, edvalue: Float, cutoff: Float, alpha: Float, result: Float)
    {
        self.lock.lock()
        let now = Date()
        let shouldLog = (self.lastLogTimes[tag].map { now.timeIntervalSince($0) >= self.logInterval }) ?? true
        if shouldLog { self.lastLogTimes[tag] = now }
        self.lock.unlock()

        guard shouldLog else { return }

        print(String(
            format: "%@: freq=%.1fHz valueScale=%.5f raw=%.4f dvalue=%.4f edvalue=%.4f cutoff=%.4fHz alpha=%.4f result=%.4f",
            tag, frequency, valueScale, rawValue, dvalue, edvalue, cutoff, alpha, result
        ))
    }
}

/// Owns one MediaPipeOneEuroFilter pair (x, y[, z]) per landmark index,
/// smoothing a fixed-size landmark array using mediapipe's own pose/face
/// landmarks-smoothing tuning (Hand has no equivalent reference, so it
/// reuses the same values).
///
/// Use the 2D overload when only x/y matter downstream; use the 3D
/// overload when z carries meaningful depth, so it gets the same
/// smoothing treatment rather than leaking raw depth. Switching which
/// overload is called on an existing instance resets its filter state
/// automatically (see `activeDimensionality`), so the two streams never
/// silently blend into each other -- but that reset also means treating
/// one instance as two independent streams by alternating overloads
/// defeats the smoothing itself. Use one instance per point stream.
///
/// Not thread-safe.
public final class MediaPipeLandmarksSmoothingFilter
{
    /// mediapipe's own pose/face landmarks-smoothing tuning.
    public static let mediaPipeDefaultMinCutoff: Float = 0.05
    public static let mediaPipeDefaultBeta: Float = 80.0
    public static let mediaPipeDefaultDerivateCutoff: Float = 1.0

    /// Below this object scale, pass landmarks through unfiltered rather
    /// than dividing by a near-zero scale.
    private static let minAllowedObjectScale: Float = 1e-6

    private let minCutoff: Float
    private let beta: Float
    private let derivateCutoff: Float
    /// Non-nil only while diagnosing over/under-smoothing -- see
    /// MediaPipeSmoothingDiagnosticLogger. Only index 0's x filter logs, to
    /// keep the output readable.
    private let debugLabel: String?
    private var xFilters: [MediaPipeOneEuroFilter] = []
    private var yFilters: [MediaPipeOneEuroFilter] = []
    private var zFilters: [MediaPipeOneEuroFilter] = []

    /// Tracks which `smooth(points:...)` overload this instance was last
    /// called with. Calling the other overload resets filter state first,
    /// so per-index filter history never blends across two unrelated point
    /// streams that happen to share one instance.
    private enum Dimensionality { case two, three }
    private var activeDimensionality: Dimensionality?

    public init(
        minCutoff: Float = MediaPipeLandmarksSmoothingFilter.mediaPipeDefaultMinCutoff,
        beta: Float = MediaPipeLandmarksSmoothingFilter.mediaPipeDefaultBeta,
        derivateCutoff: Float = MediaPipeLandmarksSmoothingFilter.mediaPipeDefaultDerivateCutoff,
        debugLabel: String? = nil
    )
    {
        self.minCutoff = minCutoff
        self.beta = beta
        self.derivateCutoff = derivateCutoff
        self.debugLabel = debugLabel
    }

    /// `points` are normalized [0,1] (any origin -- only scale matters
    /// here), full-image-relative. Filtering happens in pixel space
    /// internally (scaled by imageWidthPixels/imageHeightPixels, filtered,
    /// scaled back), since `objectScalePixels` is itself in pixels and
    /// mediapipe's tuning assumes pixel-scale deltas -- filtering
    /// normalized deltas directly would pin the filter at minCutoff
    /// regardless of real motion. Empty `points` resets filter state so a
    /// re-acquired subject snaps in immediately.
    public func smooth(points: [simd_float2], timestampNanoseconds: Int64, objectScalePixels: Float, imageWidthPixels: Float, imageHeightPixels: Float) -> [simd_float2]
    {
        guard points.isEmpty == false else
        {
            self.reset()
            return points
        }

        guard objectScalePixels >= Self.minAllowedObjectScale else { return points }

        if self.activeDimensionality != .two
        {
            self.reset()
            self.activeDimensionality = .two
        }
        if self.xFilters.count != points.count
        {
            self.xFilters = (0..<points.count).map { _ in MediaPipeOneEuroFilter(minCutoff: self.minCutoff, beta: self.beta, derivateCutoff: self.derivateCutoff) }
            self.yFilters = (0..<points.count).map { _ in MediaPipeOneEuroFilter(minCutoff: self.minCutoff, beta: self.beta, derivateCutoff: self.derivateCutoff) }
        }

        let valueScale = 1.0 / objectScalePixels
        var result = [simd_float2]()
        result.reserveCapacity(points.count)
        for index in 0..<points.count
        {
            let debugTag = (index == 0) ? self.debugLabel.map { "\($0)[0].x" } : nil
            let x = self.xFilters[index].apply(timestampNanoseconds: timestampNanoseconds, value: points[index].x * imageWidthPixels, valueScale: valueScale, debugTag: debugTag) / imageWidthPixels
            let y = self.yFilters[index].apply(timestampNanoseconds: timestampNanoseconds, value: points[index].y * imageHeightPixels, valueScale: valueScale) / imageHeightPixels
            result.append(simd_float2(x, y))
        }
        return result
    }

    /// 3D counterpart of smooth(points: [simd_float2], ...): same per-index
    /// x/y filters, plus an independent z filter per index. z is scaled by
    /// imageWidthPixels, matching mediapipe's FaceMesh convention that z
    /// is scaled like x.
    public func smooth(points: [simd_float3], timestampNanoseconds: Int64, objectScalePixels: Float, imageWidthPixels: Float, imageHeightPixels: Float) -> [simd_float3]
    {
        guard points.isEmpty == false else
        {
            self.reset()
            return points
        }

        guard objectScalePixels >= Self.minAllowedObjectScale else { return points }

        if self.activeDimensionality != .three
        {
            self.reset()
            self.activeDimensionality = .three
        }
        if self.xFilters.count != points.count
        {
            self.xFilters = (0..<points.count).map { _ in MediaPipeOneEuroFilter(minCutoff: self.minCutoff, beta: self.beta, derivateCutoff: self.derivateCutoff) }
            self.yFilters = (0..<points.count).map { _ in MediaPipeOneEuroFilter(minCutoff: self.minCutoff, beta: self.beta, derivateCutoff: self.derivateCutoff) }
            self.zFilters = (0..<points.count).map { _ in MediaPipeOneEuroFilter(minCutoff: self.minCutoff, beta: self.beta, derivateCutoff: self.derivateCutoff) }
        }

        let valueScale = 1.0 / objectScalePixels
        var result = [simd_float3]()
        result.reserveCapacity(points.count)
        for index in 0..<points.count
        {
            let debugTag = (index == 0) ? self.debugLabel.map { "\($0)[0].x" } : nil
            let x = self.xFilters[index].apply(timestampNanoseconds: timestampNanoseconds, value: points[index].x * imageWidthPixels, valueScale: valueScale, debugTag: debugTag) / imageWidthPixels
            let y = self.yFilters[index].apply(timestampNanoseconds: timestampNanoseconds, value: points[index].y * imageHeightPixels, valueScale: valueScale) / imageHeightPixels
            let z = self.zFilters[index].apply(timestampNanoseconds: timestampNanoseconds, value: points[index].z * imageWidthPixels, valueScale: valueScale) / imageWidthPixels
            result.append(simd_float3(x, y, z))
        }
        return result
    }

    /// Exposed so a caller can reset filter state on presence loss without
    /// calling smooth(points: [], ...) that frame.
    public func reset()
    {
        self.xFilters.removeAll()
        self.yFilters.removeAll()
        self.zFilters.removeAll()
    }
}
