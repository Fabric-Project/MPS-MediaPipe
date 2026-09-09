//
//  MediaPipeOneEuroFilter.swift
//  Fabric
//

import Foundation
import simd

/// Single-pole low-pass filter -- ports mediapipe/util/filtering/
/// low_pass_filter.cc verbatim (Apply/ApplyWithAlpha/SetAlpha). The first
/// call always passes its value straight through (no prior state to blend
/// against); every call after blends toward the new value by `alpha`.
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

/// The "1€ filter" (Casiez et al.) for a single scalar channel -- ports
/// mediapipe/util/filtering/one_euro_filter.cc's Apply()/GetAlpha()
/// verbatim (CreateLegacyFilter's own semantics: 0 is "no previous sample
/// yet", which is what every real wall-clock nanosecond timestamp this
/// filter will ever see satisfies). The cutoff frequency adapts to the
/// signal's own estimated speed (its low-pass-filtered derivative): a still
/// value gets heavily smoothed (cutoff near minCutoff), a fast-moving one
/// opens up toward less lag. use_filtered_derivative defaults to false in
/// MediaPipe's own landmarks-smoothing usage, so the derivative here is
/// always taken against the previous *raw* value, not the previous
/// filtered one -- matched here by reading `position.rawValue`, not
/// `position.storedValue`.
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

    /// `valueScale` is MediaPipe's own value_scale = 1/objectScale (see
    /// GetObjectScale in landmarks_smoothing_calculator_utils.cc), so the
    /// same minCutoff/beta behave consistently regardless of how large the
    /// tracked subject is in frame -- pass 1 to disable this normalization.
    /// `timestampNanoseconds` must be non-zero and should be monotonically
    /// increasing; a non-increasing timestamp is passed through unfiltered,
    /// matching OneEuroFilter::Apply's own guard. `debugTag`, when non-nil,
    /// throttled-prints every intermediate value (frequency, dvalue,
    /// edvalue, cutoff, alpha) via MediaPipeSmoothingDiagnosticLogger --
    /// temporary instrumentation for diagnosing over/under-smoothing without
    /// hand-deriving the expected numbers.
    func apply(timestampNanoseconds: Int64, value: Float, valueScale: Float = 1, debugTag: String? = nil) -> Float
    {
        guard timestampNanoseconds > self.lastTimeNanoseconds else { return value }

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

    /// Resets to the "first sample" state -- the next apply() call passes
    /// its value straight through unfiltered instead of blending toward a
    /// stale position, matching mediapipe's EMPTY_LANDMARKS_POLICY_RESET.
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
/// MediaPipeOneEuroFilter's own internals -- temporary instrumentation for
/// diagnosing why the smoothing feels over/under-aggressive without having
/// to hand-derive the expected numbers; mirrors
/// MediaPipeInferenceTimingLogger's own throttled-print shape.
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

/// Owns one MediaPipeOneEuroFilter pair (x, y) per landmark index, smoothing
/// a fixed-size landmark array the way mediapipe/modules/pose_landmark/
/// pose_landmark_filtering.pbtxt smooths its own main landmarks --
/// mediapipe's real tuning (minCutoff 0.05, beta 80.0, derivateCutoff 1.0)
/// is confirmed identical for Pose (pose_landmark_filtering.pbtxt) and Face
/// (face_landmarks_detector_graph.cc's ConfigureLandmarksSmoothingCalculator);
/// no equivalent reference exists for Hand (no smoothing pbtxt/graph config
/// found anywhere in the real repo), so Hand reuses the same proven values
/// rather than an invented, unsourced tuning.
///
/// A node with only an x/y consumer (MediaPipe Pose Landmarks'
/// outputLandmarks) uses the 2D smooth(points: [simd_float2], ...)
/// overload; a node whose landmarks carry meaningful depth downstream
/// (MediaPipe Face Landmarks' outputLandmarks3D, consumed by
/// FaceGeometryNode/FaceTransformNode) uses the 3D overload instead, so z
/// gets the same smoothing treatment rather than leaking raw, unsmoothed
/// depth alongside smoothed x/y. A given instance is only ever used with
/// one of the two overloads.
///
/// Not thread-safe -- only ever touched from execute() on the graph thread,
/// same as every other piece of per-node mutable state in these nodes.
public final class MediaPipeLandmarksSmoothingFilter
{
    /// mediapipe/modules/pose_landmark/pose_landmark_filtering.pbtxt +
    /// face_landmarks_detector_graph.cc's ConfigureLandmarksSmoothingCalculator,
    /// both verbatim-confirmed identical.
    public static let mediaPipeDefaultMinCutoff: Float = 0.05
    public static let mediaPipeDefaultBeta: Float = 80.0
    public static let mediaPipeDefaultDerivateCutoff: Float = 1.0

    /// GetObjectScale's own disable threshold (landmarks_smoothing_
    /// calculator.proto's min_allowed_object_scale default) -- below this,
    /// mediapipe passes landmarks through unfiltered for the frame rather
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

    /// `points` are normalized [0,1] (or Fabric unit space -- any origin
    /// works, only the *scale* matters here), full-image-relative, matching
    /// mediapipe's own NormalizedLandmark convention. Filtering happens in
    /// PIXEL space internally (points scaled by imageWidthPixels/
    /// imageHeightPixels, filtered, scaled back) because `objectScalePixels`
    /// -- mediapipe's own GetObjectScale(roi, imageWidth, imageHeight) --
    /// is itself in pixels: mediapipe's own tuning (minCutoff 0.05, beta
    /// 80.0 -> "~0.94 alpha when moving fast", per
    /// ConfigureLandmarksSmoothingCalculator's own comment) only produces
    /// that response for pixel-scale per-frame deltas (a few pixels/frame
    /// against a ~200px object scale) -- filtering normalized [0,1] deltas
    /// directly against a pixel-scale valueScale (1/objectScalePixels)
    /// makes dvalue three-plus orders of magnitude too small, pinning the
    /// filter at minCutoff always regardless of real motion (confirmed via
    /// MediaPipeSmoothingDiagnosticLogger output showing edvalue ~0.0001-
    /// 0.001 against the ~0.935 edvalue mediapipe's own "fast" reference
    /// implies). Empty `points` resets filter state so a re-acquired
    /// subject snaps in immediately instead of smoothing in from a stale
    /// position.
    public func smooth(points: [simd_float2], timestampNanoseconds: Int64, objectScalePixels: Float, imageWidthPixels: Float, imageHeightPixels: Float) -> [simd_float2]
    {
        guard points.isEmpty == false else
        {
            self.reset()
            return points
        }

        guard objectScalePixels >= Self.minAllowedObjectScale else { return points }

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

    /// 3D counterpart of smooth(points: [simd_float2], ...) -- same
    /// per-index x/y filters (same pixel-space rationale, see that
    /// overload's doc comment), plus an independent z filter per index so
    /// depth is smoothed with the same tuning rather than left raw. z is
    /// scaled by imageWidthPixels, matching mediapipe's own FaceMesh
    /// convention that z is "scaled like x" (see MediaPipeFaceLandmarkNode's
    /// own outputLandmarks3D doc comment).
    public func smooth(points: [simd_float3], timestampNanoseconds: Int64, objectScalePixels: Float, imageWidthPixels: Float, imageHeightPixels: Float) -> [simd_float3]
    {
        guard points.isEmpty == false else
        {
            self.reset()
            return points
        }

        guard objectScalePixels >= Self.minAllowedObjectScale else { return points }

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

    /// Exposed so a caller can reset filter state on presence loss even
    /// when it isn't calling smooth(points: [], ...) that frame (e.g. a
    /// node that simply stops sending outputLandmarks rather than resending
    /// an empty array).
    public func reset()
    {
        self.xFilters.removeAll()
        self.yFilters.removeAll()
        self.zFilters.removeAll()
    }
}
