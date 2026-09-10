//
//  MediaPipeSelfieSegmentation.swift
//  MPSMediaPipe
//

import Foundation
import simd

/// MediaPipe's Selfie Segmentation model (person-vs-background mask) —
/// owns the two model variants' geometry and the "no detector, no ROI,
/// no aspect-preserving crop" facts specific to this model.
///
/// Unlike Face/Hand/Pose, there is no detector, no ROI, and (confirmed
/// directly against mediapipe/modules/selfie_segmentation/
/// selfie_segmentation_gpu.pbtxt, not assumed) no aspect-preserving crop
/// either: `ImageToTensorCalculator` sets `keep_aspect_ratio: false`, a
/// plain non-uniform stretch of the whole frame into the tensor. That
/// degenerate case (center (0.5,0.5), size (1,1), rotation 0) is already
/// exactly what MediaPipeCropPreprocessor's forward-crop shader computes
/// for any center/size/rotation, and what MediaPipeSegmentationMaskProjector's
/// inverse-warp shader (built for BlazePose's own segmentation output)
/// computes in reverse — both reused unmodified by a caller.
///
/// Two model variants (General 256x256, Landscape 256x144 — TFLite's own
/// [N,H,W,C] shape confirms H=144,W=256, matching mediapipe's own "256x144"
/// naming as width x height).
///
/// The real graph's `TensorsToSegmentationCalculator` sets
/// `activation: NONE` (unlike BlazePose's own baked-in segmentation head,
/// which needs an external SIGMOID) — confirmed, not assumed, by reading
/// the resolved op list: the model's own last op is `LOGISTIC` (sigmoid),
/// so `applySigmoid` is false — no additional activation belongs in a
/// caller's own decode.
public enum MediaPipeSelfieSegmentation
{
    public enum Variant: String, CaseIterable
    {
        case general = "General"
        case landscape = "Landscape"

        public var inputWidth: Int { 256 }
        public var inputHeight: Int { self == .general ? 256 : 144 }
        public var resourcePrefix: String { self == .general ? "MediaPipeSelfieSegmentation" : "MediaPipeSelfieSegmentationLandscape" }

        public static func from(_ rawValue: String?) -> Variant
        {
            rawValue.flatMap(Variant.init(rawValue:)) ?? .general
        }
    }

    /// Whole-frame stretch, no letterbox — the "no crop" values to pass to
    /// both MediaPipeCropPreprocessor.encode's and
    /// MediaPipeSegmentationMaskProjector.encode's `centerNormalizedBottomLeft`/
    /// `sizeNormalized`/`rotationRadians` parameters.
    public static let fullFrameCenter = simd_float2(0.5, 0.5)
    public static let fullFrameSize = simd_float2(1, 1)
    public static let noRotation: Float = 0

    /// This model's graph already ends in LOGISTIC (sigmoid) — pass to
    /// MediaPipeSegmentationMaskProjector.encode's `applySigmoid` parameter.
    public static let applySigmoid = false
}
