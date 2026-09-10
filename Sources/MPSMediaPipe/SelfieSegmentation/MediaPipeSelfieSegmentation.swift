//
//  MediaPipeSelfieSegmentation.swift
//  MPSMediaPipe
//

import Foundation
import simd

/// MediaPipe's Selfie Segmentation model. Unlike Face/Hand/Pose, there is
/// no detector, no ROI, and no aspect-preserving crop -- the whole frame
/// is stretched into the tensor (keep_aspect_ratio: false), so the "no
/// crop" values below are the ones to pass to MediaPipeCropPreprocessor
/// and MediaPipeSegmentationMaskProjector.
///
/// Two model variants: General (256x256), Landscape (256x144).
///
/// This model's graph already ends in a sigmoid, unlike BlazePose's
/// segmentation head -- `applySigmoid` is false.
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

    /// Whole-frame stretch, no letterbox -- pass to
    /// MediaPipeCropPreprocessor's and MediaPipeSegmentationMaskProjector's
    /// centerNormalizedBottomLeft/sizeNormalized/rotationRadians parameters.
    public static let fullFrameCenter = simd_float2(0.5, 0.5)
    public static let fullFrameSize = simd_float2(1, 1)
    public static let noRotation: Float = 0

    /// Pass to MediaPipeSegmentationMaskProjector.encode's applySigmoid.
    public static let applySigmoid = false
}
