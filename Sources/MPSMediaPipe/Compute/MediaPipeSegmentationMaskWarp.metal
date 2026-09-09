//
//  MediaPipeSegmentationMaskWarp.metal
//  Fabric
//

#include <metal_stdlib>

using namespace metal;

// Inverse of MediaPipeCropPreprocess.metal's cropRotateAndNormalizeNHWC:
// that kernel maps a crop-space destination pixel back to a full-image
// source pixel via rotate(+rotationRadians) around centerPixels. Here the
// roles are reversed -- destination is a full-image pixel, source is the
// model's 256x256 crop-space mask -- so the same rotate-around-center math
// runs backwards: rotate(-rotationRadians), and anything that lands outside
// the crop's own unit square is zero (mirrors WarpAffineCalculator's
// BORDER_ZERO in MediaPipe's own pose_landmarks_and_segmentation_inverse_
// projection.pbtxt).
struct MediaPipeSegmentationMaskWarpUniforms {
    float2 centerPixels;             // (cx, cy), top-left-origin PRESENTATION PIXEL space
    float2 rectSizePixels;           // (width, height) in pixels -- same crop rect as the forward kernel
    float rotationRadians;           // MediaPipeSSDDetectorDecoder.computeRotation's own convention
    uint2 maskSize;                  // source buffer dimensions (256x256)
    uint2 outputSize;                // destination texture dimensions (full image)
    uint applySigmoid;                // 1 if maskValues holds pre-activation logits (BlazePose's own
                                      // segmentation head), 0 if the model's own graph already ends in
                                      // its own sigmoid (MediaPipe Selfie Segmentation's last op is a
                                      // real LOGISTIC, confirmed by inspecting its resolved op list --
                                      // re-applying sigmoid here would double-activate, compressing
                                      // every value toward 0.5 (visibly "grey", not the intended
                                      // near-binary confidence field).
};

kernel void warpSegmentationMaskInverseNHWC(
    device const float *maskValues [[buffer(0)]],
    constant MediaPipeSegmentationMaskWarpUniforms &uniforms [[buffer(1)]],
    texture2d<float, access::write> destination [[texture(0)]],
    uint2 position [[thread_position_in_grid]])
{
    if (position.x >= uniforms.outputSize.x || position.y >= uniforms.outputSize.y) {
        return;
    }

    // Destination pixel center, top-left-origin presentation pixel space --
    // outputSize is the full image, so position already lives in that space.
    const float2 destinationPixels = float2(position) + 0.5f;
    const float2 offsetFromCenter = destinationPixels - uniforms.centerPixels;

    const float cosA = cos(-uniforms.rotationRadians);
    const float sinA = sin(-uniforms.rotationRadians);
    const float2 unrotatedOffset = float2(
        offsetFromCenter.x * cosA - offsetFromCenter.y * sinA,
        offsetFromCenter.x * sinA + offsetFromCenter.y * cosA
    );

    // Centered [-0.5, 0.5] crop-local coordinate -- outside this range is
    // outside the crop rect entirely (BORDER_ZERO).
    const float2 cropLocal = unrotatedOffset / uniforms.rectSizePixels;

    if (any(cropLocal < -0.5f) || any(cropLocal >= 0.5f)) {
        destination.write(float4(0.0f, 0.0f, 0.0f, 1.0f), position);
        return;
    }

    // [0,1] crop UV, top-left origin, matching the forward kernel's own
    // output row/column layout (both walk the same H,W indexing the model
    // was fed and returned mask logits in).
    const float2 cropUV = cropLocal + 0.5f;
    const float2 maskPixels = cropUV * float2(uniforms.maskSize) - 0.5f;

    const float2 basePixel = floor(maskPixels);
    const float2 fraction = maskPixels - basePixel;

    const int maskWidth = int(uniforms.maskSize.x);
    const int maskHeight = int(uniforms.maskSize.y);

    auto sampleValue = [&](int x, int y) -> float
    {
        const int clampedX = clamp(x, 0, maskWidth - 1);
        const int clampedY = clamp(y, 0, maskHeight - 1);
        return maskValues[clampedY * maskWidth + clampedX];
    };

    const int x0 = int(basePixel.x);
    const int y0 = int(basePixel.y);

    const float topLeft = sampleValue(x0, y0);
    const float topRight = sampleValue(x0 + 1, y0);
    const float bottomLeft = sampleValue(x0, y0 + 1);
    const float bottomRight = sampleValue(x0 + 1, y0 + 1);

    const float top = mix(topLeft, topRight, fraction.x);
    const float bottom = mix(bottomLeft, bottomRight, fraction.x);
    const float sampled = mix(top, bottom, fraction.y);

    const float confidence = uniforms.applySigmoid != 0 ? (1.0f / (1.0f + exp(-sampled))) : sampled;

    destination.write(float4(confidence, confidence, confidence, 1.0f), position);
}
