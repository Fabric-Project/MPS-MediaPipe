//
//  MediaPipeCropPreprocess.metal
//  Fabric
//

#include <metal_stdlib>

using namespace metal;

// Inlined rather than #include'd from a shared header -- this package has
// no dependency on Fabric's own shader tree, and this is the only place
// (besides MediaPipeSegmentationMaskWarp.metal, which carries its own
// identical copy) that needs it.
inline float2 mediaPipeTextureCoordinate(float4x4 textureTransform, float2 coordinate)
{
    return (textureTransform * float4(coordinate, 0.0, 1.0)).xy;
}

// Rotation direction empirically verified against OpenCV's own
// warpAffine/boxPoints output (fasthands.pipeline.crop_rotated_rect): for a
// centered output pixel offset (px, py) in source-rect-local pixels, the
// corresponding source-image offset is R(+rotationRadians) * (px, py) --
// i.e. rotate the *output-local* offset directly by `rotationRadians` (no
// negation) to land back in source space. Confirmed against a synthetic
// marker image, not derived from OpenCV's boxPoints/getAffineTransform
// corner-correspondence algorithm directly (that convention was too easy to
// get subtly wrong to trust without independent verification).
struct MediaPipeCropUniforms {
    float2 centerPixels;             // (cx, cy), top-left-origin PRESENTATION PIXEL space
    float2 rectSizePixels;           // (width, height) in pixels
    float rotationRadians;           // MediaPipeSSDDetectorDecoder.computeRotation's own convention
    float4x4 textureTransform;
    float2 presentationSizePixels;
    uint2 outputSize;
    float2 outputPixelRange;         // (min, max) the sampled [0,1] color maps onto -- BlazePalm/
                                      // BlazeFace's landmark models both normalize to [0,1], but
                                      // BlazeFace's detector normalizes to [-1,1]; confirmed against
                                      // each model's own ImageToTensorCalculatorOptions.output_tensor_float_range.
};

kernel void cropRotateAndNormalizeNHWC(
    texture2d<float, access::sample> sourceTexture [[texture(0)]],
    device float *destination [[buffer(0)]],
    constant MediaPipeCropUniforms &uniforms [[buffer(1)]],
    uint2 position [[thread_position_in_grid]])
{
    if (position.x >= uniforms.outputSize.x || position.y >= uniforms.outputSize.y) {
        return;
    }

    constexpr sampler cropSampler(
        coord::normalized,
        address::clamp_to_edge,
        min_filter::linear,
        mag_filter::linear
    );

    // Centered, unit-square output coordinate, then scaled into source-rect-
    // local pixel offsets.
    const float2 outputCoordinate = (float2(position) + 0.5f) / float2(uniforms.outputSize) - 0.5f;
    const float2 localOffset = outputCoordinate * uniforms.rectSizePixels;

    const float cosA = cos(uniforms.rotationRadians);
    const float sinA = sin(uniforms.rotationRadians);
    const float2 rotatedOffset = float2(
        localOffset.x * cosA - localOffset.y * sinA,
        localOffset.x * sinA + localOffset.y * cosA
    );

    const float2 sourcePixels = uniforms.centerPixels + rotatedOffset;
    const float2 presentationCoordinate = sourcePixels / uniforms.presentationSizePixels;
    const float2 storedCoordinate = mediaPipeTextureCoordinate(uniforms.textureTransform, presentationCoordinate);

    const float3 sampledColor = saturate(sourceTexture.sample(cropSampler, storedCoordinate).rgb);
    const float3 color = mix(uniforms.outputPixelRange.x, uniforms.outputPixelRange.y, sampledColor);

    // NHWC (model input is [1, H, W, 3], not NCHW) — matches every bundled
    // MediaPipe model's declared input shape directly, no transpose needed
    // downstream.
    const uint pixelIndex = (position.y * uniforms.outputSize.x + position.x) * 3;
    destination[pixelIndex + 0] = color.r;
    destination[pixelIndex + 1] = color.g;
    destination[pixelIndex + 2] = color.b;
}
