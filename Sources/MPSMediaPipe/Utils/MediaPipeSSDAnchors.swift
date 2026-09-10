//
//  MediaPipeSSDAnchors.swift
//  MPSMediaPipe
//

import Foundation

/// SSD anchor grid for MediaPipe's "Blaze" family of detectors, ported
/// from ssd_anchors_calculator.cc.
///
/// BlazePalm and BlazeFace short_range: num_layers=4,
/// strides=[8,16,16,16], fixed_anchor_size=true,
/// interpolated_scale_aspect_ratio=1.0 (the defaults below). BlazeFace
/// full_range: num_layers=1, strides=[4],
/// interpolated_scale_aspect_ratio=0.0.
public enum MediaPipeSSDAnchors
{
    /// (cx, cy, w, h), all normalized [0,1] relative to detectSize, in the
    /// same top-left-origin space the detector's raw box regression uses.
    /// `strides` defaults to the value both BlazePalm and BlazeFace
    /// short_range use; re-check against the specific detector's own config
    /// before reusing this for a new model.
    public static func generate(detectSize: Int, strides: [Int] = [8, 16, 16, 16], interpolatedScaleAspectRatio: Float = 1.0) -> [(cx: Float, cy: Float, w: Float, h: Float)]
    {
        let numLayers = strides.count
        let anchorsPerLayerGroupEntry = interpolatedScaleAspectRatio > 0 ? 2 : 1 // aspect_ratio 1.0 anchor, plus an interpolated anchor (same center) only when interpolatedScaleAspectRatio > 0
        var anchors: [(cx: Float, cy: Float, w: Float, h: Float)] = []

        var layerIndex = 0
        while layerIndex < numLayers
        {
            var anchorsPerCell = 0
            var lastLayerIndex = layerIndex
            while lastLayerIndex < numLayers, strides[lastLayerIndex] == strides[layerIndex]
            {
                anchorsPerCell += anchorsPerLayerGroupEntry
                lastLayerIndex += 1
            }

            let featureMapSize = Int((Float(detectSize) / Float(strides[layerIndex])).rounded(.up))
            for y in 0..<featureMapSize
            {
                for x in 0..<featureMapSize
                {
                    for _ in 0..<anchorsPerCell
                    {
                        anchors.append((
                            cx: (Float(x) + 0.5) / Float(featureMapSize),
                            cy: (Float(y) + 0.5) / Float(featureMapSize),
                            w: 1.0, h: 1.0 // fixed_anchor_size
                        ))
                    }
                }
            }

            layerIndex = lastLayerIndex
        }

        return anchors
    }
}
