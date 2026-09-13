//
//  MediaPipeDetection.swift
//  MPSMediaPipe
//

import Foundation

/// A single decoded, full-image-projected detection from a "Blaze"-family
/// SSD detector, returned by all three detector facades
/// (MediaPipeFaceDetector/MediaPipeHandDetector/MediaPipePoseDetector).
/// `region`/`keypoints` are MediaPipe's native top-left-origin normalized
/// full-image space; a caller in a bottom-left-origin coordinate system
/// flips on its own side.
public struct MediaPipeDetection
{
    public let region: (cx: Float, cy: Float, width: Float, height: Float)
    public let rotation: Float
    public let score: Float
    public let keypoints: [(x: Float, y: Float)]

    public init(region: (cx: Float, cy: Float, width: Float, height: Float), rotation: Float, score: Float, keypoints: [(x: Float, y: Float)])
    {
        self.region = region
        self.rotation = rotation
        self.score = score
        self.keypoints = keypoints
    }
}
