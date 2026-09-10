//
//  MediaPipeSSDDetectorDecoder.swift
//  MPSMediaPipe
//

import Foundation

/// Decodes a "Blaze"-family SSD detector's raw per-anchor output and
/// merges overlapping detections via weighted NMS (score-weighted average
/// of overlapping boxes, not greedy suppression). Shared by BlazePalm and
/// BlazeFace, which use the same calculator pair with different
/// `numKeypoints`/`detectSize`.
public enum MediaPipeSSDDetectorDecoder
{
    public static let minDetectionConfidence: Float = 0.5
    static let nmsThreshold: Float = 0.3
    static let scoreClippingThreshold: Float = 100.0

    /// All coordinates normalized [0,1] in the detector's tensor space,
    /// top-left origin (MediaPipe/OpenCV convention) — callers project into
    /// full-image space separately.
    public struct Detection
    {
        public var xmin: Float
        public var ymin: Float
        public var width: Float
        public var height: Float
        public var keypoints: [(x: Float, y: Float)]
        public var score: Float
    }

    /// `rawBoxes` is the flattened `[anchors, 4 + numKeypoints*2]` box
    /// tensor, `rawScores` the flattened `[anchors]` classification tensor,
    /// both row-major as produced by the model. `detectSize` is the square
    /// detector input size (192 for BlazePalm/BlazeFace full_range, 128 for
    /// BlazeFace short_range). `minScore` defaults to 0.5; BlazeFace
    /// full_range overrides it to 0.6.
    public static func decode(rawBoxes: [Float], rawScores: [Float], anchors: [(cx: Float, cy: Float, w: Float, h: Float)], numKeypoints: Int, detectSize: Int, minScore: Float = minDetectionConfidence) -> [Detection]
    {
        let scale = Float(detectSize)
        let coordsPerAnchor = 4 + numKeypoints * 2
        var detections: [Detection] = []

        for anchorIndex in 0..<anchors.count
        {
            let logit = min(max(rawScores[anchorIndex], -scoreClippingThreshold), scoreClippingThreshold)
            // Sigmoid computed in float64, rounded once to float32.
            let score = Float(1.0 / (1.0 + exp(-Double(logit))))
            guard score >= minScore else { continue }

            let anchor = anchors[anchorIndex]
            let base = anchorIndex * coordsPerAnchor

            let xc = rawBoxes[base + 0] / scale * anchor.w + anchor.cx
            let yc = rawBoxes[base + 1] / scale * anchor.h + anchor.cy
            let width = rawBoxes[base + 2] / scale * anchor.w
            let height = rawBoxes[base + 3] / scale * anchor.h
            let xmin = xc - width / 2
            let ymin = yc - height / 2

            var keypoints: [(x: Float, y: Float)] = []
            keypoints.reserveCapacity(numKeypoints)
            for keypointIndex in 0..<numKeypoints
            {
                let kx = rawBoxes[base + 4 + keypointIndex * 2] / scale * anchor.w + anchor.cx
                let ky = rawBoxes[base + 4 + keypointIndex * 2 + 1] / scale * anchor.h + anchor.cy
                keypoints.append((x: kx, y: ky))
            }

            detections.append(Detection(xmin: xmin, ymin: ymin, width: width, height: height, keypoints: keypoints, score: score))
        }

        return detections
    }

    static func intersectionOverUnion(_ a: Detection, _ b: Detection) -> Float
    {
        let xa = max(a.xmin, b.xmin), ya = max(a.ymin, b.ymin)
        let xb = min(a.xmin + a.width, b.xmin + b.width), yb = min(a.ymin + a.height, b.ymin + b.height)
        guard xb > xa, yb > ya else { return 0 }

        let intersection = (xb - xa) * (yb - ya)
        let union = a.width * a.height + b.width * b.height - intersection
        guard union > 0 else { return 0 }
        return intersection / union
    }

    /// Each retained detection is a score-weighted average of itself and
    /// every remaining detection whose IoU exceeds `nmsThreshold` -- not
    /// greedy suppression. `top` is removed from `remaining`
    /// unconditionally (not via its own self-IoU) so the loop terminates
    /// even for a degenerate (zero-width) detection, whose self-IoU is 0.
    public static func weightedNonMaximumSuppression(_ detections: [Detection]) -> [Detection]
    {
        var remaining = detections.sorted { $0.score > $1.score }
        var merged: [Detection] = []

        while remaining.isEmpty == false
        {
            let top = remaining.removeFirst()
            let overlaps = remaining.map { intersectionOverUnion($0, top) }
            let candidates = [top] + zip(remaining, overlaps).filter { $0.1 > nmsThreshold }.map(\.0)
            remaining = zip(remaining, overlaps).filter { $0.1 <= nmsThreshold }.map(\.0)

            var result = top
            if candidates.isEmpty == false
            {
                var weightedXmin: Float = 0, weightedYmin: Float = 0, weightedXmax: Float = 0, weightedYmax: Float = 0
                var totalScore: Float = 0
                var keypointAccumulator = [(x: Float, y: Float)](repeating: (0, 0), count: top.keypoints.count)

                for candidate in candidates
                {
                    totalScore += candidate.score
                    weightedXmin += candidate.xmin * candidate.score
                    weightedYmin += candidate.ymin * candidate.score
                    weightedXmax += (candidate.xmin + candidate.width) * candidate.score
                    weightedYmax += (candidate.ymin + candidate.height) * candidate.score
                    for keypointIndex in 0..<top.keypoints.count
                    {
                        keypointAccumulator[keypointIndex].x += candidate.keypoints[keypointIndex].x * candidate.score
                        keypointAccumulator[keypointIndex].y += candidate.keypoints[keypointIndex].y * candidate.score
                    }
                }

                result.xmin = weightedXmin / totalScore
                result.ymin = weightedYmin / totalScore
                result.width = (weightedXmax / totalScore) - result.xmin
                result.height = (weightedYmax / totalScore) - result.ymin
                result.keypoints = keypointAccumulator.map { (x: $0.x / totalScore, y: $0.y / totalScore) }
            }

            merged.append(result)
        }

        return merged
    }

    /// Generalizes DetectionsToRectsCalculator::ComputeRotation over which
    /// two keypoints define the rotation vector and the target angle.
    /// BlazePalm: (wrist=0, middleMCP=2), target 90 -- raw radians, not
    /// degrees (a MediaPipe proto quirk). BlazeFace: (leftEye=0,
    /// rightEye=1), target 0.
    static func computeRotation(from startPoint: (x: Float, y: Float), to endPoint: (x: Float, y: Float), targetAngleRadians: Float) -> Float
    {
        let angle = targetAngleRadians - atan2(-(endPoint.y - startPoint.y), endPoint.x - startPoint.x)
        return normalizeRadians(angle)
    }

    static func normalizeRadians(_ angle: Float) -> Float
    {
        angle - 2 * .pi * ((angle + .pi) / (2 * .pi)).rounded(.down)
    }
}
