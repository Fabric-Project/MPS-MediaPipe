//
//  MediaPipeSSDRectTransform.swift
//  MPSMediaPipe
//

import Foundation

/// Projects a "Blaze"-family SSD detection from the detector's letterboxed
/// tensor space into full-image space, then derives the rotated square ROI
/// fed to the corresponding landmark model. Shared by BlazePalm and
/// BlazeFace, which use different rotation keypoints/target angle/scale/
/// shift.
///
/// BlazePalm: rotation keypoints (wrist=0, middleMCP=2), target 90 (raw
/// radians, not degrees -- a MediaPipe proto quirk), scale 2.6, shift_y
/// -0.5. BlazeFace: rotation keypoints (leftEye=0, rightEye=1), target 0,
/// scale 1.5 (both axes), no shift.
public enum MediaPipeSSDRectTransform
{
    /// DetectionProjectionCalculator's matrix for the full-image, non-
    /// rotated, keep-aspect-ratio letterbox ROI used for detection (side =
    /// max(imageWidth, imageHeight), centered) — maps tensor-space
    /// normalized coordinates into full-image normalized coordinates, both
    /// top-left origin.
    static func projectLetterbox(x: Float, y: Float, imageWidth: Float, imageHeight: Float) -> (x: Float, y: Float)
    {
        let side = max(imageWidth, imageHeight)
        let m0 = side / imageWidth
        let m3 = (-0.5 * side + 0.5 * imageWidth) / imageWidth
        let m5 = side / imageHeight
        let m7 = (-0.5 * side + 0.5 * imageHeight) / imageHeight
        return (x * m0 + m3, y * m5 + m7)
    }

    /// A malformed/short keypoints array (not a normal-operation case)
    /// clamps to the nearest valid index -- or (0,0) if there are none at
    /// all -- rather than trapping.
    private static func clampedKeypoint(_ keypoints: [(x: Float, y: Float)], at index: Int) -> (x: Float, y: Float)
    {
        guard keypoints.isEmpty == false else { return (0, 0) }
        return keypoints[min(max(index, 0), keypoints.count - 1)]
    }

    public struct ProjectedDetection
    {
        public var xmin: Float
        public var ymin: Float
        public var width: Float
        public var height: Float
        public var keypoints: [(x: Float, y: Float)]
        public var score: Float

        public init(xmin: Float, ymin: Float, width: Float, height: Float, keypoints: [(x: Float, y: Float)], score: Float)
        {
            self.xmin = xmin
            self.ymin = ymin
            self.width = width
            self.height = height
            self.keypoints = keypoints
            self.score = score
        }
    }

    public static func project(_ detection: MediaPipeSSDDetectorDecoder.Detection, imageWidth: Float, imageHeight: Float) -> ProjectedDetection
    {
        let corners = [
            (detection.xmin, detection.ymin),
            (detection.xmin + detection.width, detection.ymin),
            (detection.xmin + detection.width, detection.ymin + detection.height),
            (detection.xmin, detection.ymin + detection.height),
        ]
        let projectedCorners = corners.map { projectLetterbox(x: $0.0, y: $0.1, imageWidth: imageWidth, imageHeight: imageHeight) }
        let xmin = projectedCorners.map(\.x).min()!
        let ymin = projectedCorners.map(\.y).min()!
        let xmax = projectedCorners.map(\.x).max()!
        let ymax = projectedCorners.map(\.y).max()!

        let projectedKeypoints = detection.keypoints.map { projectLetterbox(x: $0.x, y: $0.y, imageWidth: imageWidth, imageHeight: imageHeight) }

        return ProjectedDetection(xmin: xmin, ymin: ymin, width: xmax - xmin, height: ymax - ymin, keypoints: projectedKeypoints, score: detection.score)
    }

    /// (cx, cy, width, height) normalized full-image, top-left origin, plus
    /// rotation in radians (MediaPipeSSDDetectorDecoder.computeRotation's
    /// own convention). Fed directly to MediaPipeCropPreprocessor for the
    /// rotated crop. `rotationKeypoints` indexes into `detection.keypoints`
    /// (BlazePalm: (0,2) wrist->middleMCP; BlazeFace: (0,1) leftEye->
    /// rightEye). `rectShiftX`/`rectShiftY` default to 0 (BlazeFace has no
    /// shift; BlazePalm passes shiftY=-0.5, shiftX=0).
    public static func rect(
        from detection: ProjectedDetection, imageWidth: Float, imageHeight: Float,
        rotationKeypoints: (start: Int, end: Int), targetAngleRadians: Float,
        rectScale: Float, rectShiftX: Float = 0, rectShiftY: Float = 0
    ) -> (cx: Float, cy: Float, width: Float, height: Float, rotation: Float)
    {
        let cx = detection.xmin + detection.width / 2
        let cy = detection.ymin + detection.height / 2

        let startKeypoint = Self.clampedKeypoint(detection.keypoints, at: rotationKeypoints.start)
        let endKeypoint = Self.clampedKeypoint(detection.keypoints, at: rotationKeypoints.end)
        let startPoint = (x: startKeypoint.x * imageWidth, y: startKeypoint.y * imageHeight)
        let endPoint = (x: endKeypoint.x * imageWidth, y: endKeypoint.y * imageHeight)
        let rotation = MediaPipeSSDDetectorDecoder.computeRotation(from: startPoint, to: endPoint, targetAngleRadians: targetAngleRadians)

        return Self.finalizeRect(
            cx: cx, cy: cy, width: detection.width, height: detection.height, rotation: rotation,
            imageWidth: imageWidth, imageHeight: imageHeight,
            rectScale: rectScale, rectShiftX: rectShiftX, rectShiftY: rectShiftY
        )
    }

    /// RectTransformationCalculator's own tail, shared by every ROI-
    /// derivation function in this file. `width`/`height` in are
    /// normalized full-image, pre-shift/pre-scale; out are post shift,
    /// square_long, and scale.
    private static func finalizeRect(
        cx: Float, cy: Float, width: Float, height: Float, rotation: Float,
        imageWidth: Float, imageHeight: Float,
        rectScale: Float, rectShiftX: Float, rectShiftY: Float
    ) -> (cx: Float, cy: Float, width: Float, height: Float, rotation: Float)
    {
        var cx = cx
        var cy = cy

        let sinA = sin(rotation)
        let cosA = cos(rotation)
        if rotation == 0
        {
            cx += width * rectShiftX
            cy += height * rectShiftY
        }
        else
        {
            let xShift = (imageWidth * width * rectShiftX * cosA - imageHeight * height * rectShiftY * sinA) / imageWidth
            let yShift = (imageWidth * width * rectShiftX * sinA + imageHeight * height * rectShiftY * cosA) / imageHeight
            cx += xShift
            cy += yShift
        }

        let longSide = max(width * imageWidth, height * imageHeight)
        let rectWidth = longSide / imageWidth * rectScale
        let rectHeight = longSide / imageHeight * rectScale

        return (cx: cx, cy: cy, width: rectWidth, height: rectHeight, rotation: rotation)
    }

    /// Ports AlignmentPointsRectsCalculator: unlike `rect()` (box-based),
    /// size and center come directly from the two rotation keypoints --
    /// center = keypoint[start] in pixel space, box size = 2x the pixel
    /// distance between the two keypoints. `detection.xmin/width/height`
    /// are unused. `rectShiftX`/`rectShiftY` default to 0.
    public static func alignmentPointsRect(
        from detection: ProjectedDetection, imageWidth: Float, imageHeight: Float,
        rotationKeypoints: (start: Int, end: Int), targetAngleRadians: Float,
        rectScale: Float, rectShiftX: Float = 0, rectShiftY: Float = 0
    ) -> (cx: Float, cy: Float, width: Float, height: Float, rotation: Float)
    {
        let startKeypoint = Self.clampedKeypoint(detection.keypoints, at: rotationKeypoints.start)
        let endKeypoint = Self.clampedKeypoint(detection.keypoints, at: rotationKeypoints.end)
        let centerPoint = (x: startKeypoint.x * imageWidth, y: startKeypoint.y * imageHeight)
        let scalePoint = (x: endKeypoint.x * imageWidth, y: endKeypoint.y * imageHeight)

        let dx = scalePoint.x - centerPoint.x
        let dy = scalePoint.y - centerPoint.y
        let boxSizePixels = 2 * sqrt(dx * dx + dy * dy)

        let cx = centerPoint.x / imageWidth
        let cy = centerPoint.y / imageHeight
        let width = boxSizePixels / imageWidth
        let height = boxSizePixels / imageHeight

        let rotation = MediaPipeSSDDetectorDecoder.computeRotation(from: centerPoint, to: scalePoint, targetAngleRadians: targetAngleRadians)

        return Self.finalizeRect(
            cx: cx, cy: cy, width: width, height: height, rotation: rotation,
            imageWidth: imageWidth, imageHeight: imageHeight,
            rectScale: rectScale, rectShiftX: rectShiftX, rectShiftY: rectShiftY
        )
    }

    /// Ports HandLandmarksToRectCalculator: rotation comes from the wrist
    /// and a weighted average of three finger MCP joints (the source's own
    /// constant names say "PIP", but they resolve to MCP joints); center/
    /// size come from an oriented bounding box over a fixed 12-point
    /// subset (excludes fingertips so extended fingers don't blow out the
    /// box). `points` must be exactly this subset, normalized full-image
    /// top-left-origin, in order: [wrist, thumbCMC, thumbMCP, thumbIP,
    /// indexMCP, indexPIP, middleMCP, middlePIP, ringMCP, ringPIP,
    /// pinkyMCP, pinkyPIP] -- the 21-point indices
    /// [0,1,2,3,5,6,9,10,13,14,17,18], the caller's responsibility to slice.
    public static func handLandmarksRect(
        points: [(x: Float, y: Float)], imageWidth: Float, imageHeight: Float,
        targetAngleRadians: Float, rectScale: Float, rectShiftX: Float = 0, rectShiftY: Float = 0
    ) -> (cx: Float, cy: Float, width: Float, height: Float, rotation: Float)?
    {
        guard points.count == 12 else { return nil }

        let wrist = (x: points[0].x * imageWidth, y: points[0].y * imageHeight)
        // (indexMCP + ringMCP) / 2, then averaged with middleMCP -- middle
        // gets 2x the weight of index/ring.
        let indexMCP = points[4], middleMCP = points[6], ringMCP = points[8]
        let averageX = (((indexMCP.x + ringMCP.x) / 2) + middleMCP.x) / 2 * imageWidth
        let averageY = (((indexMCP.y + ringMCP.y) / 2) + middleMCP.y) / 2 * imageHeight

        let rotation = MediaPipeSSDDetectorDecoder.computeRotation(
            from: wrist, to: (x: averageX, y: averageY), targetAngleRadians: targetAngleRadians
        )

        // Axis-aligned center of the un-rotated points, normalized space.
        let xs = points.map(\.x), ys = points.map(\.y)
        let axisAlignedCenterX = (xs.max()! + xs.min()!) / 2
        let axisAlignedCenterY = (ys.max()! + ys.min()!) / 2

        // Project each point into the candidate orientation (reverse_angle
        // = -rotation) around that center; the axis-aligned extent there
        // is the oriented box's width/height.
        let reverseAngle = MediaPipeSSDDetectorDecoder.normalizeRadians(-rotation)
        let cosR = cos(reverseAngle), sinR = sin(reverseAngle)

        var projected: [(x: Float, y: Float)] = []
        projected.reserveCapacity(points.count)
        for point in points
        {
            let originalX = (point.x - axisAlignedCenterX) * imageWidth
            let originalY = (point.y - axisAlignedCenterY) * imageHeight
            projected.append((
                x: originalX * cosR - originalY * sinR,
                y: originalX * sinR + originalY * cosR
            ))
        }

        let projectedXs = projected.map(\.x), projectedYs = projected.map(\.y)
        let minX = projectedXs.min()!, maxX = projectedXs.max()!
        let minY = projectedYs.min()!, maxY = projectedYs.max()!
        let projectedCenterX = (maxX + minX) / 2
        let projectedCenterY = (maxY + minY) / 2

        // Rotate back by +rotation, not -reverseAngle -- the two are
        // independently normalized, not guaranteed exact negatives after
        // wraparound.
        let cosF = cos(rotation), sinF = sin(rotation)
        let centerXPixels = projectedCenterX * cosF - projectedCenterY * sinF + imageWidth * axisAlignedCenterX
        let centerYPixels = projectedCenterX * sinF + projectedCenterY * cosF + imageHeight * axisAlignedCenterY

        let cx = centerXPixels / imageWidth
        let cy = centerYPixels / imageHeight
        let width = (maxX - minX) / imageWidth
        let height = (maxY - minY) / imageHeight

        return Self.finalizeRect(
            cx: cx, cy: cy, width: width, height: height, rotation: rotation,
            imageWidth: imageWidth, imageHeight: imageHeight,
            rectScale: rectScale, rectShiftX: rectShiftX, rectShiftY: rectShiftY
        )
    }
}
