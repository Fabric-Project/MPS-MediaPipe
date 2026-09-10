//
//  MediaPipeSSDRectTransform.swift
//  Fabric
//

import Foundation

/// Projects a "Blaze"-family SSD detection from the detector's letterboxed
/// tensor space into full-image space, then derives the rotated square ROI
/// fed to the corresponding landmark model (DetectionsToRectsCalculator +
/// RectTransformationCalculator). Shared by BlazePalm (hand) and BlazeFace,
/// which use this exact calculator pair with different rotation keypoints/
/// target angle/scale/shift.
///
/// BlazePalm: rotation keypoints (wrist=0, middleMCP=2), target 90 (a real
/// MediaPipe proto quirk — see MediaPipeSSDDetectorDecoder.computeRotation's
/// doc comment), scale 2.6, shift_y -0.5. Ported from and validated against
/// fasthands.pipeline.letterbox_projection/project_detection/
/// HandLandmarker._detect_rects (a validated third-party port) to float32
/// precision on the real bundled MediaPipeHandDetector model's output on a
/// real image.
///
/// BlazeFace: rotation keypoints (leftEye=0, rightEye=1), target 0, scale
/// 1.5 (both axes), no shift — confirmed directly against
/// mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt.
/// Unlike BlazePalm, this has no local third-party reference to check
/// against — derived from primary source and sanity-checked with hand-
/// computed cases (horizontal eyes -> 0 rotation, vertical eyes -> 90°),
/// not validated end-to-end against a real detected face.
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

        let startPoint = (x: detection.keypoints[rotationKeypoints.start].x * imageWidth, y: detection.keypoints[rotationKeypoints.start].y * imageHeight)
        let endPoint = (x: detection.keypoints[rotationKeypoints.end].x * imageWidth, y: detection.keypoints[rotationKeypoints.end].y * imageHeight)
        let rotation = MediaPipeSSDDetectorDecoder.computeRotation(from: startPoint, to: endPoint, targetAngleRadians: targetAngleRadians)

        return Self.finalizeRect(
            cx: cx, cy: cy, width: detection.width, height: detection.height, rotation: rotation,
            imageWidth: imageWidth, imageHeight: imageHeight,
            rectScale: rectScale, rectShiftX: rectShiftX, rectShiftY: rectShiftY
        )
    }

    /// RectTransformationCalculator's own tail, shared by every ROI-
    /// derivation function in this file regardless of how center/size/
    /// rotation were computed (box-based, alignment-point-based, or
    /// oriented-bounding-box-based) -- this behavior belongs to
    /// RectTransformationCalculator itself, not to whichever calculator fed
    /// it. `width`/`height` in are normalized full-image, top-left origin,
    /// pre-shift/pre-scale; same on the way out (post shift, square_long,
    /// and scale).
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

    /// (cx, cy, width, height) normalized full-image, top-left origin, plus
    /// rotation in radians — ports mediapipe/calculators/util/
    /// alignment_points_to_rects_calculator.cc's AlignmentPointsRectsCalculator
    /// exactly: unlike `rect(...)` above (box-based, used by BlazeFace/
    /// BlazePalm), size and center come directly from the two rotation
    /// keypoints themselves, not the SSD detection box — BlazePose's
    /// detector uses this because the two alignment keypoints (hip center,
    /// a body-size/rotation reference point) are the meaningful geometry,
    /// not the anchor-regressed box. `detection.xmin/width/height` are
    /// unused here. Confirmed against the real calculator source: center =
    /// keypoint[start] in pixel space, box size = 2x the pixel distance
    /// between the two keypoints (a square in pixel space, though not
    /// necessarily in normalized space when imageWidth != imageHeight —
    /// matches `rect->set_width(box_size / image_width)`, `set_height
    /// (box_size / image_height)` precisely). The final rectScale/
    /// square_long finalization (`longSide`/`rectScale` below) is shared,
    /// unmodified logic from `rect(...)` above — that behavior belongs to
    /// RectTransformationCalculator, not to which center/size calculator
    /// fed it, and BlazePose's own config also sets `square_long: true`.
    /// `rectShiftX`/`rectShiftY` default to 0 (BlazePose's own config has
    /// no shift).
    public static func alignmentPointsRect(
        from detection: ProjectedDetection, imageWidth: Float, imageHeight: Float,
        rotationKeypoints: (start: Int, end: Int), targetAngleRadians: Float,
        rectScale: Float, rectShiftX: Float = 0, rectShiftY: Float = 0
    ) -> (cx: Float, cy: Float, width: Float, height: Float, rotation: Float)
    {
        let centerPoint = (x: detection.keypoints[rotationKeypoints.start].x * imageWidth, y: detection.keypoints[rotationKeypoints.start].y * imageHeight)
        let scalePoint = (x: detection.keypoints[rotationKeypoints.end].x * imageWidth, y: detection.keypoints[rotationKeypoints.end].y * imageHeight)

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

    /// Ports mediapipe/modules/hand_landmark/calculators/
    /// hand_landmarks_to_rect_calculator.cc's HandLandmarksToRectCalculator
    /// exactly -- a third, distinct center/size/rotation algorithm (neither
    /// box-based like `rect()` nor two-point-distance-based like
    /// `alignmentPointsRect()`): rotation comes from the wrist and a
    /// *non-uniformly-weighted* average of three finger MCP joints (despite
    /// the real source's own misleading "PIP" constant names -- traced
    /// through its GetPartialLandmarks index indirection and confirmed
    /// these resolve to the MCP joints; the doc comment on this function's
    /// caller explains the trace), then width/height/center come from an
    /// oriented bounding box: every point in a fixed 12-point subset
    /// (wrist, thumb CMC/MCP/IP, and each of the other four fingers'
    /// MCP/PIP -- deliberately excluding fingertips/thumb-tip so extended
    /// fingers don't blow out the box) is rotated into the candidate
    /// orientation, axis-aligned min/max taken in that rotated frame, then
    /// the resulting center is rotated back. `points` must be exactly this
    /// 12-point subset, normalized full-image top-left-origin, in the
    /// calculator's own order: [wrist, thumbCMC, thumbMCP, thumbIP,
    /// indexMCP, indexPIP, middleMCP, middlePIP, ringMCP, ringPIP,
    /// pinkyMCP, pinkyPIP] -- i.e. Fabric's own 21-point indices
    /// [0,1,2,3,5,6,9,10,13,14,17,18], the caller's responsibility to slice.
    public static func handLandmarksRect(
        points: [(x: Float, y: Float)], imageWidth: Float, imageHeight: Float,
        targetAngleRadians: Float, rectScale: Float, rectShiftX: Float = 0, rectShiftY: Float = 0
    ) -> (cx: Float, cy: Float, width: Float, height: Float, rotation: Float)?
    {
        guard points.count == 12 else { return nil }

        let wrist = (x: points[0].x * imageWidth, y: points[0].y * imageHeight)
        // (indexMCP + ringMCP) / 2, then averaged again with middleMCP --
        // middle gets 2x the weight of index/ring, matching the real
        // calculator's own two-step averaging exactly (not a naive 1/3 each).
        let indexMCP = points[4], middleMCP = points[6], ringMCP = points[8]
        let averageX = (((indexMCP.x + ringMCP.x) / 2) + middleMCP.x) / 2 * imageWidth
        let averageY = (((indexMCP.y + ringMCP.y) / 2) + middleMCP.y) / 2 * imageHeight

        let rotation = MediaPipeSSDDetectorDecoder.computeRotation(
            from: wrist, to: (x: averageX, y: averageY), targetAngleRadians: targetAngleRadians
        )

        // Axis-aligned center of the *un-rotated* points, normalized space
        // (matches the real calculator's own `axis_aligned_center`).
        let xs = points.map(\.x), ys = points.map(\.y)
        let axisAlignedCenterX = (xs.max()! + xs.min()!) / 2
        let axisAlignedCenterY = (ys.max()! + ys.min()!) / 2

        // Project every point into the candidate orientation (reverse_angle
        // = -rotation) around that center, in pixel space, and take the
        // axis-aligned extent there -- that extent *is* the oriented box's
        // width/height in pixels.
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

        // Rotate the projected center back by +rotation and add back the
        // pixel-space axis-aligned center -- matches the real calculator's
        // own final center formula precisely (not the more common "rotate
        // by -reverseAngle" shortcut, since reverseAngle and rotation are
        // independently normalized, not guaranteed exact negatives of each
        // other after wraparound).
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
