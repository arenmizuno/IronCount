import Foundation

struct LandmarkVelocityFrame {
    static var previous: [Float]? = nil
}

enum MPIndex {
    static let leftShoulder = 11
    static let rightShoulder = 12
    static let leftElbow = 13
    static let rightElbow = 14
    static let leftWrist = 15
    static let rightWrist = 16
    static let leftHip = 23
    static let rightHip = 24
    static let leftKnee = 25
    static let rightKnee = 26
    static let leftAnkle = 27
    static let rightAnkle = 28
    static let leftFoot = 31
    static let rightFoot = 32
}

final class FeatureExtractor {

    // ============================================================
    // Distance
    // ============================================================
    static func distance(_ a: MPPoseLandmark,
                         _ b: MPPoseLandmark) -> Float {

        return sqrt(
            pow(a.x - b.x, 2) +
            pow(a.y - b.y, 2) +
            pow(a.z - b.z, 2)
        )
    }

    // ============================================================
    // Angle
    // ============================================================
    static func angle(_ a: MPPoseLandmark,
                      _ b: MPPoseLandmark,
                      _ c: MPPoseLandmark) -> Float {

        let baX = a.x - b.x
        let baY = a.y - b.y
        let baZ = a.z - b.z

        let bcX = c.x - b.x
        let bcY = c.y - b.y
        let bcZ = c.z - b.z

        let dot = baX * bcX + baY * bcY + baZ * bcZ

        let normBA = sqrt(baX * baX + baY * baY + baZ * baZ)
        let normBC = sqrt(bcX * bcX + bcY * bcY + bcZ * bcZ)

        if normBA == 0 || normBC == 0 {
            return 0
        }

        let cosine = max(-1, min(1, dot / (normBA * normBC)))

        return acos(cosine) * 180 / Float.pi
    }

    // ============================================================
    // Main feature extraction
    // Output: 300 features
    // ============================================================
    static func landmarksToFeatures(_ lms: [MPPoseLandmark]) -> [Float] {

        guard lms.count >= 33 else {
            return []
        }

        let leftHip = lms[MPIndex.leftHip]
        let rightHip = lms[MPIndex.rightHip]

        let leftShoulder = lms[MPIndex.leftShoulder]
        let rightShoulder = lms[MPIndex.rightShoulder]

        let hipCenterX = (leftHip.x + rightHip.x) / 2
        let hipCenterY = (leftHip.y + rightHip.y) / 2
        let hipCenterZ = (leftHip.z + rightHip.z) / 2

        let shoulderCenterX = (leftShoulder.x + rightShoulder.x) / 2
        let shoulderCenterY = (leftShoulder.y + rightShoulder.y) / 2
        let shoulderCenterZ = (leftShoulder.z + rightShoulder.z) / 2

        var torsoSize = sqrt(
            pow(shoulderCenterX - hipCenterX, 2) +
            pow(shoulderCenterY - hipCenterY, 2) +
            pow(shoulderCenterZ - hipCenterZ, 2)
        )

        if torsoSize < 0.000001 {
            torsoSize = 1.0
        }

        // ============================================================
        // 1. Landmark features
        // 33 * 4 = 132
        // ============================================================

        var features: [Float] = []

        for lm in lms.prefix(33) {

            let nx = (lm.x - hipCenterX) / torsoSize
            let ny = (lm.y - hipCenterY) / torsoSize
            let nz = (lm.z - hipCenterZ) / torsoSize

            features.append(nx)
            features.append(ny)
            features.append(nz)
            features.append(lm.visibility)
        }

        // ============================================================
        // 2. Angle features
        // ============================================================

        let leftElbow = angle(
            lms[MPIndex.leftShoulder],
            lms[MPIndex.leftElbow],
            lms[MPIndex.leftWrist]
        )

        let rightElbow = angle(
            lms[MPIndex.rightShoulder],
            lms[MPIndex.rightElbow],
            lms[MPIndex.rightWrist]
        )

        let leftShoulderAngle = angle(
            lms[MPIndex.leftElbow],
            lms[MPIndex.leftShoulder],
            lms[MPIndex.leftHip]
        )

        let rightShoulderAngle = angle(
            lms[MPIndex.rightElbow],
            lms[MPIndex.rightShoulder],
            lms[MPIndex.rightHip]
        )

        let leftKnee = angle(
            lms[MPIndex.leftHip],
            lms[MPIndex.leftKnee],
            lms[MPIndex.leftAnkle]
        )

        let rightKnee = angle(
            lms[MPIndex.rightHip],
            lms[MPIndex.rightKnee],
            lms[MPIndex.rightAnkle]
        )

        let leftHipAngle = angle(
            lms[MPIndex.leftShoulder],
            lms[MPIndex.leftHip],
            lms[MPIndex.leftKnee]
        )

        let rightHipAngle = angle(
            lms[MPIndex.rightShoulder],
            lms[MPIndex.rightHip],
            lms[MPIndex.rightKnee]
        )

        let leftAnkle = angle(
            lms[MPIndex.leftKnee],
            lms[MPIndex.leftAnkle],
            lms[MPIndex.leftFoot]
        )

        let rightAnkle = angle(
            lms[MPIndex.rightKnee],
            lms[MPIndex.rightAnkle],
            lms[MPIndex.rightFoot]
        )

        let torsoVecX = shoulderCenterX - hipCenterX
        let torsoVecY = shoulderCenterY - hipCenterY

        let denom = sqrt(torsoVecX * torsoVecX + torsoVecY * torsoVecY)

        var torsoAngle: Float = 0

        if denom > 0 {
            let cosine = max(-1, min(1, (-torsoVecY) / denom))
            torsoAngle = acos(cosine) * 180 / Float.pi
        }

        let angleFeatures: [Float] = [
            leftElbow / 180,
            rightElbow / 180,
            leftShoulderAngle / 180,
            rightShoulderAngle / 180,
            leftKnee / 180,
            rightKnee / 180,
            leftHipAngle / 180,
            rightHipAngle / 180,
            leftAnkle / 180,
            rightAnkle / 180,
            torsoAngle / 180
        ]

        features.append(contentsOf: angleFeatures)

        // ============================================================
        // 3. Geometry features
        // ============================================================

        let shoulderWidth = distance(leftShoulder, rightShoulder)
        let hipWidth = distance(leftHip, rightHip)

        let bodyHeight = sqrt(
            pow(shoulderCenterX - hipCenterX, 2) +
            pow(shoulderCenterY - hipCenterY, 2) +
            pow(shoulderCenterZ - hipCenterZ, 2)
        )

        let wristYMean =
            (lms[MPIndex.leftWrist].y +
             lms[MPIndex.rightWrist].y) / 2

        let shoulderYMean =
            (lms[MPIndex.leftShoulder].y +
             lms[MPIndex.rightShoulder].y) / 2

        let hipYMean =
            (lms[MPIndex.leftHip].y +
             lms[MPIndex.rightHip].y) / 2

        let kneeYMean =
            (lms[MPIndex.leftKnee].y +
             lms[MPIndex.rightKnee].y) / 2

        let wristVsShoulder = wristYMean - shoulderYMean
        let wristVsHip = wristYMean - hipYMean
        let kneeVsHip = kneeYMean - hipYMean

        let wristDistance =
            abs(lms[MPIndex.leftWrist].x -
                lms[MPIndex.rightWrist].x)

        let geometryFeatures: [Float] = [
            shoulderWidth,
            hipWidth,
            bodyHeight,
            wristVsShoulder,
            wristVsHip,
            kneeVsHip,
            wristDistance
        ]

        features.append(contentsOf: geometryFeatures)

        // ============================================================
        // Base feature count should now be 150
        // ============================================================

        while features.count < 150 {
            features.append(0)
        }

        // ============================================================
        // 4. Velocity features
        // ============================================================

        let currentBase = Array(features.prefix(150))

        var velocityFeatures: [Float] = []

        if let previous = LandmarkVelocityFrame.previous {

            for i in 0..<150 {
                velocityFeatures.append(currentBase[i] - previous[i])
            }

        } else {

            velocityFeatures = Array(repeating: 0, count: 150)
        }

        LandmarkVelocityFrame.previous = currentBase

        features.append(contentsOf: velocityFeatures)

        // ============================================================
        // Final feature count = 300
        // ============================================================

        while features.count < 300 {
            features.append(0)
        }

        if features.count > 300 {
            features = Array(features.prefix(300))
        }

        return features
    }

    // ============================================================
    // Rep counting angle
    // ============================================================

    static func countingAngle(landmarks lms: [MPPoseLandmark],
                              exercise: String) -> Float? {

        guard lms.count >= 33 else {
            return nil
        }

        let leftElbow = angle(
            lms[MPIndex.leftShoulder],
            lms[MPIndex.leftElbow],
            lms[MPIndex.leftWrist]
        )

        let rightElbow = angle(
            lms[MPIndex.rightShoulder],
            lms[MPIndex.rightElbow],
            lms[MPIndex.rightWrist]
        )

        let leftKnee = angle(
            lms[MPIndex.leftHip],
            lms[MPIndex.leftKnee],
            lms[MPIndex.leftAnkle]
        )

        let rightKnee = angle(
            lms[MPIndex.rightHip],
            lms[MPIndex.rightKnee],
            lms[MPIndex.rightAnkle]
        )

        let leftHipAngle = angle(
            lms[MPIndex.leftShoulder],
            lms[MPIndex.leftHip],
            lms[MPIndex.leftKnee]
        )

        let rightHipAngle = angle(
            lms[MPIndex.rightShoulder],
            lms[MPIndex.rightHip],
            lms[MPIndex.rightKnee]
        )

        if [
            "barbell biceps curl",
            "hammer curl",
            "push-up",
            "bench press",
            "incline bench press",
            "decline bench press",
            "tricep Pushdown",
            "tricep dips",
            "shoulder press",
            "lat pulldown",
            "pull Up",
            "t bar row"
        ].contains(exercise) {

            return (leftElbow + rightElbow) / 2
        }

        if ["squat", "leg extension"].contains(exercise) {
            return (leftKnee + rightKnee) / 2
        }

        if [
            "deadlift",
            "romanian deadlift",
            "hip thrust",
            "leg raises"
        ].contains(exercise) {

            return (leftHipAngle + rightHipAngle) / 2
        }

        return nil
    }
}
