import Foundation

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
}

final class FeatureExtractor {

    static func angle(_ a: MPPoseLandmark, _ b: MPPoseLandmark, _ c: MPPoseLandmark) -> Float {
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

    static func landmarksToFeatures(_ lms: [MPPoseLandmark]) -> [Float] {
        guard lms.count >= 33 else { return [] }

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

        var features: [Float] = []

        for lm in lms.prefix(33) {
            features.append((lm.x - hipCenterX) / torsoSize)
            features.append((lm.y - hipCenterY) / torsoSize)
            features.append((lm.z - hipCenterZ) / torsoSize)
            features.append(lm.visibility)
        }

        let leftElbow = angle(lms[MPIndex.leftShoulder], lms[MPIndex.leftElbow], lms[MPIndex.leftWrist])
        let rightElbow = angle(lms[MPIndex.rightShoulder], lms[MPIndex.rightElbow], lms[MPIndex.rightWrist])

        let leftShoulderAngle = angle(lms[MPIndex.leftElbow], lms[MPIndex.leftShoulder], lms[MPIndex.leftHip])
        let rightShoulderAngle = angle(lms[MPIndex.rightElbow], lms[MPIndex.rightShoulder], lms[MPIndex.rightHip])

        let leftKnee = angle(lms[MPIndex.leftHip], lms[MPIndex.leftKnee], lms[MPIndex.leftAnkle])
        let rightKnee = angle(lms[MPIndex.rightHip], lms[MPIndex.rightKnee], lms[MPIndex.rightAnkle])

        let leftHipAngle = angle(lms[MPIndex.leftShoulder], lms[MPIndex.leftHip], lms[MPIndex.leftKnee])
        let rightHipAngle = angle(lms[MPIndex.rightShoulder], lms[MPIndex.rightHip], lms[MPIndex.rightKnee])

        let torsoVecX = shoulderCenterX - hipCenterX
        let torsoVecY = shoulderCenterY - hipCenterY
        let verticalX: Float = 0
        let verticalY: Float = -1

        let denom = sqrt(torsoVecX * torsoVecX + torsoVecY * torsoVecY)

        var torsoAngle: Float = 0
        if denom > 0 {
            let cosine = max(-1, min(1, (torsoVecX * verticalX + torsoVecY * verticalY) / denom))
            torsoAngle = acos(cosine) * 180 / Float.pi
        }

        let shoulderWidth = distance(leftShoulder, rightShoulder)
        let hipWidth = distance(leftHip, rightHip)
        let bodyHeight = sqrt(
            pow(shoulderCenterX - hipCenterX, 2) +
            pow(shoulderCenterY - hipCenterY, 2) +
            pow(shoulderCenterZ - hipCenterZ, 2)
        )

        features.append(leftElbow / 180)
        features.append(rightElbow / 180)
        features.append(leftShoulderAngle / 180)
        features.append(rightShoulderAngle / 180)
        features.append(leftKnee / 180)
        features.append(rightKnee / 180)
        features.append(leftHipAngle / 180)
        features.append(rightHipAngle / 180)
        features.append(torsoAngle / 180)
        features.append(shoulderWidth)
        features.append(hipWidth)
        features.append(bodyHeight)

        return features
    }

    static func distance(_ a: MPPoseLandmark, _ b: MPPoseLandmark) -> Float {
        return sqrt(
            pow(a.x - b.x, 2) +
            pow(a.y - b.y, 2) +
            pow(a.z - b.z, 2)
        )
    }

    static func countingAngle(landmarks lms: [MPPoseLandmark], exercise: String) -> Float? {
        guard lms.count >= 33 else { return nil }

        let leftElbow = angle(lms[MPIndex.leftShoulder], lms[MPIndex.leftElbow], lms[MPIndex.leftWrist])
        let rightElbow = angle(lms[MPIndex.rightShoulder], lms[MPIndex.rightElbow], lms[MPIndex.rightWrist])

        let leftKnee = angle(lms[MPIndex.leftHip], lms[MPIndex.leftKnee], lms[MPIndex.leftAnkle])
        let rightKnee = angle(lms[MPIndex.rightHip], lms[MPIndex.rightKnee], lms[MPIndex.rightAnkle])

        let leftHip = angle(lms[MPIndex.leftShoulder], lms[MPIndex.leftHip], lms[MPIndex.leftKnee])
        let rightHip = angle(lms[MPIndex.rightShoulder], lms[MPIndex.rightHip], lms[MPIndex.rightKnee])

        let leftShoulder = angle(lms[MPIndex.leftElbow], lms[MPIndex.leftShoulder], lms[MPIndex.leftHip])
        let rightShoulder = angle(lms[MPIndex.rightElbow], lms[MPIndex.rightShoulder], lms[MPIndex.rightHip])

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

        if ["deadlift", "romanian deadlift", "hip thrust", "leg raises"].contains(exercise) {
            return (leftHip + rightHip) / 2
        }

        if ["lateral raise", "chest fly machine"].contains(exercise) {
            return (leftShoulder + rightShoulder) / 2
        }

        return nil
    }
}
