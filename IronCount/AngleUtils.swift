//
//  AngleUtils.swift
//  IronCount
//
//  Created by Aren Mizuno on 5/5/26.
//

import Foundation

struct Keypoint {
    let x: Float
    let y: Float
    let confidence: Float
}

class AngleUtils {

    static func calculateAngle(_ a: Keypoint, _ b: Keypoint, _ c: Keypoint) -> Float {
        let ab = (a.x - b.x, a.y - b.y)
        let cb = (c.x - b.x, c.y - b.y)

        let dot = ab.0 * cb.0 + ab.1 * cb.1
        let mag1 = sqrt(ab.0 * ab.0 + ab.1 * ab.1)
        let mag2 = sqrt(cb.0 * cb.0 + cb.1 * cb.1)

        if mag1 * mag2 == 0 { return 0 }

        let cosine = max(-1, min(1, dot / (mag1 * mag2)))
        return acos(cosine) * 180 / .pi
    }

    static func keypointsToFeatures(_ kpts: [Keypoint]) -> [Float] {
        guard kpts.count == 17 else { return [] }

        var features: [Float] = []

        // normalized XY (simple)
        for kp in kpts {
            features.append(kp.x)
            features.append(kp.y)
        }

        // angles
        let leftElbow = calculateAngle(kpts[5], kpts[7], kpts[9])
        let rightElbow = calculateAngle(kpts[6], kpts[8], kpts[10])
        let leftKnee = calculateAngle(kpts[11], kpts[13], kpts[15])
        let rightKnee = calculateAngle(kpts[12], kpts[14], kpts[16])

        features.append(leftElbow / 180)
        features.append(rightElbow / 180)
        features.append(leftKnee / 180)
        features.append(rightKnee / 180)

        return features
    }

    static func countingAngle(_ kpts: [Keypoint], exercise: String) -> Float? {
        let leftElbow = calculateAngle(kpts[5], kpts[7], kpts[9])
        let rightElbow = calculateAngle(kpts[6], kpts[8], kpts[10])
        let leftKnee = calculateAngle(kpts[11], kpts[13], kpts[15])
        let rightKnee = calculateAngle(kpts[12], kpts[14], kpts[16])

        if exercise.contains("curl") || exercise.contains("press") || exercise.contains("push") {
            return (leftElbow + rightElbow) / 2
        }

        if exercise.contains("squat") || exercise.contains("leg") {
            return (leftKnee + rightKnee) / 2
        }

        return nil
    }
}
