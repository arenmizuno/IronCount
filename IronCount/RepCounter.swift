//
//  RepCounter.swift
//  IronCount
//
//  Created by Aren Mizuno on 5/5/26.
//

import Foundation

class RepCounter {
    private var state = "start"
    private(set) var count = 0
    private let exercise: String

    init(exercise: String) {
        self.exercise = exercise
    }

    func update(angle: Float) -> Int {

        if exercise.contains("curl") {
            if angle > 150 {
                state = "down"
            }
            if state == "down" && angle < 70 {
                count += 1
                state = "up"
            }
        }

        else if exercise.contains("push") {
            if angle > 150 {
                state = "top"
            }
            if state == "top" && angle < 90 {
                count += 1
                state = "bottom"
            }
        }

        else if exercise.contains("squat") {
            if angle > 160 {
                state = "top"
            }
            if state == "top" && angle < 100 {
                count += 1
                state = "bottom"
            }
        }

        return count
    }
}
