import Foundation

final class RepCounter {
    private var state = "start"
    var count = 0
    private let exercise: String

    init(exercise: String) {
        self.exercise = exercise
    }

    func update(angle: Float) -> Int {
        if ["barbell biceps curl", "hammer curl"].contains(exercise) {
            if angle > 150 { state = "down" }
            if state == "down" && angle < 70 {
                count += 1
                state = "up"
            }
        }

        else if exercise == "push-up" {
            if angle > 125 { state = "top" }
            if state == "top" && angle < 90 {
                count += 1
                state = "bottom"
            }
        }

        else if exercise == "squat" {
            if angle > 160 { state = "top" }
            if state == "top" && angle < 100 {
                count += 1
                state = "bottom"
            }
        }

        else if ["bench press", "incline bench press", "decline bench press"].contains(exercise) {
            if angle > 150 { state = "top" }
            if state == "top" && angle < 100 {
                count += 1
                state = "bottom"
            }
        }

        else if exercise == "leg extension" {
            if angle < 100 { state = "bent" }
            if state == "bent" && angle > 155 {
                count += 1
                state = "extended"
            }
        }

        else if ["lat pulldown", "pull Up", "t bar row"].contains(exercise) {
            if angle > 150 { state = "open" }
            if state == "open" && angle < 90 {
                count += 1
                state = "closed"
            }
        }

        else if ["tricep Pushdown", "tricep dips"].contains(exercise) {
            if angle < 100 { state = "bent" }
            if state == "bent" && angle > 155 {
                count += 1
                state = "extended"
            }
        }

        else if exercise == "shoulder press" {
            if angle < 100 { state = "bent" }
            if state == "bent" && angle > 150 {
                count += 1
                state = "extended"
            }
        }

        else if ["deadlift", "romanian deadlift", "hip thrust"].contains(exercise) {
            if angle < 120 { state = "folded" }
            if state == "folded" && angle > 165 {
                count += 1
                state = "extended"
            }
        }

        else if exercise == "leg raises" {
            if angle > 160 { state = "down" }
            if state == "down" && angle < 100 {
                count += 1
                state = "up"
            }
        }

        else if ["lateral raise", "chest fly machine"].contains(exercise) {
            if angle < 40 { state = "down" }
            if state == "down" && angle > 85 {
                count += 1
                state = "up"
            }
        }

        return count
    }

    func reset() {
        count = 0
        state = "start"
    }
}
