import Foundation

struct WorkoutRecord: Identifiable, Codable {
    let id: UUID
    let date: Date
    let durationSeconds: Int
    let repsByExercise: [String: Int]
}
