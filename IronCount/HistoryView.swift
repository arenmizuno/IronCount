import SwiftUI

struct HistoryView: View {
    @State private var workouts: [WorkoutRecord] = []

    var body: some View {
        List {
            if workouts.isEmpty {
                Text("No saved workouts yet")
                    .foregroundColor(.secondary)
            } else {
                ForEach(workouts) { workout in
                    VStack(alignment: .leading, spacing: 6) {
                        Text(workout.date.formatted(date: .abbreviated, time: .shortened))
                            .bold()

                        Text("Duration: \(formattedTime(workout.durationSeconds))")
                            .foregroundColor(.secondary)

                        ForEach(workout.repsByExercise.keys.sorted(), id: \.self) { exercise in
                            Text("\(exercise): \(workout.repsByExercise[exercise] ?? 0) reps")
                                .font(.subheadline)
                        }
                    }
                    .padding(.vertical, 8)
                }
                .onDelete(perform: delete)
            }
        }
        .navigationTitle("History")
        .listStyle(.insetGrouped)
        .onAppear(perform: loadWorkouts)
    }

    private func loadWorkouts() {
        guard let data = UserDefaults.standard.data(forKey: "saved_workouts") else { return }
        if let decoded = try? JSONDecoder().decode([WorkoutRecord].self, from: data) {
            workouts = decoded
        }
    }

    private func delete(at offsets: IndexSet) {
        workouts.remove(atOffsets: offsets)
        if let data = try? JSONEncoder().encode(workouts) {
            UserDefaults.standard.set(data, forKey: "saved_workouts")
        }
    }

    private func formattedTime(_ seconds: Int) -> String {
        let minutes = seconds / 60
        let secs = seconds % 60
        return String(format: "%02d:%02d", minutes, secs)
    }
}

struct HistoryView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationStack {
            HistoryView()
        }
    }
}
