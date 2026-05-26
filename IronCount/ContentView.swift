import SwiftUI

struct ContentView: View {
    @StateObject var camera = CameraManager()
    @State private var showExerciseSummary = false
    @State private var showSavedWorkouts = false

    var body: some View {
        ZStack {
            CameraPreview(session: camera.session)
                .ignoresSafeArea()

            VStack(spacing: 14) {
                Text(camera.exerciseLabel)
                    .font(.title3)
                    .foregroundColor(.white)
                    .padding(.top, 40)

                Text("Time: \(camera.formattedTime(camera.elapsedSeconds))")
                    .font(.title2)
                    .foregroundColor(.yellow)

                Text("Current Reps: \(camera.reps)")
                    .font(.title2)
                    .foregroundColor(.green)

                // ── ADDED: counter mode toggle ──────────────────────
                // Disabled while a workout is active to avoid mid-set
                // state corruption (the active counter would be replaced).
                Button {
                    camera.usePCACounter.toggle()
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: camera.usePCACounter
                              ? "waveform.path.ecg"
                              : "angle")
                        Text(camera.usePCACounter
                             ? "Counter: PCA"
                             : "Counter: Angle")
                            .font(.caption)
                            .fontWeight(.semibold)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(camera.usePCACounter
                                ? Color.purple.opacity(0.75)
                                : Color.orange.opacity(0.75))
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .disabled(camera.isWorkoutActive)
                .opacity(camera.isWorkoutActive ? 0.4 : 1.0)
                // ── END ADDED ───────────────────────────────────────

                DisclosureGroup("Exercise Summary", isExpanded: $showExerciseSummary) {
                    VStack(alignment: .leading, spacing: 8) {
                        if camera.exerciseRepMemory.isEmpty {
                            Text("No reps yet")
                                .foregroundColor(.white)
                        } else {
                            ForEach(camera.exerciseRepMemory.keys.sorted(), id: \.self) { exercise in
                                HStack {
                                    Text(exercise)
                                    Spacer()
                                    Text("\(camera.exerciseRepMemory[exercise] ?? 0) reps")
                                }
                                .foregroundColor(.white)
                            }
                        }
                    }
                    .padding()
                }
                .padding()
                .background(Color.black.opacity(0.6))
                .cornerRadius(12)
                .foregroundColor(.white)

                if let pending = camera.pendingWorkout {
                    VStack(spacing: 10) {
                        Text("Workout ended")
                            .font(.headline)
                            .foregroundColor(.white)

                        Text("Duration: \(camera.formattedTime(pending.durationSeconds))")
                            .foregroundColor(.white)

                        HStack {
                            Button("Save") {
                                camera.savePendingWorkout()
                            }
                            .padding()
                            .frame(width: 120)
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(12)

                            Button("Delete") {
                                camera.deletePendingWorkout()
                            }
                            .padding()
                            .frame(width: 120)
                            .background(Color.gray)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                        }
                    }
                    .padding()
                    .background(Color.black.opacity(0.7))
                    .cornerRadius(14)
                }

                DisclosureGroup("Saved Workouts", isExpanded: $showSavedWorkouts) {
                    if camera.savedWorkouts.isEmpty {
                        Text("No saved workouts yet")
                            .foregroundColor(.white)
                            .padding()
                    } else {
                        ScrollView {
                            VStack(alignment: .leading, spacing: 10) {
                                ForEach(camera.savedWorkouts) { workout in
                                    VStack(alignment: .leading, spacing: 4) {
                                        HStack {
                                            VStack(alignment: .leading) {
                                                Text(workout.date.formatted(date: .abbreviated, time: .shortened))
                                                    .bold()
                                                    .foregroundColor(.white)

                                                Text("Duration: \(camera.formattedTime(workout.durationSeconds))")
                                                    .foregroundColor(.white)
                                            }

                                            Spacer()

                                            Button("Delete") {
                                                camera.deleteSavedWorkoutById(workout.id)
                                            }
                                            .foregroundColor(.red)
                                        }

                                        ForEach(workout.repsByExercise.keys.sorted(), id: \.self) { exercise in
                                            Text("\(exercise): \(workout.repsByExercise[exercise] ?? 0) reps")
                                                .foregroundColor(.white)
                                        }
                                    }
                                    .padding()
                                    .background(Color.black.opacity(0.45))
                                    .cornerRadius(10)
                                }
                            }
                            .padding()
                        }
                        .frame(height: 250)
                    }
                }
                .padding()
                .background(Color.black.opacity(0.6))
                .cornerRadius(12)
                .foregroundColor(.white)

                Spacer()

                HStack(spacing: 20) {
                    Button("Start") {
                        camera.startWorkout()
                    }
                    .padding()
                    .frame(width: 130)
                    .background(Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(12)

                    Button("End") {
                        camera.endWorkout()
                    }
                    .padding()
                    .frame(width: 130)
                    .background(Color.red)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }
                .padding(.bottom, 40)
            }
            .padding()
        }
        .onAppear {
            camera.start()
        }
    }
}
