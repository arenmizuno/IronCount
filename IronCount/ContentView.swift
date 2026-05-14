import SwiftUI

struct ContentView: View {
    @StateObject var camera = CameraManager()
    @State private var showExerciseSummary = false

    var body: some View {
        ZStack {
            CameraPreview(session: camera.session)
                .ignoresSafeArea()

            VStack(spacing: 16) {
                Text(camera.exerciseLabel)
                    .font(.title2)
                    .foregroundColor(.white)
                    .padding(.top, 40)

                Text("Current Reps: \(camera.reps)")
                    .font(.title)
                    .foregroundColor(.green)

                DisclosureGroup("Exercise Summary", isExpanded: $showExerciseSummary) {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(camera.exerciseRepMemory.keys.sorted(), id: \.self) { exercise in
                            HStack {
                                Text(exercise)
                                Spacer()
                                Text("\(camera.exerciseRepMemory[exercise] ?? 0) reps")
                            }
                            .foregroundColor(.white)
                        }
                    }
                    .padding()
                    .background(Color.black.opacity(0.5))
                    .cornerRadius(12)
                }
                .padding()
                .background(Color.black.opacity(0.6))
                .cornerRadius(12)
                .foregroundColor(.white)

                Spacer()

                HStack(spacing: 20) {
                    Button(action: {
                        camera.startWorkout()
                    }) {
                        Text("Start")
                            .font(.title2)
                            .padding()
                            .frame(width: 130)
                            .background(Color.green)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                    }

                    Button(action: {
                        camera.endWorkout()
                    }) {
                        Text("End")
                            .font(.title2)
                            .padding()
                            .frame(width: 130)
                            .background(Color.red)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                    }
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
