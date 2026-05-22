import Foundation
import SwiftUI
import AVFoundation
import Combine
import MediaPipeTasksVision

final class CameraManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {

    let session = AVCaptureSession()

    @Published var exerciseLabel = "Ready"
    @Published var reps = 0
    @Published var exerciseRepMemory: [String: Int] = [:]
    @Published var isWorkoutActive = false
    @Published var elapsedSeconds = 0
    @Published var pendingWorkout: WorkoutRecord?
    @Published var savedWorkouts: [WorkoutRecord] = []

    private var poseManager: MediaPipePoseManager?
    private let classifier = ExerciseClassifier()

    private var sequence: [[Float]] = []
    private var lockedExercise: String?
    private var repCounter: RepCounter?

    private var frameCounter = 0
    private var timer: Timer?

    private var recentPredictions: [String] = []
    private let smoothingWindow = 7
    private let minVotesToSwitch = 4

    private let minConfidence: Float = 0.60
    private let minMargin: Float = 0.15

    private let expectedFrames = 48
    private let expectedFeatures = 300

    private let storageKey = "saved_workouts"

    override init() {
        super.init()

        poseManager = MediaPipePoseManager(delegate: self)
        loadSavedWorkouts()

        if poseManager == nil {
            exerciseLabel = "MediaPipe failed to load"
        } else if classifier == nil {
            exerciseLabel = "Classifier failed to load"
        } else {
            exerciseLabel = "Ready"
        }
    }

    func start() {
        session.beginConfiguration()
        session.sessionPreset = .medium

        guard let device = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: .back
        ),
        let input = try? AVCaptureDeviceInput(device: device)
        else {
            DispatchQueue.main.async {
                self.exerciseLabel = "No camera device"
            }
            return
        }

        if session.canAddInput(input) {
            session.addInput(input)
        }

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]

        output.setSampleBufferDelegate(
            self,
            queue: DispatchQueue(label: "camera.queue")
        )

        if session.canAddOutput(output) {
            session.addOutput(output)
        }

        if let connection = output.connection(with: .video) {
            connection.videoOrientation = .portrait
        }

        session.commitConfiguration()

        DispatchQueue.global(qos: .userInitiated).async {
            self.session.startRunning()
        }
    }

    func startWorkout() {
        sequence.removeAll()
        lockedExercise = nil
        repCounter = nil
        recentPredictions.removeAll()
        exerciseRepMemory.removeAll()
        LandmarkVelocityFrame.previous = nil

        pendingWorkout = nil
        elapsedSeconds = 0

        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { _ in
            DispatchQueue.main.async {
                self.elapsedSeconds += 1
            }
        }

        DispatchQueue.main.async {
            self.reps = 0
            self.isWorkoutActive = true
            self.exerciseLabel = "Workout started"
        }
    }

    func endWorkout() {
        saveCurrentExerciseReps()

        timer?.invalidate()
        timer = nil

        let workout = WorkoutRecord(
            id: UUID(),
            date: Date(),
            durationSeconds: elapsedSeconds,
            repsByExercise: exerciseRepMemory
        )

        sequence.removeAll()
        lockedExercise = nil
        repCounter = nil
        recentPredictions.removeAll()
        LandmarkVelocityFrame.previous = nil

        DispatchQueue.main.async {
            self.isWorkoutActive = false
            self.pendingWorkout = workout
            self.exerciseLabel = "Save or delete workout?"
            self.reps = 0
        }
    }

    func savePendingWorkout() {
        guard let workout = pendingWorkout else {
            return
        }

        savedWorkouts.insert(workout, at: 0)
        saveWorkoutsToStorage()
        pendingWorkout = nil

        DispatchQueue.main.async {
            self.exerciseLabel = "Workout saved"
        }
    }

    func deletePendingWorkout() {
        pendingWorkout = nil
        exerciseRepMemory.removeAll()
        elapsedSeconds = 0

        DispatchQueue.main.async {
            self.exerciseLabel = "Workout deleted"
        }
    }

    func deleteSavedWorkoutById(_ id: UUID) {
        savedWorkouts.removeAll { $0.id == id }
        saveWorkoutsToStorage()
    }

    func reset() {
        timer?.invalidate()
        timer = nil

        sequence.removeAll()
        lockedExercise = nil
        repCounter = nil
        recentPredictions.removeAll()
        exerciseRepMemory.removeAll()
        LandmarkVelocityFrame.previous = nil

        pendingWorkout = nil
        elapsedSeconds = 0

        DispatchQueue.main.async {
            self.exerciseLabel = "Reset"
            self.reps = 0
            self.isWorkoutActive = false
        }
    }

    private func saveCurrentExerciseReps() {
        if let exercise = lockedExercise {
            exerciseRepMemory[exercise] = reps
        }
    }

    private func switchToExercise(_ newExercise: String) {
        if newExercise == lockedExercise {
            return
        }

        saveCurrentExerciseReps()

        lockedExercise = newExercise

        let savedReps = exerciseRepMemory[newExercise] ?? 0

        repCounter = RepCounter(exercise: newExercise)
        repCounter?.count = savedReps

        DispatchQueue.main.async {
            self.reps = savedReps
            self.exerciseLabel = newExercise
        }
    }

    private func saveWorkoutsToStorage() {
        if let data = try? JSONEncoder().encode(savedWorkouts) {
            UserDefaults.standard.set(data, forKey: storageKey)
        }
    }

    private func loadSavedWorkouts() {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let workouts = try? JSONDecoder().decode([WorkoutRecord].self, from: data)
        else {
            return
        }

        savedWorkouts = workouts
    }

    func formattedTime(_ seconds: Int) -> String {
        let minutes = seconds / 60
        let secs = seconds % 60

        return String(format: "%02d:%02d", minutes, secs)
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {

        frameCounter += 1

        guard frameCounter % 3 == 0 else {
            return
        }

        let timestampMs = Int(Date().timeIntervalSince1970 * 1000)

        poseManager?.detectAsync(
            sampleBuffer: sampleBuffer,
            orientation: .right,
            timestampMs: timestampMs
        )
    }
}

extension CameraManager: MediaPipePoseManagerDelegate {

    func mediaPipePoseManager(_ manager: MediaPipePoseManager,
                              didOutput landmarks: [MPPoseLandmark]) {

        guard isWorkoutActive else {
            return
        }

        let features = FeatureExtractor.landmarksToFeatures(landmarks)

        guard features.count == expectedFeatures else {
            DispatchQueue.main.async {
                self.exerciseLabel = "Wrong features: \(features.count)"
            }
            return
        }

        sequence.append(features)

        if sequence.count > expectedFrames {
            sequence.removeFirst()
        }

        if sequence.count < expectedFrames {
            DispatchQueue.main.async {
                self.exerciseLabel = "Collecting: \(self.sequence.count)/\(self.expectedFrames)"
            }
            return
        }

        if let prediction = classifier?.predict(sequence: sequence) {

            guard prediction.confidence >= minConfidence,
                  prediction.margin >= minMargin
            else {
                DispatchQueue.main.async {
                    self.exerciseLabel = "Low confidence: \(Int(prediction.confidence * 100))%"
                }
                return
            }

            recentPredictions.append(prediction.label)

            if recentPredictions.count > smoothingWindow {
                recentPredictions.removeFirst()
            }

            let voteCounts = Dictionary(grouping: recentPredictions, by: { $0 })
                .mapValues { $0.count }

            if let bestVote = voteCounts.max(by: { $0.value < $1.value }),
               bestVote.value >= minVotesToSwitch {
                switchToExercise(bestVote.key)
            }

        } else {
            DispatchQueue.main.async {
                self.exerciseLabel = "Classifier returned nil"
            }
            return
        }

        guard let exercise = lockedExercise,
              let angle = FeatureExtractor.countingAngle(
                landmarks: landmarks,
                exercise: exercise
              ),
              let counter = repCounter
        else {
            DispatchQueue.main.async {
                self.exerciseLabel = "Classifying..."
            }
            return
        }

        let currentReps = counter.update(angle: angle)

        exerciseRepMemory[exercise] = currentReps

        DispatchQueue.main.async {
            self.reps = currentReps
            self.exerciseLabel = "\(exercise) angle: \(Int(angle))"
        }
    }

    func mediaPipePoseManagerDidFail(_ manager: MediaPipePoseManager,
                                     error: Error?) {
        guard isWorkoutActive else {
            return
        }

        DispatchQueue.main.async {
            if let error = error {
                self.exerciseLabel = "Pose error: \(error.localizedDescription)"
            } else {
                self.exerciseLabel = "No pose detected"
            }
        }
    }
}
