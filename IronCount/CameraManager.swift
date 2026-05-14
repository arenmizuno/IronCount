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

    private var poseManager: MediaPipePoseManager?
    private let classifier = ExerciseClassifier()

    private var sequence: [[Float]] = []
    private var lockedExercise: String?
    private var repCounter: RepCounter?

    private var frameCounter = 0
    private var savedWorkouts: [[String: Int]] = []

    private var recentPredictions: [String] = []
    private let smoothingWindow = 7
    private let minVotesToSwitch = 4

    private let expectedFrames = 48
    private let expectedFeatures = 300

    override init() {
        super.init()
        poseManager = MediaPipePoseManager(delegate: self)

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

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device)
        else {
            DispatchQueue.main.async { self.exerciseLabel = "No camera device" }
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
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera.queue"))

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

        DispatchQueue.main.async {
            self.reps = 0
            self.isWorkoutActive = true
            self.exerciseLabel = "Workout started"
        }
    }

    func endWorkout() {
        saveCurrentExerciseReps()

        savedWorkouts.append(exerciseRepMemory)

        sequence.removeAll()
        lockedExercise = nil
        repCounter = nil
        recentPredictions.removeAll()
        LandmarkVelocityFrame.previous = nil

        DispatchQueue.main.async {
            self.isWorkoutActive = false
            self.exerciseLabel = "Workout saved"
            self.reps = 0
        }

        print("Saved workout:", exerciseRepMemory)
    }

    func reset() {
        sequence.removeAll()
        lockedExercise = nil
        repCounter = nil
        recentPredictions.removeAll()
        exerciseRepMemory.removeAll()
        LandmarkVelocityFrame.previous = nil

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

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {

        frameCounter += 1

        guard frameCounter % 3 == 0 else { return }

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

        guard isWorkoutActive else { return }

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
            recentPredictions.append(prediction)

            if recentPredictions.count > smoothingWindow {
                recentPredictions.removeFirst()
            }

            let voteCounts = Dictionary(grouping: recentPredictions, by: { $0 })
                .mapValues { $0.count }

            if let bestVote = voteCounts.max(by: { $0.value < $1.value }) {
                let bestExercise = bestVote.key
                let votes = bestVote.value

                if votes >= minVotesToSwitch {
                    switchToExercise(bestExercise)
                }
            }
        } else {
            DispatchQueue.main.async {
                self.exerciseLabel = "Classifier returned nil"
            }
            return
        }

        guard let exercise = lockedExercise,
              let angle = FeatureExtractor.countingAngle(landmarks: landmarks, exercise: exercise),
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
        guard isWorkoutActive else { return }

        DispatchQueue.main.async {
            if let error = error {
                self.exerciseLabel = "Pose error: \(error.localizedDescription)"
            } else {
                self.exerciseLabel = "No pose detected"
            }
        }
    }
}
