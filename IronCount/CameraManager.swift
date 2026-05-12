import Foundation
import SwiftUI
import AVFoundation
import Combine
import MediaPipeTasksVision

final class CameraManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {

    let session = AVCaptureSession()

    @Published var exerciseLabel = "Starting..."
    @Published var reps = 0

    private var poseManager: MediaPipePoseManager?
    private let classifier = ExerciseClassifier()

    private var sequence: [[Float]] = []
    private var lockedExercise: String?
    private var repCounter: RepCounter?

    private var frameCounter = 0

    override init() {
        super.init()

        poseManager = MediaPipePoseManager(delegate: self)

        if poseManager == nil {
            DispatchQueue.main.async {
                self.exerciseLabel = "MediaPipe failed to load"
            }
        } else if classifier == nil {
            DispatchQueue.main.async {
                self.exerciseLabel = "Classifier failed to load"
            }
        } else {
            DispatchQueue.main.async {
                self.exerciseLabel = "Ready"
            }
        }
    }

    func start() {
        DispatchQueue.main.async {
            self.exerciseLabel = "Starting camera..."
        }

        session.beginConfiguration()
        session.sessionPreset = .medium

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device)
        else {
            DispatchQueue.main.async {
                self.exerciseLabel = "No camera device"
            }
            return
        }

        if session.canAddInput(input) {
            session.addInput(input)
        } else {
            DispatchQueue.main.async {
                self.exerciseLabel = "Cannot add camera input"
            }
            return
        }

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera.queue"))

        if session.canAddOutput(output) {
            session.addOutput(output)
        } else {
            DispatchQueue.main.async {
                self.exerciseLabel = "Cannot add video output"
            }
            return
        }

        if let connection = output.connection(with: .video) {
            connection.videoOrientation = .portrait
        }

        session.commitConfiguration()

        DispatchQueue.global(qos: .userInitiated).async {
            self.session.startRunning()

            DispatchQueue.main.async {
                self.exerciseLabel = "Camera running"
            }
        }
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {

        frameCounter += 1

        if frameCounter % 30 == 0 {
            DispatchQueue.main.async {
                self.exerciseLabel = "Frames received: \(self.frameCounter)"
            }
        }

        guard frameCounter % 3 == 0 else { return }

        let timestampMs = Int(Date().timeIntervalSince1970 * 1000)

        poseManager?.detectAsync(
            sampleBuffer: sampleBuffer,
            orientation: .right,
            timestampMs: timestampMs
        )
    }

    func reset() {
        sequence.removeAll()
        lockedExercise = nil
        repCounter = nil

        DispatchQueue.main.async {
            self.exerciseLabel = "Reset"
            self.reps = 0
        }
    }
}

extension CameraManager: MediaPipePoseManagerDelegate {
    func mediaPipePoseManager(_ manager: MediaPipePoseManager, didOutput landmarks: [MPPoseLandmark]) {
        DispatchQueue.main.async {
            self.exerciseLabel = "Pose detected"
        }

        let features = FeatureExtractor.landmarksToFeatures(landmarks)

        guard features.count == 144 else {
            DispatchQueue.main.async {
                self.exerciseLabel = "Wrong features: \(features.count)"
            }
            return
        }

        sequence.append(features)

        if sequence.count > 32 {
            sequence.removeFirst()
        }

        if sequence.count < 32 {
            DispatchQueue.main.async {
                self.exerciseLabel = "Collecting: \(self.sequence.count)/32"
            }
            return
        }

        if lockedExercise == nil {
            if let predictedExercise = classifier?.predict(sequence: sequence) {
                lockedExercise = predictedExercise
                repCounter = RepCounter(exercise: predictedExercise)

                DispatchQueue.main.async {
                    self.exerciseLabel = predictedExercise
                }
            } else {
                DispatchQueue.main.async {
                    self.exerciseLabel = "Classifier returned nil"
                }
                return
            }
        }

        if let exercise = lockedExercise,
           let angle = FeatureExtractor.countingAngle(landmarks: landmarks, exercise: exercise),
           let counter = repCounter {

            let currentReps = counter.update(angle: angle)

            DispatchQueue.main.async {
                self.exerciseLabel = "\(exercise) angle: \(Int(angle))"
                self.reps = currentReps
            }
        } else {
            DispatchQueue.main.async {
                self.exerciseLabel = "No count angle"
            }
        }
    }

    func mediaPipePoseManagerDidFail(_ manager: MediaPipePoseManager, error: Error?) {
        DispatchQueue.main.async {
            if let error = error {
                self.exerciseLabel = "Pose error: \(error.localizedDescription)"
            } else {
                self.exerciseLabel = "No pose detected"
            }
        }
    }
}
