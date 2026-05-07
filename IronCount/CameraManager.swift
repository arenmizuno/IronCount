//
//  CameraManager.swift
//  IronCount
//
//  Created by Aren Mizuno on 5/4/26.
//
import Foundation
import AVFoundation
import SwiftUI
import Combine

class CameraManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {

    let session = AVCaptureSession()

    @Published var exerciseLabel = "Starting..."
    @Published var reps = 0

    private let poseEstimator = PoseEstimator()
    private let classifier = ExerciseClassifier()

    private var sequence: [[Float]] = []
    private var lockedExercise: String?
    private var repCounter: RepCounter?

    private var frameCounter = 0

    func start() {
        session.beginConfiguration()
        session.sessionPreset = .medium

        guard let device = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: device)
        else {
            print("No camera")
            return
        }

        if session.canAddInput(input) {
            session.addInput(input)
        }

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera"))

        if session.canAddOutput(output) {
            session.addOutput(output)
        }

        session.commitConfiguration()
        session.startRunning()
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {

        // Process every 3rd frame so the phone does not lag too much
        frameCounter += 1
        if frameCounter % 3 != 0 {
            return
        }

        guard let keypoints = poseEstimator?.estimatePose(from: sampleBuffer) else {
            DispatchQueue.main.async {
                self.exerciseLabel = "No pose detected"
            }
            return
        }

        let features = AngleUtils.keypointsToFeatures(keypoints)

        guard features.count == 60 else {
            print("Feature count wrong:", features.count)
            return
        }

        sequence.append(features)

        if sequence.count > 32 {
            sequence.removeFirst()
        }

        if sequence.count == 32 && lockedExercise == nil {
            if let predicted = classifier?.predict(sequence: sequence) {
                lockedExercise = predicted
                repCounter = RepCounter(exercise: predicted)

                DispatchQueue.main.async {
                    self.exerciseLabel = predicted
                }
            }
        }

        if let exercise = lockedExercise,
           let angle = AngleUtils.countingAngle(keypoints, exercise: exercise),
           let counter = repCounter {

            let newReps = counter.update(angle: angle)

            DispatchQueue.main.async {
                self.exerciseLabel = exercise
                self.reps = newReps
            }
        }
    }

    func reset() {
        sequence.removeAll()
        lockedExercise = nil
        repCounter = nil

        DispatchQueue.main.async {
            self.exerciseLabel = "Resetting..."
            self.reps = 0
        }
    }
}
