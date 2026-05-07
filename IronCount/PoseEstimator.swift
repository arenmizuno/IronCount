//
//  PoseEstimator.swift
//  IronCount
//
//  Created by Aren Mizuno on 5/5/26.
//
import Foundation
import TensorFlowLite
import AVFoundation
import UIKit
import CoreImage

final class PoseEstimator {
    private var interpreter: Interpreter
    private let inputWidth = 320
    private let inputHeight = 320
    private let outputChannels = 56
    private let outputAnchors = 2100

    init?() {
        guard let modelPath = Bundle.main.path(
            forResource: "yolov8n_pose",
            ofType: "tflite"
        ) else {
            print("Could not find yolov8n_pose.tflite")
            return nil
        }

        do {
            interpreter = try Interpreter(modelPath: modelPath)
            try interpreter.allocateTensors()
        } catch {
            print("PoseEstimator init error:", error)
            return nil
        }
    }

    func estimatePose(from sampleBuffer: CMSampleBuffer) -> [Keypoint]? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return nil
        }

        guard let inputFloats = pixelBufferToRGBFloatArray(pixelBuffer) else {
            return nil
        }

        let inputData = inputFloats.toData()

        do {
            try interpreter.copy(inputData, toInputAt: 0)
            try interpreter.invoke()

            let outputTensor = try interpreter.output(at: 0)
            let output = outputTensor.data.toArray(type: Float32.self)

            return parseYOLOPoseOutput(output)

        } catch {
            print("Pose estimation error:", error)
            return nil
        }
    }

    private func parseYOLOPoseOutput(_ output: [Float]) -> [Keypoint]? {
        // Output shape: [1, 56, 2100]
        // Channel-major layout:
        // 0: box x
        // 1: box y
        // 2: box w
        // 3: box h
        // 4: confidence
        // 5 onward: 17 keypoints x,y,conf

        guard output.count == outputChannels * outputAnchors else {
            print("Unexpected YOLO output size:", output.count)
            return nil
        }

        var bestIndex = 0
        var bestScore: Float = -Float.infinity

        for i in 0..<outputAnchors {
            let score = output[4 * outputAnchors + i]
            if score > bestScore {
                bestScore = score
                bestIndex = i
            }
        }

        if bestScore < 0.25 {
            return nil
        }

        var keypoints: [Keypoint] = []

        for j in 0..<17 {
            let baseChannel = 5 + j * 3

            let x = output[baseChannel * outputAnchors + bestIndex]
            let y = output[(baseChannel + 1) * outputAnchors + bestIndex]
            let confidence = output[(baseChannel + 2) * outputAnchors + bestIndex]

            keypoints.append(Keypoint(x: x, y: y, confidence: confidence))
        }

        return keypoints
    }

    private func pixelBufferToRGBFloatArray(_ pixelBuffer: CVPixelBuffer) -> [Float]? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()

        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            return nil
        }

        var rawBytes = [UInt8](
            repeating: 0,
            count: inputWidth * inputHeight * 4
        )

        let colorSpace = CGColorSpaceCreateDeviceRGB()

        guard let cgContext = CGContext(
            data: &rawBytes,
            width: inputWidth,
            height: inputHeight,
            bitsPerComponent: 8,
            bytesPerRow: inputWidth * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return nil
        }

        cgContext.draw(
            cgImage,
            in: CGRect(x: 0, y: 0, width: inputWidth, height: inputHeight)
        )

        var floats: [Float] = []
        floats.reserveCapacity(inputWidth * inputHeight * 3)

        for i in 0..<(inputWidth * inputHeight) {
            let r = Float(rawBytes[i * 4]) / 255.0
            let g = Float(rawBytes[i * 4 + 1]) / 255.0
            let b = Float(rawBytes[i * 4 + 2]) / 255.0

            floats.append(r)
            floats.append(g)
            floats.append(b)
        }

        return floats
    }
}
