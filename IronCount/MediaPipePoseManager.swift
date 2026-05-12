import Foundation
import AVFoundation
import UIKit
import MediaPipeTasksVision

struct MPPoseLandmark {
    let x: Float
    let y: Float
    let z: Float
    let visibility: Float
}

protocol MediaPipePoseManagerDelegate: AnyObject {
    func mediaPipePoseManager(_ manager: MediaPipePoseManager, didOutput landmarks: [MPPoseLandmark])
    func mediaPipePoseManagerDidFail(_ manager: MediaPipePoseManager, error: Error?)
}

final class MediaPipePoseManager: NSObject {

    weak var delegate: MediaPipePoseManagerDelegate?

    private var poseLandmarker: PoseLandmarker?

    init?(delegate: MediaPipePoseManagerDelegate?) {
        self.delegate = delegate
        super.init()

        guard let modelPath = Bundle.main.path(
            forResource: "pose_landmarker_lite",
            ofType: "task"
        ) else {
            print("pose_landmarker_lite.task not found")
            return nil
        }

        let options = PoseLandmarkerOptions()
        options.runningMode = .liveStream
        options.numPoses = 1
        options.minPoseDetectionConfidence = 0.5
        options.minPosePresenceConfidence = 0.5
        options.minTrackingConfidence = 0.5
        options.baseOptions.modelAssetPath = modelPath
        options.poseLandmarkerLiveStreamDelegate = self

        do {
            poseLandmarker = try PoseLandmarker(options: options)
        } catch {
            print("Failed to create PoseLandmarker:", error)
            return nil
        }
    }

    func detectAsync(sampleBuffer: CMSampleBuffer,
                     orientation: UIImage.Orientation,
                     timestampMs: Int) {
        do {
            let mpImage = try MPImage(sampleBuffer: sampleBuffer, orientation: orientation)

            try poseLandmarker?.detectAsync(
                image: mpImage,
                timestampInMilliseconds: timestampMs
            )
        } catch {
            delegate?.mediaPipePoseManagerDidFail(self, error: error)
        }
    }
}

extension MediaPipePoseManager: PoseLandmarkerLiveStreamDelegate {
    func poseLandmarker(_ poseLandmarker: PoseLandmarker,
                        didFinishDetection result: PoseLandmarkerResult?,
                        timestampInMilliseconds: Int,
                        error: Error?) {

        if let error = error {
            delegate?.mediaPipePoseManagerDidFail(self, error: error)
            return
        }

        guard let result = result,
              let firstPose = result.landmarks.first,
              firstPose.count >= 33
        else {
            delegate?.mediaPipePoseManagerDidFail(self, error: nil)
            return
        }

        let landmarks = firstPose.map { lm in
            MPPoseLandmark(
                x: lm.x,
                y: lm.y,
                z: lm.z,
                visibility: lm.visibility?.floatValue ?? 1.0
            )
        }

        delegate?.mediaPipePoseManager(self, didOutput: landmarks)
    }
}
