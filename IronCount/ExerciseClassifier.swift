import Foundation
import TensorFlowLite

final class ExerciseClassifier {
    private var interpreter: Interpreter
    private var labels: [String] = []
    private var mean: [Float] = []
    private var scale: [Float] = []

    private let expectedFrames = 48
    private let expectedFeatures = 300

    init?() {
        guard let modelPath = Bundle.main.path(
            forResource: "exercise_mediapipe_classifier_quant",
            ofType: "tflite"
        ) else {
            print("exercise_mediapipe_classifier_quant.tflite not found")
            return nil
        }

        guard let labelPath = Bundle.main.path(
            forResource: "label_names",
            ofType: "json"
        ) else {
            print("label_names.json not found")
            return nil
        }

        guard let scalerPath = Bundle.main.path(
            forResource: "mediapipe_scaler",
            ofType: "json"
        ) else {
            print("mediapipe_scaler.json not found")
            return nil
        }

        do {
            interpreter = try Interpreter(modelPath: modelPath)
            try interpreter.allocateTensors()

            let labelData = try Data(contentsOf: URL(fileURLWithPath: labelPath))
            labels = try JSONDecoder().decode([String].self, from: labelData)

            let scalerData = try Data(contentsOf: URL(fileURLWithPath: scalerPath))
            let scaler = try JSONDecoder().decode(ScalerData.self, from: scalerData)

            mean = scaler.mean
            scale = scaler.scale

            print("Classifier loaded")
            print("Labels:", labels.count)
            print("Scaler mean:", mean.count)
            print("Scaler scale:", scale.count)

        } catch {
            print("ExerciseClassifier init error:", error)
            return nil
        }
    }

    func predict(sequence: [[Float]]) -> String? {
        print("Classifier sequence count:", sequence.count)

        guard sequence.count == expectedFrames else {
            print("Wrong sequence length:", sequence.count)
            return nil
        }

        guard let first = sequence.first else {
            print("Sequence empty")
            return nil
        }

        print("Classifier feature count:", first.count)

        guard first.count == expectedFeatures else {
            print("Wrong feature count:", first.count)
            return nil
        }

        guard mean.count == expectedFeatures, scale.count == expectedFeatures else {
            print("Scaler shape mismatch. mean:", mean.count, "scale:", scale.count)
            return nil
        }

        var flat: [Float] = []
        flat.reserveCapacity(expectedFrames * expectedFeatures)

        for frame in sequence {
            guard frame.count == expectedFeatures else {
                print("Bad frame feature count:", frame.count)
                return nil
            }

            for i in 0..<expectedFeatures {
                let denom = scale[i] == 0 ? 1.0 : scale[i]
                let scaledValue = (frame[i] - mean[i]) / denom
                flat.append(scaledValue)
            }
        }

        do {
            try interpreter.copy(flat.toData(), toInputAt: 0)
            try interpreter.invoke()

            let output = try interpreter.output(at: 0)
            let probs: [Float] = output.data.toArray(type: Float.self)

            guard let maxIndex = probs.indices.max(by: { probs[$0] < probs[$1] }),
                  maxIndex < labels.count
            else {
                print("Could not find max prediction")
                return nil
            }

            print("Predicted:", labels[maxIndex], "confidence:", probs[maxIndex])

            return labels[maxIndex]

        } catch {
            print("Classifier prediction error:", error)
            return nil
        }
    }
}

struct ScalerData: Codable {
    let mean: [Float]
    let scale: [Float]
}

extension Array where Element == Float {
    func toData() -> Data {
        var copy = self
        return Data(bytes: &copy, count: copy.count * MemoryLayout<Float>.stride)
    }
}

extension Data {
    func toArray<T>(type: T.Type) -> [T] {
        return self.withUnsafeBytes { rawBufferPointer in
            let buffer = rawBufferPointer.bindMemory(to: T.self)
            return Array(buffer)
        }
    }
}
