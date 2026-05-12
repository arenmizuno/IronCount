import Foundation
import TensorFlowLite

final class ExerciseClassifier {
    private var interpreter: Interpreter
    private var labels: [String] = []
    private var mean: [Float] = []
    private var scale: [Float] = []

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

        } catch {
            print("ExerciseClassifier init error:", error)
            return nil
        }
    }

    func predict(sequence: [[Float]]) -> String? {
        guard sequence.count == 32 else { return nil }

        var flat: [Float] = []

        for frame in sequence {
            guard frame.count == 144 else {
                print("Frame feature count wrong:", frame.count)
                return nil
            }

            for i in 0..<frame.count {
                let denom = scale[i] == 0 ? 1.0 : scale[i]
                flat.append((frame[i] - mean[i]) / denom)
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
                return nil
            }

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
