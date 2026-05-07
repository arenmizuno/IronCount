import Foundation
import TensorFlowLite

final class ExerciseClassifier {
    private var interpreter: Interpreter
    private var labels: [String] = []

    init?() {
        guard let modelPath = Bundle.main.path(
            forResource: "exercise_pose_classifier_quant",
            ofType: "tflite"
        ) else {
            print("Classifier model not found")
            return nil
        }

        guard let labelPath = Bundle.main.path(
            forResource: "label_names",
            ofType: "json"
        ) else {
            print("Labels not found")
            return nil
        }

        do {
            interpreter = try Interpreter(modelPath: modelPath)
            try interpreter.allocateTensors()

            let data = try Data(contentsOf: URL(fileURLWithPath: labelPath))
            labels = try JSONDecoder().decode([String].self, from: data)

        } catch {
            print("Classifier init error:", error)
            return nil
        }
    }

    func predict(sequence: [[Float]]) -> String? {
        guard sequence.count == 32 else { return nil }

        let flat: [Float] = sequence.flatMap { $0 }
        let inputData = flat.toData()

        do {
            try interpreter.copy(inputData, toInputAt: 0)
            try interpreter.invoke()

            let output = try interpreter.output(at: 0)
            let probs: [Float] = output.data.toArray(type: Float.self)

            guard let maxIndex = probs.indices.max(by: { probs[$0] < probs[$1] }) else {
                return nil
            }

            return labels[maxIndex]

        } catch {
            print("Prediction error:", error)
            return nil
        }
    }
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
