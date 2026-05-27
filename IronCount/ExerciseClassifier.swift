import Foundation
import CoreML

struct ExercisePrediction {
    let label: String
    let confidence: Float
    let margin: Float
}

struct ScalerData: Codable {
    let mean: [Float]
    let scale: [Float]
}

final class ExerciseClassifier {

    private var model: MLModel?
    private var labels: [String] = []
    private var mean: [Float] = []
    private var scale: [Float] = []

    private let expectedFrames = 48
    private let expectedFeatures = 300

    init?() {
        do {
            let config = MLModelConfiguration()

            if let compiledURL = Bundle.main.url(
                forResource: "WorkoutClassifierCNNLSTM",
                withExtension: "mlmodelc"
            ) {
                self.model = try MLModel(contentsOf: compiledURL, configuration: config)
                print("Loaded compiled CoreML model")
            } else if let packageURL = Bundle.main.url(
                forResource: "WorkoutClassifierCNNLSTM",
                withExtension: "mlpackage"
            ) {
                let compiledURL = try MLModel.compileModel(at: packageURL)
                self.model = try MLModel(contentsOf: compiledURL, configuration: config)
                print("Loaded mlpackage CoreML model")
            } else {
                print("WorkoutClassifierCNNLSTM model not found")
                return nil
            }

            guard let labelPath = Bundle.main.path(
                forResource: "label_names",
                ofType: "json"
            ) else {
                print("label_names.json not found")
                return nil
            }

            let labelData = try Data(contentsOf: URL(fileURLWithPath: labelPath))
            self.labels = try JSONDecoder().decode([String].self, from: labelData)

            guard let scalerPath = Bundle.main.path(
                forResource: "mediapipe_scaler",
                ofType: "json"
            ) else {
                print("mediapipe_scaler.json not found")
                return nil
            }

            let scalerData = try Data(contentsOf: URL(fileURLWithPath: scalerPath))
            let scaler = try JSONDecoder().decode(ScalerData.self, from: scalerData)

            self.mean = scaler.mean
            self.scale = scaler.scale

            print("CoreML LSTM classifier loaded")
            print("Labels:", labels.count)
            print("Mean count:", mean.count)
            print("Scale count:", scale.count)

        } catch {
            print("ExerciseClassifier init error:", error)
            return nil
        }
    }

    func predict(sequence: [[Float]]) -> ExercisePrediction? {
        guard let model = model else {
            print("CoreML model not loaded")
            return nil
        }

        guard sequence.count == expectedFrames else {
            print("Wrong sequence length:", sequence.count)
            return nil
        }

        guard let first = sequence.first,
              first.count == expectedFeatures else {
            print("Wrong feature count:", sequence.first?.count ?? -1)
            return nil
        }

        guard mean.count == expectedFeatures,
              scale.count == expectedFeatures else {
            print("Scaler mismatch")
            return nil
        }

        do {
            let inputArray = try MLMultiArray(
                shape: [
                    NSNumber(value: 1),
                    NSNumber(value: expectedFrames),
                    NSNumber(value: expectedFeatures)
                ],
                dataType: .float32
            )

            var flatIndex = 0

            for t in 0..<expectedFrames {
                let frame = sequence[t]

                guard frame.count == expectedFeatures else {
                    print("Bad frame feature count:", frame.count)
                    return nil
                }

                for f in 0..<expectedFeatures {
                    let denom = scale[f] == 0 ? 1.0 : scale[f]
                    let scaled = (frame[f] - mean[f]) / denom

                    inputArray[flatIndex] = NSNumber(value: scaled)
                    flatIndex += 1
                }
            }

            let inputProvider = try MLDictionaryFeatureProvider(dictionary: [
                "input": MLFeatureValue(multiArray: inputArray)
            ])

            let output = try model.prediction(from: inputProvider)

            guard let logitsArray = output.featureValue(for: "logits")?.multiArrayValue else {
                print("Could not find logits output")
                print("Available outputs:", output.featureNames)
                return nil
            }

            let logits = multiArrayToFloatArray(logitsArray)
            let probs = softmax(logits)

            let sortedIndices = probs.indices.sorted { probs[$0] > probs[$1] }

            guard sortedIndices.count >= 2 else {
                return nil
            }

            let top1 = sortedIndices[0]
            let top2 = sortedIndices[1]

            guard top1 < labels.count else {
                return nil
            }

            let label = labels[top1]
            let confidence = probs[top1]
            let margin = probs[top1] - probs[top2]

            print("Prediction:", label, "confidence:", confidence, "margin:", margin)

            return ExercisePrediction(
                label: label,
                confidence: confidence,
                margin: margin
            )

        } catch {
            print("CoreML prediction error:", error)
            return nil
        }
    }

    private func multiArrayToFloatArray(_ array: MLMultiArray) -> [Float] {
        var values: [Float] = []
        values.reserveCapacity(array.count)

        for i in 0..<array.count {
            values.append(array[i].floatValue)
        }

        return values
    }

    private func softmax(_ logits: [Float]) -> [Float] {
        guard let maxLogit = logits.max() else {
            return logits
        }

        let expValues = logits.map { exp($0 - maxLogit) }
        let sumExp = expValues.reduce(0, +)

        guard sumExp != 0 else {
            return logits
        }

        return expValues.map { $0 / sumExp }
    }
}
