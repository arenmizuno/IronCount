import Foundation

// MARK: - JSON-decodable structs (private to this file)

private struct PCAMetadata: Decodable {
    let exercise: String
    let featureDim: Int
    let yLandmarkIndices: [Int]
    let selectedPcs: [Int]
    let schmittK: Double
    let schmittPeakMin: Double
    let schmittPeakMax: Double
    let schmittValleyMin: Double
    let schmittValleyMax: Double
    let minAmplitude: Double
    let warmupFrames: Int
    let savgolWindow: Int

    enum CodingKeys: String, CodingKey {
        case exercise
        case featureDim       = "feature_dim"
        case yLandmarkIndices = "y_landmark_indices"
        case selectedPcs      = "selected_pcs"
        case schmittK         = "schmitt_k"
        case schmittPeakMin   = "schmitt_peak_min"
        case schmittPeakMax   = "schmitt_peak_max"
        case schmittValleyMin = "schmitt_valley_min"
        case schmittValleyMax = "schmitt_valley_max"
        case minAmplitude     = "min_amplitude"
        case warmupFrames     = "warmup_frames"
        case savgolWindow     = "savgol_window"
    }
}

private struct PCAWeights: Decodable {
    let scalerMean: [Double]
    let scalerScale: [Double]
    let pcaComponents: [[Double]]   // shape: [n_components][12]
    let pcaMean: [Double]
    let nComponents: Int

    enum CodingKeys: String, CodingKey {
        case scalerMean    = "scaler_mean"
        case scalerScale   = "scaler_scale"
        case pcaComponents = "pca_components"
        case pcaMean       = "pca_mean"
        case nComponents   = "n_components"
    }
}

// MARK: - Error type

enum PCARepCounterError: Error, LocalizedError {
    case bundleFolderNotFound
    case exerciseFolderNotFound(String)
    case metadataNotFound(String)
    case weightsNotFound(String)
    case decodingFailed(Error)

    var errorDescription: String? {
        switch self {
        case .bundleFolderNotFound:
            return "coreml_bundle folder not found in app bundle"
        case .exerciseFolderNotFound(let ex):
            return "No coreml_bundle subfolder for exercise: \(ex)"
        case .metadataNotFound(let ex):
            return "metadata.json missing for: \(ex)"
        case .weightsNotFound(let ex):
            return "weights.json missing for: \(ex)"
        case .decodingFailed(let e):
            return "JSON decode failed: \(e)"
        }
    }
}

// MARK: - PCARepCounter

/// Drop-in replacement for RepCounter that uses the trained PCA + adaptive
/// Schmitt trigger instead of hard-coded angle thresholds.
///
/// Mirrors the Python PCARepCounter (V3) exactly:
///   landmarks → 12-dim feature vec → z-score → PCA dot product
///   → 60-frame rolling buffer → SavGol(21,3) → amplitude guard
///   → adaptive mean ± 0.8σ Schmitt → rep count
///
/// The profile is loaded from `coreml_bundle/<exercise>/metadata.json`
/// and `weights.json`, which must be added to the Xcode target as a
/// folder reference (blue folder, not yellow group).
final class PCARepCounter {

    // ── Public state (matches RepCounter interface) ──────────
    var count = 0

    // ── Loaded profile ───────────────────────────────────────
    private let scalerMean: [Double]
    private let scalerScale: [Double]
    private let pcaComponents: [[Double]]
    private let pcaMean: [Double]
    private let selectedPcIndex: Int
    private let yLandmarkIndices: [Int]

    // Schmitt + guard params
    private let schmittK: Double
    private let peakMin: Double
    private let peakMax: Double
    private let valleyMin: Double
    private let valleyMax: Double
    private let minAmplitude: Double
    private let warmupFrames: Int
    private let savgolWindow: Int

    // ── Rolling state ────────────────────────────────────────
    private var buffer: [Double] = []
    private var inValley = true
    private var inPeak   = false

    // ── Angle ordering (matches Python JOINT_ANGLE_DEFINITIONS) ─
    //  0 left_elbow   1 right_elbow
    //  2 left_shoulder 3 right_shoulder
    //  4 left_hip      5 right_hip
    //  6 left_knee     7 right_knee
    //  8 left_ankle    9 right_ankle
    // 10 torso
    // Bilateral pairs: (left_idx, right_idx) → averaged into one value
    private let bilateralPairs = [(0,1),(2,3),(4,5),(6,7),(8,9)]
    // symAngles[0..4] = bilateral averages, symAngles[5] = torso

    // MARK: Init

    /// Throws `PCARepCounterError` if the bundle files are missing or malformed.
    init(exercise: String) throws {
        // Locate coreml_bundle root inside the app bundle
        guard let bundleRoot = Bundle.main.url(
            forResource: "coreml_bundle", withExtension: nil
        ) else {
            throw PCARepCounterError.bundleFolderNotFound
        }

        let exDir      = bundleRoot.appendingPathComponent(exercise)
        let metaURL    = exDir.appendingPathComponent("metadata.json")
        let weightsURL = exDir.appendingPathComponent("weights.json")

        guard FileManager.default.fileExists(atPath: exDir.path) else {
            throw PCARepCounterError.exerciseFolderNotFound(exercise)
        }
        guard FileManager.default.fileExists(atPath: metaURL.path) else {
            throw PCARepCounterError.metadataNotFound(exercise)
        }
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw PCARepCounterError.weightsNotFound(exercise)
        }

        do {
            let meta    = try JSONDecoder().decode(PCAMetadata.self,
                              from: Data(contentsOf: metaURL))
            let weights = try JSONDecoder().decode(PCAWeights.self,
                              from: Data(contentsOf: weightsURL))

            scalerMean       = weights.scalerMean
            scalerScale      = weights.scalerScale
            pcaComponents    = weights.pcaComponents
            pcaMean          = weights.pcaMean
            selectedPcIndex  = meta.selectedPcs.first ?? 0
            yLandmarkIndices = meta.yLandmarkIndices

            schmittK     = meta.schmittK
            peakMin      = meta.schmittPeakMin
            peakMax      = meta.schmittPeakMax
            valleyMin    = meta.schmittValleyMin
            valleyMax    = meta.schmittValleyMax
            minAmplitude = meta.minAmplitude
            warmupFrames = meta.warmupFrames
            savgolWindow = meta.savgolWindow
        } catch let e as PCARepCounterError {
            throw e
        } catch {
            throw PCARepCounterError.decodingFailed(error)
        }
    }

    // MARK: Public API

    /// Feed one frame of MediaPipe landmarks. Returns the current rep count.
    /// Signature matches RepCounter.update(angle:) in return type.
    @discardableResult
    func update(landmarks lms: [MPPoseLandmark]) -> Int {
        guard lms.count >= 33 else { return count }

        // 1. 12-dim feature vector (Python-compatible)
        let vec = buildFeatureVec(lms)
        guard vec.count == 12 else { return count }

        // 2. Z-score normalise
        let scaled = zScore(vec)

        // 3. PCA projection (dot product — no sklearn needed)
        let projected = projectPCA(scaled)
        guard selectedPcIndex < projected.count else { return count }
        let pcVal = projected[selectedPcIndex]

        // 4. Rolling buffer (max warmupFrames = 60)
        buffer.append(pcVal)
        if buffer.count > warmupFrames { buffer.removeFirst() }
        guard buffer.count >= 10 else { return count }

        // 5. Amplitude guard — skip if signal is flat (person at rest / setting up)
        let lo  = buffer.min()!
        let hi  = buffer.max()!
        guard (hi - lo) >= minAmplitude else { return count }

        // 6. Normalise to [0, 1] for Schmitt arithmetic
        let rng  = hi - lo + 1e-8
        let norm = buffer.map { ($0 - lo) / rng }

        // 7. Savitzky-Golay smooth → take last value (mirrors scipy savgol_filter[-1])
        let val = savgolSmooth(norm)

        // 8. Adaptive Schmitt thresholds (recomputed every frame from raw buffer)
        let mu    = buffer.reduce(0, +) / Double(buffer.count)
        let sigma = sqrt(buffer.map { ($0 - mu)*($0 - mu) }.reduce(0, +)
                         / Double(buffer.count))

        let peakThr   = min(max((mu + schmittK * sigma - lo) / rng, peakMin), peakMax)
        let valleyThr = min(max((mu - schmittK * sigma - lo) / rng, valleyMin), valleyMax)

        // 9. Schmitt trigger
        if val < valleyThr           { inValley = true;  inPeak   = false }
        if val > peakThr && inValley && !inPeak {
            count   += 1
            inPeak   = true
            inValley = false
        }

        return count
    }

    func reset() {
        count    = 0
        buffer   = []
        inValley = true
        inPeak   = false
    }

    // MARK: - 12-dim feature vector
    //
    // Must match Python _get_feature_vec() exactly:
    //   [0:6]  symmetry-averaged angles in DEGREES (2D, visibility-gated)
    //   [6:12] body-centre-normalised y-coords (hip center origin, shoulder-width scale)
    //
    // Two deliberate differences from FeatureExtractor.swift:
    //   • Angles use x,y only (Python trains on 2D; Swift FeatureExtractor uses 3D)
    //   • Torso = 3-point angle at lm24 (right hip) not angle-from-vertical
    //   • norm scale = shoulder width 2D, not torso height 3D

    private func buildFeatureVec(_ lms: [MPPoseLandmark]) -> [Double] {

        // Raw angles (degrees) — 2D, visibility threshold 0.5, else 0.0
        // Order matches Python JOINT_ANGLE_DEFINITIONS exactly
        let raw: [Double] = [
            ang2D(lms[11], lms[13], lms[15]),  // 0  left_elbow   (vertex=13)
            ang2D(lms[12], lms[14], lms[16]),  // 1  right_elbow  (vertex=14)
            ang2D(lms[13], lms[11], lms[23]),  // 2  left_shoulder  (vertex=11)
            ang2D(lms[14], lms[12], lms[24]),  // 3  right_shoulder (vertex=12)
            ang2D(lms[11], lms[23], lms[25]),  // 4  left_hip   (vertex=23)
            ang2D(lms[12], lms[24], lms[26]),  // 5  right_hip  (vertex=24)
            ang2D(lms[23], lms[25], lms[27]),  // 6  left_knee  (vertex=25)
            ang2D(lms[24], lms[26], lms[28]),  // 7  right_knee (vertex=26)
            ang2D(lms[25], lms[27], lms[31]),  // 8  left_ankle  (vertex=27)
            ang2D(lms[26], lms[28], lms[32]),  // 9  right_ankle (vertex=28)
            ang2D(lms[23], lms[24], lms[12]),  // 10 torso (vertex=24: lhip→rhip→rshoulder)
        ]

        // Symmetry averaging → 6 values: [avg_elbow, avg_shoulder, avg_hip, avg_knee, avg_ankle, torso]
        var sym = [Double](repeating: 0, count: 6)
        for (i, pair) in bilateralPairs.enumerated() {
            sym[i] = (raw[pair.0] + raw[pair.1]) / 2.0
        }
        sym[5] = raw[10]    // torso is unilateral

        // Body-centre-normalised y-coords (Python convention)
        // Center = hip midpoint (x,y); scale = shoulder width (2D Euclidean)
        let hipCX = Double(lms[23].x + lms[24].x) / 2.0
        let hipCY = Double(lms[23].y + lms[24].y) / 2.0
        let dSX   = Double(lms[11].x - lms[12].x)
        let dSY   = Double(lms[11].y - lms[12].y)
        let shoulderWidth = sqrt(dSX * dSX + dSY * dSY) + 1e-8

        var normY = [Double](repeating: 0, count: 6)
        for (i, idx) in yLandmarkIndices.enumerated() {
            normY[i] = (Double(lms[idx].y) - hipCY) / shoulderWidth
        }

        return sym + normY   // (12,)
    }

    /// 2D angle in degrees at vertex `b` (x,y only).
    /// Returns 0.0 if any of the three landmarks has visibility ≤ 0.5
    /// (matches Python's visibility-gating behaviour).
    private func ang2D(_ a: MPPoseLandmark,
                       _ b: MPPoseLandmark,
                       _ c: MPPoseLandmark) -> Double {
        guard a.visibility > 0.5,
              b.visibility > 0.5,
              c.visibility > 0.5 else { return 0.0 }

        let baX = Double(a.x - b.x), baY = Double(a.y - b.y)
        let bcX = Double(c.x - b.x), bcY = Double(c.y - b.y)

        let dot    = baX * bcX + baY * bcY
        let normBA = sqrt(baX * baX + baY * baY)
        let normBC = sqrt(bcX * bcX + bcY * bcY)
        guard normBA > 0, normBC > 0 else { return 0.0 }

        return acos(max(-1.0, min(1.0, dot / (normBA * normBC)))) * 180.0 / .pi
    }

    // MARK: - Linear algebra (two dot products — the entire "model" at runtime)

    private func zScore(_ vec: [Double]) -> [Double] {
        vec.enumerated().map { i, v in
            let s = scalerScale[i] == 0 ? 1.0 : scalerScale[i]
            return (v - scalerMean[i]) / s
        }
    }

    private func projectPCA(_ scaled: [Double]) -> [Double] {
        // centered = scaled - pca_mean
        // projection[k] = dot(pcaComponents[k], centered)
        let centered = zip(scaled, pcaMean).map { $0 - $1 }
        return pcaComponents.map { row in
            zip(row, centered).reduce(0.0) { $0 + $1.0 * $1.1 }
        }
    }

    // MARK: - Savitzky-Golay smoothing (mirrors scipy savgol_filter[-1])
    //
    // scipy's savgol_filter with mode='interp' at the last position:
    // takes the last `win` elements, fits a cubic polynomial, evaluates
    // at x = win-1 (the rightmost point of the window).
    // Implemented here as a least-squares cubic fit via 4×4 normal equations.

    private func savgolSmooth(_ norm: [Double]) -> Double {
        let n = norm.count
        guard n >= 5 else { return norm.last ?? 0.5 }

        // Window: odd, capped at savgolWindow (21), minimum 5
        var win = min(n % 2 == 1 ? n : n - 1, savgolWindow)
        if win % 2 == 0 { win -= 1 }
        win = max(win, 5)

        let data = Array(norm.suffix(win))
        return fitCubicEvalLast(data)
    }

    /// Fit cubic p(x) = a + bx + cx² + dx³ to `y[0..n-1]` at x = [0..n-1],
    /// evaluate at x = n-1 (last point).
    private func fitCubicEvalLast(_ y: [Double]) -> Double {
        let n = y.count
        // Augmented normal equations [A | b] size 4×5
        var A = [[Double]](repeating: [Double](repeating: 0, count: 5), count: 4)

        for i in 0..<n {
            let xi = Double(i)
            let p  = [1.0, xi, xi*xi, xi*xi*xi]
            for r in 0..<4 {
                for c in 0..<4 { A[r][c] += p[r] * p[c] }
                A[r][4] += p[r] * y[i]
            }
        }

        let beta = gaussElim4(&A)
        let x0   = Double(n - 1)
        return beta[0] + beta[1]*x0 + beta[2]*x0*x0 + beta[3]*x0*x0*x0
    }

    /// Gaussian elimination with partial pivoting on a 4×5 augmented matrix.
    private func gaussElim4(_ mat: inout [[Double]]) -> [Double] {
        for col in 0..<4 {
            // Partial pivot
            var maxRow = col
            for row in (col+1)..<4 {
                if abs(mat[row][col]) > abs(mat[maxRow][col]) { maxRow = row }
            }
            mat.swapAt(col, maxRow)
            guard abs(mat[col][col]) > 1e-12 else { continue }

            for row in (col+1)..<4 {
                let f = mat[row][col] / mat[col][col]
                for j in col..<5 { mat[row][j] -= f * mat[col][j] }
            }
        }
        // Back substitution
        var x = [Double](repeating: 0, count: 4)
        for i in stride(from: 3, through: 0, by: -1) {
            var s = mat[i][4]
            for j in (i+1)..<4 { s -= mat[i][j] * x[j] }
            x[i] = abs(mat[i][i]) > 1e-12 ? s / mat[i][i] : 0.0
        }
        return x
    }
}
