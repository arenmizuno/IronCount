# Iron Count

## ADSP 32023 – Advanced Computer Vision Using Deep Learning  
### Final Project – University of Chicago  

**Instructor:** Ashish Pujari  
**Term:** Spring 2026  

---

## Team Members

- Aren Mizuno  
- Devyani Rastogi  
- Kennedy Damtse  
- Rohan Muralidhar  

---

# Project Overview

Iron Count is a pose-based workout exercise classification and repetition counting system built using deep learning, MediaPipe pose estimation, and CoreML deployment for iOS.

The project analyzes workout videos and live camera streams to:
- classify exercises in real time,
- estimate body pose landmarks,
- generate temporal pose sequences,
- and support repetition counting for gym exercises.

The system is designed for efficient on-device inference using CoreML and Swift, enabling real-time mobile deployment without requiring cloud computation.

---

# Features

- Real-time exercise classification
- MediaPipe pose landmark extraction
- Temporal sequence modeling
- Rep-counting support
- iOS CoreML deployment
- Multiple deep learning architectures
- Train/test evaluation pipeline
- EDA and preprocessing notebooks
- Support for noisy real-world test videos

---

# Pipeline Overview

## 1. Exploratory Data Analysis (EDA)

EDA and preprocessing were performed in:

```text
src/eda.ipynb
```

### Dataset Source

```text
Kaggle: philosopher0808/gym-workoutexercises-video
```

### Dataset Overview

- No corrupted videos detected
- 22 exercise classes used (plank removed)
- Separate train and noisy real-world test splits

### Train / Verified Split

- 1,571 videos
- Cleaned and segmented exercise clips (10–13 seconds max)
- 817 videos from the original Workout/Exercises Video dataset (Hasyim Abdillah)
- 754 videos crawled from YouTube

### Test Split

- 61 noisy real-world videos
- Includes longer durations, varied resolutions, and more realistic recording conditions

### EDA Tasks Performed

- Video metadata extraction
- FPS analysis
- Resolution analysis
- Duration distribution analysis
- Exercise frequency analysis
- Class imbalance analysis
- Train/test split validation
- Filename overlap validation
- Corrupted video detection

### Key Findings

- Most training videos were standardized around:
  - ~10 second durations
  - 25–30 FPS
  - 720p and 1080p resolutions

- The noisy test set showed:
  - larger variation in duration
  - more diverse resolutions
  - less standardized recording conditions

- Class imbalance was moderate but manageable across the 22 exercise categories.

### Generated Outputs

EDA outputs are automatically saved to:

```text
data/eda/
graphs/eda/
```

Including:
- metadata CSV files
- exercise statistics
- class imbalance summaries
- duration/FPS/resolution graphs
- train/test visualizations

## 2. Pose Extraction & Feature Engineering

Pose extraction and preprocessing were performed in:

```text
src/mediapipe_landmark_extractor.ipynb
```

### MediaPipe Pose Extraction

MediaPipe Pose was used to extract:
- 33 body landmarks per frame
- x, y, z coordinates
- landmark visibility scores

Landmarks were normalized relative to:
- hip center position
- torso/body scale

to improve robustness across:
- different camera distances
- body sizes
- resolutions
- recording environments

### Engineered Features

Additional pose-engineered features were generated for each frame:

#### Joint Angle Features
Angles were computed for:
- elbows
- shoulders
- hips
- knees
- ankles
- torso orientation

#### Geometry Features
Additional body geometry features included:
- shoulder width
- hip width
- wrist distance
- relative wrist/body positions
- body height estimates

#### Velocity Features
Temporal motion information was added using:
- frame-to-frame feature differences
- movement velocity encoding

This helped capture:
- exercise motion dynamics
- exercise tempo
- directional movement patterns

### Temporal Sequence Construction

Videos were converted into overlapping temporal sequences using sliding windows.

Configuration:
- 48 frames per sequence
- 50% overlap between windows
- short videos padded when necessary

This transformed frame-level pose data into temporal motion sequences suitable for:
- LSTM models
- CNN-LSTM hybrids
- TCN architectures

### Final Feature Representation

Each frame contained:
- 150 base pose features
- 150 velocity features

Final sequence shape:

```text
(sequence_length, feature_dim)
(48, 300)
```

### Generated Outputs

The preprocessing pipeline generated:
- train/test `.npy` sequence tensors
- label arrays
- metadata CSV files
- scaler configurations
- label mappings

Outputs are saved to:

```text
data/preprocessed/
```

### Notes

Large preprocessed `.npy` files were not uploaded to GitHub because they exceeded GitHub's file size limits.

These files can be regenerated using:

```text
src/mediapipe_landmark_extractor.ipynb
```

## 3. Sequence Generation

Videos are converted into overlapping temporal sequences using sliding windows.

Example:
- 48 frames per sequence
- 50% overlap between sequences

## 4. Model Training

Several deep learning architectures were explored:

| Model | Purpose |
|---|---|
| CNN | Spatial-temporal feature extraction |
| LSTM | Sequential temporal modeling |
| CNN + LSTM | Combined spatial and temporal learning |
| TCN | Temporal convolution-based sequence learning |

## 5. CoreML Deployment

The best-performing models were converted to CoreML for integration into an iOS application using Swift.

---

# Repository Structure

```text
IronCount/
│
├── README.md
│
├── data/
│   ├── eda/
│   ├── torch_cnn_coreml/
│   ├── torch_lstm_coreml/
│   ├── torch_cnn_lstm_coreml/
│   └── torch_tcn_coreml/
│   └── preprocessed/
│
├── graphs/
│   └── eda/
│   ├── torch_cnn_coreml/
│   ├── torch_lstm_coreml/
│   ├── torch_cnn_lstm_coreml/
│   └── torch_tcn_coreml/
│
├── models/
│   ├── torch_cnn_coreml/
│   ├── torch_lstm_coreml/
│   ├── torch_cnn_lstm_coreml/
│   └── torch_tcn_coreml/
│
├── src/
│   ├── eda.ipynb
│   ├── mediapipe_landmark_extractor.ipynb
│   └── models/
│       ├── torch_cnn_coreml_mediapipe.ipynb
│       ├── torch_lstm_coreml_mediapipe.ipynb
│       ├── torch_cnn_lstm_coreml_mediapipe.ipynb
│       └── torch_tcn_coreml_mediapipe.ipynb
│
└── IronCount/
    │
    ├── IronCountApp.swift
    ├── ContentView.swift
    ├── CameraManager.swift
    ├── CameraPreview.swift
    ├── ExerciseClassifier.swift
    ├── FeatureExtractor.swift
    ├── MediaPipePoseManager.swift
    ├── RepCounter.swift
    ├── WorkoutRecord.swift
    ├── WorkoutClassifierLSTM.mlpackage
    ├── label_names.json
    └──  mediapipe_scaler.json
```

---

## Notes

Large preprocessed `.npy` feature files were not uploaded to GitHub because they exceeded GitHub's file size limits.

Examples include:
- `X_verified_mediapipe.npy`
- `X_test_mediapipe.npy`

These files can be regenerated using:

```text
src/mediapipe_landmark_extractor.ipynb
```

All preprocessing scripts, metadata files, label mappings, and feature extraction pipelines are included in this repository.

---

# Technologies Used

## Computer Vision
- MediaPipe
- OpenCV

## Deep Learning
- PyTorch
- CoreMLTools

## Data Science
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Mobile Deployment
- Swift
- CoreML
- Xcode

---

# Model Deployment

The final models were exported as:
- `.mlpackage`
- `.json` label mappings
- scaler configuration files

and integrated into an iOS application for real-time inference.

---

# Results

EDIT

---

# Future Improvements

Potential future work includes:
- full repetition counting integration,
- exercise form evaluation,
- multi-person detection,
- workout tracking analytics,
- and transformer-based temporal architectures.

---

