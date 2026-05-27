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

# Business Problem & Project Goal

Users often lose track of repetitions during workouts, especially during high-intensity or high-volume training sessions. Additionally, many current fitness applications lack accurate real-time visual analysis and instead rely on:
- manual repetition input,
- wearable sensors,
- or limited motion tracking systems.

As a result, gym users need intelligent systems capable of:
- automated exercise recognition,
- real-time repetition counting,
- and motion-aware workout monitoring using only a mobile camera.

To address this problem, Iron Count was developed as an AI-based fitness assistant that uses computer vision, pose estimation, and deep learning to:
- classify exercises in real time,
- track workout repetitions,
- analyze pose-based movement patterns,
- and support intelligent workout monitoring directly from a mobile device.

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

## 3. Model Training

Multiple deep learning architectures were explored for exercise classification using MediaPipe pose sequences.  
All models were trained using PyTorch on temporal pose representations with shape:

```text
(sequence_length, feature_dim)
(48, 300)
```

where:
- 48 = number of frames per sequence
- 300 = engineered pose + velocity features per frame

Training pipelines for the models are included in:

```text
src/torch_cnn_coreml_mediapipe.ipynb
src/torch_lstm_coreml_mediapipe.ipynb
src/torch_cnn_lstm_coreml_mediapipe.ipynb
src/torch_tcn_coreml_mediapipe.ipynb
```

### Architectures Evaluated

| Model | Purpose |
|---|---|
| CNN | Lightweight temporal feature extraction with fast mobile inference |
| LSTM | Sequential motion modeling using bidirectional temporal learning |
| CNN + LSTM | Combines local motion extraction with temporal sequence modeling |
| TCN | Efficient temporal modeling with dilated convolutions and residual connections |

### Hyperparameter Search

Multiple hyperparameter configurations were evaluated across all architectures using GroupKFold cross-validation.

| Model | Hyperparameters Tested |
|---|---|
| CNN | Conv channels: `(96,128,192,256)`, `(64,128,192,256)`, `(96,160,224,288)`, `(128,160,224,320)`; kernels: `(3,5,3,3)`, `(3,5,5,3)`, `(5,5,3,3)`; dropout: `0.30–0.40`; LR: `1e-3`, `5e-4`; weight decay: `1e-5` |
| LSTM | Hidden size: `128`, `256`; LSTM layers: `1`, `2`; dropout: `0.30–0.40`; LR: `1e-3`, `5e-4`; weight decay: `1e-5`; bidirectional attention-based LSTM |
| CNN + LSTM | CNN channels: `(96,128,192)`, `(96,160,224)`, `(128,160,224)`; hidden size: `128`, `256`; LSTM layers: `1`, `2`; dropout: `0.30–0.40`; LR: `1e-3`, `5e-4`; weight decay: `1e-5` |
| TCN | Projection channels: `128`, `160`; TCN channels: `(128,160,192,256)`, `(128,192,224,256)`, `(160,192,224,288)`; dilations: `(1,2,4,8)`; kernel sizes: `3`, `5`; dropout: `0.30–0.40`; LR: `1e-3`, `5e-4`; weight decay: `1e-5` |

#### Best Hyperparameter Configurations

The highest-performing configuration for each architecture after hyperparameter search was:

| Model | Best Configuration |
|---|---|
| CNN | Channels: `(64, 128, 192, 256)`; kernels: `(3, 5, 5, 3)`; dropout: `0.30`; learning rate: `0.001`; weight decay: `1e-5` |
| LSTM | Hidden size: `128`; LSTM layers: `2`; dropout: `0.35`; learning rate: `0.001`; weight decay: `1e-5` |
| CNN + LSTM | CNN channels: `(128, 160, 224)`; hidden size: `256`; LSTM layers: `1`; dropout: `0.35`; learning rate: `0.0005`; weight decay: `1e-5` |
| TCN | Projection channels: `160`; TCN channels: `(160, 192, 224, 288)`; dilations: `(1, 2, 4, 8)`; kernel size: `3`; dropout: `0.35`; learning rate: `0.0005`; weight decay: `1e-5` |

All models shared a common training pipeline:

- 5-Fold GroupKFold cross-validation grouped by source video
- Fold-specific StandardScaler normalization
- Pose-sequence augmentation for stronger generalization
- Weighted cross-entropy loss for class imbalance
- Adam optimizer with early stopping and LR scheduling
- Final retraining using the best CV-selected hyperparameters

### Pose-Sequence Augmentation

Pose-sequence augmentation was applied during training to improve robustness and generalization.

Augmentations included:
- Gaussian noise injection
- Temporal shifting
- Frame dropout
- Temporal masking
- Feature masking

### Evaluation Metrics

Models were evaluated using:
- Mean cross-validation accuracy
- Official held-out test accuracy
- Macro F1-score
- Weighted F1-score
- Confusion matrices
- Per-class classification reports

### Final Results

| Model | Mean CV Accuracy | Official Test Accuracy |
|---|---:|---:|
| CNN | 88.50% | 57.00% |
| LSTM | 87.31% | 59.40% |
| CNN + LSTM | 88.22% | 57.70% |
| TCN | 88.22% | 57.39% |

The LSTM architecture achieved the highest official test accuracy and was selected for final deployment within the iOS application. While CNN + LSTM and TCN achieved slightly stronger cross-validation performance, the LSTM model demonstrated the best generalization on the noisy real-world held-out test set while maintaining efficient CoreML deployment compatibility.

### Deployment

Final PyTorch models were:
- exported as CoreML `.mlpackage` models,
- integrated into the Swift iOS application,
- and used for real-time on-device workout classification.

The deployment pipeline included:
- MediaPipe landmark extraction,
- feature scaling using saved StandardScaler statistics,
- real-time sequence buffering,
- and CoreML inference directly on-device.

### Final Model Retraining

After cross-validation:
- the best-performing configuration was retrained on the full verified dataset
- augmented training data was regenerated
- a final scaler was fit on all training data
- the final model was exported to CoreML

Final outputs included:
- `.mlpackage`
- scaler JSON files
- label mappings

for deployment in the iOS application.


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

# Future Improvements

Potential future work includes:
- full repetition counting integration,
- exercise form evaluation,
- multi-person detection,
- workout tracking analytics,
- and transformer-based temporal architectures.

---

