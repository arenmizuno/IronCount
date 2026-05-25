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

# Dataset

Dataset used:

```text
philosopher0808/gym-workoutexercises-video
```

The dataset contains:
- verified/preprocessed exercise clips,
- crawled YouTube workout videos,
- noisy real-world test videos.

## Dataset Summary

| Split | Videos |
|---|---|
| Train | 1,571 |
| Test | 61 |
| Exercises | 22 |

Exercises include:
- squat
- push-up
- deadlift
- bench press
- shoulder press
- pull up
- leg raises
- and more.

---

# Pipeline Overview

## 1. Exploratory Data Analysis (EDA)

- Video metadata extraction
- FPS analysis
- Resolution analysis
- Duration distributions
- Class imbalance analysis
- Train/test split validation

## 2. Pose Extraction

MediaPipe Pose was used to extract:
- 33 body landmarks
- visibility scores
- normalized pose coordinates

Additional engineered features:
- joint angles
- body geometry features
- velocity features
- temporal sequences

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

