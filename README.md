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

---

### Training Strategy

All models shared a common training pipeline:

- 5-fold GroupKFold cross-validation
- StandardScaler normalization
- pose sequence augmentation
- weighted cross-entropy loss
- Adam optimization
- learning rate scheduling
- gradient clipping
- early stopping

The training process used:
- grouping by original video filepath to avoid sequence leakage between train/validation folds
- ReduceLROnPlateau learning rate scheduling
- gradient clipping (`max_norm = 1.0`) for training stability
- early stopping patience of 15 epochs

---

### Data Scaling & Augmentation

Pose features were standardized using `StandardScaler` across the feature dimension.

Augmentation techniques included:
- Gaussian noise injection
- temporal shifting
- frame duplication
- temporal masking
- feature masking

Each training sequence generated:
- 2 augmented copies

This increased robustness to:
- noisy pose extraction
- varying movement speed
- inconsistent recording conditions
- dropped frames

---

### CNN Model

The CNN-only model focused on extracting local temporal motion patterns using stacked Conv1D layers.

#### Architecture

The CNN architecture included:
- 4 Conv1D layers
- Batch Normalization
- ReLU activations
- Dropout regularization
- global average pooling
- global max pooling
- fully connected classifier head

#### CNN Depth

| Layer | Channels | Kernel Size |
|---|---|---|
| Conv1D | 96 | 3 |
| Conv1D | 128 | 5 |
| Conv1D | 192 | 3 |
| Conv1D | 256 | 3 |

Classifier head:
- 512 → 256
- 256 → 128
- 128 → num_classes

#### CNN Hyperparameters

| Hyperparameter | Value |
|---|---|
| Batch Size | 32 |
| Epochs | 80 |
| Final Retraining Epochs | 40 |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-5 |
| Dropout | 0.30 |

The CNN model emphasized:
- short-term motion pattern learning
- efficient inference
- lightweight deployment

---

### LSTM + Attention Model

The LSTM model focused on learning long-term temporal exercise dynamics.

#### Architecture

The model used:
- bidirectional LSTM layers
- Layer Normalization
- temporal attention mechanism
- fully connected classifier

#### LSTM Hyperparameters

| Hyperparameter | Value |
|---|---|
| Hidden Size | 128 |
| Number of Layers | 1 |
| Bidirectional | Yes |
| Batch Size | 32 |
| Epochs | 80 |
| Dropout | 0.30 |

#### Attention Mechanism

An attention layer learned:
- which frames were most important
- where exercise-defining motion occurred

This improved:
- temporal focus
- robustness to noisy frames
- sequence interpretability

---

### CNN + LSTM + Attention Hybrid Model

The hybrid CNN-LSTM architecture combined:
- local spatial-temporal feature extraction
- long-term temporal modeling
- attention-based frame weighting

#### Architecture Pipeline

```text
Conv1D Blocks
→ Bidirectional LSTM
→ Attention Layer
→ Fully Connected Classifier
```

#### CNN Feature Extractor

| Layer | Channels |
|---|---|
| Conv1D | 96 |
| Conv1D | 128 |
| Conv1D | 192 |

#### LSTM Component

| Parameter | Value |
|---|---|
| Hidden Size | 128 |
| Layers | 1 |
| Bidirectional | Yes |

#### Classifier Head

```text
256 → 128 → num_classes
```

This hybrid model attempted to:
- capture short-term motion cues with CNNs
- model exercise flow with LSTMs
- emphasize important frames using attention

---

### Temporal Convolutional Network (TCN)

The TCN model used dilated temporal convolutions and residual connections to model long-range motion dependencies.

#### TCN Architecture

The TCN included:
- input projection Conv1D
- residual TCN blocks
- dilated convolutions
- BatchNorm
- dropout
- residual skip connections

#### Dilated Convolution Blocks

| Block | Channels | Dilation |
|---|---|---|
| TCN Block 1 | 128 | 1 |
| TCN Block 2 | 160 | 2 |
| TCN Block 3 | 192 | 4 |
| TCN Block 4 | 256 | 8 |

Increasing dilation allowed the model to:
- capture larger temporal receptive fields
- model exercise motion over longer time horizons
- reduce sequential bottlenecks seen in RNNs

#### TCN Advantages

Compared to LSTMs:
- more parallelizable
- faster training
- stable gradients
- efficient long-range sequence modeling

The TCN also used:
- global average pooling
- global max pooling
- dense classifier head

---

### Optimization & Evaluation

All models used:

| Component | Configuration |
|---|---|
| Optimizer | Adam |
| LR Scheduler | ReduceLROnPlateau |
| Loss Function | Weighted CrossEntropy |
| Gradient Clipping | 1.0 |
| Early Stopping | Patience = 15 |
| Validation | 5-Fold GroupKFold |

Evaluation metrics included:
- validation accuracy
- macro F1-score
- weighted F1-score
- classification reports

---

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

---

### Model Performance

Final models were evaluated on the official noisy test split.

| Model | Test Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| CNN | 54.52% | 52.36% | 54.18% |
| LSTM + Attention | 59.40% | 57.00% | 58.57% |
| CNN + LSTM + Attention | 57.31% | 55.25% | 57.40% |
| TCN | 58.70% | 55.85% | 57.64% |

The LSTM + Attention model achieved the strongest overall test performance among the completed model runs, with the highest official test accuracy and macro F1-score.

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

