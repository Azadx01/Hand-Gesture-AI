# 🖐 Hand Gesture AI – Virtual Mouse Controller

> **B.Tech CSE AI/ML Portfolio Project** | Python · OpenCV · MediaPipe · Scikit-learn · CustomTkinter

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10-green?logo=opencv)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange)](https://mediapipe.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📋 Project Abstract

This project implements a **real-time AI-powered hand gesture recognition system** that replaces a physical mouse with natural hand gestures captured via a standard webcam. The system uses **MediaPipe** for 21-point hand landmark detection, a **Random Forest / SVM** classifier trained on user-collected gesture data, and **PyAutoGUI** for safe desktop automation.

---

## 🎯 Problem Statement

Conventional mouse input requires a physical device and a flat surface, creating accessibility barriers and ergonomic discomfort. This project enables **touchless, intuitive human-computer interaction** through computer vision and machine learning, with applications in accessibility, presentations, and sterile environments.

---

## 🏗️ System Architecture

```
Webcam Frame
     │
     ▼
┌─────────────┐     ┌──────────────────┐     ┌────────────────────┐
│ Camera      │────▶│  HandTracker     │────▶│ Feature Extraction │
│ (camera.py) │     │ (hand_tracker.py)│     │(feature_extraction)│
└─────────────┘     └──────────────────┘     └────────────────────┘
                           │                          │
                           │                          ▼
                           │               ┌────────────────────────┐
                           │               │  Gesture Classifier    │
                           │               │ (gesture_classifier.py)│
                           │               │  Random Forest / SVM   │
                           │               └────────────────────────┘
                           │                          │
                           ▼                          ▼
                    ┌─────────────┐     ┌──────────────────────────┐
                    │  Desktop    │     │   Mouse Controller        │
                    │  GUI (app)  │◀────│   (mouse_controller.py)  │
                    └─────────────┘     └──────────────────────────┘
```

---

## 🚀 Objectives

1. Detect and track one hand in real time using MediaPipe's 21-landmark model
2. Collect a labelled dataset of 7 hand gestures using a custom webcam script
3. Train and compare Random Forest and SVM classifiers with full evaluation metrics
4. Map hand gestures to mouse actions (move, click, scroll, drag, right-click)
5. Provide a professional desktop GUI for controlling all features
6. Ensure safety with fail-safes, debouncing, and emergency stop

---

## 📁 Folder Structure

```
HandGestureAI/
├── .venv/                     ← Python virtual environment
├── data/
│   └── gestures.csv           ← Collected training samples
├── models/
│   ├── gesture_model.joblib   ← Trained Random Forest model
│   └── gesture_model_svm.joblib
├── src/
│   ├── __init__.py
│   ├── camera.py              ← Webcam capture & FPS tracking
│   ├── hand_tracker.py        ← MediaPipe 21-landmark detection
│   ├── feature_extraction.py  ← Landmark → 47-dim feature vector
│   ├── data_collection.py     ← Interactive dataset collection
│   ├── train_model.py         ← ML training, evaluation, saving
│   ├── gesture_classifier.py  ← Real-time inference + smoothing
│   ├── mouse_controller.py    ← Safe PyAutoGUI mouse automation
│   └── app.py                 ← CustomTkinter desktop GUI
├── tests/
│   ├── __init__.py
│   ├── test_features.py       ← Feature extraction unit tests
│   └── test_gesture_logic.py  ← Classifier & mouse logic tests
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10–3.12 | Core language |
| OpenCV | 4.10.0 | Frame capture, image processing |
| MediaPipe | 0.10.14 | 21-point hand landmark tracking |
| NumPy | 1.26.4 | Numerical operations |
| Scikit-learn | 1.5.2 | Random Forest, SVM, evaluation |
| PyAutoGUI | 0.9.54 | Safe mouse automation |
| Pandas | 2.2.3 | Dataset management |
| Joblib | 1.4.2 | Model serialization |
| CustomTkinter | 5.2.2 | Modern dark-mode GUI |
| Matplotlib | 3.9.2 | Confusion matrix plots |
| Pillow | 10.4.0 | Image format conversion for GUI |

---

## ⚙️ Installation – Windows CMD (Copy-Paste Ready)

### Step 1 – Check Python
```cmd
python --version
python -c "import sys; print(sys.executable)"
```
> **Required**: Python 3.10, 3.11, or 3.12 (64-bit). MediaPipe 0.10.x does not support Python 3.13+.
> Download from https://www.python.org/downloads/release/python-3120/

### Step 2 – Create Project Directory
```cmd
cd %USERPROFILE%\Desktop
mkdir HandGestureAI
cd HandGestureAI
```

### Step 3 – Create & Activate Virtual Environment
```cmd
python -m venv .venv
.venv\Scripts\activate
```
You should see `(.venv)` at the start of your prompt.

### Step 4 – Install All Required Packages
```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

Expected time: 2–5 minutes depending on internet speed.

### Step 5 – Open in VS Code
```cmd
code .
```

### Step 6 – Verify Installation
```cmd
python -c "import cv2, mediapipe, sklearn, pyautogui, customtkinter; print('All packages OK')"
```

---

## 📸 Dataset Collection

Collect **200+ samples per gesture** for best accuracy.

### Collect each gesture (run once per gesture):
```cmd
python src/data_collection.py --gesture MOVE        --samples 200
python src/data_collection.py --gesture CLICK        --samples 200
python src/data_collection.py --gesture RIGHT_CLICK  --samples 200
python src/data_collection.py --gesture SCROLL       --samples 200
python src/data_collection.py --gesture PAUSE        --samples 200
python src/data_collection.py --gesture DRAG         --samples 200
python src/data_collection.py --gesture IDLE         --samples 200
```

**In the collection window:**
- Position your hand for the gesture shown
- Press **SPACE** to start/pause recording
- Press **Q** to quit early
- Data is appended to `data/gestures.csv` automatically

### Gesture Guide:

| Gesture | Hand Pose |
|---|---|
| **MOVE** | Point index finger up, all others curled |
| **CLICK** | Pinch thumb + index fingertips together |
| **RIGHT_CLICK** | Pinch thumb + middle fingertips together |
| **SCROLL** | Index + middle fingers extended (peace sign), move hand up/down |
| **PAUSE** | Closed fist |
| **DRAG** | Index + pinky extended (rock-on sign) |
| **IDLE** | Open palm, relaxed |

---

## 🧠 ML Training Workflow

### Train and evaluate:
```cmd
python src/train_model.py
```

**Expected output:**
```
Hand Gesture AI – Model Training
============================================================
Sample counts per class:
MOVE          210
CLICK         205
RIGHT_CLICK   200
SCROLL        198
PAUSE         212
DRAG          201
IDLE          195

Dataset loaded: 1421 samples, 47 features, 7 classes

Split → Train: 1001, Val: 214, Test: 206

============================================================
  Training Random Forest ...
  [Random Forest]  Val Acc: 0.9720   Test Acc: 0.9660

  Classification Report (test set):
                precision    recall  f1-score   support
  CLICK            0.97      0.95      0.96        30
  DRAG             0.96      0.97      0.96        29
  IDLE             0.99      1.00      0.99        30
  MOVE             0.96      0.97      0.96        30
  PAUSE            0.97      0.97      0.97        30
  RIGHT_CLICK      0.95      0.93      0.94        28
  SCROLL           0.97      0.96      0.96        29

  5-fold CV Accuracy: 0.9701 ± 0.0088

Models saved:
  Random Forest → models/gesture_model.joblib
  SVM           → models/gesture_model_svm.joblib
```

> ⚠️ Actual accuracy values will depend on the quality and quantity of your collected samples. The above is a representative example.

---

## 🖥️ Starting the Application
```cmd
python src/app.py
```

---

## 🖱️ Using Each Gesture

| Gesture | Action |
|---|---|
| ☝️ **MOVE** | Move cursor with index fingertip |
| 👌 **CLICK** | Left-click (pinch index+thumb) |
| 🤏 **RIGHT_CLICK** | Right-click menu |
| ✌️ **SCROLL** | Scroll page up/down |
| ✊ **PAUSE** | Freeze mouse control |
| 🤘 **DRAG** | Hold + drag files/windows |
| 🤚 **IDLE** | No action |

**Tip:** Keep your hand 30–60 cm from the webcam, well-lit, and within the center frame for best detection.

---

## 🧪 Running Tests
```cmd
pip install pytest
python -m pytest tests/ -v
```

Expected output:
```
tests/test_features.py::TestExtractFeatures::test_returns_correct_shape PASSED
tests/test_features.py::TestExtractFeatures::test_dtype_is_float32 PASSED
tests/test_features.py::TestExtractFeatures::test_none_on_wrong_count PASSED
...
tests/test_gesture_logic.py::TestGestureClassifier::test_unloaded_predicts_idle PASSED
...
=================== 20 passed in 2.14s ===================
```

---

## 🔬 Feature Engineering

The feature vector (47 dimensions) is derived from MediaPipe's 21 normalised landmarks:

```
Feature Index | Description
0–41          | (x, y) of all 21 landmarks, wrist-centred + scale-normalised
42            | Thumb extended (1.0) or curled (0.0)
43            | Index extended bit
44            | Middle extended bit
45            | Ring extended bit
46            | Pinky extended bit
```

**Normalisation steps:**
1. Subtract wrist (landmark 0) position → translation-invariant
2. Divide by wrist→middle-MCP distance → scale-invariant

---

## 🚨 Troubleshooting

### Camera won't open
```cmd
python -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened())"
```
- Close Zoom, Teams, or other apps using the webcam
- Try `Camera(index=1)` if you have multiple cameras

### MediaPipe installation fails (Python 3.13+)
```cmd
python --version
# If 3.13+, install Python 3.12 from python.org
# Then recreate .venv with that interpreter
py -3.12 -m venv .venv
```

### `No module named 'src'` error
```cmd
cd HandGestureAI   # Must be run from the project root
python src/app.py
```

### PyAutoGUI FailSafe triggered
Move the mouse away from the top-left screen corner. The fail-safe is triggered when the cursor reaches pixel (0, 0).

### Model accuracy is low (< 85%)
- Collect more samples (300+ per class)
- Ensure consistent lighting and hand distance
- Vary hand angles slightly during collection for robustness

### GUI appears frozen during training
Training runs in a background thread. The progress is printed to the terminal. Wait until "Training complete!" appears in the status log.

---

## 📊 Model Evaluation

Confusion matrix plots are saved to `models/` after training:
- `models/confusion_matrix_Random_Forest.png`
- `models/confusion_matrix_SVM.png`

---

## 🔒 Safety & Ethics

- **PyAutoGUI fail-safe is always enabled** – move mouse to screen corner to abort
- **ESC key** triggers emergency stop from the GUI
- Click debouncing (600ms default) prevents accidental triggers
- No webcam images are saved unless you explicitly enable it
- Model files should be loaded only from trusted sources
- Never disable cooldowns for automation of safety-critical systems

---

## 🚀 Future Enhancements

- [ ] Two-hand gesture recognition
- [ ] Custom gesture definition via GUI (no code required)
- [ ] Voice feedback for gesture confirmation
- [ ] Presentation slide control mode
- [ ] Gesture history timeline in the GUI
- [ ] ONNX model export for cross-platform deployment
- [ ] Hand landmark visualizer with 3D view

---

## ⚠️ Limitations

- Works best in good lighting conditions (avoid strong backlighting)
- Single-hand detection only (first detected hand is used)
- MediaPipe 0.10.x requires Python 3.10–3.12 (not 3.13+)
- Accuracy depends on dataset quality and quantity
- Latency is typically 20–40ms on a modern CPU

---

## 👨‍💻 Author

**B.Tech CSE AI/ML Student Portfolio Project**
Built for demonstrating: Computer Vision · Supervised ML · Real-time Systems · Desktop Automation
#   H a n d - G e s t u r e - A I  
 