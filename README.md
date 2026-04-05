# 🤟 Indian Sign Language Recognition System

A complete end-to-end deep learning system for real-time Indian Sign Language (ISL) alphabet recognition using computer vision and neural networks.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.0-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-100%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Project Overview

This system recognizes ISL alphabet gestures (A-Z) in real-time using hand landmark detection and deep learning. It supports both one-handed and two-handed gestures with 100% test accuracy.

### ✨ Key Features

- ✅ **100% Test Accuracy** - CNN model trained on 31,000+ samples
- ✅ **Real-Time Recognition** - Instant predictions via webcam
- ✅ **2-Hand Support** - Detects both 1-handed and 2-handed gestures
- ✅ **Web Application** - Beautiful Flask-based UI
- ✅ **Hand Tracking** - MediaPipe for robust landmark detection
- ✅ **Visual Reference** - Gesture guide generator included

## 🏗️ Architecture

```
┌─────────────────┐
│  Webcam Input   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   MediaPipe     │  ← Hand Landmark Detection
│  (42 landmarks) │     (21 points × 2 hands)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   CNN Model     │  ← Deep Learning Classifier
│  512→256→128→64 │     (100% accuracy)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Prediction     │  → A-Z Classification
│  (A-Z + conf)   │
└─────────────────┘
```

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: MediaPipe, OpenCV
- **Web Framework**: Flask, Flask-CORS
- **Data Science**: NumPy, Pandas, scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Database**: SQLite

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | **100.00%** |
| Training Samples | 31,130 |
| Gestures | 26 (A-Z) |
| Parameters | 210,778 |
| Architecture | CNN (4 layers) |
| Input Shape | (42, 3) |

## 🚀 Quick Start

### Prerequisites

- Python 3.11
- Miniconda/Anaconda
- Webcam

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/isl-recognition.git
cd isl-recognition
```

2. **Create conda environment**
```bash
conda create -n sign-language python=3.11 -y
conda activate sign-language
```

3. **Install dependencies**
```bash
conda install -c conda-forge numpy pandas pillow matplotlib seaborn flask flask-cors scikit-learn opencv -y
pip install tensorflow==2.17.0 mediapipe==0.10.14 python-dotenv tqdm
```

4. **Download dataset** (if training from scratch)
- Download ISL dataset from [Kaggle](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl)
- Place in project root as `Indian/` folder

### Usage

#### Option 1: Real-Time Recognition
```bash
python modules/realtime_recognition_2hands.py
```

#### Option 2: Web Application
```bash
python app.py
# Then open: http://localhost:5000
```

#### Option 3: Train from Scratch
```bash
# 1. Process dataset
python reprocess_2hands.py

# 2. Train model
python modules/training_2hands.py

# 3. Test recognition
python modules/realtime_recognition_2hands.py
```

## 📁 Project Structure

```
isl-recognition/
├── app.py                          # Flask web application
├── config.py                       # Configuration settings
├── schema.sql                      # Database schema
│
├── database/
│   ├── db_manager.py              # Database operations
│   └── sign_language.db           # SQLite database (generated)
│
├── models/
│   ├── cnn_model.py               # CNN architecture
│   └── trained_model.h5           # Trained model (generated)
│
├── modules/
│   ├── data_collection.py         # Data collection tools
│   ├── training_2hands.py         # Training pipeline
│   └── realtime_recognition_2hands.py  # Webcam recognition
│
├── templates/
│   └── index.html                 # Web UI
│
├── static/                        # Static files
├── data/                          # Dataset storage
└── Indian/                        # Original dataset (not in repo)
```

## 🎓 How It Works

### 1. Hand Landmark Detection
- MediaPipe detects up to 2 hands in frame
- Extracts 21 landmarks per hand (42 total)
- Each landmark has (x, y, z) coordinates

### 2. Deep Learning Classification
- CNN processes 42×3 landmark coordinates
- Architecture: Dense layers with BatchNorm & Dropout
- Outputs probability distribution over 26 classes (A-Z)

### 3. Real-Time Prediction
- Smoothing buffer for stable predictions
- Confidence thresholding
- Top-N prediction display

## 📈 Training Details

- **Dataset**: 31,130 ISL gesture images
- **Split**: 70% train, 15% validation, 15% test
- **Batch Size**: 32
- **Epochs**: 100 (with early stopping)
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Sparse Categorical Crossentropy
- **Callbacks**: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 👨‍💻 Author

**VK**
- GitHub: [@yourusername](https://github.com/yourusername)
- Institution: SRM Institute of Science and Technology

## 🙏 Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) for hand tracking
- [TensorFlow](https://www.tensorflow.org/) for deep learning framework
- [Kaggle ISL Dataset](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl) for training data

---

⭐ **If you found this project helpful, please give it a star!**
