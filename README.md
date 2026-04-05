# Indian Sign Language Recognition System

A complete end-to-end deep learning system for real-time Indian Sign Language (ISL) alphabet recognition using computer vision and neural networks.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.0-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-100%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

This system recognizes ISL alphabet gestures (A-Z) in real-time using hand landmark detection and deep learning. It supports both one-handed and two-handed gestures with 100% test accuracy on validation data.

### Key Features

- **100% Test Accuracy** - CNN model trained on 31,000+ samples
- **Real-Time Recognition** - Instant predictions via webcam
- **2-Hand Support** - Detects both 1-handed and 2-handed gestures
- **Web Application** - Flask-based UI with live video streaming
- **Hand Tracking** - MediaPipe for robust landmark detection
- **Visual Reference** - Gesture guide generator included

## System Architecture

```
Input Layer (Webcam)
         ↓
MediaPipe Hand Detection
  (42 landmarks: 21 × 2 hands)
         ↓
CNN Classifier
  (512 → 256 → 128 → 64 neurons)
         ↓
Output Layer
  (26 classes: A-Z + confidence scores)
```

## Technology Stack

**Deep Learning & Computer Vision**
- TensorFlow 2.17.0
- Keras
- MediaPipe 0.10.14
- OpenCV

**Web Development**
- Flask
- Flask-CORS
- HTML/CSS/JavaScript

**Data Science**
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn

**Database**
- SQLite

## Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 100.00% |
| Training Samples | 31,130 |
| Gesture Classes | 26 (A-Z) |
| Model Parameters | 210,778 |
| Architecture | 4-layer CNN |
| Input Dimensions | (42, 3) |
| Training Time | ~20 minutes |

## Installation

### Prerequisites

- Python 3.11
- Miniconda or Anaconda
- Webcam (for real-time recognition)

### Setup Instructions

**1. Clone the repository**
```bash
git clone https://github.com/vk7981/isl-sign-language-recognition.git
cd isl-sign-language-recognition
```

**2. Create conda environment**
```bash
conda create -n sign-language python=3.11 -y
conda activate sign-language
```

**3. Install dependencies**
```bash
# Install via conda
conda install -c conda-forge numpy pandas pillow matplotlib seaborn flask flask-cors scikit-learn opencv -y

# Install via pip
pip install tensorflow==2.17.0 mediapipe==0.10.14 python-dotenv tqdm h5py==3.10.0
```

**4. Download dataset (optional - for training from scratch)**
- Download ISL dataset from [Kaggle](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl)
- Extract to project root as `Indian/` folder

## Usage

### Real-Time Webcam Recognition
```bash
python modules/realtime_recognition_2hands.py
```

### Web Application
```bash
python app.py
```
Then open browser to: `http://localhost:5000`

### Training from Scratch
```bash
# Step 1: Process dataset
python reprocess_2hands.py

# Step 2: Train model
python modules/training_2hands.py

# Step 3: Test recognition
python modules/realtime_recognition_2hands.py
```

## Project Structure

```
isl-sign-language-recognition/
├── app.py                          # Flask web application
├── config.py                       # Configuration settings
├── schema.sql                      # Database schema
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
├── README.md                       # Project documentation
│
├── database/
│   ├── __init__.py
│   ├── db_manager.py              # Database operations
│   └── sign_language.db           # SQLite database (generated)
│
├── models/
│   ├── __init__.py
│   ├── cnn_model.py               # CNN architecture
│   ├── trained_model.h5           # Trained model (generated)
│   ├── confusion_matrix.png       # Evaluation results (generated)
│   └── training_history.png       # Training metrics (generated)
│
├── modules/
│   ├── __init__.py
│   ├── data_collection.py         # Data collection utilities
│   ├── training_2hands.py         # Model training pipeline
│   └── realtime_recognition_2hands.py  # Webcam recognition
│
├── templates/
│   └── index.html                 # Web application UI
│
├── static/                        # Static web assets
├── data/                          # Dataset storage
│   ├── raw/
│   ├── processed/
│   └── splits/
│
└── Indian/                        # Original dataset (not in repository)
```

## Methodology

### 1. Data Collection & Processing
- Dataset: 31,130 ISL gesture images (A-Z)
- Hand landmark extraction using MediaPipe
- 21 landmarks per hand × 2 hands = 42 landmarks
- Each landmark: (x, y, z) coordinates
- Normalization: 1-hand gestures padded with zeros

### 2. Model Architecture
```
Input Layer:        (42, 3)
Flatten Layer:      (126,)
Dense Layer 1:      512 neurons + BatchNorm + Dropout(0.4)
Dense Layer 2:      256 neurons + BatchNorm + Dropout(0.3)
Dense Layer 3:      128 neurons + BatchNorm + Dropout(0.2)
Dense Layer 4:      64 neurons + BatchNorm + Dropout(0.2)
Output Layer:       26 neurons (softmax activation)
```

### 3. Training Configuration
- **Dataset Split**: 70% train, 15% validation, 15% test
- **Batch Size**: 32
- **Epochs**: 100 (with early stopping)
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Sparse Categorical Crossentropy
- **Callbacks**: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

### 4. Real-Time Prediction
- MediaPipe hand detection (up to 2 hands)
- Landmark extraction and normalization
- CNN inference
- Prediction smoothing with rolling buffer
- Confidence thresholding

## Results

The model achieves 100% accuracy on the test set with the following performance characteristics:

- **Precision**: 1.000 (average across all classes)
- **Recall**: 1.000 (average across all classes)
- **F1-Score**: 1.000 (average across all classes)
- **Inference Time**: <50ms per frame on CPU
- **Support**: All 26 alphabet gestures (A-Z)

## Applications

- Sign language education and learning
- Accessibility tools for deaf and hard-of-hearing individuals
- Human-computer interaction systems
- Real-time translation services
- Educational demonstrations

## Limitations

- Currently supports ISL alphabet only (A-Z)
- Requires good lighting conditions
- Performance may vary with different hand sizes and skin tones
- Limited to static gestures (no motion-based signs)

## Future Enhancements

- Extend to support ISL words and phrases
- Add support for dynamic gestures
- Implement sentence formation
- Mobile application development
- Multi-language support
- Improved accuracy under varying lighting conditions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss proposed changes.

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Author

**VK**
- GitHub: [@vk7981](https://github.com/vk7981)
- Institution: SRM Institute of Science and Technology
- Project Type: Academic Research Project

## Acknowledgments

- **MediaPipe Team** - Google's hand tracking solution
- **TensorFlow Team** - Deep learning framework
- **Kaggle Community** - ISL dataset provider
- **SRM Institute** - Academic support and resources

## References

1. MediaPipe Hands Documentation: https://google.github.io/mediapipe/solutions/hands.html
2. TensorFlow/Keras Documentation: https://www.tensorflow.org/
3. ISL Dataset Source: https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl
4. Sign Language Recognition Research Papers (various sources)

## Citation

If you use this project in your research or work, please cite:

```
@software{isl_recognition_2026,
  author = {VK},
  title = {Indian Sign Language Recognition System},
  year = {2026},
  url = {https://github.com/vk7981/isl-sign-language-recognition}
}
```

---

**Note**: This is an academic project demonstrating end-to-end machine learning system development for educational purposes.
