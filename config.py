"""
Configuration settings for Sign Language Recognition System
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Database configuration
DATABASE_PATH = BASE_DIR / 'database' / 'sign_language.db'
SCHEMA_PATH = BASE_DIR / 'schema.sql'

# Data directories
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
SPLITS_DIR = DATA_DIR / 'splits'

# Model directories
MODELS_DIR = BASE_DIR / 'models'
TRAINED_MODEL_PATH = MODELS_DIR / 'trained_model.h5'

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ISL Alphabet labels (26 letters)
ISL_LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Data collection settings
SAMPLES_PER_GESTURE = 500
IMAGE_SIZE = (224, 224)
CAPTURE_DELAY = 0.1

# MediaPipe configuration
MEDIAPIPE_CONFIDENCE = 0.5
NUM_LANDMARKS = 21

# Model training hyperparameters - INCREASED FOR BETTER ACCURACY
BATCH_SIZE = 32
EPOCHS = 100  # Increased from 50 to 100 for better accuracy
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Model architecture
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = len(ISL_LABELS)

# Real-time recognition settings
CONFIDENCE_THRESHOLD = 0.7
PREDICTION_BUFFER_SIZE = 5

# Flask configuration
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
DEBUG_MODE = True
SECRET_KEY = 'your-secret-key-change-in-production'

# Upload settings
UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_SIZE = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Analytics configuration
CHART_COLORS = [
    '#9D4EDD', '#7209B7', '#560BAD', '#3C096C',
    '#10002B', '#240046', '#3A0CA3', '#4361EE',
    '#4CC9F0', '#F72585', '#B5179E', '#7209B7'
]
