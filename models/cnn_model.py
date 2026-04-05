"""
CNN Model for ISL Recognition - TWO HAND SUPPORT
Input: (42, 3) = 2 hands × 21 landmarks × 3 coordinates
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path
from typing import Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class SignLanguageCNN:
    """CNN model with 2-hand support"""
    
    def __init__(self, input_shape: Tuple = (42, 3), num_classes: int = config.NUM_CLASSES):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self) -> keras.Model:
        """Build model for 2-hand input"""
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Flatten(),
            
            # Larger architecture for 2-hand complexity
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate: float = config.LEARNING_RATE):
        """Compile model"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def get_callbacks(self, model_path: str):
        """Get callbacks"""
        return [
            ModelCheckpoint(filepath=model_path, monitor='val_accuracy',
                          save_best_only=True, mode='max', verbose=1),
            EarlyStopping(monitor='val_loss', patience=15,
                         restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                            patience=7, min_lr=1e-7, verbose=1)
        ]
    
    def summary(self):
        """Print summary"""
        if self.model:
            self.model.summary()
    
    def get_model(self) -> keras.Model:
        """Get model"""
        return self.model


def load_trained_model(model_path: str) -> keras.Model:
    """Load saved model"""
    return keras.models.load_model(model_path)
