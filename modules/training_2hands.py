"""
Training Pipeline for ISL Sign Language Recognition - 2 HAND SUPPORT
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

import config
from database.db_manager import db

# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


class ModelTrainer:
    """Handles model training pipeline"""
    
    def __init__(self):
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.model = None
        self.history = None
    
    def load_dataset_from_db(self):
        """Load dataset from database"""
        print("\n" + "="*60)
        print("LOADING DATASET FROM DATABASE")
        print("="*60)
        
        stats = db.get_dataset_statistics()
        
        print("\nDataset statistics:")
        for label, count in sorted(stats.items()):
            print(f"  {label}: {count} samples")
        
        total = sum(stats.values())
        print(f"\nTotal samples: {total}")
        
        # Load all gestures
        X = []
        y = []
        
        print("\nLoading landmarks...")
        for label in config.ISL_LABELS:
            gestures = db.get_gestures_by_label(label)
            
            for gesture in gestures:
                if gesture['landmarks']:
                    X.append(gesture['landmarks'])
                    y.append(config.ISL_LABELS.index(label))
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✓ Loaded {len(X)} samples")
        print(f"  Shape: {X.shape}")
        print(f"  Labels shape: {y.shape}")
        
        return X, y
    
    def split_dataset(self, X, y):
        """Split dataset into train/val/test"""
        print("\n" + "="*60)
        print("SPLITTING DATASET")
        print("="*60)
        
        # Split: 70% train, 15% val, 15% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=config.RANDOM_SEED, stratify=y
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=config.RANDOM_SEED, stratify=y_temp
        )
        
        print(f"Training set:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"Test set:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def build_model_2hands(self):
        """Build CNN model for 2-hand input - FORCED (42, 3)"""
        print("\n" + "="*60)
        print("BUILDING MODEL - 2 HAND SUPPORT")
        print("="*60)
        print(f"Input shape: (42, 3) = 42 landmarks × 3 coordinates")
        print(f"Output classes: 26 (A-Z)")
        
        model = models.Sequential([
            layers.Input(shape=(42, 3)),  # FORCED 2-hand input
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
            
            layers.Dense(26, activation='softmax')
        ])
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nModel architecture:")
        model.summary()
        
        self.model = model
        return model
    
    def get_callbacks(self):
        """Get training callbacks"""
        callbacks = [
            ModelCheckpoint(
                filepath=str(config.TRAINED_MODEL_PATH),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_model(self):
        """Train the model"""
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        print(f"Epochs: {config.EPOCHS}")
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"Learning rate: {config.LEARNING_RATE}")
        print("="*60 + "\n")
        
        callbacks = self.get_callbacks()
        
        start_time = time.time()
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Training time: {training_time/60:.2f} minutes")
        print(f"Model saved to: {config.TRAINED_MODEL_PATH}")
        
        return training_time
    
    def evaluate_model(self):
        """Evaluate model on test set"""
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)
        
        # Evaluate
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        
        # Predictions
        y_pred = np.argmax(self.model.predict(self.X_test, verbose=0), axis=1)
        
        # Classification report
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        
        report = classification_report(
            self.y_test, y_pred,
            target_names=config.ISL_LABELS,
            digits=3
        )
        print(report)
        
        # Confusion matrix
        self.plot_confusion_matrix(self.y_test, y_pred)
        self.plot_training_history()
        
        return test_acc
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=config.ISL_LABELS,
            yticklabels=config.ISL_LABELS,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - ISL Alphabet Recognition (2-Hand Support)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        save_path = config.MODELS_DIR / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Confusion matrix saved to: {save_path}")
        plt.close()
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Train')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy (2-Hand Support)')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Train')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss (2-Hand Support)')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = config.MODELS_DIR / 'training_history.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training history saved to: {save_path}")
        plt.close()


def main():
    print("\n" + "="*60)
    print("ISL SIGN LANGUAGE RECOGNITION - 2-HAND SUPPORT")
    print("="*60)
    
    trainer = ModelTrainer()
    
    # Step 1: Load dataset
    X, y = trainer.load_dataset_from_db()
    
    # Step 2: Split dataset
    trainer.split_dataset(X, y)
    
    # Step 3: Build model
    trainer.build_model_2hands()
    
    # Step 4: Train
    print("\nReady to train?")
    input("Press ENTER to start training...")
    
    training_time = trainer.train_model()
    
    # Step 5: Evaluate
    test_acc = trainer.evaluate_model()
    
    print("\n" + "="*60)
    print("✅ ALL DONE!")
    print("="*60)
    print(f"Model saved: {config.TRAINED_MODEL_PATH}")
    print(f"Test accuracy: {test_acc*100:.2f}%")
    print("\nNext steps:")
    print("  1. Test real-time recognition with webcam")
    print("  2. Deploy Flask web application")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
