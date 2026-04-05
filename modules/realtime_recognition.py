"""
Real-Time Sign Language Recognition using Webcam
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow import keras
import config


class RealTimeRecognizer:
    """Real-time ISL recognition from webcam"""
    
    def __init__(self, model_path: str):
        # Load trained model
        print("Loading trained model...")
        self.model = keras.models.load_model(model_path)
        print("✓ Model loaded successfully")
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=config.MEDIAPIPE_CONFIDENCE,
            min_tracking_confidence=config.MEDIAPIPE_CONFIDENCE
        )
        
        # Prediction smoothing buffer
        self.prediction_buffer = deque(maxlen=config.PREDICTION_BUFFER_SIZE)
        
        # Stats
        self.frame_count = 0
        self.prediction_count = 0
    
    def extract_landmarks(self, frame):
        """Extract hand landmarks from frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks), hand_landmarks
        
        return None, None
    
    def predict_gesture(self, landmarks):
        """Predict gesture from landmarks"""
        # Reshape for model input
        landmarks_reshaped = landmarks.reshape(1, 21, 3)
        
        # Get prediction
        predictions = self.model.predict(landmarks_reshaped, verbose=0)[0]
        
        # Get top prediction
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        predicted_label = config.ISL_LABELS[predicted_idx]
        
        return predicted_label, confidence, predictions
    
    def get_smoothed_prediction(self, current_prediction, confidence):
        """Smooth predictions using buffer"""
        if confidence >= config.CONFIDENCE_THRESHOLD:
            self.prediction_buffer.append(current_prediction)
        
        if len(self.prediction_buffer) > 0:
            # Most common prediction in buffer
            most_common = max(set(self.prediction_buffer), key=self.prediction_buffer.count)
            return most_common
        
        return None
    
    def draw_ui(self, frame, prediction, confidence, predictions, hand_landmarks):
        """Draw UI overlay on frame"""
        height, width, _ = frame.shape
        
        # Draw hand landmarks
        if hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Background for prediction display
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        
        # Display prediction
        if prediction:
            # Main prediction
            cv2.putText(frame, f"Prediction: {prediction}", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # Confidence
            cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Confidence bar
            bar_width = int((width - 40) * confidence)
            cv2.rectangle(frame, (20, 100), (width - 20, 120), (50, 50, 50), -1)
            
            # Color based on confidence
            if confidence >= 0.9:
                color = (0, 255, 0)  # Green
            elif confidence >= 0.7:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 165, 255)  # Orange
            
            cv2.rectangle(frame, (20, 100), (20 + bar_width, 120), color, -1)
            
            # Top 3 predictions
            top_3_idx = np.argsort(predictions)[-3:][::-1]
            y_pos = 145
            
            cv2.putText(frame, "Top 3:", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            for idx in top_3_idx:
                label = config.ISL_LABELS[idx]
                conf = predictions[idx] * 100
                cv2.putText(frame, f"{label}: {conf:.1f}%", (120, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                y_pos += 25
        else:
            cv2.putText(frame, "No hand detected", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'Q' to quit | 'S' to save screenshot", 
                   (20, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Stats
        cv2.putText(frame, f"FPS: {self.frame_count}", 
                   (width - 150, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Run real-time recognition"""
        print("\n" + "="*60)
        print("REAL-TIME ISL RECOGNITION")
        print("="*60)
        print("Instructions:")
        print("  • Show your hand gesture to the camera")
        print("  • Press 'Q' to quit")
        print("  • Press 'S' to save screenshot")
        print("="*60 + "\n")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("✓ Webcam opened. Starting recognition...\n")
        
        screenshot_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            self.frame_count += 1
            
            # Extract landmarks
            landmarks, hand_landmarks = self.extract_landmarks(frame)
            
            prediction = None
            confidence = 0.0
            predictions = None
            
            if landmarks is not None:
                # Predict gesture
                prediction, confidence, predictions = self.predict_gesture(landmarks)
                
                # Get smoothed prediction
                smoothed = self.get_smoothed_prediction(prediction, confidence)
                if smoothed:
                    prediction = smoothed
                
                self.prediction_count += 1
            
            # Draw UI
            frame = self.draw_ui(frame, prediction, confidence, predictions, hand_landmarks)
            
            # Display
            cv2.imshow('ISL Recognition - Press Q to Quit', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n✓ Quitting...")
                break
            
            elif key == ord('s'):
                # Save screenshot
                screenshot_path = config.MODELS_DIR / f'screenshot_{screenshot_count}.jpg'
                cv2.imwrite(str(screenshot_path), frame)
                print(f"✓ Screenshot saved: {screenshot_path}")
                screenshot_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("SESSION COMPLETE")
        print("="*60)
        print(f"Total frames: {self.frame_count}")
        print(f"Predictions made: {self.prediction_count}")
        print("="*60 + "\n")
    
    def cleanup(self):
        """Clean up resources"""
        self.hands.close()


def main():
    model_path = str(config.TRAINED_MODEL_PATH)
    
    if not Path(model_path).exists():
        print(f"❌ Error: Trained model not found at {model_path}")
        print("Please train the model first using modules/training.py")
        return
    
    recognizer = RealTimeRecognizer(model_path)
    
    try:
        recognizer.run()
    finally:
        recognizer.cleanup()


if __name__ == "__main__":
    main()
