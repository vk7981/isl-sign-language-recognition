"""
Data Collection Module
"""
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from datetime import datetime
import uuid
from typing import Optional, Tuple, List
import sys

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from database.db_manager import db


class DataCollector:
    """Handles data collection from webcam"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=config.MEDIAPIPE_CONFIDENCE,
            min_tracking_confidence=config.MEDIAPIPE_CONFIDENCE
        )
        
        self.session_id = None
        self.current_label = None
        self.samples_collected = 0
    
    def start_session(self, label: str) -> str:
        """Start a new collection session"""
        self.session_id = str(uuid.uuid4())
        self.current_label = label
        self.samples_collected = 0
        
        gesture_dir = config.RAW_DATA_DIR / label
        gesture_dir.mkdir(parents=True, exist_ok=True)
        
        return self.session_id
    
    def capture_frame(self, frame: np.ndarray, save: bool = True) -> Tuple[bool, Optional[List], Optional[str]]:
        """Process and save a frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            
            if save and self.current_label:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{self.current_label}_{timestamp}.jpg"
                image_path = config.RAW_DATA_DIR / self.current_label / filename
                
                cv2.imwrite(str(image_path), frame)
                
                db.insert_gesture(
                    label=self.current_label,
                    image_path=str(image_path),
                    landmarks=landmarks,
                    session_id=self.session_id
                )
                
                self.samples_collected += 1
                
                return True, landmarks, str(image_path)
            
            return True, landmarks, None
        
        return False, None, None
    
    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """Draw hand landmarks on frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return frame
    
    def add_text_overlay(self, frame: np.ndarray, label: str, count: int, target: int) -> np.ndarray:
        """Add status overlay"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        cv2.putText(frame, f"Gesture: {label}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Collected: {count}/{target}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        progress = min(count / target, 1.0)
        bar_width = int(360 * progress)
        cv2.rectangle(frame, (20, 90), (380, 110), (100, 100, 100), -1)
        cv2.rectangle(frame, (20, 90), (20 + bar_width, 110), (0, 255, 0), -1)
        
        return frame
    
    def collect_gesture_dataset(self, label: str, target_samples: int = config.SAMPLES_PER_GESTURE):
        """Collect dataset for a gesture"""
        self.start_session(label)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"\nStarting collection for gesture '{label}'")
        print(f"Target: {target_samples} samples")
        print("Press 'SPACE' to capture, 'Q' to quit\n")
        
        while self.samples_collected < target_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            success, landmarks, _ = self.capture_frame(frame, save=False)
            
            if success:
                frame = self.draw_landmarks(frame)
            
            frame = self.add_text_overlay(frame, label, self.samples_collected, target_samples)
            
            cv2.putText(frame, "Press SPACE to capture", (20, 460),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow(f'Collecting Gesture: {label}', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                if success:
                    self.capture_frame(frame, save=True)
                    print(f"Captured sample {self.samples_collected}/{target_samples}")
                else:
                    print("No hand detected! Please show your hand clearly.")
            
            elif key == ord('q'):
                print("\nCollection stopped by user")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nCollection complete for '{label}'")
        print(f"Total samples collected: {self.samples_collected}")
        
        return self.samples_collected
    
    def cleanup(self):
        """Clean up resources"""
        self.hands.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect ISL gesture dataset")
    parser.add_argument('--label', type=str, help='Gesture to collect (A-Z)')
    parser.add_argument('--samples', type=int, default=config.SAMPLES_PER_GESTURE,
                       help='Number of samples')
    
    args = parser.parse_args()
    
    collector = DataCollector()
    
    try:
        if args.label:
            if args.label.upper() not in config.ISL_LABELS:
                print(f"Error: Invalid label '{args.label}'. Must be A-Z.")
            else:
                collector.collect_gesture_dataset(args.label.upper(), args.samples)
        else:
            print("Error: Please specify --label")
            parser.print_help()
    
    finally:
        collector.cleanup()
