"""
Flask Web Application for ISL Sign Language Recognition - 2 HAND SUPPORT
"""
from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
import config

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = config.SECRET_KEY

# Load model
print("Loading trained model...")
model = keras.models.load_model(str(config.TRAINED_MODEL_PATH))
print("✓ Model loaded (2-hand support)")

# Initialize MediaPipe for 2 hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # DETECT UP TO 2 HANDS
    min_detection_confidence=config.MEDIAPIPE_CONFIDENCE,
    min_tracking_confidence=config.MEDIAPIPE_CONFIDENCE
)

# Global state
current_prediction = {"label": None, "confidence": 0.0, "num_hands": 0}


def extract_landmarks(frame):
    """Extract hand landmarks - supports 1 or 2 hands"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)
        
        # Extract landmarks for all detected hands
        all_landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            hand_coords = []
            for landmark in hand_landmarks.landmark:
                hand_coords.append([landmark.x, landmark.y, landmark.z])
            all_landmarks.append(hand_coords)
        
        # Normalize to always have 2 hands (pad with zeros if only 1 hand)
        if num_hands == 1:
            landmarks = all_landmarks[0] + [[0.0, 0.0, 0.0]] * 21
        else:
            landmarks = all_landmarks[0] + all_landmarks[1]
        
        return np.array(landmarks), results.multi_hand_landmarks, num_hands
    
    return None, None, 0


def predict_gesture(landmarks):
    """Predict gesture"""
    landmarks_reshaped = landmarks.reshape(1, 42, 3)  # 2-hand input
    predictions = model.predict(landmarks_reshaped, verbose=0)[0]
    
    predicted_idx = np.argmax(predictions)
    confidence = predictions[predicted_idx]
    predicted_label = config.ISL_LABELS[predicted_idx]
    
    # Get top 5
    top_5_idx = np.argsort(predictions)[-5:][::-1]
    top_5 = [(config.ISL_LABELS[idx], float(predictions[idx])) for idx in top_5_idx]
    
    return predicted_label, confidence, top_5


def generate_frames():
    """Generate video frames with predictions"""
    global current_prediction
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Extract and predict
        landmarks, hand_landmarks_list, num_hands = extract_landmarks(frame)
        
        if landmarks is not None:
            prediction, confidence, top_5 = predict_gesture(landmarks)
            
            # Update global state
            current_prediction = {
                "label": prediction,
                "confidence": float(confidence),
                "top_5": top_5,
                "num_hands": num_hands,
                "timestamp": time.time()
            }
            
            # Draw landmarks for ALL hands
            if hand_landmarks_list:
                for hand_landmarks in hand_landmarks_list:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            # Draw prediction
            cv2.putText(frame, f"{prediction} ({confidence*100:.1f}%)", 
                       (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # Draw hand count
            cv2.putText(frame, f"{num_hands} hand(s)", 
                       (frame.shape[1] - 200, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        else:
            current_prediction = {"label": None, "confidence": 0.0, "num_hands": 0}
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/prediction')
def get_prediction():
    """Get current prediction as JSON"""
    return jsonify(current_prediction)


@app.route('/stats')
def get_stats():
    """Get model statistics"""
    from database.db_manager import db
    
    stats = db.get_dataset_statistics()
    total_samples = sum(stats.values())
    
    return jsonify({
        "total_samples": total_samples,
        "gestures": len(stats),
        "samples_per_gesture": stats,
        "model_accuracy": 100.00,  # From your training results!
        "supports_2_hands": True
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("ISL SIGN LANGUAGE RECOGNITION - WEB APP (2-Hand Support)")
    print("="*60)
    print(f"Server starting on http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    print("Open your browser and go to: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.DEBUG_MODE
    )
