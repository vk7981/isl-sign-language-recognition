-- Sign Language Recognition System Database Schema

CREATE TABLE IF NOT EXISTS gestures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL,
    image_path TEXT NOT NULL,
    landmarks TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT,
    session_id TEXT,
    is_validated BOOLEAN DEFAULT 0
);

CREATE TABLE IF NOT EXISTS training_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_path TEXT NOT NULL,
    training_accuracy REAL,
    validation_accuracy REAL,
    test_accuracy REAL,
    total_epochs INTEGER,
    batch_size INTEGER,
    learning_rate REAL,
    dataset_size INTEGER,
    training_duration REAL,
    hyperparameters TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    predicted_label TEXT NOT NULL,
    confidence REAL NOT NULL,
    actual_label TEXT,
    is_correct BOOLEAN,
    image_path TEXT,
    landmarks TEXT,
    model_id INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES training_sessions(id)
);

CREATE TABLE IF NOT EXISTS gesture_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gesture_label TEXT NOT NULL,
    total_samples INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    accuracy REAL DEFAULT 0.0,
    avg_confidence REAL DEFAULT 0.0,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_gestures_label ON gestures(label);
CREATE INDEX IF NOT EXISTS idx_gestures_session ON gestures(session_id);
CREATE INDEX IF NOT EXISTS idx_predictions_label ON predictions(predicted_label);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
