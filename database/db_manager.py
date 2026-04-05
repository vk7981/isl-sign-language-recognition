"""
Database Manager for Sign Language Recognition System
"""
import sqlite3
from pathlib import Path
import json
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class DatabaseManager:
    """Manages database operations for the sign language recognition system"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.db_path = config.DATABASE_PATH
        self.schema_path = config.SCHEMA_PATH
        self._initialize_database()
        self._initialized = True
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _initialize_database(self):
        """Initialize database with schema"""
        if not self.db_path.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.schema_path.exists():
                with open(self.schema_path, 'r') as f:
                    schema = f.read()
                
                with self.get_connection() as conn:
                    conn.executescript(schema)
                
                print(f"✓ Database initialized: {self.db_path}")
    
    def insert_gesture(self, label: str, image_path: str, landmarks: List, 
                      session_id: Optional[str] = None) -> int:
        """Insert a gesture record"""
        landmarks_json = json.dumps(landmarks)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO gestures (label, image_path, landmarks, session_id)
                VALUES (?, ?, ?, ?)
            """, (label, image_path, landmarks_json, session_id))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_gestures_by_label(self, label: str) -> List[Dict]:
        """Get all gestures for a specific label"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, label, image_path, landmarks, session_id
                FROM gestures
                WHERE label = ?
            """, (label,))
            
            rows = cursor.fetchall()
            
            gestures = []
            for row in rows:
                gestures.append({
                    'id': row['id'],
                    'label': row['label'],
                    'image_path': row['image_path'],
                    'landmarks': json.loads(row['landmarks']) if row['landmarks'] else None,
                    'session_id': row['session_id']
                })
            
            return gestures
    
    def get_dataset_statistics(self) -> Dict[str, int]:
        """Get count of samples per gesture label"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT label, COUNT(*) as count
                FROM gestures
                GROUP BY label
                ORDER BY label
            """)
            
            stats = {row['label']: row['count'] for row in cursor.fetchall()}
            conn.close()
            
            return stats
    
    def get_total_samples(self) -> int:
        """Get total number of collected samples"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as total FROM gestures")
            total = cursor.fetchone()['total']
            conn.close()
            
            return total
    
    def clear_all_gestures(self):
        """Delete all gesture data from database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM gestures")
            conn.commit()
            rows_deleted = cursor.rowcount
        
        print(f"✓ Deleted {rows_deleted} gesture records from database")
        return rows_deleted
    
    def insert_training_session(self, model_name: str, model_path: str,
                                training_acc: float, val_acc: float, test_acc: float,
                                epochs: int, batch_size: int, learning_rate: float,
                                dataset_size: int, duration: float,
                                hyperparameters: Dict) -> int:
        """Insert a training session record"""
        hyperparams_json = json.dumps(hyperparameters)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_sessions 
                (model_name, model_path, training_acc, val_acc, test_acc,
                 epochs, batch_size, learning_rate, dataset_size, duration, hyperparameters)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (model_name, model_path, training_acc, val_acc, test_acc,
                  epochs, batch_size, learning_rate, dataset_size, duration, hyperparams_json))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_latest_training_session(self) -> Optional[Dict]:
        """Get the most recent training session"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM training_sessions
                ORDER BY id DESC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            
            if row:
                return {
                    'id': row['id'],
                    'model_name': row['model_name'],
                    'model_path': row['model_path'],
                    'training_acc': row['training_acc'],
                    'val_acc': row['val_acc'],
                    'test_acc': row['test_acc'],
                    'epochs': row['epochs'],
                    'batch_size': row['batch_size'],
                    'learning_rate': row['learning_rate'],
                    'dataset_size': row['dataset_size'],
                    'duration': row['duration'],
                    'hyperparameters': json.loads(row['hyperparameters']) if row['hyperparameters'] else None
                }
            
            return None
    
    def insert_prediction(self, gesture_label: str, confidence: float,
                         model_id: Optional[int] = None) -> int:
        """Insert a prediction record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (gesture_label, confidence, model_id)
                VALUES (?, ?, ?)
            """, (gesture_label, confidence, model_id))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_prediction_statistics(self, limit: int = 100) -> List[Dict]:
        """Get recent prediction statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT gesture_label, confidence
                FROM predictions
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            
            predictions = []
            for row in cursor.fetchall():
                predictions.append({
                    'gesture_label': row['gesture_label'],
                    'confidence': row['confidence']
                })
            
            return predictions


# Singleton instance
db = DatabaseManager()
