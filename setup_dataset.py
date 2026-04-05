"""
ISL Dataset Processor
Downloads and processes pre-collected ISL dataset
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import shutil
import uuid
import config
from database.db_manager import db

class DatasetProcessor:
    """Process pre-collected ISL dataset"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        self.session_id = str(uuid.uuid4())
        self.processed_count = 0
        self.failed_count = 0
    
    def extract_landmarks(self, image_path):
        """Extract hand landmarks from image"""
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            return landmarks
        
        return None
    
    def process_dataset_folder(self, dataset_path):
        """Process dataset from folder"""
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            print(f"❌ Dataset path not found: {dataset_path}")
            return False
        
        print("\n" + "="*60)
        print("PROCESSING ISL DATASET")
        print("="*60)
        print(f"Source: {dataset_path}")
        print(f"Destination: {config.RAW_DATA_DIR}")
        print("="*60 + "\n")
        
        # Get letter folders (A-Z only)
        letter_folders = [f for f in dataset_path.iterdir() if f.is_dir() and f.name.upper() in config.ISL_LABELS]
        
        if not letter_folders:
            print("❌ No letter folders found in dataset!")
            return False
        
        print(f"Found {len(letter_folders)} gesture folders\n")
        
        for folder in sorted(letter_folders, key=lambda x: x.name.upper()):
            label = folder.name.upper()
            
            print(f"\n📁 Processing gesture: {label}")
            
            # Get all images
            image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
            
            if not image_files:
                print(f"  ⚠️  No images found")
                continue
            
            print(f"  Found {len(image_files)} images")
            
            # Create destination folder
            dest_folder = config.RAW_DATA_DIR / label
            dest_folder.mkdir(parents=True, exist_ok=True)
            
            # Process each image
            processed = 0
            failed = 0
            
            for img_path in tqdm(image_files, desc=f"  Processing {label}", unit="img"):
                landmarks = self.extract_landmarks(str(img_path))
                
                if landmarks is not None:
                    # Copy image
                    dest_path = dest_folder / f"{label}_{img_path.stem}.jpg"
                    shutil.copy2(img_path, dest_path)
                    
                    # Save to database
                    db.insert_gesture(
                        label=label,
                        image_path=str(dest_path),
                        landmarks=landmarks,
                        session_id=self.session_id
                    )
                    
                    processed += 1
                    self.processed_count += 1
                else:
                    failed += 1
                    self.failed_count += 1
            
            print(f"  ✓ Processed: {processed} | ✗ Failed: {failed}")
        
        print("\n" + "="*60)
        print("✅ DATASET PROCESSING COMPLETE!")
        print("="*60)
        print(f"Total processed: {self.processed_count}")
        print(f"Total failed: {self.failed_count}")
        print(f"Success rate: {(self.processed_count / (self.processed_count + self.failed_count) * 100):.1f}%")
        print("="*60 + "\n")
        
        # Show statistics
        stats = db.get_dataset_statistics()
        print("Samples per gesture:")
        for label, count in sorted(stats.items()):
            print(f"  {label}: {count}")
        
        print(f"\nTotal samples in database: {sum(stats.values())}")
        
        return True
    
    def cleanup(self):
        """Clean up resources"""
        self.mp_hands.close()


def main():
    print("\n" + "="*60)
    print("ISL DATASET SETUP")
    print("="*60)
    print("\nThis script will process your ISL dataset")
    print("and prepare it for model training.\n")
    
    print("Enter dataset path:")
    print("Example: D:\\Projects\\sign-language-recognition\\Indian")
    dataset_path = input("\nPath: ").strip()
    
    if not dataset_path:
        print("\n❌ No path provided. Exiting.")
        return
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"\n❌ Path not found: {dataset_path}")
        return
    
    print("\nThis will:")
    print("  • Extract hand landmarks from all images")
    print("  • Copy images to project folder")
    print("  • Save data to database")
    print("  • May take 10-20 minutes")
    print()
    
    confirm = input("Continue? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("\n❌ Cancelled.")
        return
    
    processor = DatasetProcessor()
    
    try:
        success = processor.process_dataset_folder(dataset_path)
        
        if success:
            print("\n✅ SUCCESS! Dataset ready for training.")
            print("\nNext step: Create training pipeline")
        else:
            print("\n❌ Processing failed.")
    
    finally:
        processor.cleanup()


if __name__ == "__main__":
    main()
