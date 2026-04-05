"""
ISL Gesture Reference Guide Generator
Shows sample images for each letter from the training dataset
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import config
import random


def create_reference_guide(dataset_path: str, output_path: str = None):
    """Create a visual reference guide showing all ISL gestures"""
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return
    
    print("\n" + "="*60)
    print("CREATING ISL GESTURE REFERENCE GUIDE")
    print("="*60)
    
    # Create figure with grid
    fig, axes = plt.subplots(5, 6, figsize=(18, 15))
    fig.suptitle('ISL Alphabet Gesture Reference Guide', fontsize=20, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, label in enumerate(config.ISL_LABELS):
        letter_folder = dataset_path / label
        
        if not letter_folder.exists():
            print(f"⚠️  Folder not found: {label}")
            continue
        
        # Get all images for this letter
        images = list(letter_folder.glob("*.jpg")) + list(letter_folder.glob("*.png"))
        
        if not images:
            print(f"⚠️  No images found for: {label}")
            continue
        
        # Pick a random sample image
        sample_image = random.choice(images)
        
        # Load and display
        img = cv2.imread(str(sample_image))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display on subplot
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(f'Letter: {label}', fontsize=14, fontweight='bold')
        axes[idx].axis('off')
        
        print(f"✓ Added {label}")
    
    # Hide extra subplots (we have 26 letters, 30 subplot spaces)
    for idx in range(26, 30):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save
    if output_path is None:
        output_path = config.MODELS_DIR / 'ISL_Reference_Guide.png'
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Reference guide saved to: {output_path}")
    
    # Also display
    plt.show()
    
    print("="*60 + "\n")


def create_individual_samples(dataset_path: str, samples_per_letter: int = 3):
    """Create individual sample sheets for each letter"""
    
    dataset_path = Path(dataset_path)
    samples_dir = config.MODELS_DIR / 'gesture_samples'
    samples_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("CREATING INDIVIDUAL GESTURE SAMPLES")
    print("="*60)
    
    for label in config.ISL_LABELS:
        letter_folder = dataset_path / label
        
        if not letter_folder.exists():
            continue
        
        images = list(letter_folder.glob("*.jpg")) + list(letter_folder.glob("*.png"))
        
        if not images:
            continue
        
        # Pick random samples
        num_samples = min(samples_per_letter, len(images))
        sample_images = random.sample(images, num_samples)
        
        # Create figure for this letter
        fig, axes = plt.subplots(1, num_samples, figsize=(12, 4))
        if num_samples == 1:
            axes = [axes]
        
        fig.suptitle(f'ISL Gesture: {label}', fontsize=16, fontweight='bold')
        
        for idx, img_path in enumerate(sample_images):
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[idx].imshow(img_rgb)
            axes[idx].set_title(f'Sample {idx+1}')
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # Save
        output_path = samples_dir / f'gesture_{label}.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Created samples for {label}: {output_path}")
    
    print(f"\n✅ Individual samples saved to: {samples_dir}")
    print("="*60 + "\n")


def main():
    print("\n" + "="*60)
    print("ISL GESTURE REFERENCE GUIDE GENERATOR")
    print("="*60)
    print("\nThis will create:")
    print("  1. Complete reference guide (all 26 letters in one image)")
    print("  2. Individual sample sheets for each letter")
    print()
    
    dataset_path = input("Enter dataset path (or press ENTER for default): ").strip()
    
    if not dataset_path:
        dataset_path = "D:\\Projects\\sign-language-recognition\\Indian"
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"\n❌ Dataset not found: {dataset_path}")
        return
    
    print("\nOption 1: Create complete reference guide (all letters)")
    print("Option 2: Create individual sample sheets")
    print("Option 3: Both")
    
    choice = input("\nChoice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        create_reference_guide(str(dataset_path))
    
    if choice in ['2', '3']:
        create_individual_samples(str(dataset_path))
    
    print("\n✅ Done! Check the 'models' folder for your reference guide(s).")


if __name__ == "__main__":
    main()
