"""
ISL Gesture Reference Guide - SIMPLE VERSION
Creates a single image showing all 26 ISL alphabet gestures
"""
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random

# Paths
DATASET_PATH = Path("D:/Projects/sign-language-recognition/Indian")
OUTPUT_PATH = Path("D:/Projects/sign-language-recognition/models/ISL_Reference_Guide.png")

print("\n" + "="*60)
print("ISL GESTURE REFERENCE GUIDE GENERATOR")
print("="*60)
print("\nThis creates ONE image with all 26 ISL letters (A-Z)")
print("You can use this as a cheat sheet!\n")

# Check if dataset exists
if not DATASET_PATH.exists():
    print(f"❌ Dataset not found at: {DATASET_PATH}")
    print("Please update DATASET_PATH in this script to your dataset location")
    exit()

# Create figure
print("Creating reference guide...\n")
fig, axes = plt.subplots(5, 6, figsize=(20, 16))
fig.suptitle('ISL Alphabet Gesture Reference Guide\n(What Each Letter Looks Like)', 
             fontsize=24, fontweight='bold', y=0.98)

axes = axes.flatten()

# Process each letter A-Z
letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

for idx, letter in enumerate(letters):
    letter_folder = DATASET_PATH / letter
    
    if not letter_folder.exists():
        print(f"⚠️  Folder not found: {letter}")
        axes[idx].text(0.5, 0.5, f'{letter}\nNot Found', 
                      ha='center', va='center', fontsize=20)
        axes[idx].axis('off')
        continue
    
    # Get all images
    images = list(letter_folder.glob("*.jpg")) + list(letter_folder.glob("*.png"))
    
    if not images:
        print(f"⚠️  No images in folder: {letter}")
        axes[idx].text(0.5, 0.5, f'{letter}\nNo Images', 
                      ha='center', va='center', fontsize=20)
        axes[idx].axis('off')
        continue
    
    # Pick a random sample
    sample_image = random.choice(images)
    
    # Load and display
    img = cv2.imread(str(sample_image))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Show image
    axes[idx].imshow(img_rgb)
    axes[idx].set_title(f'Letter: {letter}', fontsize=16, fontweight='bold', pad=10)
    axes[idx].axis('off')
    
    print(f"✓ Added gesture: {letter}")

# Hide extra subplots (we have 26 letters, 30 subplot spaces)
for idx in range(26, 30):
    axes[idx].axis('off')

plt.tight_layout()

# Save
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')

print("\n" + "="*60)
print("✅ SUCCESS!")
print("="*60)
print(f"\nReference guide saved to:")
print(f"  {OUTPUT_PATH}")
print("\nOpen this image to see what each ISL gesture looks like!")
print("Keep it next to your webcam while testing recognition!")
print("="*60 + "\n")

# Also try to open it automatically
try:
    import os
    os.startfile(OUTPUT_PATH)
    print("✓ Opening image automatically...\n")
except:
    print("Please open the file manually to view it.\n")
