"""
Setup Script for Sign Language Recognition Project
Run this to create the correct folder structure
"""
from pathlib import Path

def create_structure():
    """Create project folder structure"""
    
    base = Path.cwd()
    print(f"Setting up project in: {base}\n")
    
    # Create folders
    folders = [
        'database',
        'models',
        'modules',
        'data',
        'data/raw',
        'data/processed',
        'static',
        'templates'
    ]
    
    for folder in folders:
        folder_path = base / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {folder}/")
    
    # Create __init__.py files
    init_folders = ['database', 'models', 'modules']
    for folder in init_folders:
        init_file = base / folder / '__init__.py'
        if not init_file.exists():
            init_file.write_text('# This file makes the directory a Python package\n')
            print(f"✓ Created: {folder}/__init__.py")
    
    print("\n" + "="*60)
    print("FOLDER STRUCTURE CREATED!")
    print("="*60)
    print("\nNext steps:")
    print("1. Place config.py in the main folder")
    print("2. Place schema.sql in the main folder")
    print("3. Place db_manager.py in database/ folder")
    print("4. Place data_collection.py in modules/ folder")
    print("5. Place cnn_model.py in models/ folder")
    print("\nThen run: python modules\\data_collection.py --label A --samples 3")

if __name__ == "__main__":
    create_structure()
