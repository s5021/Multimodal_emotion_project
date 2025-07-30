#!/usr/bin/env python3
"""
Setup script for Multi-Modal Emotion Analysis Project
Run this to install dependencies and test the setup
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return success status"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f" {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f" {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print(" Setting up Multi-Modal Emotion Analysis Project")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print(" Python 3.8+ required")
        return False
    else:
        print(f" Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Install dependencies
    print("\n Installing dependencies...")
    if not run_command("pip install -r requirements.txt"):
        print(" Failed to install dependencies")
        return False
    
    # Test imports
    print("\n Testing imports...")
    test_imports = [
        "import streamlit",
        "import cv2", 
        "import torch",
        "import transformers",
        "import deepface",
        "import plotly",
        "import pandas",
        "import numpy"
    ]
    
    for import_test in test_imports:
        try:
            exec(import_test)
            print(f" {import_test}")
        except ImportError as e:
            print(f" {import_test} - {e}")
            return False
    
    # Create sample data directory
    if not os.path.exists("sample_data"):
        os.makedirs("sample_data")
        print(" Created sample_data directory")
    
    print("\n Setup completed successfully!")
    print("\n To run the application:")
    print("   streamlit run app.py")
    print("\n Open your browser and navigate to the URL shown")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)