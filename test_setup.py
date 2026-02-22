#!/usr/bin/env python
"""
Quick test script to verify Face Mask Detection setup
"""

import os
import sys

def test_environment():
    """Test if environment is properly configured"""
    
    print("\n" + "="*70)
    print("🔍 TESTING FACE MASK DETECTION SETUP")
    print("="*70 + "\n")
    
    checks = {
        "Python Version": False,
        "Flask": False,
        "OpenCV": False,
        "TensorFlow": False,
        "NumPy": False,
        "Project Structure": False,
        "Models": False,
        "Face Detector Files": False,
    }
    
    # Check Python version
    try:
        print(f"✓ Python version: {sys.version.split()[0]}")
        checks["Python Version"] = True
    except:
        print("❌ Python version check failed")
    
    # Check Flask
    try:
        import flask
        print(f"✓ Flask: {flask.__version__}")
        checks["Flask"] = True
    except:
        print("❌ Flask not installed - run: pip install flask")
    
    # Check OpenCV
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
        checks["OpenCV"] = True
    except:
        print("❌ OpenCV not installed - run: pip install opencv-python")
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow: {tf.__version__}")
        checks["TensorFlow"] = True
    except:
        print("❌ TensorFlow not installed - run: pip install tensorflow")
    
    # Check NumPy
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
        checks["NumPy"] = True
    except:
        print("❌ NumPy not installed - run: pip install numpy")
    
    # Check project structure
    print("\n📁 Checking project structure...")
    required_dirs = [
        "app",
        "app/templates",
        "app/static",
        "src",
        "src/components",
        "src/pipelines",
        "artifact/models",
        "face_detector"
    ]
    
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ NOT FOUND")
    checks["Project Structure"] = all(os.path.isdir(d) for d in required_dirs)
    
    # Check models
    print("\n🤖 Checking trained models...")
    model_dir = "artifact/models"
    models = []
    
    if os.path.isdir(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith(('.h5', '.keras'))]
        if model_files:
            for model in model_files:
                print(f"  ✓ {model}")
                models.append(model)
            checks["Models"] = True
        else:
            print("  ❌ No trained models found in artifact/models/")
    else:
        print("  ❌ artifact/models/ directory not found")
    
    # Check face detector files
    print("\n🔍 Checking face detector files...")
    face_detector_files = [
        "face_detector/deploy.prototxt",
        "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    ]
    
    for file_path in face_detector_files:
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"  ✓ {file_path} ({size:.1f} MB)")
        else:
            print(f"  ❌ {file_path} NOT FOUND")
    
    checks["Face Detector Files"] = all(os.path.isfile(f) for f in face_detector_files)
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    
    for check, status in checks.items():
        symbol = "✓" if status else "❌"
        print(f"{symbol} {check}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All checks passed! Ready to run the app!")
        print("\nTo start the app, run:")
        print("  python run_app.py")
        print("\nThen open your browser to: http://localhost:5000")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nCommon issues:")
        print("  • Missing dependencies: pip install -r requirements.txt")
        print("  • Missing models: Train models or download pretrained versions")
        print("  • Wrong directory: Make sure you're in the project root")
    
    print("\n" + "="*70 + "\n")
    
    return passed == total


def test_imports():
    """Test if core imports work"""
    
    print("\n" + "="*70)
    print("🧪 TESTING IMPORTS")
    print("="*70 + "\n")
    
    try:
        print("Importing Flask app...")
        sys.path.insert(0, os.getcwd())
        from app.app import app
        print("✓ Flask app imported successfully")
    except Exception as e:
        print(f"❌ Failed to import Flask app: {str(e)}")
        return False
    
    try:
        print("Importing PredictPipeline...")
        from src.pipelines.predict_pipeline import PredictPipeline
        print("✓ PredictPipeline imported successfully")
    except Exception as e:
        print(f"❌ Failed to import PredictPipeline: {str(e)}")
        return False
    
    try:
        print("Importing FaceCropper...")
        from src.components.face_crop import FaceCropper
        print("✓ FaceCropper imported successfully")
    except Exception as e:
        print(f"❌ Failed to import FaceCropper: {str(e)}")
        return False
    
    print("\n✓ All imports successful!")
    print("\n" + "="*70 + "\n")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Face Mask Detection Setup")
    parser.add_argument('--import-test', action='store_true', help='Test imports only')
    parser.add_argument('--env-test', action='store_true', help='Test environment only')
    
    args = parser.parse_args()
    
    if args.env_test:
        test_environment()
    elif args.import_test:
        test_imports()
    else:
        env_ok = test_environment()
        if env_ok:
            test_imports()
