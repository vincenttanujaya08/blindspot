#!/usr/bin/env python3
"""
ADAS Project Setup Validator
Checks if all requirements are met before running the system
"""

import os
import sys
from pathlib import Path

print("="*60)
print("🚗 ADAS Project Setup Validator")
print("="*60)

errors = []
warnings = []

# ============================================================
# 1. Check Python Version
# ============================================================
print("\n📌 Checking Python version...")
py_version = sys.version_info
if py_version.major == 3 and py_version.minor >= 8:
    print(f"  ✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
else:
    errors.append(f"Python 3.8+ required, found {py_version.major}.{py_version.minor}")
    print(f"  ❌ Python {py_version.major}.{py_version.minor} (Need 3.8+)")

# ============================================================
# 2. Check Dependencies
# ============================================================
print("\n📌 Checking dependencies...")

required_packages = [
    ('cv2', 'opencv-python'),
    ('numpy', 'numpy'),
    ('flask', 'flask'),
    ('ultralytics', 'ultralytics'),
    ('torch', 'torch'),
]

for module_name, package_name in required_packages:
    try:
        __import__(module_name)
        print(f"  ✓ {package_name}")
    except ImportError:
        errors.append(f"Missing package: {package_name}")
        print(f"  ❌ {package_name} (NOT INSTALLED)")

# Check CUDA
try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available (GPU: {torch.cuda.get_device_name(0)})")
    else:
        warnings.append("CUDA not available - will use CPU (slower)")
        print(f"  ⚠ CUDA not available (will use CPU)")
except:
    pass

# ============================================================
# 3. Check Project Structure
# ============================================================
print("\n📌 Checking project structure...")

required_files = {
    'app.py': 'Flask server',
    'requirements.txt': 'Dependencies list',
    'logic/unified_detector.py': 'Main detector',
    'static/index.html': 'Frontend UI',
}

for file_path, description in required_files.items():
    if os.path.exists(file_path):
        print(f"  ✓ {file_path} ({description})")
    else:
        errors.append(f"Missing file: {file_path}")
        print(f"  ❌ {file_path} (NOT FOUND)")

# ============================================================
# 4. Check Model File
# ============================================================
print("\n📌 Checking YOLO model...")

model_path = 'best_yolov11n_BDD100K_50.pt'
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  ✓ {model_path} ({size_mb:.1f} MB)")
else:
    errors.append(f"Model file not found: {model_path}")
    print(f"  ❌ {model_path} (NOT FOUND)")
    print(f"     Please download the YOLO model file!")

# ============================================================
# 5. Check Video Files
# ============================================================
print("\n📌 Checking video files...")

video_dir = 'camera'
required_videos = ['front.mp4', 'back.mp4', 'left.mp4', 'right.mp4']

if not os.path.exists(video_dir):
    warnings.append(f"Camera directory not found: {video_dir}")
    print(f"  ⚠ Directory 'camera/' not found")
    print(f"     System will use placeholder frames")
else:
    for video_file in required_videos:
        video_path = os.path.join(video_dir, video_file)
        if os.path.exists(video_path):
            size_mb = os.path.getsize(video_path) / (1024 * 1024)
            print(f"  ✓ {video_file} ({size_mb:.1f} MB)")
        else:
            warnings.append(f"Video not found: {video_path}")
            print(f"  ⚠ {video_file} (NOT FOUND - will use placeholder)")

# ============================================================
# 6. Test Model Loading
# ============================================================
if os.path.exists(model_path):
    print("\n📌 Testing model loading...")
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Model type: {model.task}")
    except Exception as e:
        errors.append(f"Model loading failed: {str(e)}")
        print(f"  ❌ Failed to load model")
        print(f"     Error: {str(e)}")

# ============================================================
# 7. Test Video Reading
# ============================================================
if os.path.exists(video_dir):
    print("\n📌 Testing video reading...")
    test_video = None
    for video_file in required_videos:
        video_path = os.path.join(video_dir, video_file)
        if os.path.exists(video_path):
            test_video = video_path
            break
    
    if test_video:
        try:
            import cv2
            cap = cv2.VideoCapture(test_video)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    h, w, _ = frame.shape
                    print(f"  ✓ Video readable: {os.path.basename(test_video)} ({w}x{h})")
                else:
                    warnings.append(f"Cannot read frames from {test_video}")
                    print(f"  ⚠ Cannot read frames")
            else:
                warnings.append(f"Cannot open video {test_video}")
                print(f"  ⚠ Cannot open video")
            cap.release()
        except Exception as e:
            warnings.append(f"Video test failed: {str(e)}")
            print(f"  ⚠ Video test failed: {str(e)}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("📊 VALIDATION SUMMARY")
print("="*60)

if errors:
    print(f"\n❌ ERRORS ({len(errors)}):")
    for i, error in enumerate(errors, 1):
        print(f"   {i}. {error}")

if warnings:
    print(f"\n⚠ WARNINGS ({len(warnings)}):")
    for i, warning in enumerate(warnings, 1):
        print(f"   {i}. {warning}")

if not errors and not warnings:
    print("\n✓ All checks passed! System ready to run.")
    print("\n🚀 To start the system:")
    print("   python app.py")
elif not errors:
    print("\n✓ No critical errors. System can run with warnings.")
    print("\n🚀 To start the system:")
    print("   python app.py")
else:
    print("\n❌ Please fix errors before running the system.")
    print("\n📝 To install missing packages:")
    print("   pip install -r requirements.txt --break-system-packages")

print("\n" + "="*60)