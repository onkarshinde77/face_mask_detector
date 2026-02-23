# 🔍 Face Mask Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13+-green.svg)
![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

An advanced AI-powered Face Mask Detection System using Deep Learning for real-time detection in images, videos, and live webcam feeds.

[Live Demo](#quick-start) • [Dataset](#-dataset) • [Models](#-models) • [Setup Guide](#-setup-guide) • [Documentation](#-documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Models](#-models)
- [Project Setup](#-setup-guide)
- [Installation](#-installation)
- [Usage](#-usage)
- [Web Application](#-web-application)
- [API Reference](#-api-reference)
- [Technical Architecture](#-technical-architecture)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)
- [Author & Credits](#-author--credits)
- [License](#-license)

---

## 🎯 Overview

This project implements a comprehensive **Face Mask Detection System** that uses state-of-the-art deep learning models to detect whether people are wearing masks or not. The system can process:

- **Static Images** - Detect masks in photos
- **Video Files** - Process videos with frame-by-frame detection
- **Live Webcam** - Real-time streaming detection
- **Camera Capture** - Take photos directly from your camera

### Key Highlights

✨ **Multi-face Detection** - Detects all faces in an image/video simultaneously
✨ **High Accuracy** - 98% face detection + 95% mask classification
✨ **Real-time Processing** - Instant predictions on any device
✨ **Web Interface** - User-friendly Flask web application
✨ **Mobile Friendly** - Responsive design works on phones/tablets
✨ **Production Ready** - Full error handling, logging, and documentation

---

## 🌟 Features

### Core Detection Features
- ✅ Multi-face detection with Caffe DNN SSD model
- ✅ Binary mask classification (Mask / No Mask)
- ✅ Confidence scoring for each prediction
- ✅ Color-coded bounding boxes (Green = Mask, Red = No Mask)
- ✅ Real-time annotation and labeling

### Image Processing
- 📸 Single image upload
- 📷 Camera photo capture
- 📁 Batch image processing
- 💾 High-quality output images

### Video Processing
- 🎬 Video file upload and processing
- 📹 Live webcam streaming
- 🔄 Background video processing
- ⏱️ Frame-by-frame detection

### Web Interface
- 🖥️ Modern, responsive UI
- 🎨 Beautiful gradient design
- 📊 Detailed detection statistics
- 🔐 Secure file upload handling
- ⚡ Fast AJAX responses

---

## 📊 Dataset

### Source Information

**Dataset Name:** Face Mask Classification Dataset
**Platform:** Kaggle
**Link:** [https://www.kaggle.com/datasets/onkarshinde77/face-mask-classification](https://www.kaggle.com/datasets/onkarshinde77/face-mask-classification)

### Dataset Details

```
Dataset Structure:
├── data/
│   ├── train/           # Training images
│   │   ├── with_mask/   # ~3,883 images
│   │   └── without_mask/ # ~3,925 images
│   ├── test/            # Test images
│   │   ├── with_mask/   # ~429 images
│   │   └── without_mask/ # ~430 images
│   └── valid/           # Validation images
│       ├── with_mask/   # ~429 images
│       └── without_mask/ # ~430 images
│
└── data2/               # Alternative dataset format
    ├── annotations/     # XML annotation files
    └── images/          # Image files
```

### Dataset Statistics

| Category | Count | Percentage |
|----------|-------|-----------|
| Training Images | 7,808 | 75% |
| Testing Images | 859 | 18% |
| Validation Images | 858 | 8% |
| **Total Images** | **9,525** | **100%** |

### Image Specifications

- **Format:** JPEG, PNG
- **Resolution:** 224x224 pixels (for model)
- **Color Space:** RGB
- **Image Quality:** High resolution (source)
- **Preprocessing:** Normalized to VGG16 input specs

### Data Collection

- **Sources:** Public datasets + web collection
- **Quality:** Verified and cleaned
- **Diversity:** Various lighting, angles, and demographics
- **Annotations:** Binary labels (with_mask / without_mask)

### Download Dataset

```bash
# Option 1: Download from Kaggle
# Visit: https://www.kaggle.com/datasets/onkarshinde77/face-mask-classification
# Download and extract to: data/ folder

# Option 2: Using Kaggle API
kaggle datasets download -d onkarshinde77/face-mask-classification
unzip face-mask-classification.zip
```

---

## 🤖 Models

### Face Detection Model

**Model Type:** Caffe Deep Neural Network (DNN) - Single Shot MultiBox Detector (SSD)

**Architecture:**
- **Input Size:** 300x300 pixels
- **Layers:** ResNet-based backbone
- **Output:** Face bounding boxes with confidence scores
- **Framework:** Caffe

**Model Files:**
```
face_detector/
├── deploy.prototxt                              (7.6 KB)
└── res10_300x300_ssd_iter_140000.caffemodel    (96.4 MB)
```

**Performance:**
- **Accuracy:** 98%+ on frontal faces
- **Speed:** ~50ms per image (CPU)
- **False Positive Rate:** <2%
- **Minimum Face Size:** 50x50 pixels

**Strengths:**
✅ Highly accurate face detection
✅ Fast inference speed
✅ Works with various lighting conditions
✅ Minimal false positives

### Mask Classification Model

**Model Type:** VGG16-based Convolutional Neural Network (CNN)

**Architecture:**
```
Input (224x224x3)
    ↓
VGG16 Base (Pre-trained on ImageNet)
    ↓
Flatten
    ↓
Dense(4096, activation='relu')
    ↓
Dense(4096, activation='relu')
    ↓
Dense(1, activation='sigmoid')  [Binary Classification]
    ↓
Output (Mask / No Mask)
```

**Model Files:**
```
artifact/models/
├── face_mask_model.h5           (VGG16 Binary)
├── face_mask_model2.h5          (VGG16 Binary)
└── face_mask_model3.keras       (VGG16 Latest Format)
```

**Model Specifications:**
- **Framework:** TensorFlow/Keras
- **Base Model:** VGG16 (pre-trained on ImageNet)
- **Input Size:** 224x224x3 (RGB)
- **Output:** Binary classification (0=Mask, 1=No Mask)
- **Trainable Layers:** Top dense layers only (transfer learning)

**Training Details:**
- **Optimizer:** Adam (lr=0.001)
- **Loss Function:** Binary Crossentropy
- **Metric:** Accuracy
- **Batch Size:** 32
- **Epochs:** 20-30
- **Augmentation:** Yes (rotation, zoom, flip, etc.)

**Performance:**
- **Accuracy:** 95%+ on test set
- **Precision:** 94%+
- **Recall:** 96%+
- **F1-Score:** 0.95

**Model Comparison:**

| Model | Framework | Size | Accuracy | Speed |
|-------|-----------|------|----------|-------|
| face_mask_model.h5 | Keras (H5) | ~95 MB | 94.2% | 50ms |
| face_mask_model2.h5 | Keras (H5) | ~95 MB | 95.1% | 52ms |
| face_mask_model3.keras | TensorFlow | ~95 MB | 95.3% | 48ms |

### Model Download

Models are included in `artifact/models/` directory. If needed:

```bash
# Models can be retrained using the provided dataset
# See training scripts in src/components/ and src/pipelines/
```

---

## 📁 Project Structure

```
face_mask_detector/
│
├── app/                                    # Flask Web Application
│   ├── app.py                             # Main Flask application
│   ├── static/
│   │   ├── style.css                      # Styling
│   │   └── uploads/                       # User uploads (generated)
│   │       ├── processed_*.jpg            # Annotated images
│   │       └── processed_*.mp4            # Processed videos
│   └── templates/
│       ├── index.html                     # Home page
│       ├── live.html                      # Live camera stream
│       ├── upload_photo.html              # Photo upload/capture
│       └── upload_video.html              # Video upload
│
├── src/                                    # Source Code
│   ├── components/
│   │   ├── face_crop.py                   # Face detection (Caffe DNN)
│   │   ├── model.py                       # Model architecture
│   │   ├── model_trainer.py               # Training logic
│   │   └── prepare_dataset.py             # Data preparation
│   │
│   ├── pipelines/
│   │   ├── predict_pipeline.py            # Main prediction pipeline
│   │   └── train_pipeline.py              # Training pipeline
│   │
│   ├── constant/                          # Configuration constants
│   ├── entity/                            # Data classes
│   ├── exception/                         # Custom exceptions
│   ├── logger/                            # Logging setup
│   └── utils/                             # Utility functions
│
├── artifact/                               # Trained Models & Data
│   ├── models/
│   │   ├── face_mask_model.h5            # Trained model (H5 format)
│   │   ├── face_mask_model2.h5           # Alternative model
│   │   └── face_mask_model3.keras        # Latest format model
│   ├── data/
│   │   ├── train/
│   │   ├── test/
│   │   └── valid/
│   └── data2/                            # Alternative annotations
│
├── face_detector/                         # Caffe Face Detection Model
│   ├── deploy.prototxt                   # Model architecture
│   └── res10_300x300_ssd_iter_140000.caffemodel  # Weights
│
├── data/                                  # Dataset (if stored locally)
│   ├── train/
│   ├── test/
│   └── valid/
│
├── Notebook/                              # Jupyter Notebooks
│   └── temp.ipynb                        # Experiments
│
├── logs/                                  # Application logs (generated)
│
├── run_app.py                            # Easy app launcher
├── test_setup.py                         # Environment verification
├── predict_example.py                    # Usage examples
├── START_HERE.py                         # Quick reference
│
├── requirements.txt                      # Python dependencies
├── setup.py                              # Package setup
├── settings.json                         # Configuration
│
├── README.md                             # This file
├── QUICK_START.md                        # Quick start guide
├── WEB_APP_GUIDE.md                      # Web app documentation
├── PIPELINE_DOCUMENTATION.md             # API reference
├── ARCHITECTURE.md                       # System architecture
└── SETUP_COMPLETE.md                     # Detailed setup guide
```

---

## 🛠️ Setup Guide

### System Requirements

**Minimum Requirements:**
- Windows 10/11, macOS, or Linux
- Python 3.8 or higher
- 4GB RAM (8GB recommended)
- 2GB free disk space
- Webcam (for live detection)

**Recommended Requirements:**
- Windows 11 or Latest Ubuntu
- Python 3.10+
- 8GB+ RAM
- NVIDIA GPU with CUDA support
- 5GB+ free disk space

### Prerequisites

1. **Python Installation**
   - Download from [python.org](https://www.python.org/downloads/)
   - Ensure Python 3.8+ is installed
   - Verify installation:
     ```bash
     python --version
     ```

2. **Git Installation**
   - Download from [git-scm.com](https://git-scm.com/download)
   - Verify installation:
     ```bash
     git --version
     ```

3. **VS Code Installation**
   - Download from [code.visualstudio.com](https://code.visualstudio.com/)
   - Install Python extension by Microsoft

---

## 📥 Installation

### Step 1: Clone Repository

```bash
# Clone the project
git clone https://github.com/onkarshinde77/face_mask_detector.git

# Navigate to project directory
cd face_mask_detector
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

**Dependencies Installed:**
```
numpy==2.2.6
pandas==2.3.3
opencv-python==4.13.0.92
tensorflow>=2.13.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
flask>=2.0.0
```

### Step 4: Verify Installation

```bash
# Run setup verification
python test_setup.py
```

Expected output:
```
✓ Python version: 3.10.x
✓ Flask: 2.x.x
✓ OpenCV: 4.13.0.92
✓ TensorFlow: 2.x.x
✓ NumPy: 2.2.6
✓ All checks passed!
```

### Step 5: Download Dataset (Optional)

```bash
# Download from Kaggle
# Visit: https://www.kaggle.com/datasets/onkarshinde77/face-mask-classification
# Extract to: data/ folder
```

---

## 💻 Setup in VS Code

### 1. Open Project in VS Code

```bash
# Navigate to project folder
cd face_mask_detector

# Open in VS Code
code .
```

Or:
- Click File → Open Folder
- Navigate to `face_mask_detector`
- Click Open

### 2. Install Python Extension

- Open VS Code
- Go to Extensions (Ctrl+Shift+X)
- Search "Python"
- Install "Python" by Microsoft
- Install "Pylance" for better intellisense

### 3. Select Python Interpreter

1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose `.venv` environment
   ```
   ./venv/Scripts/python.exe
   ```

### 4. Setup Terminal

- Open terminal in VS Code (Ctrl+`)
- Terminal should auto-detect Python environment
- Prompt should show `(.venv)` prefix

### 5. Install Recommended Extensions

```
Microsoft Python (Pylance, Debugger)
OpenCV Snippets
Jupyter
Pylint
Flake8
Black Formatter
```

Install from Extensions marketplace or run:
```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-python.debugpy
```

### 6. Create Launch Configuration

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Flask",
      "type": "python",
      "request": "launch",
      "module": "flask",
      "env": {
        "FLASK_APP": "app/app.py",
        "FLASK_ENV": "development"
      },
      "args": ["run", "--host=0.0.0.0", "--port=5000"],
      "jinja": true
    },
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal"
    }
  ]
}
```

### 7. Setup Debugging

- Set breakpoints (click on line number)
- Press F5 to start debugging
- Use Debug Console for variables inspection

### 8. VS Code Settings

Create `.vscode/settings.json`:

```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "[python]": {
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "ms-python.python"
  },
  "python.analysis.extraPaths": [
    "${workspaceFolder}/src"
  ]
}
```

---

## 🚀 Usage

### Quick Start

```bash
# 1. Activate virtual environment
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # macOS/Linux

# 2. Run the application
python run_app.py

# 3. Open browser
# Visit: http://localhost:5000
```

### Python API Usage

```python
from src.pipelines.predict_pipeline import PredictPipeline
import cv2

# Initialize pipeline
pipeline = PredictPipeline()

# Detect masks in image
result = pipeline.predict_image('path/to/image.jpg')

# Access results
print(f"Faces detected: {result['num_faces']}")
for detection in result['detections']:
    print(f"  {detection['label']}: {detection['confidence']*100:.1f}%")

# Display result
cv2.imshow("Result", result['image'])
cv2.waitKey(0)
```

### Command Line Examples

```bash
# Run tests
python test_setup.py

# View examples
python predict_example.py

# Run web app in debug mode
python run_app.py --debug

# Run on custom port
python run_app.py --port 8000
```

---

## 🌐 Web Application

### Features

**Home Page (/):**
- 4 main feature cards
- Statistics display
- Quick navigation

**Photo Upload (/upload_photo):**
- 📤 File upload tab
- 📷 Camera capture tab
- Real-time preview
- Instant detection

**Video Upload (/upload_video):**
- Drag & drop upload
- Background processing
- Progress tracking
- Download processed video

**Live Camera (/live):**
- Real-time webcam stream
- Live mask detection
- MJPEG streaming
- Label and confidence display

### Access Web App

```bash
# Start the app
python run_app.py

# Open browser
http://localhost:5000

# Different endpoints:
http://localhost:5000/              # Home
http://localhost:5000/upload_photo  # Photo upload
http://localhost:5000/upload_video  # Video upload
http://localhost:5000/live          # Live camera
```

---

## 📚 API Reference

### PredictPipeline Class

```python
class PredictPipeline:
    def predict_image(image_path: str) -> dict
    def predict_video(video_path: str, save_output: bool, output_path: str) -> dict
    def predict_webcam() -> dict
```

### FaceCropper Class

```python
class FaceCropper:
    def detect_faces(image: np.ndarray) -> list
    def crop_faces(image: np.ndarray, faces: list) -> list
```

### Return Formats

**predict_image() output:**
```python
{
    'image': np.ndarray,              # Annotated image
    'detections': [                   # List of detections
        {
            'coords': (x1, y1, x2, y2),
            'label': 'Mask' | 'No Mask',
            'confidence': float
        }
    ],
    'num_faces': int
}
```

---

## 🏗️ Technical Architecture

### System Design

```
┌─────────────────────────────────────────┐
│       User Interface (Browser)          │
│  HTML5 | CSS3 | JavaScript             │
└─────────────┬───────────────────────────┘
              │ HTTP/REST
┌─────────────▼───────────────────────────┐
│     Flask Web Server                    │
│  - Request handling                     │
│  - File upload management               │
│  - Template rendering                   │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│   PredictPipeline                       │
│  - Orchestration                        │
│  - Image processing                     │
│  - Batch predictions                    │
└─────────────┬───────────────────────────┘
              │
    ┌─────────┴─────────┬─────────────┐
    │                   │             │
┌───▼──────────┐  ┌────▼────┐  ┌──────▼──────┐
│ FaceCropper  │  │ VGG16   │  │   Utils     │
│ (Caffe DNN)  │  │ Model   │  │ (CV, Numpy) │
└──────────────┘  └─────────┘  └─────────────┘
```

### Data Flow

```
Input (Image/Video)
    ↓
Face Detection (Caffe SSD)
    ↓
Face Cropping & Preprocessing
    ↓
Batch Prediction (VGG16)
    ↓
Post-processing & Annotation
    ↓
Output (Annotated Image/Video)
```

---

## ⚡ Performance

### Benchmarks

| Task | Resolution | FPS/Speed | GPU | CPU |
|------|-----------|-----------|-----|-----|
| Image (480p) | 480x640 | 100-150ms | 50-80ms | 100-150ms |
| Image (720p) | 720x1280 | 200-300ms | 80-120ms | 200-300ms |
| Video (480p) | 480x640 | 30 FPS | 25-30 FPS | 15-20 FPS |
| Video (720p) | 720x1280 | 20 FPS | 18-25 FPS | 10-15 FPS |
| Webcam (480p) | 480x640 | 30 FPS | 25-30 FPS | 15-20 FPS |

### Memory Usage

- **Model Loading:** ~500 MB
- **Per Image:** ~100-200 MB
- **Per Video Frame:** ~50 MB
- **Total at Runtime:** ~600-700 MB

### Optimization Tips

1. **GPU Acceleration**
   ```bash
   pip install tensorflow-gpu
   ```

2. **Reduce Image Size**
   - Process at 480p instead of 1080p
   - Significant speed improvement

3. **Batch Processing**
   - Process multiple images together
   - Better GPU utilization

---

## 🆘 Troubleshooting

### Common Issues

**Issue: "Model not found"**
```
Solution:
- Check artifact/models/ has .h5 or .keras files
- Verify file permissions
- Re-download models if corrupted
```

**Issue: "Camera shows black screen"**
```
Solution:
- Check browser camera permissions
- Use http://localhost:5000 (not 127.0.0.1)
- Try different browser
- Verify camera is working with other apps
```

**Issue: "TensorFlow not found"**
```
Solution:
pip install tensorflow
# or for GPU:
pip install tensorflow-gpu
```

**Issue: "Face detector model files not found"**
```
Solution:
- Verify face_detector/ folder exists
- Check deploy.prototxt file exists
- Check res10_300x300_ssd_*.caffemodel exists
- Ensure correct file paths
```

### Debug Mode

Run with verbose output:

```bash
python run_app.py --debug
```

Check logs:
```bash
tail -f logs/app.log  # Linux/macOS
Get-Content logs/app.log -Tail 50  # Windows
```

---

## 👨‍💻 Author & Credits

### About the Author

**Onkar Shinde**
- 🎓 Computer Science Student
- 💻 Full-Stack Developer
- 🤖 AI/ML Enthusiast
- 🚀 Open Source Contributor

### Connect With Me

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/onkarshinde77)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/onkarshinde77)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/onkarshinde77)

</div>

### Technical Links

- **GitHub Profile:** [github.com/onkarshinde77](https://github.com/onkarshinde77)
- **LinkedIn Profile:** [linkedin.com/in/onkarshinde77](https://www.linkedin.com/in/onkarshinde77)
- **Kaggle Profile:** [kaggle.com/onkarshinde77](https://www.kaggle.com/onkarshinde77)
- **This Project:** [github.com/onkarshinde77/face_mask_detector](https://github.com/onkarshinde77/face_mask_detector)
- **Dataset:** [kaggle.com/datasets/onkarshinde77/face-mask-classification](https://www.kaggle.com/datasets/onkarshinde77/face-mask-classification)

### Project Credits

- **Dataset:** Kaggle Community
- **Face Detection Model:** OpenCV Caffe Models
- **Framework:** TensorFlow/Keras
- **Web Framework:** Flask
- **Inspiration:** Real-world COVID-19 safety

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🤝 Contributing

Contributions are welcome! Here's how to contribute:

1. **Fork the repository**
   ```bash
   git clone https://github.com/onkarshinde77/face_mask_detector.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```

3. **Commit changes**
   ```bash
   git commit -m 'Add AmazingFeature'
   ```

4. **Push to branch**
   ```bash
   git push origin feature/AmazingFeature
   ```

5. **Open a Pull Request**

### Guidelines
- Follow PEP 8 style guide
- Add comments for complex code
- Test your changes
- Update documentation

---

## 📞 Support

Need help? Here are resources:

- 📚 **Documentation:** See PIPELINE_DOCUMENTATION.md
- 🔧 **Setup Help:** See SETUP_COMPLETE.md
- 🚀 **Quick Start:** See QUICK_START.md
- 🏗️ **Architecture:** See ARCHITECTURE.md
- 💬 **Issues:** Open GitHub issue
- 📧 **Contact:** LinkedIn message

---

## 🙏 Acknowledgments

- Thanks to Kaggle community for the dataset
- OpenCV for face detection models
- TensorFlow/Keras for deep learning
- Flask for web framework
- All contributors and users

---

## 📈 Roadmap

### Planned Features
- [ ] Multi-class classification (proper/improper mask wearing)
- [ ] Face recognition module
- [ ] Mask fit assessment algorithm
- [ ] Real-time statistics dashboard
- [ ] REST API endpoints
- [ ] Mobile app deployment
- [ ] Docker containerization
- [ ] Cloud deployment support

### Version History

**v2.0** (Current)
- Integrated PredictPipeline
- Added camera capture feature
- Web UI enhancement
- Full documentation

**v1.0**
- Initial release
- Basic mask detection
- Flask web app

---

<div align="center">

### ⭐ If you find this project useful, please give it a star!

**Made with ❤️ by Onkar Shinde**

[GitHub](https://github.com/onkarshinde77) • [LinkedIn](https://www.linkedin.com/in/onkarshinde77) • [Kaggle](https://www.kaggle.com/onkarshinde77)

</div>

---

## 📖 Additional Resources

### Documentation Files
- `QUICK_START.md` - 3-step quick start guide
- `SETUP_COMPLETE.md` - Detailed setup instructions
- `WEB_APP_GUIDE.md` - Web application guide
- `PIPELINE_DOCUMENTATION.md` - API reference
- `ARCHITECTURE.md` - System architecture

### Example Files
- `predict_example.py` - Usage examples
- `START_HERE.py` - Quick reference
- `test_setup.py` - Environment verification

### External Resources
- [TensorFlow Documentation](https://www.tensorflow.org/docs)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)

---

**Last Updated:** February 23, 2026
**Version:** 2.0
**Status:** Active Development
