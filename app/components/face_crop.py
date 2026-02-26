"""
face_crop.py — Face detection with angle tolerance

Detection capability comparison:
    Angle            Caffe SSD    MediaPipe
    Front  0°          95%          99%
    Tilt  30°          60%          90%
    Side  45°          30%          85%
    Profile 90°        20%          70%
    Back of head        0%           0%
"""
import cv2
import numpy as np
import os
import urllib.request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models", "face_detector")

PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
CAFFEMODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
PROTOTXT_NAME = "deploy.prototxt"
CAFFEMODEL_NAME = "res10_300x300_ssd_iter_140000.caffemodel"

MEDIAPIPE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
MEDIAPIPE_MODEL_PATH = os.path.join(MODELS_DIR, "blaze_face_short_range.tflite")

_MEDIAPIPE_AVAILABLE = False
_MEDIAPIPE_NEW_API = False

try:
    try:
        _MEDIAPIPE_AVAILABLE = True
        _MEDIAPIPE_NEW_API = True
    except ImportError:
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
            _MEDIAPIPE_AVAILABLE = True
except ImportError:
    pass


def _download(url, dest):
    print(f"[FaceCropper] Downloading {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest)

def _ensure_caffe_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    prototxt_path = os.path.join(MODELS_DIR, PROTOTXT_NAME)
    caffemodel_path = os.path.join(MODELS_DIR, CAFFEMODEL_NAME)

    def _is_valid_prototxt(path):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.readline().strip().startswith(("name", "input", "layer", "#", "{"))
        except Exception:
            return False

    def _is_valid_caffemodel(path):
        try:
            return os.path.getsize(path) > 1_000_000
        except Exception:
            return False

    if not os.path.exists(prototxt_path) or not _is_valid_prototxt(prototxt_path):
        _download(PROTOTXT_URL, prototxt_path)
    if not os.path.exists(caffemodel_path) or not _is_valid_caffemodel(caffemodel_path):
        _download(CAFFEMODEL_URL, caffemodel_path)

    return prototxt_path, caffemodel_path

def _ensure_mediapipe_model():
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(MEDIAPIPE_MODEL_PATH) or os.path.getsize(MEDIAPIPE_MODEL_PATH) < 100_000:
        _download(MEDIAPIPE_MODEL_URL, MEDIAPIPE_MODEL_PATH)
    return MEDIAPIPE_MODEL_PATH


class FaceCropper:
    """
    Face detection and cropping.

    Backend priority:
      1. MediaPipe v0.10+ (Tasks API)     — best angle tolerance
      2. MediaPipe v0.9   (solutions API) — good angle tolerance
      3. Caffe ResNet-10 SSD             — front-facing only, always available
    """

    def __init__(self):
        self.confidence_threshold = 0.5

        if _MEDIAPIPE_AVAILABLE and _MEDIAPIPE_NEW_API:
            self._init_mediapipe_new()
        elif _MEDIAPIPE_AVAILABLE:
            self._init_mediapipe_old()
        else:
            print("[FaceCropper] MediaPipe not found — using Caffe SSD fallback.")
            print("              Install with: pip install mediapipe")
            self._init_caffe()

    def _init_mediapipe_new(self):
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            options = mp_vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=_ensure_mediapipe_model()),
                min_detection_confidence=self.confidence_threshold,
            )
            self.detector = mp_vision.FaceDetector.create_from_options(options)
            self.backend = "mediapipe_new"
            print("[FaceCropper] Using MediaPipe FaceDetector (Tasks API v0.10+)")
        except Exception as e:
            print(f"[FaceCropper] MediaPipe new API failed ({e}) — falling back to Caffe.")
            self._init_caffe()

    def _init_mediapipe_old(self):
        try:
            self.detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=self.confidence_threshold,
            )
            self.backend = "mediapipe_old"
            print("[FaceCropper] Using MediaPipe FaceDetection (solutions API v0.9)")
        except Exception as e:
            print(f"[FaceCropper] MediaPipe old API failed ({e}) — falling back to Caffe.")
            self._init_caffe()

    def _init_caffe(self):
        try:
            prototxt_path, caffemodel_path = _ensure_caffe_models()
            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
            self.backend = "caffe"
            print("[FaceCropper] Using Caffe ResNet-10 SSD (front-facing only)")
        except Exception as e:
            raise Exception(f"Failed to load Caffe face detector from '{MODELS_DIR}': {e}") from e

    def detect_faces(self, image: np.ndarray) -> list:
        """Detect faces in a BGR image. Returns list of (startX, startY, endX, endY, confidence)."""
        if self.backend == "mediapipe_new":
            return self._detect_mediapipe_new(image)
        elif self.backend == "mediapipe_old":
            return self._detect_mediapipe_old(image)
        else:
            return self._detect_caffe(image)

    def _detect_mediapipe_new(self, image: np.ndarray) -> list:
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        faces = []
        for detection in result.detections or []:
            score = detection.categories[0].score
            if score < self.confidence_threshold:
                continue
            bb = detection.bounding_box
            startX = max(0, bb.origin_x)
            startY = max(0, bb.origin_y)
            endX = min(w, bb.origin_x + bb.width)
            endY = min(h, bb.origin_y + bb.height)
            if endX > startX and endY > startY:
                faces.append((startX, startY, endX, endY, float(score)))
        return faces

    def _detect_mediapipe_old(self, image: np.ndarray) -> list:
        h, w = image.shape[:2]
        results = self.detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        faces = []
        for detection in results.detections or []:
            score = detection.score[0]
            if score < self.confidence_threshold:
                continue
            bb = detection.location_data.relative_bounding_box
            startX = max(0, int(bb.xmin * w))
            startY = max(0, int(bb.ymin * h))
            endX = min(w, int((bb.xmin + bb.width) * w))
            endY = min(h, int((bb.ymin + bb.height) * h))
            if endX > startX and endY > startY:
                faces.append((startX, startY, endX, endY, float(score)))
        return faces

    def _detect_caffe(self, image: np.ndarray) -> list:
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                startX, startY, endX, endY = box.astype("int")
                faces.append((max(0, startX), max(0, startY), min(w, endX), min(h, endY), float(confidence)))
        return faces

    def crop_faces(self, image: np.ndarray, faces: list) -> list:
        """Crop detected faces from the image. Returns list of dicts with 'face', 'coords', and 'confidence'."""
        cropped = []
        for startX, startY, endX, endY, confidence in faces:
            face = image[startY:endY, startX:endX]
            if face.size > 0:
                cropped.append({"face": face, "coords": (startX, startY, endX, endY), "confidence": float(confidence)})
        return cropped