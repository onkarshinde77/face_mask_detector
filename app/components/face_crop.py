import cv2
import numpy as np
import os
from ultralytics import YOLO

# Keep same directory structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models", "face_detector")

MODEL_NAME = "model.pt"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)


class FaceCropper:
    """
    Face detection and cropping using YOLOv8
    """

    def __init__(self):
        self.confidence_threshold = 0.5

        # Ensure model exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"YOLO model not found at {MODEL_PATH}\n"
                f"Place '{MODEL_NAME}' inside models/face_detector/"
            )

        # Load YOLO model
        self.model = YOLO(MODEL_PATH)

        print(f"[FaceCropper] Using YOLOv8 model: {MODEL_NAME}")

    def detect_faces(self, image: np.ndarray) -> list:
        """
        Detect faces using YOLOv8

        Returns:
            List of (startX, startY, endX, endY, confidence)
        """
        h, w = image.shape[:2]

        results = self.model(image, verbose=False)

        faces = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                conf = float(box.conf[0])

                if conf < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Clamp within image
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                faces.append((x1, y1, x2, y2, conf))

        return faces

    def crop_faces(self, image: np.ndarray, faces: list) -> list:
        """
        Crop detected faces

        Returns:
            List of dicts:
            {
                'face': cropped_image,
                'coords': (x1, y1, x2, y2),
                'confidence': float
            }
        """
        cropped = []

        for x1, y1, x2, y2, conf in faces:
            face = image[y1:y2, x1:x2]

            if face.size > 0:
                cropped.append({
                    "face": face,
                    "coords": (x1, y1, x2, y2),
                    "confidence": conf
                })

        return cropped