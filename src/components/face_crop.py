import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO
from src.logger.logger import logging
from src.exception.exception import CustomException


class FaceCropper:
    """
    Face detection and cropping using YOLOv8 Face model
    """
    def __init__(self):
        try:
            # Project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            face_detector_dir = os.path.join(project_root, "face_detector")

            model_path = os.path.join(face_detector_dir, "model.pt")

            logging.info(f"Looking for YOLO model in: {model_path}")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"YOLO model not found: {model_path}")

            # Load YOLO model
            self.model = YOLO(model_path)
            self.confidence_threshold = 0.5
            logging.info("YOLOv8 FaceCropper initialized successfully")

        except Exception as e:
            logging.error(f"Error initializing YOLO FaceCropper: {str(e)}")
            raise CustomException(e, sys)

    def detect_faces(self, image):
        """
        Detect faces using YOLOv8
        """
        try:
            h, w = image.shape[:2]
            results = self.model(image, verbose=False)
            faces = []

            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    conf = float(box.conf[0])

                    if conf > self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # Clamp values
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(w, x2)
                        y2 = min(h, y2)

                        faces.append((x1, y1, x2, y2, conf))
            logging.info(f"Detected {len(faces)} face(s)")
            return faces

        except Exception as e:
            logging.error(f"Error detecting faces: {str(e)}")
            raise CustomException(e, sys)

    def crop_faces(self, image, faces):
        """
        Crop detected faces
        """
        try:
            cropped_faces = []

            for x1, y1, x2, y2, conf in faces:
                face = image[y1:y2, x1:x2]

                cropped_faces.append({
                    "face": face,
                    "coords": (x1, y1, x2, y2),
                    "confidence": conf
                })

            logging.info(f"Cropped {len(cropped_faces)} face(s)")
            return cropped_faces

        except Exception as e:
            logging.error(f"Error cropping faces: {str(e)}")
            raise CustomException(e, sys)