import os
import sys
import base64
import threading
import cv2
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b4

from src.constant import ARTIFACT_DIR, MODEL_DIR, IMG_SIZE, MODEL_NAME
from src.exception.exception import CustomException
from src.logger.logger import logging
from src.components.face_crop import FaceCropper


# ImageNet normalization — same values used during training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Preprocessing pipeline applied to every face crop before model inference
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def draw_bounding_box(frame, coords, label, confidence):
    """Draw a colored bounding box and label on the frame."""
    startX, startY, endX, endY = coords
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)

    text = f"{label}: {confidence * 100:.1f}%"
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(
        frame,
        (startX, startY - text_height - 15),
        (startX + text_width + 10, startY),
        color, cv2.FILLED,
    )
    cv2.putText(frame, text, (startX + 5, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def parse_model_output(raw_score):
    """Convert raw sigmoid score to (label, confidence)."""
    score = float(raw_score)
    if score <= 0.5:
        return "Mask", 1.0 - score
    return "No Mask", score


def load_pytorch_model(model_path, device):
    """Reconstruct the EfficientNetB4 architecture and load saved weights."""
    model = efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


class PredictPipeline:
    """Thread-safe prediction pipeline using PyTorch EfficientNetB4."""

    def __init__(self):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"PredictPipeline using device: {self.device}")

            model_path = os.path.join(ARTIFACT_DIR, MODEL_DIR, MODEL_NAME)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")

            logging.info(f"Loading model from: {model_path}")
            self.model = load_pytorch_model(model_path, self.device)

            # Warm-up: run one dummy forward pass to initialize the model graph
            dummy = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                self.model(dummy)
            logging.info("Model warm-up complete")

            self.face_cropper = FaceCropper()
            self.inference_lock = threading.Lock()

            logging.info("PredictPipeline initialized successfully")

        except Exception as e:
            logging.error(f"PredictPipeline init error: {e}")
            raise CustomException(e, sys)

    def preprocess_face_crops(self, face_crops):
        """
        Convert a list of BGR face crops (numpy arrays) into a
        normalized PyTorch batch tensor ready for the model.
        """
        tensors = []
        for face in face_crops:
            # Convert BGR to RGB for torchvision
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            tensor = preprocess(rgb)
            tensors.append(tensor)

        # Stack into a batch of shape (N, 3, IMG_SIZE, IMG_SIZE)
        return torch.stack(tensors).to(self.device)

    def run_inference(self, face_crops):
        """Run the model on preprocessed face crops. Returns scores as a numpy array."""
        if not face_crops:
            return []

        batch_tensor = self.preprocess_face_crops(face_crops)

        with self.inference_lock:
            with torch.no_grad():
                logits = self.model(batch_tensor)
                scores = torch.sigmoid(logits).cpu().numpy()

        return scores

    def predict_frame(self, frame):
        """Detect faces in a frame and classify each one as Mask/No Mask."""
        try:
            faces = self.face_cropper.detect_faces(frame)
            annotated_frame = frame.copy()
            detections = []

            if not faces:
                return annotated_frame, detections

            cropped_faces = self.face_cropper.crop_faces(frame, faces)
            face_images   = [crop["face"] for crop in cropped_faces]
            scores        = self.run_inference(face_images)

            for idx, score in enumerate(scores):
                label, confidence = parse_model_output(score[0])
                coords = cropped_faces[idx]["coords"]
                draw_bounding_box(annotated_frame, coords, label, confidence)

                detections.append({
                    "face_num"  : idx + 1,
                    "label"     : label,
                    "confidence": f"{confidence * 100:.1f}%",
                    "coords"    : coords,
                })

            return annotated_frame, detections

        except Exception as e:
            logging.error(f"predict_frame error: {e}")
            raise CustomException(e, sys)

    def predict_image(self, image_path):
        """Load an image from disk and run face mask detection."""
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")

            annotated_frame, detections = self.predict_frame(frame)

            return {
                "image"     : annotated_frame,
                "detections": detections,
                "num_faces" : len(detections),
            }

        except Exception as e:
            logging.error(f"predict_image error: {e}")
            raise CustomException(e, sys)

    def predict_base64(self, b64_string):
        """Decode a base64 image from the browser and run detection."""
        try:
            if "," in b64_string:
                b64_string = b64_string.split(",", 1)[1]

            img_bytes = np.frombuffer(base64.b64decode(b64_string), np.uint8)
            frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

            if frame is None:
                raise ValueError("Could not decode base64 image")

            annotated_frame, detections = self.predict_frame(frame)

            return {
                "image"     : annotated_frame,
                "detections": detections,
                "num_faces" : len(detections),
            }

        except Exception as e:
            logging.error(f"predict_base64 error: {e}")
            raise CustomException(e, sys)

    def predict_video_frames(self, video_path, frame_skip=0):
        """Generator that yields (annotated_frame, detections) for each video frame."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frame_idx       = 0
        last_annotated  = None
        last_detections = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_skip > 0 and frame_idx % (frame_skip + 1) != 0:
                    # Reuse the last result to maintain output FPS
                    out = last_annotated if last_annotated is not None else frame
                    yield out, last_detections
                else:
                    annotated_frame, detections = self.predict_frame(frame)
                    last_annotated  = annotated_frame
                    last_detections = detections
                    yield annotated_frame, detections

                frame_idx += 1

        finally:
            cap.release()
            logging.info(f"predict_video_frames: {frame_idx} frames processed")

    def generate_live_frames(self, camera_index=0, frame_skip=1, jpeg_quality=80, flip=True):
        """Generator for a live MJPEG webcam stream to send to the browser."""
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No Camera Found", (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 128), 2)
            _, buf = cv2.imencode(".jpg", placeholder)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        frame_idx      = 0
        last_annotated = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if flip:
                    frame = cv2.flip(frame, 1)

                if frame_skip > 0 and frame_idx % (frame_skip + 1) != 0:
                    out = last_annotated if last_annotated is not None else frame
                else:
                    out, _ = self.predict_frame(frame)
                    last_annotated = out

                encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                _, buf = cv2.imencode(".jpg", out, encode_params)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
                frame_idx += 1

        finally:
            cap.release()
            logging.info("generate_live_frames: camera released")
