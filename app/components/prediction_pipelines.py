import os
import base64
import threading
import cv2
import numpy as np

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b4,
    EfficientNet_B4_Weights
)
from components.face_crop import FaceCropper
from src import constant

IMG_SIZE   = constant.FINE_TUNE_IMG_SIZE
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_NAME = "EfficientNetB4.pth"

# --- Helper Functions ---

def draw_bounding_box(frame, coords, label, confidence):
    """Draw a box around the detected face with its label and confidence."""
    startX, startY, endX, endY = coords
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)

    text = f"{label}: {confidence * 100:.1f}%"
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (startX, startY - text_height - 15), (startX + text_width + 10, startY), color, cv2.FILLED)
    cv2.putText(frame, text, (startX + 5, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

def parse_model_prediction(prediction_score):
    """Convert raw model score (0 to 1) into Mask/No Mask label."""
    score = float(prediction_score[0])
    if score <= 0.5:
        return "Mask", 1.0 - score
    return "No Mask", score

def enhance_image_quality(frame):
    """Improve contrast and remove noise for dark or low-quality images."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21)
    return enhanced

def decode_base64_to_image(b64_string):
    """Convert a base64 string from the browser into an OpenCV image."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = np.frombuffer(base64.b64decode(b64_string), np.uint8)
    frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode base64 image")
    return frame

def load_face_mask_model(model_path, device):
    """Load the pre-trained PyTorch model into memory."""
    model = efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 1)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# --- Main Pipeline ---

class PredictPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.path.join(MODEL_DIR, MODEL_NAME)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = load_face_mask_model(model_path, self.device)

        # Run a dummy prediction to initialize PyTorch
        dummy_input = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            self.model(dummy_input)

        self.face_cropper = FaceCropper()
        self.inference_lock = threading.Lock()

    def run_inference_on_faces(self, face_images):
        """Pass the cropped faces through the PyTorch model."""
        if not face_images:
            return np.empty((0, 1), dtype=np.float32)

        preprocessed = []
        for face in face_images:
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
            normalized = resized.astype(np.float32) / 255.0
            chw = np.transpose(normalized, (2, 0, 1))
            preprocessed.append(chw)

        batch_tensor = torch.from_numpy(np.stack(preprocessed)).to(self.device)

        with self.inference_lock:
            with torch.no_grad():
                logits = self.model(batch_tensor)
                probs = torch.sigmoid(logits)

        return probs.cpu().numpy()

    def detect_in_single_frame(self, frame):
        """Detect faces and predict masks for a single frame."""
        faces = self.face_cropper.detect_faces(frame)
        annotated_frame = frame.copy()
        detections = []

        if not faces:
            return annotated_frame, detections

        cropped_faces = self.face_cropper.crop_faces(frame, faces)
        face_images = [crop["face"] for crop in cropped_faces]
        predictions = self.run_inference_on_faces(face_images)

        for idx, prediction in enumerate(predictions):
            label, confidence = parse_model_prediction(prediction)
            coords = cropped_faces[idx]["coords"]
            draw_bounding_box(annotated_frame, coords, label, confidence)
            
            detections.append({
                "face_num": idx + 1,
                "label": label,
                "confidence": f"{confidence * 100:.1f}%",
                "coords": tuple(int(c) for c in coords),
            })

        return annotated_frame, detections

    def detect_with_enhancement_retry(self, frame):
        """If no faces are found, try enhancing the image and detect again."""
        annotated_frame, detections = self.detect_in_single_frame(frame)
        if not detections:
            enhanced_frame = enhance_image_quality(frame)
            annotated_frame, detections = self.detect_in_single_frame(enhanced_frame)
        return annotated_frame, detections

    def detect_with_multiscale(self, frame):
        """Resize large images to help the face detector find smaller faces."""
        height, width = frame.shape[:2]
        annotated_frame, detections = self.detect_with_enhancement_retry(frame)

        if not detections and max(height, width) > 1920:
            scale = 1920 / max(height, width)
            small_frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            small_annotated, small_detections = self.detect_with_enhancement_retry(small_frame)

            if small_detections:
                inv_scale = 1.0 / scale
                for det in small_detections:
                    sx, sy, ex, ey = det["coords"]
                    det["coords"] = (int(sx * inv_scale), int(sy * inv_scale), int(ex * inv_scale), int(ey * inv_scale))
                
                annotated_frame = frame.copy()
                for det in small_detections:
                    conf_float = float(det["confidence"].rstrip("%")) / 100
                    draw_bounding_box(annotated_frame, det["coords"], det["label"], conf_float)
                detections = small_detections

        return annotated_frame, detections

    def detect_mask_in_image(self, frame):
        """Public method for checking standard uploaded images."""
        return self.detect_with_multiscale(frame)

    def detect_mask_in_base64(self, b64_string, flip_horizontal=False):
        """Public method for checking images from browser webcams."""
        frame = decode_base64_to_image(b64_string)

        if flip_horizontal:
            frame = cv2.flip(frame, 1)

        annotated_frame, detections = self.detect_with_multiscale(frame)

        if flip_horizontal:
            annotated_frame = cv2.flip(annotated_frame, 1)
            height, width = annotated_frame.shape[:2]
            for det in detections:
                sx, sy, ex, ey = det["coords"]
                det["coords"] = (width - ex, sy, width - sx, ey)

        return {"image": annotated_frame, "detections": detections, "num_faces": len(detections)}

    def detect_mask_in_video(self, video_path, frame_skip=0):
        """Generator to process a video file frame by frame."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frame_idx = 0
        last_detections = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_skip > 0 and frame_idx % (frame_skip + 1) != 0:
                    out_frame = frame.copy()
                    for det in last_detections:
                        conf_float = float(det["confidence"].rstrip("%")) / 100
                        draw_bounding_box(out_frame, det["coords"], det["label"], conf_float)
                    yield out_frame, last_detections
                else:
                    annotated_frame, detections = self.detect_mask_in_image(frame)
                    last_detections = detections
                    yield annotated_frame, detections

                frame_idx += 1
        finally:
            cap.release()

    def detect_mask_live_stream(self, camera_index=0, frame_skip=2, jpeg_quality=80, flip=True):
        """Generator to stream live webcam video to the browser."""
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No Camera Found", (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 128), 2)
            _, buf = cv2.imencode(".jpg", placeholder)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        frame_idx = 0
        last_detections = []
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if flip:
                    frame = cv2.flip(frame, 1)

                if frame_skip > 0 and frame_idx % (frame_skip + 1) != 0:
                    out_frame = frame.copy()
                    for det in last_detections:
                        conf_float = float(det["confidence"].rstrip("%")) / 100
                        draw_bounding_box(out_frame, det["coords"], det["label"], conf_float)
                else:
                    out_frame, last_detections = self.detect_in_single_frame(frame)

                _, buf = cv2.imencode(".jpg", out_frame, encode_params)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
                frame_idx += 1
        finally:
            cap.release()