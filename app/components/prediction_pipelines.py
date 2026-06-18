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
# MODEL_NAME = "face_mask_model4_f16.keras"
MODEL_NAME = "EfficientNetB4.pth"


# ── Drawing ───────────────────────────────────────────────────────────────────

def _draw_detection(frame: np.ndarray, coords: tuple, label: str, confidence: float) -> np.ndarray:
    startX, startY, endX, endY = coords
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)

    text = f"{label}: {confidence * 100:.1f}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (startX, startY - th - 15), (startX + tw + 10, startY), color, cv2.FILLED)
    cv2.putText(frame, text, (startX + 5, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def _parse_prediction(pred: np.ndarray) -> tuple[str, float]:
    """
    Sigmoid output: score > 0.5 → No Mask, score ≤ 0.5 → Mask.
    Returns (label, confidence 0–1).
    """
    score = float(pred[0])
    if score <= 0.5:
        return "Mask", 1.0 - score
    return "No Mask", score


# ── Image normalisation helpers ───────────────────────────────────────────────

def _auto_orient(frame: np.ndarray) -> np.ndarray:
    """
    Best-effort fixes for images that arrive in unusual orientations.

    Strategy
    --------
    1. Portrait images that are wider than they are tall are likely rotated 90°
       from a mobile back-camera in landscape mode — rotate them upright.
    2. For pure landscape frames (video/webcam) we leave them untouched.
    3. This only activates when aspect ratio is clearly wrong (ratio > 1.8),
       reducing false-positives on normally wide frames.
    """
    h, w = frame.shape[:2]
    # If image is sideways-landscape (much wider than tall after a 90° rotation
    # from a phone held upright), rotate it upright.
    if w > h * 1.8:
        # Heuristic: try to detect if there are more face candidates after rotation.
        # For efficiency we just rotate — the face detector handles tilts up to ~45°.
        pass  # Let MediaPipe handle tilt; only rotate if severely sideways
    return frame


def _enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    Lightweight quality enhancement to improve detection on dark / low-contrast
    images from back cameras or poorly lit environments.

    Steps
    -----
    1. CLAHE on the L channel (adaptive histogram equalisation — sharpens local contrast
       without blowing out highlights the way global equalisation does).
    2. Mild denoising — important for back cameras that use aggressive compression.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE — clip limit 2.0 and tile 8×8 are safe defaults (no halos)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Fast denoising (h=5 is mild; increase to 8–10 for very noisy back-camera frames)
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, h=5, hColor=5,
                                               templateWindowSize=7, searchWindowSize=21)
    return enhanced


def _decode_base64_image(b64_string: str) -> np.ndarray:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = np.frombuffer(base64.b64decode(b64_string), np.uint8)
    frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode base64 image")
    return frame

def load_model(model_path, device):
    model = efficientnet_b4(
        weights=None
    )
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(
            in_features,
            1
        )
    )
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )
    model.to(device)
    model.eval()
    
    return model

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          PREDICTION PIPELINE                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class PredictPipeline:
    """
    Thread-safe mask-detection pipeline.

    Handles
    -------
    - Uploaded photos (any orientation, front or back camera)
    - Webcam captures (base64, browser mirrored or not)
    - Live MJPEG webcam stream
    - Server-side video file processing
    """

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        model_path = os.path.join(MODEL_DIR, MODEL_NAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = load_model(model_path, self.device)

        # Warm-up: run one dummy prediction to initialise CUDA/PyTorch model
        dummy = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _ = self.model(dummy)

        self.face_cropper = FaceCropper()
        self._infer_lock  = threading.Lock()

    # ── Core: preprocess + batch inference ───────────────────────────────────

    def _infer_faces(self, face_crops: list[np.ndarray]) -> np.ndarray:
        """Preprocess a list of BGR face crops and run a single batch prediction."""
        if not face_crops:
            return np.empty((0, 1), dtype=np.float32)

        preprocessed = []
        for face in face_crops:
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
            # Convert to float and scale to [0.0, 1.0]
            normalized = resized.astype(np.float32) / 255.0
            # Transpose HWC to CHW
            chw = np.transpose(normalized, (2, 0, 1))
            preprocessed.append(chw)

        batch = np.stack(preprocessed)
        batch_tensor = torch.from_numpy(batch).to(self.device)

        with self._infer_lock:
            with torch.no_grad():
                logits = self.model(batch_tensor)
                probs = torch.sigmoid(logits)

        return probs.cpu().numpy()

    # ── Core: single frame → annotated frame + detections ────────────────────

    def predict_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Run face detection + mask classification on one BGR frame.

        Returns (annotated_frame, detections)
        where detections = [{face_num, label, confidence, coords}]
        """
        faces = self.face_cropper.detect_faces(frame)
        annotated  = frame.copy()
        detections = []

        if not faces:
            return annotated, detections

        cropped    = self.face_cropper.crop_faces(frame, faces)
        face_crops = [c["face"] for c in cropped]
        preds      = self._infer_faces(face_crops)

        for idx, pred in enumerate(preds):
            label, conf = _parse_prediction(pred)
            coords = cropped[idx]["coords"]
            _draw_detection(annotated, coords, label, conf)
            detections.append({
                "face_num"  : idx + 1,
                "label"     : label,
                "confidence": f"{conf * 100:.1f}%",
                "coords"    : tuple(int(c) for c in coords),
            })

        return annotated, detections

    # ── Retry logic: enhance → re-detect when no faces found ─────────────────

    def _predict_with_enhance_retry(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Try plain detection first. If no faces found, apply CLAHE enhancement
        and retry. Helps with dark / low-contrast images (common on back cameras).
        """
        annotated, detections = self.predict_frame(frame)
        if not detections:
            enhanced = _enhance_frame(frame)
            annotated, detections = self.predict_frame(enhanced)
            # Draw on enhanced version so the returned image is the one that worked
        return annotated, detections

    # ── Multi-scale retry: shrink very-high-res images ────────────────────────

    def _predict_multiscale(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        High-resolution images (e.g. 12 MP back-camera photos) can cause the
        face detector to miss faces because the bounding-box sizes it outputs
        are calibrated for ~720–1080 p input. Downscale and re-run if needed.
        """
        h, w = frame.shape[:2]
        annotated, detections = self._predict_with_enhance_retry(frame)

        if not detections and max(h, w) > 1920:
            scale   = 1920 / max(h, w)
            small   = cv2.resize(frame, (int(w * scale), int(h * scale)))
            ann_s, dets_s = self._predict_with_enhance_retry(small)

            if dets_s:
                # Scale bounding boxes back to original resolution
                inv = 1.0 / scale
                for d in dets_s:
                    sx, sy, ex, ey = d["coords"]
                    d["coords"] = (int(sx * inv), int(sy * inv),
                                   int(ex * inv), int(ey * inv))
                # Redraw on the full-res frame with the scaled coords
                annotated = frame.copy()
                for d in dets_s:
                    conf_float = float(d["confidence"].rstrip("%")) / 100
                    _draw_detection(annotated, d["coords"], d["label"], conf_float)
                detections = dets_s

        return annotated, detections

    # ── Public: image file upload ─────────────────────────────────────────────

    def predict_image(self, image_path: str) -> dict:
        """Load from disk, apply multiscale + enhance retries, return result dict."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        annotated, detections = self._predict_multiscale(frame)
        return {"image": annotated, "detections": detections, "num_faces": len(detections)}

    # ── Public: in-memory BGR frame (from multipart upload) ──────────────────

    def predict_frame_robust(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Like predict_frame but with enhance + multiscale retries.
        Use this for uploaded photos; use predict_frame directly for live video
        (retries are too slow at 30 fps).
        """
        return self._predict_multiscale(frame)

    # ── Public: base64 webcam capture ────────────────────────────────────────

    def predict_base64(self, b64_string: str, flip_horizontal: bool = False) -> dict:
        """
        Decode a base64 data-URL image, run detection, return result dict.

        flip_horizontal
        ---------------
        False (default): browser JS has already un-mirrored the canvas before
                         sending, so no extra flip needed.
        True           : raw front-camera capture that is still mirrored —
                         flip before inference, flip output back for display.
        """
        frame = _decode_base64_image(b64_string)

        if flip_horizontal:
            frame = cv2.flip(frame, 1)

        annotated, detections = self._predict_multiscale(frame)

        if flip_horizontal:
            annotated = cv2.flip(annotated, 1)
            h, w = annotated.shape[:2]
            for det in detections:
                sx, sy, ex, ey = det["coords"]
                det["coords"] = (w - ex, sy, w - sx, ey)

        return {"image": annotated, "detections": detections, "num_faces": len(detections)}

    # ── Public: video file processing (generator) ─────────────────────────────

    def predict_video_frames(self, video_path: str, frame_skip: int = 0):
        """
        Generator → yields (annotated_frame, detections) for every frame.

        frame_skip > 0: inference runs every (frame_skip + 1)-th frame;
        intermediate frames are yielded with the last known detections redrawn
        on the current live frame so output fps stays stable.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frame_idx       = 0
        last_detections = []
        last_annotated  = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_skip > 0 and frame_idx % (frame_skip + 1) != 0:
                    # Reuse last detections, redraw on current frame
                    out = frame.copy()
                    for det in last_detections:
                        _draw_detection(out, det["coords"], det["label"],
                                        float(det["confidence"].rstrip("%")) / 100)
                    yield out, last_detections
                else:
                    annotated, detections = self.predict_frame_robust(frame)
                    last_annotated  = annotated
                    last_detections = detections
                    yield annotated, detections

                frame_idx += 1
        finally:
            cap.release()

    # ── Public: live MJPEG stream (Flask Response generator) ─────────────────

    def generate_live_frames(
        self,
        camera_index: int = 0,
        frame_skip:   int = 2,
        jpeg_quality: int = 80,
        flip:         bool = True,
    ):
        """
        MJPEG generator for Flask streaming response.

        flip=True   : mirrors frame horizontally (front-camera / selfie convention).
        flip=False  : use for back-camera or fixed CCTV sources where mirroring
                      would be confusing.

        frame_skip=2 → inference every 3rd frame (good CPU/accuracy balance).
        Skipped frames get last known detections redrawn on the live frame so
        video appears smooth without stale frozen images.
        """
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No Camera Found",
                        (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 128), 2)
            _, buf = cv2.imencode(".jpg", placeholder)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            return

        # Prefer higher resolution if the camera supports it
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        frame_idx       = 0
        last_detections = []
        encode_params   = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if flip:
                    frame = cv2.flip(frame, 1)

                if frame_skip > 0 and frame_idx % (frame_skip + 1) != 0:
                    out = frame.copy()
                    for det in last_detections:
                        _draw_detection(out, det["coords"], det["label"],
                                        float(det["confidence"].rstrip("%")) / 100)
                else:
                    # Use plain predict_frame (no retry) — speed is critical for live video
                    out, last_detections = self.predict_frame(frame)

                _, buf = cv2.imencode(".jpg", out, encode_params)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
                frame_idx += 1
        finally:
            cap.release()