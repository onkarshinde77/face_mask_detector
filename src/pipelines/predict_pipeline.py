import os
import sys
import base64
import threading
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

from src.constant import ARTIFACT_DIR, MODEL_DIR, IMG_SIZE, MODEL_NAME
from src.exception.exception import CustomException
from src.logger.logger import logging
from src.components.face_crop import FaceCropper


# ── CPU / GPU config ─────────────────────────────────────────────────────────
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'          # Force CPU
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(4)
logging.info("CPU-optimized inference mode enabled")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                      NUMPY-TO-JSON CONVERTER                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _to_json_serializable(val):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    elif isinstance(val, (np.floating, np.float32, np.float64)):
        return float(val)
    elif isinstance(val, np.ndarray):
        return val.tolist()
    return val


def _draw_detection(frame: np.ndarray, coords: tuple,
                    label: str, confidence: float) -> np.ndarray:
    """
    Draw a bounding box + filled label banner on *frame* (in-place).
    Returns the same frame for convenience.
    """
    startX, startY, endX, endY = coords
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)   # BGR

    # Bounding box
    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)

    # Label text
    text = f"{label}: {confidence * 100:.1f}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    # Filled background rectangle for text
    cv2.rectangle(
        frame,
        (startX, startY - th - 15),
        (startX + tw + 10, startY),
        color,
        cv2.FILLED,
    )

    # White text on top
    cv2.putText(
        frame, text,
        (startX + 5, startY - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (255, 255, 255), 2,
    )
    return frame


def _parse_prediction(pred: np.ndarray) -> tuple[str, float]:
    if pred.shape[0] == 1:
        score = float(pred[0])
        label = "No Mask" if score > 0.5 else "Mask"
        conf  = score if score > 0.5 else 1.0 - score
    else:
        idx   = int(np.argmax(pred))
        label = "No Mask" if idx == 1 else "Mask"
        conf  = float(pred[idx])
    return label, conf

class PredictPipeline:
    _instance_lock = threading.Lock()

    def __init__(self):
        try:
            self.model_path = os.path.join(ARTIFACT_DIR, MODEL_DIR, MODEL_NAME)
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")

            logging.info(f"Loading model from: {self.model_path}")
            self.model = load_model(self.model_path)
            self.output_shape = self.model.output_shape

            # Warm-up: run one dummy prediction to initialise TF graph
            dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32")
            self.model.predict(dummy, verbose=0)
            logging.info("Model warm-up complete")

            # Face detectors
            self.face_cropper = FaceCropper()

            # Haar cascade as backup
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            # Per-instance lock for thread-safe inference
            self._infer_lock = threading.Lock()

            logging.info("PredictPipeline initialised successfully")

        except Exception as e:
            logging.error(f"PredictPipeline init error: {e}")
            raise CustomException(e, sys)

    # ── Core: preprocess + batch-predict a list of face crops ────────────────
    def _infer_faces(self, face_crops: list[np.ndarray]) -> np.ndarray:
        """
        Preprocess a list of BGR face crops and run a single batch prediction.
        Thread-safe via self._infer_lock.
        """
        preprocessed = []
        for face in face_crops:
            rgb     = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
            arr     = np.asarray(resized, dtype="float32")
            arr     = preprocess_input(arr)
            preprocessed.append(arr)

        batch = np.array(preprocessed, dtype="float32")
        with self._infer_lock:
            preds = self.model.predict(batch, verbose=0)
        return preds

    # ── Public: predict a single raw BGR frame ───────────────────────────────
    def predict_frame(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, list[dict]]:
        try:
            faces = self.face_cropper.detect_faces(frame)
            annotated  = frame.copy()
            detections = []

            if not faces or len(faces) == 0:
                return annotated, detections

            cropped = self.face_cropper.crop_faces(frame, faces)
            face_crops = [c["face"] for c in cropped]
            preds      = self._infer_faces(face_crops)

            for idx, pred in enumerate(preds):
                label, conf = _parse_prediction(pred)
                coords = cropped[idx]["coords"]          # (startX, startY, endX, endY)
                _draw_detection(annotated, coords, label, conf)

                # Convert numpy int64 to Python int for JSON serialization
                coords_serializable = [int(c) for c in coords]

                detections.append({
                    "face_num"  : idx + 1,
                    "label"     : label,
                    "confidence": f"{conf * 100:.1f}%",
                    "coords"    : coords_serializable,
                })

            return annotated, detections

        except Exception as e:
            logging.error(f"predict_frame error: {e}")
            raise CustomException(e, sys)

    # ── Public: predict from image path (used by photo-upload route) ──────────
    def predict_image(self, image_path: str) -> dict:
        """
        Load an image from disk, run detection, return result dict.

        Returns
        -------
        {
            'image'      : annotated BGR ndarray,
            'detections' : [{ label, face_num, confidence, coords }],
            'num_faces'  : int,
        }
        """
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")

            annotated, detections = self.predict_frame(frame)

            return {
                "image"     : annotated,
                "detections": detections,
                "num_faces" : len(detections),
            }

        except Exception as e:
            logging.error(f"predict_image error: {e}")
            raise CustomException(e, sys)

    # ── Public: predict from base64 string (used by webcam-capture route) ─────
    def predict_base64(self, b64_string: str) -> dict:
        """
        Decode a data-URL base64 image string, run detection, return result dict.

        Returns same shape as predict_image().
        """
        try:
            # Strip data-URL header if present
            if "," in b64_string:
                b64_string = b64_string.split(",", 1)[1]

            img_bytes = np.frombuffer(base64.b64decode(b64_string), np.uint8)
            frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

            if frame is None:
                raise ValueError("Could not decode base64 image")

            annotated, detections = self.predict_frame(frame)

            return {
                "image"     : annotated,
                "detections": detections,
                "num_faces" : len(detections),
            }

        except Exception as e:
            logging.error(f"predict_base64 error: {e}")
            raise CustomException(e, sys)

    # ── Public: generator for server-side video processing ───────────────────
    def predict_video_frames(
        self,
        video_path: str,
        frame_skip: int = 0,
    ):
        """
        Generator that yields (annotated_frame, detections) for each frame.

        Parameters
        ----------
        video_path  : str   path to input video file
        frame_skip  : int   process every (frame_skip + 1)-th frame;
                            skipped frames are yielded with the last detection
                            drawn on them (keeps output fps stable).

        Yields
        ------
        (annotated_frame: np.ndarray, detections: list[dict])
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frame_idx      = 0
        last_annotated = None
        last_detections: list[dict] = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_skip > 0 and frame_idx % (frame_skip + 1) != 0:
                    # Re-use previous detections on this raw frame to maintain fps
                    yield (frame if last_annotated is None else last_annotated.copy(),
                           last_detections)
                else:
                    annotated, detections = self.predict_frame(frame)
                    last_annotated  = annotated
                    last_detections = detections
                    yield annotated, detections

                frame_idx += 1

        finally:
            cap.release()
            logging.info(f"predict_video_frames: {frame_idx} frames processed")

    # ── Public: live webcam generator (for Flask MJPEG stream) ───────────────
    def generate_live_frames(
        self,
        camera_index: int  = 0,
        frame_skip: int    = 1,
        jpeg_quality: int  = 80,
    ):
        """
        Generator that yields MJPEG-ready bytes for the Flask Response.

        frame_skip=1  → process every other frame (halves CPU on low-end hw)
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            # Yield a single "no camera" placeholder frame and stop
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder, "No Camera Found",
                (140, 240), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 200, 128), 2,
            )
            _, buf = cv2.imencode(".jpg", placeholder)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            return

        frame_idx      = 0
        last_annotated = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_skip > 0 and frame_idx % (frame_skip + 1) != 0:
                    out = last_annotated if last_annotated is not None else frame
                else:
                    out, _ = self.predict_frame(frame)
                    last_annotated = out

                encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                _, buf = cv2.imencode(".jpg", out, encode_params)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )
                frame_idx += 1

        finally:
            cap.release()
            logging.info("generate_live_frames: camera released")
