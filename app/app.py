import os
import sys
import uuid
import threading
import cv2
from flask import (
    Flask, render_template, request, Response,
    jsonify, send_from_directory, url_for,
)
from werkzeug.utils import secure_filename

# ── Import real pipeline ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from src.pipelines.predict_pipeline import PredictPipeline
from src.logger.logger import logging

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                            APP CONFIG                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB
app.config["UPLOAD_FOLDER"]      = os.path.join(os.path.dirname(__file__), "uploads")
app.config["PROCESSED_FOLDER"]   = os.path.join(os.path.dirname(__file__), "processed")

ALLOWED_IMAGE = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}
ALLOWED_VIDEO = {"mp4", "avi", "mov", "mkv", "webm"}

os.makedirs(app.config["UPLOAD_FOLDER"],    exist_ok=True)
os.makedirs(app.config["PROCESSED_FOLDER"], exist_ok=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║            PIPELINE — loaded ONCE at startup (not per-request)          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

try:
    pipeline = PredictPipeline()
    logging.info("PredictPipeline ready.")
except Exception as _exc:
    logging.error(f"Failed to load PredictPipeline: {_exc}")
    pipeline = None     # app returns 503 on any inference route


def _pipeline_required(fn):
    """Decorator: return 503 JSON if the pipeline failed to initialise."""
    from functools import wraps
    @wraps(fn)
    def _wrapper(*args, **kwargs):
        if pipeline is None:
            return jsonify({"error": "Model not loaded. Check server logs."}), 503
        return fn(*args, **kwargs)
    return _wrapper

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       IN-MEMORY VIDEO JOB STORE                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

VIDEO_JOBS: dict = {}   # { video_id: { status, output, message } }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                             HELPERS                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _allowed(filename: str, allowed_set: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


def _build_summary(detections: list) -> str:
    """Build a human-readable summary string from a detection list."""
    if not detections:
        return "⚠️ No faces detected."
    mask_n    = sum(1 for d in detections if d["label"] == "Mask")
    no_mask_n = len(detections) - mask_n
    return (
        f"✅ {len(detections)} Face(s) detected — "
        f"{mask_n} With Mask / {no_mask_n} Without Mask"
    )


def _save_annotated(frame, folder: str, prefix: str = "result") -> str:
    """Save an annotated BGR frame as JPEG and return the filename."""
    filename = f"{prefix}_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(os.path.join(folder, filename), frame)
    return filename


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                              ROUTES                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Home ──────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── Live camera ───────────────────────────────────────────────────────────────
@app.route("/live")
def live():
    return render_template("live.html")


@app.route("/video_feed")
@_pipeline_required
def video_feed():
    """
    MJPEG stream consumed by <img src="/video_feed"> in live.html.

    frame_skip=1  → run inference every 2nd frame (halves CPU load).
    jpeg_quality=75 → good quality/bandwidth balance for streaming.
    """
    return Response(
        pipeline.generate_live_frames(
            camera_index=0,
            frame_skip=1,
            jpeg_quality=75,
        ),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ── Photo detection ───────────────────────────────────────────────────────────
@app.route("/upload_photo", methods=["GET", "POST"])
def upload_photo():
    if request.method == "GET":
        return render_template(
            "upload_photo.html",
            image_path=None, detection=None, detections=[],
        )

    if pipeline is None:
        return render_template(
            "upload_photo.html",
            image_path=None,
            detection="❌ Model not loaded. Check server logs.",
            detections=[],
        )

    # ── Branch A: multipart image file upload ────────────────────────────────
    file = request.files.get("file")
    if file and file.filename:
        if not _allowed(file.filename, ALLOWED_IMAGE):
            return render_template(
                "upload_photo.html",
                image_path=None,
                detection="❌ Invalid file type. Upload a valid image.",
                detections=[],
            )

        orig_name = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
        orig_path = os.path.join(app.config["UPLOAD_FOLDER"], orig_name)
        file.save(orig_path)
        logging.info(f"Saved upload: {orig_path}")

        try:
            result = pipeline.predict_image(orig_path)
        except Exception as exc:
            logging.error(f"predict_image error: {exc}")
            return render_template(
                "upload_photo.html",
                image_path=None,
                detection=f"❌ Detection failed: {exc}",
                detections=[],
            )

        out_name = _save_annotated(
            result["image"], app.config["UPLOAD_FOLDER"], prefix="result"
        )
        return render_template(
            "upload_photo.html",
            image_path=url_for("uploaded_file", filename=out_name),
            detection=_build_summary(result["detections"]),
            detections=result["detections"],
        )

    # ── Branch B: JSON base64 from webcam capture ────────────────────────────
    data = request.get_json(silent=True)
    if data and "image" in data:
        try:
            result = pipeline.predict_base64(data["image"])
        except Exception as exc:
            logging.error(f"predict_base64 error: {exc}")
            return jsonify({"success": False, "error": str(exc)}), 500

        out_name = _save_annotated(
            result["image"], app.config["UPLOAD_FOLDER"], prefix="capture"
        )
        return jsonify({
            "success"   : True,
            "image_path": url_for("uploaded_file", filename=out_name),
            "detection" : _build_summary(result["detections"]),
            "detections": result["detections"],
        })

    return render_template(
        "upload_photo.html", image_path=None, detection=None, detections=[]
    )


# ── Static file serving ───────────────────────────────────────────────────────
@app.route("/uploads/<filename>")
def uploaded_file(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/processed/<filename>")
def processed_file(filename: str):
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename)


# ── Video upload + background processing ─────────────────────────────────────

def _process_video_job(video_id: str, input_path: str) -> None:
    """
    Background thread.
    Uses predict_video_frames() generator — no cv2.imshow, fully headless.

    frame_skip=2 → process every 3rd frame.
    Increases throughput ~3× on CPU. Set to 0 for per-frame accuracy.
    """
    try:
        # Read video properties before starting generator
        cap    = cv2.VideoCapture(input_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        out_name = f"processed_{video_id}.mp4"
        out_path = os.path.join(app.config["PROCESSED_FOLDER"], out_name)
        writer   = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        frame_count = 0
        for ann_frame, _ in pipeline.predict_video_frames(input_path, frame_skip=2):
            writer.write(ann_frame)
            frame_count += 1

        writer.release()
        logging.info(f"Job {video_id}: {frame_count} frames → {out_path}")
        VIDEO_JOBS[video_id] = {"status": "done", "output": out_name}

    except Exception as exc:
        logging.error(f"Video job {video_id} failed: {exc}")
        VIDEO_JOBS[video_id] = {
            "status": "error", "output": None, "message": str(exc)
        }


@app.route("/upload_video", methods=["GET", "POST"])
def upload_video():
    if request.method == "GET":
        return render_template("upload_video.html", processing=False, video_id=None)

    if pipeline is None:
        return jsonify({"success": False, "error": "Model not loaded."}), 503

    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"success": False, "error": "No file received."}), 400

    if not _allowed(file.filename, ALLOWED_VIDEO):
        return jsonify({"success": False, "error": "Invalid file type."}), 400

    video_id  = uuid.uuid4().hex
    safe_name = secure_filename(f"{video_id}_{file.filename}")
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    file.save(save_path)
    logging.info(f"Saved video upload: {save_path}")

    VIDEO_JOBS[video_id] = {"status": "processing", "output": None}
    threading.Thread(
        target=_process_video_job,
        args=(video_id, save_path),
        daemon=True,
    ).start()

    return jsonify({"success": True, "video_id": video_id})


@app.route("/video_status/<video_id>")
def video_status(video_id: str):
    job = VIDEO_JOBS.get(video_id)
    if job is None:
        return jsonify({"status": "not_found"}), 404

    if job["status"] == "done":
        return jsonify({
            "status"      : "done",
            "download_url": url_for("processed_file", filename=job["output"]),
        })
    if job["status"] == "error":
        return jsonify({
            "status" : "error",
            "message": job.get("message", "Unknown error"),
        })
    return jsonify({"status": "processing"})


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          ERROR HANDLERS                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@app.errorhandler(413)
def too_large(e):
    return jsonify({"success": False, "error": "File too large (max 500 MB)."}), 413

@app.errorhandler(404)
def not_found(e):
    return render_template("index.html"), 404


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                           ENTRY POINT                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    # threaded=True is mandatory: MJPEG stream + background video jobs must
    # run concurrently.  Never use debug=True in production.
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
