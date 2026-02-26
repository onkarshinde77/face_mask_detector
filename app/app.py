import os
import sys
import uuid
import base64
import threading
import tempfile
import cv2
import numpy as np
from flask import (
    Flask, render_template, request, Response,
    jsonify, stream_with_context, url_for,
)
from functools import wraps
from werkzeug.utils import secure_filename
sys.path.insert(0, os.path.dirname(__file__))
from components.prediction_pipelines import PredictPipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

ALLOWED_IMAGE = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}
ALLOWED_VIDEO = {"mp4", "avi", "mov", "mkv", "webm"}
TEMP_DIR      = tempfile.mkdtemp(prefix="facemask_")

# ── Pipeline (loaded once at startup) ────────────────────────────────────────
try:
    pipeline = PredictPipeline()
except Exception:
    import traceback
    print("=" * 60)
    print("PIPELINE LOAD ERROR — model will be unavailable")
    traceback.print_exc()
    print("=" * 60)
    pipeline = None


def _pipeline_required(fn):
    @wraps(fn)
    def _w(*a, **kw):
        if pipeline is None:
            return jsonify({"error": "Model not loaded."}), 503
        return fn(*a, **kw)
    return _w

# ── Video job store: { id: {status, progress, total_frames, output_path, error} }
VIDEO_JOBS: dict = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _allowed(filename: str, allowed_set: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


def _frame_to_b64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _build_summary(detections: list) -> str:
    if not detections:
        return "⚠️ No faces detected."
    m = sum(1 for d in detections if d["label"] == "Mask")
    n = len(detections) - m
    return f"✅ {len(detections)} Face(s) — {m} With Mask / {n} Without Mask"


def _stream_file(path: str, mimetype: str):
    """Generic file streamer with HTTP Range support (needed for <video> seek)."""
    size         = os.path.getsize(path)
    range_header = request.headers.get("Range")

    def _read(start: int, length: int):
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining:
                chunk = f.read(min(8192, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    if range_header:
        parts = range_header.strip().replace("bytes=", "").split("-")
        byte1 = int(parts[0])
        byte2 = int(parts[1]) if parts[1] else size - 1
        length = byte2 - byte1 + 1
        return Response(
            stream_with_context(_read(byte1, length)),
            206, mimetype=mimetype,
            headers={
                "Content-Range" : f"bytes {byte1}-{byte2}/{size}",
                "Accept-Ranges" : "bytes",
                "Content-Length": str(length),
            },
        )

    return Response(
        stream_with_context(_read(0, size)),
        mimetype=mimetype,
        headers={"Content-Length": str(size), "Accept-Ranges": "bytes"},
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/live")
def live():
    return render_template("live.html")


@app.route("/video_feed")
@_pipeline_required
def video_feed():
    camera_index = int(request.args.get("camera", 0))
    flip         = request.args.get("flip", "true").lower() != "false"

    return Response(
        pipeline.generate_live_frames(
            camera_index=camera_index,
            frame_skip=1,
            jpeg_quality=75,
            flip=flip,
        ),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@app.route("/upload_photo", methods=["GET", "POST"])
def upload_photo():
    if request.method == "GET":
        return render_template("upload_photo.html")

    if pipeline is None:
        return jsonify({"success": False, "error": "Model not loaded."}), 503

    # Branch A: multipart image file (from file picker or camera capture)
    file = request.files.get("file")
    if file and file.filename and _allowed(file.filename, ALLOWED_IMAGE):
        try:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if frame is None:
                return jsonify({"success": False, "error": "Cannot decode image."}), 400

            # Use robust path: multiscale + enhance retries
            # This handles both front-camera (low res, mirrored) and
            # back-camera (high res, dark, rotated) photos correctly.
            annotated, detections = pipeline.predict_frame_robust(frame)
            return jsonify({
                "success"   : True,
                "image_b64" : _frame_to_b64(annotated),
                "detection" : _build_summary(detections),
                "detections": detections,
            })
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 500

    # Branch B: base64 JSON (webcam capture from browser canvas)
    data = request.get_json(silent=True)
    if data and "image" in data:
        try:
            # flip_horizontal reflects whether the JS already un-mirrored the
            # canvas. Default False = JS handled it; pass True from the client
            # if the canvas was NOT flipped before encoding.
            flip = bool(data.get("flip_horizontal", False))
            result = pipeline.predict_base64(data["image"], flip_horizontal=flip)
            return jsonify({
                "success"   : True,
                "image_b64" : _frame_to_b64(result["image"]),
                "detection" : _build_summary(result["detections"]),
                "detections": result["detections"],
            })
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 500

    return jsonify({"success": False, "error": "No valid file or image data."}), 400


# ── Video processing ──────────────────────────────────────────────────────────
def _process_video_job(video_id: str, input_path: str):
    """Background thread: process video, write annotated output to TEMP_DIR."""
    try:
        cap    = cv2.VideoCapture(input_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.release()

        out_path = os.path.join(TEMP_DIR, f"out_{video_id}.mp4")
        writer   = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (width, height),
        )
        VIDEO_JOBS[video_id].update({"total_frames": total, "progress": 0})
        done = 0
        for ann_frame, _ in pipeline.predict_video_frames(input_path, frame_skip=1):
            writer.write(ann_frame)
            done += 1
            VIDEO_JOBS[video_id]["progress"] = min(int(done / total * 100), 99)

        writer.release()
        VIDEO_JOBS[video_id].update({
            "status"     : "done",
            "progress"   : 100,
            "output_path": out_path,
        })
    except Exception as exc:
        VIDEO_JOBS[video_id].update({"status": "error", "error": str(exc)})


@app.route("/upload_video", methods=["GET", "POST"])
def upload_video():
    if request.method == "GET":
        return render_template("upload_video.html")

    if pipeline is None:
        return jsonify({"success": False, "error": "Model not loaded."}), 503

    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"success": False, "error": "No file received."}), 400
    if not _allowed(file.filename, ALLOWED_VIDEO):
        return jsonify({"success": False, "error": "Invalid file type."}), 400

    video_id = uuid.uuid4().hex
    in_path  = os.path.join(TEMP_DIR, f"in_{video_id}_{secure_filename(file.filename)}")
    file.save(in_path)

    VIDEO_JOBS[video_id] = {
        "status"      : "processing",
        "progress"    : 0,
        "total_frames": 0,
        "output_path" : None,
        "error"       : None,
    }
    threading.Thread(
        target=_process_video_job, args=(video_id, in_path), daemon=True
    ).start()

    return jsonify({"success": True, "video_id": video_id})


@app.route("/video_status/<video_id>")
def video_status(video_id: str):
    job = VIDEO_JOBS.get(video_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    resp = {"status": job["status"], "progress": job.get("progress", 0)}
    if job["status"] == "done":
        resp["stream_url"]   = url_for("video_stream",   video_id=video_id)
        resp["download_url"] = url_for("video_download", video_id=video_id)
    if job["status"] == "error":
        resp["error"] = job.get("error", "Unknown error")
    return jsonify(resp)


@app.route("/video_stream/<video_id>")
def video_stream(video_id: str):
    job = VIDEO_JOBS.get(video_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
    return _stream_file(job["output_path"], "video/mp4")


@app.route("/video_download/<video_id>")
def video_download(video_id: str):
    job = VIDEO_JOBS.get(video_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
    path = job["output_path"]
    size = os.path.getsize(path)

    def _gen():
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                yield chunk

    return Response(
        stream_with_context(_gen()),
        mimetype="video/mp4",
        headers={
            "Content-Disposition": f'attachment; filename="detected_{video_id}.mp4"',
            "Content-Length"     : str(size),
        },
    )

# ── Error handlers ────────────────────────────────────────────────────────────
@app.errorhandler(413)
def too_large(e):
    return jsonify({"success": False, "error": "File too large (max 500 MB)."}), 413

@app.errorhandler(404)
def not_found(e):
    return render_template("index.html"), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)