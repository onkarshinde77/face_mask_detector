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

# Add the current directory to sys.path so we can import components
sys.path.insert(0, os.path.dirname(__file__))
from components.prediction_pipelines import PredictPipeline
from src import constant

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = constant.MAX_CONTENT_LENGTH

ALLOWED_IMAGE_TYPES = constant.ALLOWED_IMAGE_TYPES
ALLOWED_VIDEO_TYPES = constant.ALLOWED_VIDEO_TYPES

# Create a temporary directory to store uploaded videos
TEMP_DIR = tempfile.mkdtemp(prefix="facemask_")

# Load the model pipeline once when the server starts
try:
    pipeline = PredictPipeline()
except Exception as e:
    import traceback
    print("============================================================")
    print("PIPELINE LOAD ERROR: The face mask model will be unavailable.")
    traceback.print_exc()
    print("============================================================")
    pipeline = None


def require_pipeline(func):
    """Decorator to ensure the pipeline is loaded before running a route."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if pipeline is None:
            return jsonify({"error": "Model not loaded. Please check server logs."}), 503
        return func(*args, **kwargs)
    return wrapper


# Dictionary to store the progress of video processing jobs
VIDEO_JOBS = {}


# --- Helper Functions ---

def check_allowed_file(filename, allowed_extensions):
    """Check if the uploaded file has a valid extension."""
    if "." in filename:
        extension = filename.rsplit(".", 1)[1].lower()
        if extension in allowed_extensions:
            return True
    return False


def convert_frame_to_base64(frame):
    """Encode an OpenCV image frame to a base64 string for the browser."""
    success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    encoded_string = base64.b64encode(buffer).decode()
    return "data:image/jpeg;base64," + encoded_string


def create_summary_text(detections):
    """Create a readable summary of what was detected in the image."""
    if not detections:
        return "⚠️ No faces detected."
    
    mask_count = 0
    for detection in detections:
        if detection["label"] == "Mask":
            mask_count += 1
            
    no_mask_count = len(detections) - mask_count
    return f"✅ {len(detections)} Face(s) detected — {mask_count} With Mask / {no_mask_count} Without Mask"


def stream_video_file(file_path, mimetype):
    """Reads a video file in chunks so the browser can seek and stream it."""
    file_size = os.path.getsize(file_path)
    range_header = request.headers.get("Range")

    def read_chunks(start_byte, length):
        with open(file_path, "rb") as file:
            file.seek(start_byte)
            remaining_bytes = length
            while remaining_bytes > 0:
                chunk_size = min(8192, remaining_bytes)
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                remaining_bytes -= len(chunk)
                yield chunk

    if range_header:
        # Handle video seeking (e.g., when the user clicks the video timeline)
        range_value = range_header.strip().replace("bytes=", "")
        parts = range_value.split("-")
        
        start_byte = int(parts[0])
        if len(parts) > 1 and parts[1]:
            end_byte = int(parts[1])
        else:
            end_byte = file_size - 1
            
        length = end_byte - start_byte + 1
        
        return Response(
            stream_with_context(read_chunks(start_byte, length)),
            206, 
            mimetype=mimetype,
            headers={
                "Content-Range": f"bytes {start_byte}-{end_byte}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
            },
        )

    # If no range header is provided, stream the entire file
    return Response(
        stream_with_context(read_chunks(0, file_size)),
        mimetype=mimetype,
        headers={
            "Content-Length": str(file_size), 
            "Accept-Ranges": "bytes"
        },
    )


# --- Web Routes ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/live")
def live():
    return render_template("live.html")


@app.route("/video_feed")
@require_pipeline
def video_feed():
    camera_index = int(request.args.get("camera", 0))
    flip_param = request.args.get("flip", "true").lower()
    flip_video = (flip_param != "false")

    return Response(
        pipeline.detect_mask_live_stream(
            camera_index=camera_index,
            frame_skip=1,
            jpeg_quality=75,
            flip=flip_video,
        ),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/upload_photo", methods=["GET", "POST"])
def upload_photo():
    if request.method == "GET":
        return render_template("upload_photo.html")

    if pipeline is None:
        return jsonify({"success": False, "error": "Model not loaded."}), 503

    # Handle standard file upload (e.g., from file picker)
    uploaded_file = request.files.get("file")
    if uploaded_file and uploaded_file.filename:
        if not check_allowed_file(uploaded_file.filename, ALLOWED_IMAGE_TYPES):
            return jsonify({"success": False, "error": "Invalid image type."}), 400
            
        try:
            # Read the image bytes directly into an OpenCV frame
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({"success": False, "error": "Cannot decode image."}), 400

            # Process the image with our face mask detection pipeline
            annotated_frame, detections = pipeline.detect_mask_in_image(frame)
            return jsonify({
                "success": True,
                "image_b64": convert_frame_to_base64(annotated_frame),
                "detection": create_summary_text(detections),
                "detections": detections,
            })
        except Exception as error:
            return jsonify({"success": False, "error": str(error)}), 500

    # Handle base64 image (e.g., webcam capture from the browser)
    json_data = request.get_json(silent=True)
    if json_data and "image" in json_data:
        try:
            flip_horizontal = bool(json_data.get("flip_horizontal", False))
            result = pipeline.detect_mask_in_base64(json_data["image"], flip_horizontal=flip_horizontal)
            
            return jsonify({
                "success": True,
                "image_b64": convert_frame_to_base64(result["image"]),
                "detection": create_summary_text(result["detections"]),
                "detections": result["detections"],
            })
        except Exception as error:
            return jsonify({"success": False, "error": str(error)}), 500

    return jsonify({"success": False, "error": "No valid file or image data provided."}), 400


# --- Video Processing ---

def process_video_in_background(video_id, input_video_path):
    """Background task to run face mask detection on each frame of a video."""
    try:
        video_capture = cv2.VideoCapture(input_video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        if not fps:
            fps = 25.0  # Fallback to 25 frames per second
            
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            total_frames = 1
            
        video_capture.release()

        # Prepare to save the annotated output video
        output_video_path = os.path.join(TEMP_DIR, f"out_{video_id}.mp4")
        video_writer = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, 
            (width, height),
        )
        
        # Initialize job status
        VIDEO_JOBS[video_id]["total_frames"] = total_frames
        VIDEO_JOBS[video_id]["progress"] = 0
        
        frames_processed = 0
        # Iterate over each frame and process it
        for annotated_frame, detections in pipeline.detect_mask_in_video(input_video_path, frame_skip=0):
            video_writer.write(annotated_frame)
            frames_processed += 1
            
            # Update the progress percentage
            progress_percent = int((frames_processed / total_frames) * 100)
            VIDEO_JOBS[video_id]["progress"] = min(progress_percent, 99)

        video_writer.release()
        
        # Mark the job as completely finished
        VIDEO_JOBS[video_id]["status"] = "done"
        VIDEO_JOBS[video_id]["progress"] = 100
        VIDEO_JOBS[video_id]["output_path"] = output_video_path
        
    except Exception as error:
        VIDEO_JOBS[video_id]["status"] = "error"
        VIDEO_JOBS[video_id]["error"] = str(error)


@app.route("/upload_video", methods=["GET", "POST"])
def upload_video():
    if request.method == "GET":
        return render_template("upload_video.html")

    if pipeline is None:
        return jsonify({"success": False, "error": "Model not loaded."}), 503

    uploaded_file = request.files.get("file")
    if not uploaded_file or not uploaded_file.filename:
        return jsonify({"success": False, "error": "No file received."}), 400
        
    if not check_allowed_file(uploaded_file.filename, ALLOWED_VIDEO_TYPES):
        return jsonify({"success": False, "error": "Invalid file type."}), 400

    # Save the incoming video to a temporary path
    video_id = uuid.uuid4().hex
    safe_filename = secure_filename(uploaded_file.filename)
    input_video_path = os.path.join(TEMP_DIR, f"in_{video_id}_{safe_filename}")
    uploaded_file.save(input_video_path)

    # Initialize the video processing job record
    VIDEO_JOBS[video_id] = {
        "status": "processing",
        "progress": 0,
        "total_frames": 0,
        "output_path": None,
        "error": None,
    }
    
    # Start video processing in a new thread so we don't block the web server
    processing_thread = threading.Thread(
        target=process_video_in_background, 
        args=(video_id, input_video_path), 
        daemon=True
    )
    processing_thread.start()

    return jsonify({"success": True, "video_id": video_id})


@app.route("/video_status/<video_id>")
def video_status(video_id):
    """Endpoint for the frontend to poll the video processing progress."""
    job = VIDEO_JOBS.get(video_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
        
    response_data = {
        "status": job["status"], 
        "progress": job.get("progress", 0)
    }
    
    if job["status"] == "done":
        response_data["stream_url"] = url_for("video_stream", video_id=video_id)
        response_data["download_url"] = url_for("video_download", video_id=video_id)
        
    if job["status"] == "error":
        response_data["error"] = job.get("error", "Unknown error")
        
    return jsonify(response_data)


@app.route("/video_stream/<video_id>")
def video_stream(video_id):
    """Endpoint to stream the processed video back to the frontend."""
    job = VIDEO_JOBS.get(video_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
        
    return stream_video_file(job["output_path"], "video/mp4")


@app.route("/video_download/<video_id>")
def video_download(video_id):
    """Endpoint to download the processed video file."""
    job = VIDEO_JOBS.get(video_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
        
    output_path = job["output_path"]
    file_size = os.path.getsize(output_path)

    def generate_file_content():
        with open(output_path, "rb") as file:
            while True:
                chunk = file.read(8192)
                if not chunk:
                    break
                yield chunk

    return Response(
        stream_with_context(generate_file_content()),
        mimetype="video/mp4",
        headers={
            "Content-Disposition": f'attachment; filename="detected_{video_id}.mp4"',
            "Content-Length": str(file_size),
        },
    )


# --- Error Handlers ---

@app.errorhandler(413)
def too_large(error):
    return jsonify({"success": False, "error": "File too large (max 500 MB)."}), 413


@app.errorhandler(404)
def not_found(error):
    return render_template("index.html"), 404


if __name__ == "__main__":
    app.run(
        host=constant.APP_HOST, 
        port=constant.APP_PORT, 
        debug=constant.APP_DEBUG, 
        threaded=constant.APP_THREADED
    )