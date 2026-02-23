import os
import uuid
import json
import threading
import time
import cv2
import numpy as np
from flask import (
    Flask, render_template, request, Response,
    jsonify, send_from_directory, url_for
)
from werkzeug.utils import secure_filename

# ── Optional: load your real model here ─────────────────────────────────────
# Example using a Caffe-based face + mask classifier:
#
#   FACE_NET  = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
#   MASK_NET  = tensorflow / keras model
# For portability this file ships with a DEMO detector that draws a green/red
# box based on simple brightness heuristics so every route still works without
# any model weights.  Replace `detect_masks_in_frame()` with your real logic.
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024   # 200 MB
app.config['UPLOAD_FOLDER']     = 'uploads'
app.config['PROCESSED_FOLDER']  = 'processed'

ALLOWED_IMAGE = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
ALLOWED_VIDEO = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

os.makedirs(app.config['UPLOAD_FOLDER'],    exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# ── In-memory job store ──────────────────────────────────────────────────────
# { video_id: { 'status': 'processing'|'done'|'error', 'output': path } }
VIDEO_JOBS: dict[str, dict] = {}


# ── Helper: allowed extension checks ────────────────────────────────────────
def allowed_image(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE

def allowed_video(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        MASK DETECTION LOGIC                             ║
# ║  Replace the body of detect_masks_in_frame() with your real model.      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def detect_masks_in_frame(frame: np.ndarray):
    """
    Demo detector: uses OpenCV's Haar-cascade face detector.
    Returns:
        annotated_frame  – BGR numpy array with bounding boxes drawn
        detections       – list of dicts  { label, face_num, confidence }
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Haar cascade ships with OpenCV – no extra download needed
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    detections = []
    annotated  = frame.copy()

    for i, (x, y, w, h) in enumerate(faces):
        face_roi = gray[y:y+h, x:x+w]
        mean_brightness = float(np.mean(face_roi))

        # ── Demo heuristic (replace with your model inference) ───────────
        # Your model would do something like:
        #   blob = cv2.dnn.blobFromImage(frame[y:y+h, x:x+w], ...)
        #   mask_net.setInput(blob)
        #   (no_mask_prob, mask_prob) = mask_net.forward()[0]
        has_mask   = mean_brightness > 100          # placeholder
        confidence = round(abs(mean_brightness - 127.5) / 127.5 * 40 + 60, 1)
        label      = 'Mask' if has_mask else 'No Mask'
        color      = (0, 200, 128) if has_mask else (60, 69, 255)   # BGR

        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            annotated,
            f'{label} {confidence}%',
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )
        detections.append({'label': label, 'face_num': i + 1, 'confidence': f'{confidence}%'})

    return annotated, detections


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                              ROUTES                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Home ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

# ── Live camera feed ─────────────────────────────────────────────────────────
def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Yield a placeholder frame if no camera is available
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, 'No Camera Found', (140, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 128), 2)
        _, buf = cv2.imencode('.jpg', placeholder)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            annotated, _ = detect_masks_in_frame(frame)
            _, buffer = cv2.imencode('.jpg', annotated)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()


@app.route('/live')
def live():
    return render_template('live.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ── Photo upload / capture ───────────────────────────────────────────────────
@app.route('/upload_photo', methods=['GET', 'POST'])
def upload_photo():
    if request.method == 'GET':
        return render_template('upload_photo.html',
                               image_path=None, detection=None, detections=[])

    # ── POST: file upload ────────────────────────────────────────────────────
    if 'file' in request.files and request.files['file'].filename:
        file = request.files['file']
        if not allowed_image(file.filename):
            return render_template('upload_photo.html',
                                   image_path=None,
                                   detection='❌ Invalid file type.',
                                   detections=[])

        filename  = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        frame = cv2.imread(save_path)
        if frame is None:
            return render_template('upload_photo.html',
                                   image_path=None,
                                   detection='❌ Could not read image.',
                                   detections=[])

        annotated, detections = detect_masks_in_frame(frame)

        out_name = f"result_{filename}"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        cv2.imwrite(out_path, annotated)

        mask_count    = sum(1 for d in detections if d['label'] == 'Mask')
        no_mask_count = len(detections) - mask_count
        summary = (
            f"✅ {len(detections)} Face(s) — {mask_count} With Mask / {no_mask_count} Without Mask"
            if detections else "⚠️ No faces detected."
        )

        image_url = url_for('uploaded_file', filename=out_name)
        return render_template('upload_photo.html',
                               image_path=image_url,
                               detection=summary,
                               detections=detections)

    # ── POST: base64 capture from webcam ────────────────────────────────────
    data = request.get_json(silent=True)
    if data and 'image' in data:
        header, encoded = data['image'].split(',', 1)
        img_bytes = np.frombuffer(__import__('base64').b64decode(encoded), np.uint8)
        frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'success': False, 'error': 'Could not decode image'})

        annotated, detections = detect_masks_in_frame(frame)

        out_name = f"capture_{uuid.uuid4()}.jpg"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        cv2.imwrite(out_path, annotated)

        mask_count    = sum(1 for d in detections if d['label'] == 'Mask')
        no_mask_count = len(detections) - mask_count
        summary = (
            f"✅ {len(detections)} Face(s) — {mask_count} With Mask / {no_mask_count} Without Mask"
            if detections else "⚠️ No faces detected."
        )

        return jsonify({
            'success'   : True,
            'image_path': url_for('uploaded_file', filename=out_name),
            'detection' : summary,
            'detections': detections,
        })

    return render_template('upload_photo.html',
                           image_path=None, detection=None, detections=[])


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

# ── Video upload ─────────────────────────────────────────────────────────────
def process_video_job(video_id: str, input_path: str):
    """Runs in a background thread. Writes output to processed/."""
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            VIDEO_JOBS[video_id] = {'status': 'error', 'output': None}
            return

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_name = f"processed_{video_id}.mp4"
        out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_name)
        fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
        writer   = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated, _ = detect_masks_in_frame(frame)
            writer.write(annotated)
        cap.release()
        writer.release()
        VIDEO_JOBS[video_id] = {'status': 'done', 'output': out_name}
    except Exception as e:
        VIDEO_JOBS[video_id] = {'status': 'error', 'output': None, 'message': str(e)}


@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'GET':
        return render_template('upload_video.html', processing=False, video_id=None)

    file = request.files.get('file')
    if not file or not file.filename:
        return render_template('upload_video.html', processing=False, video_id=None)

    if not allowed_video(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400

    video_id  = str(uuid.uuid4())
    filename  = secure_filename(f"{video_id}_{file.filename}")
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    VIDEO_JOBS[video_id] = {'status': 'processing', 'output': None}
    thread = threading.Thread(target=process_video_job, args=(video_id, save_path), daemon=True)
    thread.start()
    return jsonify({'success': True, 'video_id': video_id})

@app.route('/video_status/<video_id>')
def video_status(video_id: str):
    job = VIDEO_JOBS.get(video_id)
    if not job:
        return jsonify({'status': 'not_found'}), 404

    if job['status'] == 'done':
        download_url = url_for('processed_file', filename=job['output'])
        return jsonify({'status': 'done', 'download_url': download_url})
    return jsonify({'status': job['status']})


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
