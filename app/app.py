import os
import cv2
import sys
import numpy as np
import threading
import json
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from src.pipelines.predict_pipeline import PredictPipeline

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the new predict pipeline

# Initialize the prediction pipeline
try:
    predict_pipeline = PredictPipeline()
    print("✓ PredictPipeline initialized successfully")
except Exception as e:
    print(f"⚠ Warning: PredictPipeline initialization failed: {str(e)}")
    print("Falling back to legacy prediction method")
    predict_pipeline = None

# Legacy model for fallback
MODEL_PATH = "artifact/models/face_mask_model4.keras"
try:
    model = load_model(MODEL_PATH)
except:
    model = None

# FACE DETECTOR (Legacy)
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# FLASK APP
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Video processing status tracker
video_processing_status = {}
video_lock = threading.Lock()


# PREDICTION FUNCTION
# def predict_mask(frame):
#     img = frame.copy()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_detector.detectMultiScale(gray, 1.3, 5)

#     final_prediction = 0  # default mask

#     for (x, y, w, h) in faces:
#         face = img[y:y+h, x:x+w]

#         # Resize for VGG16
#         face = cv2.resize(face, (224, 224))
#         face = np.array(face, dtype=np.float32)
#         face = np.expand_dims(face, axis=0)
#         face = preprocess_input(face)

#         preds = model.predict(face)[0]

#         # Assuming model output = [mask, no_mask]
#         mask_prob = preds[0]
#         no_mask_prob = preds[1]

#         if mask_prob > no_mask_prob:
#             label = "Mask"
#             color = (0, 255, 0)
#             final_prediction = 0
#         else:
#             label = "No Mask"
#             color = (0, 0, 255)
#             final_prediction = 1

#         # Draw bounding box
#         cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
#         cv2.putText(img, label, (x, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.8, color, 2)

#     return img, final_prediction


def predict_mask(frame):
    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    final_prediction = 0

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]

        face = cv2.resize(face, (224, 224))
        face = np.array(face, dtype=np.float32)
        face = np.expand_dims(face, axis=0)
        face = preprocess_input(face)

        preds = model.predict(face)[0][0]   # Single value

        # If sigmoid output > 0.5 → No Mask (example logic)
        if preds > 0.5:
            label = "No Mask"
            color = (0, 0, 255)
            final_prediction = 1
        else:
            label = "Mask"
            color = (0, 255, 0)
            final_prediction = 0

        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)

    return img, final_prediction
# LIVE CAMERA STREAM
def generate_frames():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Use new pipeline if available, otherwise use legacy method
        if predict_pipeline:
            faces = predict_pipeline.face_cropper.detect_faces(frame)
            
            if len(faces) > 0:
                cropped_faces = predict_pipeline.face_cropper.crop_faces(frame, faces)
                faces_list = []
                
                for cropped_face_dict in cropped_faces:
                    face = cropped_face_dict['face']
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_resized = cv2.resize(face_rgb, (224, 224))
                    face_array = np.asarray(face_resized, dtype="float32")
                    face_array = preprocess_input(face_array)
                    faces_list.append(face_array)
                
                if faces_list:
                    faces_array = np.array(faces_list, dtype="float32")
                    predictions = predict_pipeline.model.predict(faces_array, verbose=0)
                    
                    for idx, pred in enumerate(predictions):
                        startX, startY, endX, endY = cropped_faces[idx]['coords']
                        
                        if len(pred) == 1:
                            score = float(pred[0])
                            label = "No Mask" if score > 0.5 else "Mask"
                            confidence = score if score > 0.5 else 1 - score
                        else:
                            label_idx = np.argmax(pred)
                            confidence = float(pred[label_idx])
                            label = "No Mask" if label_idx == 1 else "Mask"
                        
                        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                        
                        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
                        
                        label_text = f"{label}: {confidence*100:.1f}%"
                        (text_width, text_height), _ = cv2.getTextSize(
                            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )
                        
                        cv2.rectangle(
                            frame,
                            (startX, startY - text_height - 12),
                            (startX + text_width + 8, startY),
                            color,
                            -1
                        )
                        
                        cv2.putText(
                            frame,
                            label_text,
                            (startX + 4, startY - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )
        else:
            # Legacy method
            frame, _ = predict_mask(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()

def process_and_save_video(input_path, output_path):
    """Process video using new PredictPipeline or legacy method"""
    
    if predict_pipeline:
        # Use new PredictPipeline for video processing
        predict_pipeline.predict_video(
            video_path=input_path,
            save_output=True,
            output_path=output_path
        )
    else:
        # Fallback to legacy method
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise Exception("Cannot open video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame, _ = predict_mask(frame)
            out.write(frame)

        cap.release()
        out.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/live')
def live():
    return render_template('live.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload_photo', methods=['GET', 'POST'])
def upload_photo():
    if request.method == 'POST':

        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            if predict_pipeline:
                # Use new PredictPipeline with Caffe DNN face detection
                result = predict_pipeline.predict_image(filepath)
                
                output_image = result['image']
                detections = result['detections']
                num_faces = result['num_faces']
                
                # Save processed image
                processed_filename = "processed_" + file.filename
                processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
                cv2.imwrite(processed_path, output_image)
                
                relative_path = url_for('static', filename=f'uploads/{processed_filename}')
                
                # Format detection information for display
                detection_info = []
                mask_count = 0
                no_mask_count = 0
                
                for i, det in enumerate(detections, 1):
                    label = det['label']
                    confidence = det['confidence'] * 100
                    detection_info.append({
                        'face_num': i,
                        'label': label,
                        'confidence': f"{confidence:.1f}%"
                    })
                    
                    if label == 'Mask':
                        mask_count += 1
                    else:
                        no_mask_count += 1
                
                summary = f"{num_faces} face(s) detected | Mask: {mask_count} | No Mask: {no_mask_count}"
                
                return render_template('upload_photo.html',
                                     image_path=relative_path,
                                     detection=summary,
                                     detections=detection_info,
                                     num_faces=num_faces)
            else:
                # Fallback to legacy prediction method
                img = cv2.imread(filepath)
                if img is not None:
                    img, pred = predict_mask(img)
                    
                    processed_filename = "processed_" + file.filename
                    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
                    
                    cv2.imwrite(processed_path, img)
                    
                    relative_path = url_for('static', filename=f'uploads/{processed_filename}')
                    
                    if pred == 0:
                        flag = "Mask"
                    else:
                        flag = "No Mask"
                    
                    return render_template('upload_photo.html',
                                         image_path=relative_path,
                                         detection=flag)
        
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return render_template('upload_photo.html', error=f"Error: {str(e)}")

    return render_template('upload_photo.html')


@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':

        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        processed_filename = "processed_" + file.filename
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
        
        # Create a unique ID for this processing task
        video_id = processed_filename
        
        # Mark as processing
        with video_lock:
            video_processing_status[video_id] = {
                'status': 'processing',
                'processed_filename': processed_filename,
                'processed_path': processed_path,
                'error': None
            }
        
        # Start background thread to process video
        thread = threading.Thread(target=process_video_background, args=(filepath, processed_path, video_id))
        thread.daemon = True
        thread.start()
        
        # Return immediately with processing status
        return render_template('upload_video.html', processing=True, video_id=video_id)

    return render_template('upload_video.html')


def process_video_background(filepath, processed_path, video_id):
    """Process video in background thread"""
    try:
        process_and_save_video(filepath, processed_path)
        with video_lock:
            video_processing_status[video_id]['status'] = 'completed'
    except Exception as e:
        with video_lock:
            video_processing_status[video_id]['status'] = 'error'
            video_processing_status[video_id]['error'] = str(e)


@app.route('/check_video_status/<video_id>')
def check_video_status(video_id):
    """Check video processing status"""
    with video_lock:
        if video_id not in video_processing_status:
            return jsonify({'status': 'not_found'}), 404
        
        status_info = video_processing_status[video_id]
        
        result = {
            'status': status_info['status'],
            'error': status_info.get('error')
        }
        
        if status_info['status'] == 'completed':
            result['video_path'] = url_for('static', 
                                          filename=f'uploads/{status_info["processed_filename"]}')
        
        return jsonify(result)


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)