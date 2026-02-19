
import os
import sys
from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np

# Add project root to sys.path to allow importing src modules
# __file__ is absolute path to app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.pipelines.predict_pipeline import PredictPipeline
from src.logger.logger import logging

app = Flask(__name__)
# Define upload folder relative to app.py location
app.config['UPLOAD_FOLDER'] = os.path.join(current_dir, 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize pipeline
pipeline = PredictPipeline()

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = pipeline.predict(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/upload_photo', methods=['GET', 'POST'])
def upload_photo():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read and process image
            img = cv2.imread(filepath)
            if img is not None:
                img = pipeline.predict(img)
                
                # Save processed image
                processed_filename = 'processed_' + filename
                processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
                cv2.imwrite(processed_filepath, img)
                
                # Generate URL for the processed image
                # 'static' endpoint serves from 'app/static'
                # Filename should be relative to 'static' folder, so 'uploads/processed_filename'
                relative_path = url_for('static', filename=f'uploads/{processed_filename}')
                return render_template('upload_photo.html', image_path=relative_path)
    return render_template('upload_photo.html')

def gen_video(path):
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = pipeline.predict(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Stream the processed video
            return Response(gen_video(filepath), mimetype='multipart/x-mixed-replace; boundary=frame')
            
    return render_template('upload_video.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
