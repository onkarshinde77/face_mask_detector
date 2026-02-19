import os
import cv2
import sys
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from src.exception.exception import CustomException
from src.pipelines.predict_pipeline import PredictPipeline

prediction = PredictPipeline()

# Load Model ONCE
# try:
#     MODEL_PATH = "artifact/models/face_mask_model.h5"
#     model = load_model(MODEL_PATH)
# except Exception as e:
#     raise CustomException(e,sys)

# Load Face Detector (Haarcascade)
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Flask App Setup
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Initialize Camera ONCE

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Prediction Function
def predict_mask(frame):
    return prediction.predict(frame=frame),0
    
    
# def predict_mask(frame):

#     # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     faces = face_detector.detectMultiScale(frame, 1.3, 5)

#     for (x, y, w, h) in faces:

#         face = frame[y:y+h, x:x+w]
#         face_resized = cv2.resize(face, (224, 224))

#         img = preprocess_input(face_resized)
#         img = np.expand_dims(img, axis=0)

#         prediction = model.predict(img, verbose=0)[0][0]

#         if prediction == 0 :
#             label = "Mask"
#             color = (0, 255, 0)
#         else:
#             label = "No Mask"
#             color = (0, 0, 255)

#         cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
#         cv2.putText(frame, label, (x, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.9, color, 2)

#     return frame , prediction


def generate_frames():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        # Fallback to default backend if DSHOW fails
        camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame,_ = predict_mask(frame)

        ret, buffer = cv2.imencode(
            '.jpg',
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        )

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    camera.release()

# Video File Processing
def generate_video(path):
    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame,prediction = predict_mask(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

# Routes
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

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        if img is not None:
            img,prediction = predict_mask(img)
            
            processed_filename = "processed_" + file.filename
            processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            cv2.imwrite(processed_path, img)

            relative_path = url_for('static', filename=f'uploads/{processed_filename}')
            if prediction==0: flag = "Mask"
            else: flag = "No Mask"
            return render_template('upload_photo.html', image_path=relative_path,detection=flag)

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

        return Response(generate_video(filepath),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    return render_template('upload_video.html')

# Cleanup Camera on Exit
@app.teardown_appcontext
def cleanup(exception=None):
    if camera.isOpened():
        camera.release()

# Run App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
