
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from src.constant import ARTIFACT_DIR, MODEL_DIR, IMG_SIZE , MODEL_NAME
from src.exception.exception import CustomException
from src.logger.logger import logging

# CPU Optimization: Disable GPU and limit threading for CPU efficiency
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only mode
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(4)
logging.info("CPU-optimized inference mode enabled")

class PredictPipeline:
    def __init__(self):
        try:
            self.model_path = os.path.join(ARTIFACT_DIR, MODEL_DIR, MODEL_NAME)
            if not os.path.exists(self.model_path):
                 raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            self.model = load_model(self.model_path)
            # Check model output shape
            self.output_shape = self.model.output_shape
            
            # Optimize model for inference on CPU
            try:
                self.model.make_predict_function()  # Pre-compile for faster predictions
            except AttributeError:
                pass  # Newer TensorFlow versions don't require this
            
            # Load face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                logging.warning(f"Cascade file not found at {cascade_path}, using default path if available in cv2")
            
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Cache for last frame processing to reduce redundant work
            self.last_gray_frame = None
            
            logging.info(f"Model loaded from {self.model_path} - CPU Optimized Mode")
        except Exception as e:
            logging.error(f"Error initializing PredictPipeline: {str(e)}")
            raise CustomException(e, sys)

    def predict(self, frame):
        try:
            # Detect faces in grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            
            faces_list = []
            faces_rects = []

            for (x, y, w, h) in faces:
                # Preprocess face for model (resize to 224x224, RGB)
                face_img = frame[y:y+h, x:x+w]
                # Convert BGR to RGB for model
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
                face_img = img_to_array(face_img)
                face_img = preprocess_input(face_img) # VGG16 preprocess expects RGB
                faces_list.append(face_img)
                faces_rects.append((x, y, w, h))
            
            if len(faces_list) > 0:
                faces_array = np.array(faces_list, dtype="float32")
                preds = self.model.predict(faces_array, batch_size=32)
            else:
                return frame # No faces detected

            for (i, (x, y, w, h)) in enumerate(faces_rects):
                pred = preds[i]
                
                # Logic for class determination
                # Assuming 0: "Mask" (with_mask), 1: "No Mask" (without_mask)
                if pred.shape[0] == 1 or (len(pred.shape) == 0):
                     # Binary case
                     score = pred[0] if len(pred) > 0 else pred
                     label = "No Mask" if score > 0.5 else "Mask"
                     confidence = score if score > 0.5 else 1 - score
                else:
                    # Multi-class or 2-node softmax
                    # Index 0: Mask, Index 1: No Mask (alphabetical)
                    # wait, 'with_mask' < 'without_mask'
                    label_idx = np.argmax(pred)
                    label = "No Mask" if label_idx == 1 else "Mask"
                    confidence = pred[label_idx]

                # Color: Green for Mask, Red for No Mask
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                
                label_text = "{}: {:.2f}%".format(label, confidence * 100)
                
                cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            return frame
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            return frame
