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

class PredictPipeline:
    def __init__(self):
        try:
            self.model_path = os.path.join(ARTIFACT_DIR, MODEL_DIR, MODEL_NAME)
            if not os.path.exists(self.model_path):
                 raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            self.model = load_model(self.model_path)
            # Check model output shape
            self.output_shape = self.model.output_shape
            
            # Load face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                logging.warning(f"Cascade file not found at {cascade_path}, using default path if available in cv2")
            
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            logging.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logging.error(f"Error initializing PredictPipeline: {str(e)}")
            raise CustomException(e, sys)
        
    def predict(self, frame):
        try:
            # 🔹 Resize frame for faster detection (important)
            small_frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )

            if len(faces) == 0:
                return small_frame

            faces_list = []
            faces_rects = []

            for (x, y, w, h) in faces:
                face = small_frame[y:y+h, x:x+w]

                # Convert BGR → RGB
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                # Resize once
                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

                # Convert to array and preprocess
                face = np.asarray(face, dtype="float32")
                face = preprocess_input(face)

                faces_list.append(face)
                faces_rects.append((x, y, w, h))

            faces_array = np.array(faces_list, dtype="float32")

            # 🔹 Use batch_size = len(faces) (faster)
            preds = self.model.predict(faces_array, batch_size=len(faces_array), verbose=0)

            for i, (x, y, w, h) in enumerate(faces_rects):

                pred = preds[i]

                # 🔹 Binary Classification (Most Common)
                if len(pred) == 1:
                    score = float(pred[0])
                    label = "No Mask" if score > 0.5 else "Mask"
                    confidence = score if score > 0.5 else 1 - score
                else:
                    label_idx = np.argmax(pred)
                    confidence = float(pred[label_idx])
                    label = "No Mask" if label_idx == 1 else "Mask"

                # 🔹 Better Color Control
                if label == "Mask":
                    color = (0, 255, 0)
                elif label == 'No Mask':
                    color = (0, 0, 255)
                text = f"{label}: {confidence*100:.1f}%"
                # Draw rectangle
                cv2.rectangle(small_frame, (x, y), (x+w, y+h), color, 2)
                # Draw filled label background
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(small_frame, (x, y - th - 10), (x + tw, y), color, -1)
                cv2.putText(small_frame,text,(x, y - 5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255, 255, 255),2)
            return small_frame

        except Exception as e:
            logging.error(f"Prediction Error: {str(e)}")
            return frame
    def predict_main(self, frame):
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
        
    def detect_mask(self,img):
        sample = cv2.resize(img,(224,224))
        y_pred = self.model.predict_classes(img.reshape(1,224,224,3))
        return y_pred
    
    def predict2(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret,frame = cap.read()
            # prediction function fo frame
            
            cv2.imshow("window",frame)
            if cv2.waitKey(1) & 0xFF==ord('x'):
                break
        cv2.destroyAllWindows()
            

# path = '../artifact/data/test/images/19-with-mask_jpg.rf.c15f92c5014adda1b32d128e903bd3ce.jpg'
# obj = PredictPipeline()     
# # obj.predict2()
# y_pred = obj.detect_mask(path)
