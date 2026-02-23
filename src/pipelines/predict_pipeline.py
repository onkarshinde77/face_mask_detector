import os
import sys
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


class PredictPipeline:
    def __init__(self):
        try:
            self.model_path = os.path.join(ARTIFACT_DIR, MODEL_DIR, MODEL_NAME)
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            self.model = load_model(self.model_path)
            self.output_shape = self.model.output_shape
            
            # Initialize Face Cropper using Caffe DNN model
            self.face_cropper = FaceCropper()
            
            # Load face cascade as backup
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            logging.info(f"PredictPipeline initialized successfully with model: {self.model_path}")
        
        except Exception as e:
            logging.error(f"Error initializing PredictPipeline: {str(e)}")
            raise CustomException(e, sys)
    
    def predict_image(self, image_path):
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image not found at {image_path}")
            
            logging.info(f"Processing image: {image_path}")
            
            # Detect faces using Caffe DNN model
            faces = self.face_cropper.detect_faces(image)
            
            if len(faces) == 0:
                logging.warning("No faces detected in the image")
                return {
                    'image': image,
                    'detections': [],
                    'num_faces': 0
                }
            
            # Crop faces
            cropped_faces = self.face_cropper.crop_faces(image, faces)
            
            # Prepare faces for prediction
            faces_list = []
            detections = []
            
            for cropped_face_dict in cropped_faces:
                face = cropped_face_dict['face']
                startX, startY, endX, endY = cropped_face_dict['coords']
                
                # Preprocess face for mask detection model
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
                face_array = np.asarray(face_resized, dtype="float32")
                face_array = preprocess_input(face_array)
                faces_list.append(face_array)
            
            # Batch prediction on all faces
            faces_array = np.array(faces_list, dtype="float32")
            predictions = self.model.predict(faces_array, verbose=0)
            
            # Draw bounding boxes and labels on image
            output_image = image.copy()
            
            for idx, pred in enumerate(predictions):
                startX, startY, endX, endY = cropped_faces[idx]['coords']
                
                # Determine label and confidence
                if len(pred) == 1:
                    score = float(pred[0])
                    label = "No Mask" if score > 0.5 else "Mask"
                    confidence = score if score > 0.5 else 1 - score
                else:
                    label_idx = np.argmax(pred)
                    confidence = float(pred[label_idx])
                    label = "No Mask" if label_idx == 1 else "Mask"
                
                # Color: Green (0, 255, 0) for Mask, Red (0, 0, 255) for No Mask
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                
                # Draw rectangle
                cv2.rectangle(output_image, (startX, startY), (endX, endY), color, 3)
                
                # Prepare label text
                label_text = f"{label}: {confidence*100:.1f}%"
                
                # Draw filled label background
                (text_width, text_height), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                
                cv2.rectangle(
                    output_image,
                    (startX, startY - text_height - 15),
                    (startX + text_width + 10, startY),
                    color,
                    -1
                )
                
                # Put text on image
                cv2.putText(
                    output_image,
                    label_text,
                    (startX + 5, startY - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # Store detection info
                detections.append({
                    'coords': (startX, startY, endX, endY),
                    'label': label,
                    'confidence': confidence
                })
            
            logging.info(f"Detected {len(detections)} face(s) - Processing complete")
            
            return {
                'image': output_image,
                'detections': detections,
                'num_faces': len(detections)
            }
        
        except Exception as e:
            logging.error(f"Error in predict_image: {str(e)}")
            raise CustomException(e, sys)
    
    def predict_video(self, video_path=None, save_output=False, output_path=None):
        try:
            # Open video source
            if video_path is None:
                cap = cv2.VideoCapture(0)  # Webcam
                logging.info("Using webcam as video source")
            else:
                cap = cv2.VideoCapture(video_path)
                logging.info(f"Reading video from: {video_path}")
            
            if not cap.isOpened():
                raise ValueError("Cannot open video source")
            
            # Video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Video writer for output
            out = None
            if save_output and output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect faces
                faces = self.face_cropper.detect_faces(frame)
                
                if len(faces) > 0:
                    # Crop and predict
                    cropped_faces = self.face_cropper.crop_faces(frame, faces)
                    faces_list = []
                    
                    for cropped_face_dict in cropped_faces:
                        face = cropped_face_dict['face']
                        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
                        face_array = np.asarray(face_resized, dtype="float32")
                        face_array = preprocess_input(face_array)
                        faces_list.append(face_array)
                    
                    # Batch prediction
                    faces_array = np.array(faces_list, dtype="float32")
                    predictions = self.model.predict(faces_array, verbose=0)
                    
                    # Draw detections
                    for idx, pred in enumerate(predictions):
                        startX, startY, endX, endY = cropped_faces[idx]['coords']
                        
                        # Determine label and confidence
                        if len(pred) == 1:
                            score = float(pred[0])
                            label = "No Mask" if score > 0.5 else "Mask"
                            confidence = score if score > 0.5 else 1 - score
                        else:
                            label_idx = np.argmax(pred)
                            confidence = float(pred[label_idx])
                            label = "No Mask" if label_idx == 1 else "Mask"
                        
                        # Color: Green for Mask, Red for No Mask
                        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                        
                        # Draw rectangle
                        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
                        
                        # Prepare label text
                        label_text = f"{label}: {confidence*100:.1f}%"
                        
                        # Draw filled label background
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
                        
                        # Put text
                        cv2.putText(
                            frame,
                            label_text,
                            (startX + 4, startY - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )
                
                # Display current frame
                cv2.imshow("Face Mask Detection", frame)
                # Save frame if output writer is active
                if out:
                    out.write(frame)
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # Release resources
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            logging.info(f"Video processing complete - {frame_count} frames processed")
            
            return {
                'status': 'completed',
                'frames_processed': frame_count,
                'output_path': output_path if save_output else None
            }
        
        except Exception as e:
            logging.error(f"Error in predict_video: {str(e)}")
            raise CustomException(e, sys)
    
    def predict_webcam(self):
        return self.predict_video(video_path=None)

