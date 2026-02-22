import cv2
import numpy as np
import os
import sys
from src.logger.logger import logging
from src.exception.exception import CustomException


class FaceCropper:
    """
    Face detection and cropping using Caffe Deep Neural Network model
    """
    
    def __init__(self):
        """Initialize the Caffe face detection model"""
        try:
            # Get the project root directory path (face_mask_detector/)
            # __file__ = src/components/face_crop.py
            # dirname 1: src/components
            # dirname 2: src
            # dirname 3: face_mask_detector (PROJECT ROOT)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            face_detector_dir = os.path.join(project_root, "face_detector")
            
            config_path = os.path.join(face_detector_dir, "deploy.prototxt")
            model_path = os.path.join(face_detector_dir, "res10_300x300_ssd_iter_140000.caffemodel")
            
            logging.info(f"Looking for face detector files in: {face_detector_dir}")
            
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
            self.confidence_threshold = 0.5
            logging.info("FaceCropper initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing FaceCropper: {str(e)}")
            raise CustomException(e, sys)
    
    def detect_faces(self, image):
        """
        Detect all faces in an image using Caffe DNN model
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            list: List of face bounding boxes [(startX, startY, endX, endY, confidence), ...]
        """
        try:
            h, w = image.shape[:2]
            
            # Create blob (300x300 required)
            blob = cv2.dnn.blobFromImage(
                image,
                scalefactor=1.0,
                size=(300, 300),
                mean=(104.0, 177.0, 123.0)
            )
            
            self.net.setInput(blob)
            detections = self.net.forward()
            
            faces = []
            
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > self.confidence_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Ensure box stays inside image
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)
                    
                    faces.append((startX, startY, endX, endY, confidence))
            
            logging.info(f"Detected {len(faces)} face(s)")
            return faces
        
        except Exception as e:
            logging.error(f"Error detecting faces: {str(e)}")
            raise CustomException(e, sys)
    
    def crop_faces(self, image, faces):
        """
        Crop detected faces from the image
        
        Args:
            image: Input image (BGR format)
            faces: List of face bounding boxes
            
        Returns:
            list: List of cropped face images with their coordinates
        """
        try:
            cropped_faces = []
            
            for startX, startY, endX, endY, confidence in faces:
                face = image[startY:endY, startX:endX]
                cropped_faces.append({
                    'face': face,
                    'coords': (startX, startY, endX, endY),
                    'confidence': confidence
                })
            
            logging.info(f"Cropped {len(cropped_faces)} face(s)")
            return cropped_faces
        
        except Exception as e:
            logging.error(f"Error cropping faces: {str(e)}")
            raise CustomException(e, sys)