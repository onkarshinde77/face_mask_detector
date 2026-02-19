import os
import sys
from src.exception.exception import CustomException
from src import constant
from src.logger.logger import logging
from src.entity.artifact_entity import DataIngestionArtifact
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

class ModelTrainer:
    def __init__(self,artifact:DataIngestionArtifact):
        self.train_data_artifact:str = artifact.train_dir_path
        self.test_data_artifact:str = artifact.test_dir_path
        self.valid_data_artifact:str = artifact.valid_dir_path
        self.train_lable_artifact:str = artifact.train_lable_path
        self.test_lable_artifact:str = artifact.test_lable_path
        self.valid_lable_artifact:str = artifact.valid_lable_path
    
    def model(self):
        try:

            base_model = VGG16(
                weights="imagenet",
                include_top=False,
                input_shape=(constant.IMG_SIZE, constant.IMG_SIZE, 3)
            )
            base_model.trainable = False
            model = models.Sequential()

            model.add(base_model)
            model.add(layers.Flatten())
            model.add(layers.Dense(256, activation="relu"))
            model.add(layers.Dropout(0.5))
            model.add(layers.Dense(constant.NUM_CLASSES, activation="softmax"))

            model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
            print(model.summary())
            logging.info("Model Initialization done")
        except Exception as e:
            raise CustomException(e,sys)
    
    def initialize_model(self):
        pass
    
    

