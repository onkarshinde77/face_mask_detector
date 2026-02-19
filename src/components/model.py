import os
import sys
from src.exception.exception import CustomException
from src import constant
from src.logger.logger import logging
from src.entity.artifact_entity import DataIngestionArtifact
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

class Model:
    def __init__(self):
        pass
    
    def model(self)->models:
        try:
            logging.info("Start model Initialization")
            base_model = VGG16(
                weights="imagenet",
                include_top=False,
                input_shape=(constant.IMG_SIZE, constant.IMG_SIZE, 3)
            )
            base_model.trainable = False
            model = models.Sequential()

            model.add(base_model)
            model.add(layers.Flatten())
            model.add(layers.Dense(4096, activation="relu"))
            model.add(layers.Dropout(0.5))
            model.add(layers.Dense(4096, activation="relu"))
            model.add(layers.Dropout(0.5))
            model.add(layers.Dense(constant.NUM_CLASSES, activation="softmax"))

            model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )
            logging.info(f"model compile done")
            print(model.summary())
            logging.info("Model Initialization done")
        except Exception as e:
            raise CustomException(e, sys)
    


    def create_data_generator(image_dir_path: str,label_csv_path: str,img_size: int = 224,
                              batch_size: int = 32,shuffle: bool = True):
        
        df = pd.read_csv(label_csv_path)
        # Create ImageDataGenerator
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        generator = datagen.flow_from_dataframe(
            dataframe=df,
            directory=image_dir_path,
            x_col="filename",
            y_col="label",
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode="binary",   # since you have 0 & 1
            shuffle=shuffle
        )

        return generator

    
    def initialize_model(self):
        self.model()
    

