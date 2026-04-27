import os
import sys
from src.exception.exception import CustomException
from src import constant
from src.logger.logger import logging
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input


class Model:
    def __init__(self):
        pass

    def model(self) -> models.Model:
        try:
            logging.info("Start model Initialization (EfficientNet)")

            # Load EfficientNet base model
            base_model = EfficientNetB0(
                weights="imagenet",
                include_top=False,
                input_shape=(constant.IMG_SIZE, constant.IMG_SIZE, 3)
            )

            # Freeze base model
            base_model.trainable = False

            # Build model
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),   # better than Flatten
                layers.BatchNormalization(),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(1, activation="sigmoid")  # binary classification
            ])

            # Compile
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )

            logging.info("Model compile done")
            model.summary()
            logging.info("Model Initialization done")

            return model

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def create_data_generator(
        image_dir_path: str,
        label_csv_path: str,
        img_size: int = 224,
        batch_size: int = 32,
        shuffle: bool = True
    ):
        try:
            if not os.path.exists(image_dir_path) or not os.path.exists(label_csv_path):
                raise FileNotFoundError("File path not found")

            df = pd.read_csv(label_csv_path)

            # EfficientNet preprocessing
            datagen = ImageDataGenerator(
                preprocessing_function=preprocess_input
            )

            generator = datagen.flow_from_dataframe(
                dataframe=df,
                directory=image_dir_path,
                x_col="filename",
                y_col="lable",   # make sure column name is correct
                target_size=(img_size, img_size),
                batch_size=batch_size,
                class_mode="binary",
                shuffle=shuffle
            )

            return generator

        except Exception as e:
            raise CustomException(e, sys)