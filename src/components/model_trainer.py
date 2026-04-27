import os
import sys
import tensorflow as tf
from datetime import datetime
from src.exception.exception import CustomException
from src import constant
from src.logger.logger import logging
from src.entity.artifact_entity import DataIngestionArtifact
from src.components.model import Model
from tensorflow.keras import layers, models

class ModelTrainer:
    def __init__(self,artifact:DataIngestionArtifact):
        try:
            self.artifact = artifact
            self.train_data_artifact:str = artifact.train_dir_path
            self.test_data_artifact:str = artifact.test_dir_path
            self.valid_data_artifact:str = artifact.valid_dir_path

            self.model_save_path:str = os.path.join(
                constant.ARTIFACT_DIR,constant.MODEL_DIR
            )
            os.makedirs(self.model_save_path,exist_ok=True)
            self.model_obj = Model()
            
        except Exception as e:
            raise CustomException(e,sys)

    def model_training(self):
        try:
            model = self.model_obj.model()
            train_data_generation = self.model_obj.create_data_generator(
                image_dir_path=self.train_data_artifact
            )
            test_data_generation = self.model_obj.create_data_generator(
                image_dir_path=self.test_data_artifact
            )
            valid_data_generation = self.model_obj.create_data_generator(
                image_dir_path=self.valid_data_artifact
            )

            # --- Create runs directory for history ---
            runs_dir = "runs"
            time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            current_run_dir = os.path.join(runs_dir, f"run_{time_stamp}")
            os.makedirs(current_run_dir, exist_ok=True)

            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=current_run_dir, histogram_freq=1)
            csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(current_run_dir, 'training_history.csv'))
            callbacks = [tensorboard_callback, csv_logger]
            # -----------------------------------------

            history = model.fit(
                train_data_generation,
                validation_data=valid_data_generation,
                epochs=constant.EPOCHS,
                callbacks=callbacks
            )

            logging.info("Model Training Completed")
            # Evaluate on Test Data
            test_loss, test_accuracy = model.evaluate(test_data_generation)
            logging.info(f"Test Accuracy: {test_accuracy}")
            # Save Model
            model_path:str = os.path.join(self.model_save_path,constant.MODEL_NAME)
            model.save(model_path)
            logging.info(f"Model Saved Successfully in {model_path}")
            return history
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initialize_training(self):
        return self.model_training()

