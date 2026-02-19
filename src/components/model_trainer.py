import os
import sys
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
            self.train_lable_artifact:str = artifact.train_lable_path
            self.test_lable_artifact:str = artifact.test_lable_path
            self.valid_lable_artifact:str = artifact.valid_lable_path

            self.model_save_path:str = os.path.join(
                constant.ARTIFACT_DIR,constant.MODEL_DIR
            )
            self.model_obj = Model()
            
        except Exception as e:
            raise CustomException(e,sys)

    def model_training(self):
        try:
            model:tensorflow.keras.models = self.model_obj.model
            train_data_generation = self.model_obj.create_data_generator(
                image_dir_path=self.train_data_artifact,
                label_csv_path=self.train_lable_artifact
            )
            test_data_generation = self.model_obj.create_data_generator(
                image_dir_path=self.test_data_artifact,
                label_csv_path=self.test_lable_artifact
            )
            valid_data_generation = self.model_obj.create_data_generator(
                image_dir_path=self.valid_data_artifact,
                label_csv_path=self.valid_lable_artifact
            )

            history = model.fit(
                train_data_generation,
                validation_data=valid_data_generation,
                epochs=constant.EPOCHS
            )

            logging.info("Model Training Completed")
            # Evaluate on Test Data
            test_loss, test_accuracy = model.evaluate(test_data_generation)
            logging.info(f"Test Accuracy: {test_accuracy}")
            # Save Model
            model_path:str = os.path.join(self.model_save_path,"face_mask_model.h5")
            model.save(model_path)
            logging.info(f"Model Saved Successfully in {model_path}")
            return history
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initialize_training(self):
        return self.model_training()
    
    

