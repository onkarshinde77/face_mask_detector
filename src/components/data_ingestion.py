import sys
import shutil
import os
from src.exception.exception import CustomException
from src.logger.logger import logging
from src import constant
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.train_dir_path: str = config.train_dir_path
        self.test_dir_path: str = config.test_dir_path
        self.valid_dir_path: str = config.valid_dir_path
        
        self.artifact_dir = constant.ARTIFACT_DIR
        self.artifact_data_dir = os.path.join(constant.ARTIFACT_DIR, constant.DATA_DIR)
        self.artifact_train_dir = os.path.join(self.artifact_data_dir, constant.TRAIN_DATA_DIR)
        self.artifact_test_dir = os.path.join(self.artifact_data_dir, constant.TEST_DATA_DIR)
        self.artifact_valid_dir = os.path.join(self.artifact_data_dir, constant.VALID_DATA_DIR)

    def init_data_ingestion(self) -> DataIngestionArtifact:
        try:
            if (os.path.exists(self.artifact_train_dir) and 
                os.path.exists(self.artifact_test_dir) and 
                os.path.exists(self.artifact_valid_dir)):
                
                logging.info("Data already ingested. Skipping ingestion step.")
                return DataIngestionArtifact(
                    train_dir_path=self.artifact_train_dir,
                    test_dir_path=self.artifact_test_dir,
                    valid_dir_path=self.artifact_valid_dir
                )

            logging.info("Started Data Ingestion")
            
            if os.path.exists(self.artifact_data_dir):
                shutil.rmtree(self.artifact_data_dir)
            os.makedirs(self.artifact_data_dir, exist_ok=True)
            
            logging.info(f"Copying train data from {self.train_dir_path} to {self.artifact_train_dir}")
            shutil.copytree(self.train_dir_path, self.artifact_train_dir)
            
            logging.info(f"Copying test data from {self.test_dir_path} to {self.artifact_test_dir}")
            shutil.copytree(self.test_dir_path, self.artifact_test_dir)
            
            logging.info(f"Copying valid data from {self.valid_dir_path} to {self.artifact_valid_dir}")
            shutil.copytree(self.valid_dir_path, self.artifact_valid_dir)
            
            logging.info("Data Ingestion Completed")
            return DataIngestionArtifact(
                train_dir_path=self.artifact_train_dir,
                test_dir_path=self.artifact_test_dir,
                valid_dir_path=self.artifact_valid_dir
            )

        except Exception as e:
            raise CustomException(e, sys)
