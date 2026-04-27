import sys
from src.exception.exception import CustomException
from src.logger.logger import logging
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        try:
            self.data_ingestion_config = DataIngestionConfig()
        except Exception as e:
             raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("Start Data Ingestion")
            data_ingestion = DataIngestion(config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.init_data_ingestion()
            logging.info(f"Data Ingestion Complete {data_ingestion_artifact}")

            logging.info("Start Model Training")
            model_trainer = ModelTrainer(artifact=data_ingestion_artifact)
            history = model_trainer.initialize_training()
            logging.info("Model Training Complete")
            return history
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = TrainPipeline()
    history = obj.run_pipeline()
    logging.info(f"history : {history}")
    print(history)
