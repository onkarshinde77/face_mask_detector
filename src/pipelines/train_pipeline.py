import sys
from src.exception.exception import CustomException
from src.logger.logger import logging

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelBuilderConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_builder import ModelBuilder
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation


class TrainPipeline:
    def __init__(self):
        self.ingestion_config    = DataIngestionConfig()
        self.validation_config   = DataValidationConfig()
        self.transformation_config = DataTransformationConfig()
        self.builder_config      = ModelBuilderConfig()
        self.trainer_config      = ModelTrainerConfig()
        self.evaluation_config   = ModelEvaluationConfig()

    def run_pipeline(self):
        try:
            # ── Step 1: Data Ingestion ─────────────────────────────────────────
            logging.info("=" * 60)
            logging.info("STEP 1 — Data Ingestion")
            data_ingestion = DataIngestion(config=self.ingestion_config)
            ingestion_artifact = data_ingestion.init_data_ingestion()
            logging.info(f"Data Ingestion Done: {ingestion_artifact}")

            # ── Step 2: Data Validation ────────────────────────────────────────
            logging.info("=" * 60)
            logging.info("STEP 2 — Data Validation")
            data_validation = DataValidation(config=self.validation_config, artifact=ingestion_artifact)
            validation_artifact = data_validation.init_data_validation()
            logging.info(f"Data Validation Done. Report: {validation_artifact.report_file_path}")

            if not validation_artifact.is_valid:
                raise Exception("Data Validation Failed. Check the validation report.")

            # ── Step 3: Data Transformation ────────────────────────────────────
            logging.info("=" * 60)
            logging.info("STEP 3 — Data Transformation (Face Cropping)")
            data_transformation = DataTransformation(
                config=self.transformation_config,
                artifact=validation_artifact,
            )
            transformation_artifact = data_transformation.init_data_transformation()
            logging.info(f"Data Transformation Done: {transformation_artifact}")

            # ── Step 4: Model Builder ──────────────────────────────────────────
            logging.info("=" * 60)
            logging.info("STEP 4 — Model Builder")
            model_builder = ModelBuilder(config=self.builder_config)
            builder_artifact = model_builder.build()
            logging.info("Model Build Done")

            # ── Step 5: Model Trainer ──────────────────────────────────────────
            logging.info("=" * 60)
            logging.info("STEP 5 — Model Trainer")
            model_trainer = ModelTrainer(
                config=self.trainer_config,
                builder_artifact=builder_artifact,
                transformation_artifact=transformation_artifact,
            )
            trainer_artifact = model_trainer.initialize_training()
            logging.info(f"Model Training Done. Saved at: {trainer_artifact.model_path}")

            # ── Step 6: Model Evaluation ───────────────────────────────────────
            logging.info("=" * 60)
            logging.info("STEP 6 — Model Evaluation")
            model_evaluation = ModelEvaluation(
                config=self.evaluation_config,
                trainer_artifact=trainer_artifact,
                transformation_artifact=transformation_artifact,
            )
            evaluation_artifact = model_evaluation.init_model_evaluation()
            logging.info(
                f"Model Evaluation Done — "
                f"Test Accuracy: {evaluation_artifact.test_accuracy:.4f}, "
                f"Test Loss: {evaluation_artifact.test_loss:.4f}"
            )

            logging.info("=" * 60)
            logging.info("Pipeline Complete!")
            return evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    result = pipeline.run_pipeline()
    print(f"\nFinal Test Accuracy : {result.test_accuracy:.4f}")
    print(f"Final Test Loss     : {result.test_loss:.4f}")
    print(f"Evaluation Report   : {result.report_file_path}")
