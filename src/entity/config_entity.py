import os
from dataclasses import dataclass, field
from src import constant


@dataclass
class DataIngestionConfig:
    train_dir_path: str = os.path.join(constant.DATA_DIR, constant.TRAIN_DATA_DIR)
    test_dir_path: str  = os.path.join(constant.DATA_DIR, constant.TEST_DATA_DIR)
    valid_dir_path: str = os.path.join(constant.DATA_DIR, constant.VALID_DATA_DIR)


@dataclass
class DataValidationConfig:
    # Validated images will be copied here (bad images are excluded)
    output_dir: str       = os.path.join(constant.ARTIFACT_DIR, constant.DATA_VALIDATION_DIR, "validated_data")
    report_file_path: str = os.path.join(constant.ARTIFACT_DIR, constant.DATA_VALIDATION_DIR, "validation_report.txt")
    blur_threshold: float = constant.BLUR_THRESHOLD
    min_width: int        = constant.MIN_IMG_WIDTH
    min_height: int       = constant.MIN_IMG_HEIGHT
    valid_labels: set     = field(default_factory=lambda: constant.VALID_LABELS)


@dataclass
class DataTransformationConfig:
    # Cropped face images will be saved here
    output_dir: str = os.path.join(constant.ARTIFACT_DIR, constant.DATA_TRANSFORMATION_DIR, constant.CROPPED_DATA_DIR)


@dataclass
class ModelBuilderConfig:
    img_size: int        = constant.IMG_SIZE
    learning_rate: float = constant.LEARNING_RATE


@dataclass
class ModelTrainerConfig:
    model_save_dir: str  = os.path.join(constant.ARTIFACT_DIR, constant.MODEL_DIR)
    model_name: str      = constant.MODEL_NAME
    epochs: int          = constant.EPOCHS
    runs_dir: str        = "runs"


@dataclass
class ModelEvaluationConfig:
    report_file_path: str = os.path.join(constant.ARTIFACT_DIR, constant.EVALUATION_DIR, "evaluation_report.txt")