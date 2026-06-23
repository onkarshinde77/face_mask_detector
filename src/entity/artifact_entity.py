from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    train_dir_path: str
    test_dir_path: str
    valid_dir_path: str


@dataclass
class DataValidationArtifact:
    train_dir_path: str
    test_dir_path: str
    valid_dir_path: str
    report_file_path: str
    is_valid: bool


@dataclass
class DataTransformationArtifact:
    train_dir_path: str
    test_dir_path: str
    valid_dir_path: str


@dataclass
class ModelBuilderArtifact:
    model_object: object      # compiled keras model, not saved to disk yet


@dataclass
class ModelTrainerArtifact:
    model_path: str
    history: object


@dataclass
class ModelEvaluationArtifact:
    test_loss: float
    test_accuracy: float
    report_file_path: str
