import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b4

from src.exception.exception import CustomException
from src.logger.logger import logging
from src import constant
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact, ModelEvaluationArtifact


class ModelEvaluation:
    def __init__(
        self,
        config: ModelEvaluationConfig,
        trainer_artifact: ModelTrainerArtifact,
        transformation_artifact: DataTransformationArtifact,
    ):
        self.config = config
        self.model_path = trainer_artifact.model_path
        self.test_dir   = transformation_artifact.test_dir_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        """Reconstruct the model architecture and load saved weights."""
        model = efficientnet_b4(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1),
        )
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model

    def make_test_loader(self):
        """Create a DataLoader for the test set."""
        test_transform = transforms.Compose([
            transforms.Resize((constant.IMG_SIZE, constant.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        test_dataset = datasets.ImageFolder(self.test_dir, transform=test_transform)
        return DataLoader(
            test_dataset, batch_size=constant.BATCH_SIZE,
            shuffle=False, num_workers=4, pin_memory=True
        )

    def init_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            report_dir = os.path.dirname(self.config.report_file_path)

            # Skip if evaluation report already exists
            if os.path.exists(self.config.report_file_path):
                logging.info("Evaluation report already exists. Skipping evaluation step.")
                with open(self.config.report_file_path) as f:
                    lines = f.readlines()
                test_loss     = float(lines[1].split(":")[1].strip())
                test_accuracy = float(lines[2].split(":")[1].strip())
                return ModelEvaluationArtifact(
                    test_loss=test_loss,
                    test_accuracy=test_accuracy,
                    report_file_path=self.config.report_file_path,
                )

            logging.info(f"Starting Model Evaluation on device: {self.device}")
            os.makedirs(report_dir, exist_ok=True)

            model = self.load_model()
            test_loader = self.make_test_loader()
            criterion = nn.BCEWithLogitsLoss()

            total_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(self.device)
                    labels = labels.float().unsqueeze(1).to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    preds = (torch.sigmoid(outputs) > 0.5)
                    correct += (preds == labels.bool()).sum().item()
                    total += labels.size(0)

            test_loss     = total_loss / len(test_loader)
            test_accuracy = correct / total

            logging.info(f"Test Loss:     {test_loss:.4f}")
            logging.info(f"Test Accuracy: {test_accuracy:.4f}")

            # Save report
            report_content = (
                "==== MODEL EVALUATION REPORT ====\n"
                f"Test Loss:     {test_loss:.4f}\n"
                f"Test Accuracy: {test_accuracy:.4f}\n"
            )
            with open(self.config.report_file_path, "w") as f:
                f.write(report_content)

            logging.info(f"Evaluation report saved at: {self.config.report_file_path}")

            return ModelEvaluationArtifact(
                test_loss=test_loss,
                test_accuracy=test_accuracy,
                report_file_path=self.config.report_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
