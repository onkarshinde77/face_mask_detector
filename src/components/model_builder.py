import sys
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from src.exception.exception import CustomException
from src.logger.logger import logging
from src.entity.config_entity import ModelBuilderConfig
from src.entity.artifact_entity import ModelBuilderArtifact


class ModelBuilder:
    def __init__(self, config: ModelBuilderConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build(self) -> ModelBuilderArtifact:
        """Build EfficientNetB0 with a custom binary classification head."""
        try:
            logging.info(f"Starting Model Builder (device: {self.device})")

            # Load pretrained EfficientNetB0
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

            # Freeze all backbone layers
            for param in model.features.parameters():
                param.requires_grad = False

            # Replace the classifier head with a binary output
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),        # single sigmoid output for binary classification
            )

            model = model.to(self.device)

            logging.info("Model Build Complete")
            logging.info(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

            return ModelBuilderArtifact(model_object=model)

        except Exception as e:
            raise CustomException(e, sys)
