import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime

from src.exception.exception import CustomException
from src.logger.logger import logging
from src import constant
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelBuilderArtifact, DataTransformationArtifact, ModelTrainerArtifact


class ModelTrainer:
    def __init__(
        self,
        config: ModelTrainerConfig,
        builder_artifact: ModelBuilderArtifact,
        transformation_artifact: DataTransformationArtifact,
    ):
        self.config = config
        self.model  = builder_artifact.model_object
        self.device = next(self.model.parameters()).device   # reuse device from model
        self.train_dir = transformation_artifact.train_dir_path
        self.valid_dir = transformation_artifact.valid_dir_path
        os.makedirs(self.config.model_save_dir, exist_ok=True)

    def make_data_loaders(self):
        """Create PyTorch DataLoaders for train and validation sets."""
        # EfficientNet expects [0,1] normalized with ImageNet stats
        train_transform = transforms.Compose([
            transforms.Resize((constant.IMG_SIZE, constant.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        valid_transform = transforms.Compose([
            transforms.Resize((constant.IMG_SIZE, constant.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = datasets.ImageFolder(self.train_dir, transform=train_transform)
        valid_dataset = datasets.ImageFolder(self.valid_dir, transform=valid_transform)

        logging.info(f"Training classes: {train_dataset.classes}")
        logging.info(f"Train samples: {len(train_dataset)} | Valid samples: {len(valid_dataset)}")

        train_loader = DataLoader(
            train_dataset, batch_size=constant.BATCH_SIZE,
            shuffle=True, num_workers=4, pin_memory=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=constant.BATCH_SIZE,
            shuffle=False, num_workers=4, pin_memory=True
        )
        return train_loader, valid_loader

    def train_one_epoch(self, loader, criterion, optimizer):
        """Run one training epoch and return (loss, accuracy)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.float().unsqueeze(1).to(self.device)

            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5)
            correct += (preds == labels.bool()).sum().item()
            total += labels.size(0)

        return total_loss / len(loader), correct / total

    def validate_one_epoch(self, loader, criterion):
        """Run one validation pass and return (loss, accuracy)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5)
                correct += (preds == labels.bool()).sum().item()
                total += labels.size(0)

        return total_loss / len(loader), correct / total

    def initialize_training(self) -> ModelTrainerArtifact:
        try:
            model_save_path = os.path.join(self.config.model_save_dir, self.config.model_name)

            # Skip training if model already saved
            if os.path.exists(model_save_path):
                logging.info(f"Model already exists at {model_save_path}. Skipping training step.")
                return ModelTrainerArtifact(model_path=model_save_path, history=None)

            logging.info(f"Starting Model Training on device: {self.device}")

            train_loader, valid_loader = self.make_data_loaders()
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=constant.LEARNING_RATE,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=2, factor=0.5
            )

            best_val_loss = float("inf")
            patience_counter = 0
            early_stop_patience = 5
            best_weights = copy.deepcopy(self.model.state_dict())
            history = []

            for epoch in range(self.config.epochs):
                train_loss, train_acc = self.train_one_epoch(train_loader, criterion, optimizer)
                val_loss, val_acc     = self.validate_one_epoch(valid_loader, criterion)
                scheduler.step(val_loss)

                epoch_log = (
                    f"Epoch [{epoch + 1}/{self.config.epochs}] "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                )
                logging.info(epoch_log)
                history.append({"epoch": epoch + 1, "train_loss": train_loss,
                                 "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        logging.info("Early stopping triggered.")
                        break

            # Restore best weights and save model
            self.model.load_state_dict(best_weights)
            torch.save(self.model.state_dict(), model_save_path)
            logging.info(f"Model saved at: {model_save_path}")

            return ModelTrainerArtifact(model_path=model_save_path, history=history)

        except Exception as e:
            raise CustomException(e, sys)
