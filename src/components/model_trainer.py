import os
import sys
from src.exception.exception import CustomException
from src import constant
from src.logger.logger import logging
from src.entity.artifact_entity import DataIngestionArtifact
from src.components.model import Model
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

class ModelTrainer:
    def __init__(self,artifact:DataIngestionArtifact):
        self.artifact = artifact
        self.train_data_artifact:str = artifact.train_dir_path
        self.test_data_artifact:str = artifact.test_dir_path
        self.valid_data_artifact:str = artifact.valid_dir_path
        self.train_lable_artifact:str = artifact.train_lable_path
        self.test_lable_artifact:str = artifact.test_lable_path
        self.valid_lable_artifact:str = artifact.valid_lable_path

    def initialize_training(self):
        
    
    

