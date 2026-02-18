import sys
from typing import List, Tuple
from src.exception.exception import CustomException
from src.logger.logger import logging
from src import constant
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.utils.helper import read_img_path , read_img_from_csv
import cv2
import shutil
import os
import pandas as pd


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.train_dir_path: str = config.train_dir_path
        self.test_dir_path: str = config.test_dir_path
        self.valid_dir_path: str = config.valid_dir_path
        self.train_lable_file_path: str = config.train_lable_path
        self.test_lable_file_path: str = config.test_lable_path
        self.valid_lable_file_path: str = config.valid_lable_path

        self.img_extenstion: Tuple = (".jpg", ".jpeg", ".png")
        self.img_list: List[str] = []
        self.lable_list: List[dict] = []
        self.raw_img_list: List[str] = []
        self.raw_lable_list: List[str] = []
        self.corrupted_list: List[str] = []

        self.artifact_dir = constant.ARTIFACT_DIR
        self.artifact_data_dir = os.path.join(constant.ARTIFACT_DIR, constant.DATA_DIR)
        self.lable_file: str = "lable.csv"
        self.artifact_train_dir = os.path.join(self.artifact_data_dir, constant.TRAIN_DATA_DIR)
        self.artifact_test_dir = os.path.join(self.artifact_data_dir, constant.TEST_DATA_DIR)
        self.artifact_valid_dir = os.path.join(self.artifact_data_dir, constant.VALID_DATA_DIR)

        os.makedirs(self.artifact_dir, exist_ok=True)
        os.makedirs(self.artifact_train_dir, exist_ok=True)
        os.makedirs(self.artifact_test_dir, exist_ok=True)
        os.makedirs(self.artifact_valid_dir, exist_ok=True)

    def valid_paths(self, img_dir_path: str, lable_file_path: str):
        if not os.path.exists(lable_file_path):
            logging.info(f"file path not found: {lable_file_path}")
            raise FileNotFoundError(f"file path not found: {lable_file_path}")

        self.lable_list: List[dict] = read_img_from_csv(lable_file_path)
        self.img_list, _ = read_img_path(img_dir_path)

        img_set = set(self.img_list)
        label_img_set = set()
        
        # Identify images present in the label file
        for item in self.lable_list:
            img_path = item.get("filename")
            if img_path:
                label_img_set.add(img_path)

        # Filter label list to keep only those that have corresponding images in the directory
        filtered_label_list = []
        self.raw_lable_list = []

        for item in self.lable_list:
            img_path = item.get("filename")

            if img_path in img_set:
                filtered_label_list.append(item)
            else:
                logging.info(f"Label path not found in folder: {img_path}")
                self.raw_lable_list.append(item)

        self.lable_list = filtered_label_list

        # Remove images from directory that are not in the label file (or set)
        self.raw_img_list = []
        for img_path in self.img_list:
            if img_path not in label_img_set:
                logging.info(f"Image not present in label file, removing: {img_path}")
                self.raw_img_list.append(img_path)
                file_to_remove = os.path.join(img_dir_path, img_path)
                if os.path.exists(file_to_remove):
                    os.remove(file_to_remove)

    def remove_img(self):
        # Placeholder or legacy method
        pass

    def remove_currupt_image(self, dir_path: str):
        corrupted_removed = 0
        valid_items = []
        self.corrupted_list = []

        # Iterate over the valid label list to check physical file integrity
        for item in self.lable_list:
            img_name = item.get("filename")
            file_path = os.path.join(dir_path, img_name)
            
            try:
                img = cv2.imread(file_path)
                if img is None:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    corrupted_removed += 1
                    self.corrupted_list.append(img_name)
                    logging.debug(f"Removed corrupted image: {file_path}")
                else:
                    valid_items.append(item)
            except Exception:
                if os.path.exists(file_path):
                    os.remove(file_path)
                corrupted_removed += 1
                self.corrupted_list.append(img_name)
                logging.debug(f"Removed unreadable image: {file_path}")
        
        self.lable_list = valid_items
        logging.info(f"Removed {corrupted_removed} corrupted/unreadable images.")

    def init_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Started Data Ingestion")
            
            # Helper to process each split
            def _process_split(dir_path, label_path, artifact_dir):
                logging.info(f"Processing data from {dir_path} to {artifact_dir}")
                self.valid_paths(dir_path, label_path)
                self.remove_currupt_image(dir_path)

                if self.lable_list:
                    # Save validated labels
                    df = pd.DataFrame(self.lable_list)
                    df.to_csv(os.path.join(artifact_dir, self.lable_file), index=False)

                    # Copy validated images
                    for item in self.lable_list:
                        img_name = item.get("filename")
                        src_img = os.path.join(dir_path, img_name)
                        dst_img = os.path.join(artifact_dir, img_name)
                        if os.path.exists(src_img):
                            shutil.copy(src_img, dst_img)
                        else:
                            logging.warning(f"Image {src_img} expected but not found during copy")
            
            # Process Train
            _process_split(self.train_dir_path, self.train_lable_file_path, self.artifact_train_dir)
            
            # Process Test
            _process_split(self.test_dir_path, self.test_lable_file_path, self.artifact_test_dir)
            
            # Process Valid
            _process_split(self.valid_dir_path, self.valid_lable_file_path, self.artifact_valid_dir)

            logging.info("Data Ingestion Completed")
            
            return DataIngestionArtifact(
                train_dir_path=self.artifact_train_dir,
                test_dir_path=self.artifact_test_dir,
                valid_dir_path=self.artifact_valid_dir,
                train_lable_path=os.path.join(self.artifact_train_dir, self.lable_file),
                test_lable_path=os.path.join(self.artifact_test_dir, self.lable_file),
                valid_lable_path=os.path.join(self.artifact_valid_dir, self.lable_file)
            )

        except Exception as e:
            raise CustomException(e, sys)
