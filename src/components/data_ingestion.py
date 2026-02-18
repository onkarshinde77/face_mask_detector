import sys
from typing import List, Tuple
from src.exception.exception import CustomException
from src.logger.logger import logging
from src import constant
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.utils.helper import read_img_path, read_img_from_csv
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
        self.raw_lable_list: List[dict] = []
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

        self.lable_list = read_img_from_csv(lable_file_path)
        self.img_list, _ = read_img_path(img_dir_path)

        img_set = set(self.img_list)
        filtered_label_list = []
        self.raw_lable_list = []

        for item in self.lable_list:
            img_name = item.get("filename")
            if img_name in img_set:
                filtered_label_list.append(item)
            else:
                logging.info(f"Label path not found in folder: {img_name}")
                self.raw_lable_list.append(item)

        self.lable_list = filtered_label_list

        self.raw_img_list = []
        for img_name in self.img_list:
            if img_name not in {item.get("filename") for item in self.lable_list}:
                file_to_remove = os.path.join(img_dir_path, img_name)
                if os.path.exists(file_to_remove):
                    os.remove(file_to_remove)
                self.raw_img_list.append(img_name)

    def remove_currupt_image(self, dir_path: str):
        valid_items = []
        self.corrupted_list = []

        for item in self.lable_list:
            img_name = item.get("filename")
            file_path = os.path.join(dir_path, img_name)

            try:
                img = cv2.imread(file_path)
                if img is None:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    self.corrupted_list.append(img_name)
                else:
                    valid_items.append(item)
            except Exception:
                if os.path.exists(file_path):
                    os.remove(file_path)
                self.corrupted_list.append(img_name)

        self.lable_list = valid_items

    def _process_split(self, dir_path: str, label_path: str, artifact_dir: str):
        logging.info(f"Processing data from {dir_path} to {artifact_dir}")

        self.valid_paths(dir_path, label_path)
        self.remove_currupt_image(dir_path)

        if self.lable_list:
            df = pd.DataFrame(self.lable_list)
            df.to_csv(os.path.join(artifact_dir, self.lable_file), index=False)

            for item in self.lable_list:
                img_name = item.get("filename")
                src_img = os.path.join(dir_path, img_name)
                dst_img = os.path.join(artifact_dir, img_name)
                if os.path.exists(src_img):
                    shutil.copy(src_img, dst_img)

    def init_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Started Data Ingestion")

            self._process_split(
                self.train_dir_path,
                self.train_lable_file_path,
                self.artifact_train_dir,
            )

            self._process_split(
                self.test_dir_path,
                self.test_lable_file_path,
                self.artifact_test_dir,
            )

            self._process_split(
                self.valid_dir_path,
                self.valid_lable_file_path,
                self.artifact_valid_dir,
            )

            logging.info("Data Ingestion Completed")

            return DataIngestionArtifact(
                train_dir_path=self.artifact_train_dir,
                test_dir_path=self.artifact_test_dir,
                valid_dir_path=self.artifact_valid_dir,
                train_lable_path=os.path.join(self.artifact_train_dir, self.lable_file),
                test_lable_path=os.path.join(self.artifact_test_dir, self.lable_file),
                valid_lable_path=os.path.join(self.artifact_valid_dir, self.lable_file),
            )

        except Exception as e:
            raise CustomException(e, sys)
