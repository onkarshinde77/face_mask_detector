import os
from typing import List,Tuple
from src.exception.exception import CustomException
from src.logger.logger import logging
from src import constant
from src.entity.config_entity import DataIngestionConfig
from src.utils.helper import read_img_path , read_img_from_csv
import cv2
import shutil
import pandas as pd


class DataIngestion:
    def __init__(self, config:DataIngestionConfig):
        
        self.tain_dir_path:str = config.train_dir_path
        self.tain_test_path:str = config.test_dir_path
        self.tain_valid_path:str = config.valid_dir_path
        self.train_lable_file_path:str = config.train_lable_path
        self.test_lable_file_path:str = config.test_lable_path
        self.valid_dir_file_path:str = config.valid_dir_path
        
        self.img_extenstion:Tuple = (".jpg",".jpeg",".png")
        self.img_list:List[str] = []
        self.lable_list:List[dict] = []
        self.raw_img_list:List[str] = []
        self.raw_lable_list:List[str] = []
        
        self.artifact_dir = constant.ARTIFACT_DIR
        self.artifact_data_dir = os.path.join(constant.ARTIFACT_DIR,constant.DATA_DIR)
        self.lable_file:str = "lable.csv"
        self.artifact_train_dir = os.path.join(self.artifact_data_dir,constant.TRAIN_DATA_DIR)
        self.artifact_test_dir = os.path.join(self.artifact_data_dir,constant.TEST_DATA_DIR)
        self.artifact_valid_dir = os.path.join(self.artifact_data_dir,constant.VALID_DATA_DIR)
        
        os.makedirs(self.artifact_dir,exist_ok=True)
        os.makedirs(self.artifact_train_dir,exist_ok=True)
        os.makedirs(self.artifact_test_dir,exist_ok=True)
        os.makedirs(self.artifact_valid_dir,exist_ok=True)
        
    

    def valid_paths(self, img_dir_path: str, lable_file_path: str):
        if not os.path.exists(lable_file_path):
            logging.info(f"file path not found: {lable_file_path}")
            raise FileNotFoundError(f"file path not found: {lable_file_path}")

        self.lable_list: List[dict] = read_img_from_csv(lable_file_path)
        self.img_list, _ = read_img_path(img_dir_path)

        img_set = set(self.img_list)
        label_img_set = set()
        for item in self.lable_list:
            img_path = item.get("image_path")
            if img_path:
                label_img_set.add(img_path)

        filtered_label_list = []
        self.raw_lable_list = []

        for item in self.lable_list:
            img_path = item.get("image_path")

            if img_path in img_set:
                filtered_label_list.append(item)
            else:
                logging.info(f"Label path not found in folder: {img_path}")
                self.raw_lable_list.append(item)

        self.lable_list = filtered_label_list

        self.raw_img_list = []
        for img_path in self.img_list:
            if img_path not in label_img_set:
                logging.info(f"Image not present in label file, removing: {img_path}")
                self.raw_img_list.append(img_path)
                os.remove(img_path)

    def remove_img(self):
        if not self.raw_img_list:
            logging.info("Not present img for remove")
            return
        for i in self.raw_img_list:
            os.remove(i)
                  
    def remove_currupt_image(self):
        corrupted_removed = 0
        for file_path in self.img_list:
            try:
                img = cv2.imread(file_path)
                if img is None:
                    os.remove(file_path)
                    corrupted_removed += 1
                    logging.debug(f"Removed corrupted image: {file_path}")
            except Exception:
                os.remove(file_path)
                corrupted_removed += 1
                logging.debug(f"Removed unreadable image: {file_path}")

    def process_split(self, img_dir_path: str, label_file_path: str, artifact_dir: str):
        logging.info(f"Processing split: {img_dir_path}")
        # Step 1: Validate paths and filter
        self.valid_paths(img_dir_path, label_file_path)
        valid_images = []
        valid_labels = []

        for item in self.lable_list:
            img_path = item.get("image_path")
            if not os.path.exists(img_path):
                continue

            # Check corruption
            img = cv2.imread(img_path)
            if img is None:
                logging.info(f"Corrupted image skipped: {img_path}")
                continue

            # Copy image to artifact folder
            file_name = os.path.basename(img_path)
            dest_path = os.path.join(artifact_dir, file_name)

            shutil.copy2(img_path, dest_path)

            valid_images.append(dest_path)
            valid_labels.append(item)

        # Save cleaned CSV
        df = pd.DataFrame(valid_labels)
        csv_path = os.path.join(artifact_dir, self.lable_file)
        df.to_csv(csv_path, index=False)

        logging.info(f"Saved cleaned data to {artifact_dir}")

    def init_data_ingestion(self):

        # Process train
        self.process_split(
            self.tain_dir_path,
            self.train_lable_file_path,
            self.artifact_train_dir
        )

        # Process test
        self.process_split(
            self.tain_test_path,
            self.test_lable_file_path,
            self.artifact_test_dir
        )

        # Process valid
        self.process_split(
            self.tain_valid_path,
            self.valid_dir_file_path,
            self.artifact_valid_dir
        )

        logging.info("Data ingestion completed successfully.")

    
    
    
