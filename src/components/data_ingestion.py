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

        self.image_size: Tuple[int, int] = (224, 224)
    
        self.artifact_dir = constant.ARTIFACT_DIR
        self.artifact_data_dir = os.path.join(constant.ARTIFACT_DIR, constant.DATA_DIR)
        self.artifact_train_dir = os.path.join(self.artifact_data_dir, constant.TRAIN_DATA_DIR)
        self.artifact_test_dir = os.path.join(self.artifact_data_dir, constant.TEST_DATA_DIR)
        self.artifact_valid_dir = os.path.join(self.artifact_data_dir, constant.VALID_DATA_DIR)
        
        self.train_lable_file: str = "train.csv"
        self.test_lable_file: str = "test.csv"
        self.valid_lable_file: str = "valid.csv"
        
        self.img_dir: str = constant.IMG_DIR
        self.label_dir: str = constant.LABLE_DIR

        os.makedirs(self.artifact_dir, exist_ok=True)
        os.makedirs(self.artifact_train_dir, exist_ok=True)
        os.makedirs(self.artifact_test_dir, exist_ok=True)
        os.makedirs(self.artifact_valid_dir, exist_ok=True)

    def valid_paths(self, img_dir_path: str, lable_file_path: str):
        try:
            logging.info(f"Validating paths: {img_dir_path}, {lable_file_path}")

            if not os.path.exists(lable_file_path):
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
                    logging.warning(f"Label without image removed: {img_name}")
                    self.raw_lable_list.append(item)

            self.lable_list = filtered_label_list

        except Exception as e:
            raise CustomException(e, sys)

    def remove_currupt_image(self, dir_path: str):
        try:
            logging.info(f"Checking corrupted images in {dir_path}")

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
                        logging.warning(f"Corrupted image removed: {img_name}")
                        self.corrupted_list.append(img_name)
                    else:
                        valid_items.append(item)
                except Exception:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    logging.warning(f"Unreadable image removed: {img_name}")
                    self.corrupted_list.append(img_name)

            self.lable_list = valid_items

        except Exception as e:
            raise CustomException(e, sys)

    def _process_split(self, dir_path: str, label_path: str, artifact_dir: str, lable_file: str):
        try:
            logging.info(f"Processing split: {dir_path}")

            self.valid_paths(dir_path, label_path)
            self.remove_currupt_image(dir_path)

            if not self.lable_list:
                logging.warning("No valid data found.")
                return

            label_save_dir = os.path.join(artifact_dir, self.label_dir)
            os.makedirs(label_save_dir, exist_ok=True)
            
            df = pd.DataFrame(self.lable_list)
            label_save_path = os.path.join(label_save_dir, lable_file)
            df.to_csv(label_save_path, index=False)
            logging.info(f"Label file saved at {label_save_path}")

            img_save_dir = os.path.join(artifact_dir, self.img_dir)
            os.makedirs(img_save_dir, exist_ok=True)

            for item in self.lable_list:
                img_name = item.get("filename")
                src_img_path = os.path.join(dir_path, img_name)
                temp_img_path = os.path.join(img_save_dir, f"temp_{img_name}")
                final_img_path = os.path.join(img_save_dir, img_name)

                if os.path.exists(src_img_path):
                    img = cv2.imread(src_img_path)
                    if img is not None:
                        resized_img = cv2.resize(img, self.image_size)
                        cv2.imwrite(temp_img_path, resized_img)

                        shutil.move(temp_img_path, final_img_path)
                        logging.info(f"Image saved after resize: {final_img_path}")
                    else:
                        logging.warning(f"Image read failed: {img_name}")
                else:
                    logging.warning(f"Image not found: {src_img_path}")

        except Exception as e:
            raise CustomException(e, sys)

    def init_data_ingestion(self) -> DataIngestionArtifact:
        try:
            train_img_dir = os.path.join(self.artifact_train_dir, self.img_dir)
            test_img_dir = os.path.join(self.artifact_test_dir, self.img_dir)
            valid_img_dir = os.path.join(self.artifact_valid_dir, self.img_dir)
            
            if (os.path.exists(train_img_dir) and 
                os.path.exists(test_img_dir) and 
                os.path.exists(valid_img_dir)):
                
                logging.info("Data already ingested. Skipping ingestion step.")
                return DataIngestionArtifact(
                    train_dir_path=train_img_dir,
                    test_dir_path=test_img_dir,
                    valid_dir_path=valid_img_dir,
                    train_lable_path=os.path.join(self.artifact_train_dir, self.label_dir, self.train_lable_file),
                    test_lable_path=os.path.join(self.artifact_test_dir, self.label_dir, self.test_lable_file),
                    valid_lable_path=os.path.join(self.artifact_valid_dir, self.label_dir, self.valid_lable_file),
                )

            logging.info("Started Data Ingestion")

            self._process_split(
                self.train_dir_path,
                self.train_lable_file_path,
                self.artifact_train_dir,
                self.train_lable_file
            )

            self._process_split(
                self.test_dir_path,
                self.test_lable_file_path,
                self.artifact_test_dir,
                self.test_lable_file
            )

            self._process_split(
                self.valid_dir_path,
                self.valid_lable_file_path,
                self.artifact_valid_dir,
                self.valid_lable_file
            )
            logging.info(f"Valid Image count : {self.img_list}")
            logging.info(f"lable count : {self.lable_list}")
            
            logging.info("Data Ingestion Completed")
            return DataIngestionArtifact(
                train_dir_path=train_img_dir,
                test_dir_path=test_img_dir,
                valid_dir_path=valid_img_dir,
                train_lable_path=os.path.join(self.artifact_train_dir, self.label_dir, self.train_lable_file),
                test_lable_path=os.path.join(self.artifact_test_dir, self.label_dir, self.test_lable_file),
                valid_lable_path=os.path.join(self.artifact_valid_dir, self.label_dir, self.valid_lable_file),
            )

        except Exception as e:
            raise CustomException(e, sys)


# if __name__ == "__main__":
#     config = DataIngestionConfig()
#     obj = DataIngestion(config=config)
#     obj.init_data_ingestion()
