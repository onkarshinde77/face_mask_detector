from dataclasses import dataclass
from typing import List
import os
from src import constant

@dataclass
class DataIngestionConfig:
    train_dir_path:str = os.path.join(constant.DATA_DIR,constant.TRAIN_DATA_DIR)
    test_dir_path:str = os.path.join(constant.DATA_DIR,constant.TEST_DATA_DIR)
    valid_dir_path:str = os.path.join(constant.DATA_DIR,constant.VALID_DATA_DIR)
    train_lable_path:str = os.path.join(constant.DATA_DIR,constant.TRAIN_DATA_DIR,constant.DATA_CLASSES_FILE_PATH)
    test_lable_path:str = os.path.join(constant.DATA_DIR,constant.TEST_DATA_DIR,constant.DATA_CLASSES_FILE_PATH)
    valid_lable_path:str = os.path.join(constant.DATA_DIR,constant.VALID_DATA_DIR,constant.DATA_CLASSES_FILE_PATH)