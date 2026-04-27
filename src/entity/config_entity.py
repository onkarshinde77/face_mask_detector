from dataclasses import dataclass
from typing import List
import os
from src import constant

@dataclass
class DataIngestionConfig:
    train_dir_path:str = os.path.join(constant.DATA_DIR,constant.TRAIN_DATA_DIR)
    test_dir_path:str = os.path.join(constant.DATA_DIR,constant.TEST_DATA_DIR)
    valid_dir_path:str = os.path.join(constant.DATA_DIR,constant.VALID_DATA_DIR)