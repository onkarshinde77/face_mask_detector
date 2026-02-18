import os
from src import constant
from src.exception.exception import CustomException
from src.logger.logger import logging
from typing import List,Tuple,Dict
import numpy as np
import pandas as pd

def read_img_path(dir_path:str)->Tuple[List[str], List[str]]:
    logging.info("entering in read_img_path function")
    if not os.path.exists(dir_path) :
        logging.info(f"file path not found : {dir_path}")
        raise FileNotFoundError(f"file path not found : {dir_path}")
    valid_paths:List[str]=[]
    raw_path:List[str]=[]
    for file in os.listdir(dir_path):
        if file.endswith(".csv"):
            continue
        if not file.endswith(constant.img_extention):
            raw_path.append(file)
            continue
        valid_paths.append(str(file))
    return valid_paths,raw_path

def read_img_from_csv(file_path:str)->List[Dict]:
    logging.info("entering in read_img_from_csv function")
    file = pd.read_csv(file_path)
    valid_path: List[Dict] = file.to_dict(orient="records")
    return valid_path
    
    