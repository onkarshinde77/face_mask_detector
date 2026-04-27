from dataclasses import dataclass
from typing import List
import os
from src import constant

@dataclass
class DataIngestionArtifact:
    train_dir_path:str 
    test_dir_path:str 
    valid_dir_path:str 

