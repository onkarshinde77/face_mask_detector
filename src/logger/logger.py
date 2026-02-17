import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m%d%Y%H%M%S')}"  # ex o/p format : 07252025181830
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE)    # getcwd() stands for "get current working directory".
# creating directory contains timestamp
os.makedirs(logs_path,exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level =logging.INFO,
)

'''
Logging in Python is used to record information about your program’s execution—such as events,
errors, and status messages—in a structured, timestamped way. It offers several key advantages
over simple print statements:

# In machine learning workflows (like your code example), logging typically documents:

1) The start/stop of data processes (e.g., data loading),
2) Information about data or models (shapes, sample values, hyperparameters),
3) Warnings about bad or missing data,
4) Errors encountered during ingestion, training, or prediction.
'''