# unprocess data path
DATA_DIR:str = "face_mask_dataset"
TRAIN_DATA_DIR:str = "train"
TEST_DATA_DIR:str = "test"
VALID_DATA_DIR:str = "valid"
DATA_CLASSES_FILE_PATH:str = "_classes.csv"

# process data
ARTIFACT_DIR:str = "artifact"
IMG_DIR:str = "images"
LABLE_DIR:str = "labels"
LABLE_FILE_NAME:str = "label.csv"
# other
img_extention = (".jpg",".jpeg",".png")
# model 
IMG_SIZE = 224
NUM_CLASSES = 1
EPOCHS=10
MODEL_DIR:str = "models"
MODEL_NAME:str = "face_mask_model2.h5"
