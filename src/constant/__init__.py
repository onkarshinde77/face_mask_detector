# unprocess data path
DATA_DIR:str = "face_mask_dataset"
TRAIN_DATA_DIR:str = "train"
TEST_DATA_DIR:str = "test"
VALID_DATA_DIR:str = "valid"

# process data
ARTIFACT_DIR:str = "artifact"
IMG_DIR:str = "images"

# other
img_extention = (".jpg",".jpeg",".png")
# model 
IMG_SIZE = 224
NUM_CLASSES = 2
EPOCHS=10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MODEL_DIR:str = "models"
MODEL_NAME:str = "efficientnetb0_model.h5"

# fine-tune model (EfficientNetB4)
FINE_TUNE_IMG_SIZE = 380
FINE_TUNE_BATCH_SIZE = 16
FINE_TUNE_LEARNING_RATE_CLASSIFIER = 1e-4
FINE_TUNE_LEARNING_RATE_FINETUNE = 1e-5
FINE_TUNE_EPOCHS_CLASSIFIER = 10
FINE_TUNE_EPOCHS_FINETUNE = 5
FINE_TUNE_PATIENCE = 6

# App settings
APP_HOST = "0.0.0.0"
APP_PORT = 7860
APP_DEBUG = False
APP_THREADED = True
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB
ALLOWED_IMAGE_TYPES = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}
ALLOWED_VIDEO_TYPES = {"mp4", "avi", "mov", "mkv", "webm"}
