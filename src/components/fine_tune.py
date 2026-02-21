import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

TRAIN_DATASET_PATH =os.path.join("artifact" ,"data2","train")
VALID_DATASET_PATH =os.path.join("artifact" ,"data2","valid")

MODEL_PATH = os.path.join("artifact" ,"models","face_mask_model2.h5")
NEW_MODEL_PATH = os.path.join("artifact" ,"models","face_mask_model3.h5")

# Load existing model
model = load_model(MODEL_PATH)
base_model = model.get_layer("vgg16")

for layer in base_model.layers[:-4]:
    layer.trainable = False
    
for layer in base_model.layers[-4:]:
    layer.trainable = True
    
for layer in base_model.layers[:]:
    print(layer)

# Freeze first layers
for layer in model.layers:
    print(layer.name, layer.trainable)

# # Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    TRAIN_DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    VALID_DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

model.save(NEW_MODEL_PATH)
print("Fine-tuning completed.")