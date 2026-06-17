# import os
# import tensorflow as tf
# from tensorflow.keras.applications import EfficientNetB4
# from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping

# TRAIN_DATASET_PATH = os.path.join("face_mask_dataset", "train")
# VALID_DATASET_PATH = os.path.join("face_mask_dataset", "valid")

# NEW_MODEL_PATH = os.path.join(
#     "artifact",
#     "models",
#     "EfficientNetB4.keras"
# )
# IMG_SIZE = (380, 380)

# # ------------------------
# # Data Augmentation
# # ------------------------

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     zoom_range=0.2,
#     width_shift_range=0.15,
#     height_shift_range=0.15,
#     horizontal_flip=True,
#     brightness_range=[0.2, 1.9],
#     fill_mode="nearest"
# )

# valid_datagen = ImageDataGenerator(
#     rescale=1./255
# )

# train_generator = train_datagen.flow_from_directory(
#     TRAIN_DATASET_PATH,
#     target_size=IMG_SIZE,
#     batch_size=16,
#     class_mode="binary"
# )

# val_generator = valid_datagen.flow_from_directory(
#     VALID_DATASET_PATH,
#     target_size=IMG_SIZE,
#     batch_size=16,
#     class_mode="binary"
# )

# # ------------------------
# # EfficientNetB4
# # ------------------------

# base_model = EfficientNetB4(
#     include_top=False,
#     weights="imagenet",
#     input_shape=(380, 380, 3)
# )

# base_model.trainable = False

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.3)(x)

# output = Dense(
#     1,
#     activation="sigmoid"
# )(x)

# model = Model(
#     inputs=base_model.input,
#     outputs=output
# )

# model.compile(
#     optimizer=Adam(learning_rate=1e-4),
#     loss="binary_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# # ------------------------
# # Training
# # ------------------------

# history = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=10,
#     callbacks=[
#         EarlyStopping(
#             monitor="val_loss",
#             patience=6
#         )
#     ],
#     verbose=1 # it use to show epocs and progress bar in less detalied
#     # verbose=2 # it use to show epoch number and progress bar and Loss and validation accuracy and validation Loss and loss
# )

# # ------------------------
# # Fine-tuning
# # ------------------------

# base_model.trainable = True

# for layer in base_model.layers[:-50]:
#     layer.trainable = False

# model.compile(
#     optimizer=Adam(learning_rate=1e-5),
#     loss="binary_crossentropy",
#     metrics=["accuracy"]
# )

# history_finetune = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=5,
#     callbacks=[
#         EarlyStopping(
#             monitor="val_loss",
#             patience=6
#         )
#     ],
#     verbose=1 # it use to show epocs and progress bar in less detalied
# )

# model.save(NEW_MODEL_PATH)

# print("EfficientNetB4 training completed.")


import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.models import (
    efficientnet_b4,
    EfficientNet_B4_Weights
)

 
# Config
 

TRAIN_DATASET_PATH = "face_mask_dataset/train"
VALID_DATASET_PATH = "face_mask_dataset/valid"

MODEL_PATH = "artifact/models/EfficientNetB4.pth"

IMG_SIZE = 380
BATCH_SIZE = 16

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Using Device:", device)
 
# Data Augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.15, 0.15),
        scale=(0.8, 1.2)
    ),
    transforms.ColorJitter(
        brightness=(0.2, 1.9)
    ),
    transforms.ToTensor()
])

valid_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])


# Dataset
train_dataset = datasets.ImageFolder(
    TRAIN_DATASET_PATH,
    transform=train_transform
)

valid_dataset = datasets.ImageFolder(
    VALID_DATASET_PATH,
    transform=valid_transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

print("Classes:", train_dataset.classes)

 
# EfficientNetB4
model = efficientnet_b4(
    weights=EfficientNet_B4_Weights.IMAGENET1K_V1
)

# Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier
in_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, 1)
)
model = model.to(device)

# Loss & Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

 
# Training Function
def train_one_epoch():
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5)
        correct += (preds == labels.bool()).sum().item()
        total += labels.size(0)
    return running_loss / len(train_loader), correct / total

 
# Validation Function
def validate():
    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5)
            correct += (preds == labels.bool()).sum().item()
            total += labels.size(0)
    return running_loss / len(valid_loader), correct / total


# Phase 1 Training
best_loss = float("inf")
patience = 6
counter = 0

best_weights = copy.deepcopy(model.state_dict())

print("\nTraining Classifier Head...\n")

for epoch in range(10):

    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = validate()
    print(
        f"Epoch [{epoch+1}/10] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_acc:.4f}"
    )

    if val_loss < best_loss:
        best_loss = val_loss
        best_weights = copy.deepcopy(model.state_dict())
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early Stopping")
            break
 
# Fine Tuning
print("\nFine-Tuning...\n")

# Freeze everything
for param in model.features.parameters():
    param.requires_grad = False

# for layer in list(model.features.children())[-3:]:
#     for param in layer.parameters():
#         param.requires_grad = True

# Unfreeze last 3 blocks
for layer in model.features[6:]:
    for param in layer.parameters():
        param.requires_grad = True

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-5
)

for epoch in range(5):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = validate()
    print(
        f"FineTune [{epoch+1}/5] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_acc:.4f}"
    )
 
# Save
os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

torch.save(
    model.state_dict(),
    MODEL_PATH
)
print("\nModel Saved:", MODEL_PATH)

# this command execute in another terminal to see GPU utilization
# watch -n 1 nvidia-smi