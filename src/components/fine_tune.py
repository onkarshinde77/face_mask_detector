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

 
from src import constant
 

TRAIN_DATASET_PATH = os.path.join(constant.DATA_DIR, constant.TRAIN_DATA_DIR)
VALID_DATASET_PATH = os.path.join(constant.DATA_DIR, constant.VALID_DATA_DIR)

MODEL_PATH = os.path.join(constant.ARTIFACT_DIR, constant.MODEL_DIR, "EfficientNetB4.pth")

IMG_SIZE = constant.FINE_TUNE_IMG_SIZE
BATCH_SIZE = constant.FINE_TUNE_BATCH_SIZE

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Using Device:", device)
 
# Data Augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1
    ),
    transforms.GaussianBlur(
        kernel_size=3
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
    lr=constant.FINE_TUNE_LEARNING_RATE_CLASSIFIER
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
patience = constant.FINE_TUNE_PATIENCE
counter = 0

best_weights = copy.deepcopy(model.state_dict())

print("\nTraining Classifier Head...\n")

for epoch in range(constant.FINE_TUNE_EPOCHS_CLASSIFIER):

    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = validate()
    print(
        f"Epoch [{epoch+1}/{constant.FINE_TUNE_EPOCHS_CLASSIFIER}] "
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
    lr=constant.FINE_TUNE_LEARNING_RATE_FINETUNE
)

for epoch in range(constant.FINE_TUNE_EPOCHS_FINETUNE):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = validate()
    print(
        f"FineTune [{epoch+1}/{constant.FINE_TUNE_EPOCHS_FINETUNE}] "
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