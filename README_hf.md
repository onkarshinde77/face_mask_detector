## Flow of EfficientNetB4
Image
 ↓
Resize 380x380
 ↓
Data Augmentation
 ↓
Tensor Conversion
 ↓
EfficientNet-B4 (ImageNet Weights)
 ↓
Feature Extraction
 ↓
Dropout
 ↓
Linear Layer
 ↓
1 Logit
 ↓
BCEWithLogitsLoss
 ↓
Backpropagation
 ↓
Adam Optimizer
 ↓
Fine Tuning
 ↓
Save Model




FineTune [4/5] Train Loss: 0.0385 Train Acc: 0.9867 Val Loss: 0.0068 Val Acc: 1.0000
/home/meow/miniconda3/envs/tfgpu/lib/python3.11/site-packages/PIL/Image.py:1137: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
  warnings.warn(
/home/meow/miniconda3/envs/tfgpu/lib/python3.11/site-packages/PIL/Image.py:1137: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
  warnings.warn(
FineTune [5/5] Train Loss: 0.0314 Train Acc: 0.9895 Val Loss: 0.0054 Val Acc: 1.0000