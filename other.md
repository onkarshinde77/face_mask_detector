# All about the VGG/Project

- VGG = Visual Geometry Group
- It was trained on ImageNet (1.2M images, 1000 classes).
✅ Uses only 3×3 convolution filters
- Instead of large filters like 7×7 or 11×11.
Why?
- More non-linearity
- Fewer parameters
- Better feature extraction

📏 Input Size
- 224 × 224 × 3
- RGB images
- Preprocessing: subtract ImageNet mean (via preprocess_input)

🧮How Many Parameters?
- 👉 ~138 Million parameters
- Very heavy model.

⚖ Advantages
- ✔ Simple architecture
- ✔ Good feature extractor
- ✔ Easy to fine-tune
- ✔ Works well with transfer learning

❌ Disadvantages
- ❌ Very large size (500+ MB)
- ❌ High memory usage
- ❌ Slow compared to modern models
- ❌ Not suitable for mobile devices


💼 16️⃣ Common Interview Questions
Q1: Why small filters (3×3)?
- Because stacked small filters increase non-linearity and reduce parameters.
Q2: Why does VGG16 use MaxPooling?
- To reduce spatial size and computation.
Q3: Why does it have so many parameters?
- Fully connected layers contribute most parameters.
Q4: Why freeze early layers in transfer learning?
- Because early layers learn general features like edges and textures.

🚀 How To Improve VGG16 Performance?
- Data augmentation
- Fine-tune Block5
- Use lower learning rate
- Add dropout
- Use batch normalization

🏁 Final Interview Summary (Memorize This)
VGG16 is a 16-layer CNN developed by Oxford’s VGG group.
It uses stacked 3×3 convolution layers and was trained on ImageNet.
It has 138M parameters and is widely used for transfer learning.
It extracts hierarchical features and performs well on small datasets.


Hierarchical Feature Learning : Layer-wise representation:
Layer Depth	Learns
Block1	Edges
Block2	Textures
Block3	Shapes
Block4	Parts
Block5	Objects

![alt text](assets/image.png)

Binary Face Mask Classification Dataset

This dataset consists of over 13,000 face images divided into two categories: with_mask and without_mask. It includes diverse subjects across age groups and genders with variations in lighting, pose, and background. The dataset is structured for supervised learning and can be used for training CNN-based models such as VGG16, MobileNet, or ResNet for real-time face mask detection.