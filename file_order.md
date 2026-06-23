# Codebase Flow and Reading Order Guide

To understand this codebase systematically, it is recommended to read the files in the following order. This flow follows the pipeline architecture from initial configuration to dataset preparation, model training, evaluation, and finally the web application.

---

### Phase 1: Configuration & Structure

1. **[constant/\_\_init\_\_.py](file:///f:/Projects/face_mask_detector/src/constant/__init__.py)**
   * **Purpose**: Central configurations, thresholds, training hyperparameters (learning rates, batch sizes), image sizes, paths, and application host/port parameters. Start here to understand the global configuration of the project.
2. **[entity/config\_entity.py](file:///f:/Projects/face_mask_detector/src/entity/config_entity.py)**
   * **Purpose**: Declares Python dataclasses wrapping constants for each pipeline stage (e.g. data validation folders, trainer parameters, evaluation paths).
3. **[entity/artifact\_entity.py](file:///f:/Projects/face_mask_detector/src/entity/artifact_entity.py)**
   * **Purpose**: Defines intermediate artifact dataclasses output by one pipeline step and consumed by another (e.g. dataset directory locations, PyTorch model objects, evaluation score text files).

---

### Phase 2: Data Component Pipeline

4. **[components/data\_ingestion.py](file:///f:/Projects/face_mask_detector/src/components/data_ingestion.py)**
   * **Purpose**: Pipeline Step 1. Checks for source dataset directory structures (train, validation, test) and creates ingestion metadata artifacts.
5. **[components/data\_validation.py](file:///f:/Projects/face_mask_detector/src/components/data_validation.py)**
   * **Purpose**: Pipeline Step 2. Cleans the source dataset by checking image resolutions, verifying file format/extensions, and filtering out blurry images using Laplacian variance thresholds.
6. **[components/face\_crop.py](file:///f:/Projects/face_mask_detector/src/components/face_crop.py)**
   * **Purpose**: Core utility component. Uses a pre-trained OpenCV Caffe SSD model (`face_detector/`) to localize faces and return cropped bounding boxes.
7. **[components/data\_transformation.py](file:///f:/Projects/face_mask_detector/src/components/data_transformation.py)**
   * **Purpose**: Pipeline Step 3. Takes validated images, crops all faces found using `FaceCropper`, and outputs a clean face-only dataset (with/without mask classes) inside the artifact directory.

---

### Phase 3: Model Building, Training, & Evaluation

8. **[components/model\_builder.py](file:///f:/Projects/face_mask_detector/src/components/model_builder.py)**
   * **Purpose**: Pipeline Step 4. Instantiates PyTorch `EfficientNetB4` with pre-trained ImageNet weights, freezes backbone feature extraction layers, and appends a custom binary classification head.
9. **[components/model\_trainer.py](file:///f:/Projects/face_mask_detector/src/components/model_trainer.py)**
   * **Purpose**: Pipeline Step 5. Trains the built classifier using standard BCEWithLogitsLoss loss, Adam optimizer, early stopping, and plateau scheduling. Saves the final weights state dict.
10. **[components/model\_evaluation.py](file:///f:/Projects/face_mask_detector/src/components/model_evaluation.py)**
    * **Purpose**: Pipeline Step 6. Evaluates the trained model on the test dataset, computes final loss/accuracy metrics, and writes them to an evaluation report artifact.

---

### Phase 4: Pipelines Integration

11. **[pipelines/train\_pipeline.py](file:///f:/Projects/face_mask_detector/src/pipelines/train_pipeline.py)**
    * **Purpose**: Integrates components 4 to 10 into a cohesive, sequential pipeline. Executing this file performs data ingestion, cleaning, transformation, model training, and performance reporting from scratch.
12. **[pipelines/predict\_pipeline.py](file:///f:/Projects/face_mask_detector/src/pipelines/predict_pipeline.py)**
    * **Purpose**: Production inference pipeline using PyTorch. Loads the trained weights, instantiates the face cropper, preprocesses images, and outputs annotated bounding boxes (Green = Mask, Red = No Mask) with confidence scores for single images, videos, base64 strings, or webcam frames.
13. **[components/fine\_tune.py](file:///f:/Projects/face_mask_detector/src/components/fine_tune.py)**
    * **Purpose**: Standalone advanced script. Rather than just training the classifier head, this script fine-tunes the network. It trains the classification head first, then unfreezes the final blocks (layers 6 onwards) of EfficientNetB4 to specialize them for mask detection.

---

### Phase 5: Web Server & UI Application

14. **[app/components/prediction\_pipelines.py](file:///f:/Projects/face_mask_detector/app/components/prediction_pipelines.py)**
    * **Purpose**: Custom inference pipeline tailored specifically for the Flask app. It adds web features, such as image contrast enhancement (CLAHE + Denoising) for low-light images, and multi-scale scaling for very large high-resolution uploads.
15. **[app/app.py](file:///f:/Projects/face_mask_detector/app/app.py)**
    * **Purpose**: The web server launcher. Initializes a Flask server that exposes API routes for real-time video feeds, webcam streaming, video uploads processing in background threads, and image processing.
