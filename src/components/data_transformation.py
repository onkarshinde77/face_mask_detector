import os
import sys
import cv2

from src.exception.exception import CustomException
from src.logger.logger import logging
from src import constant
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.components.face_crop import FaceCropper


class DataTransformation:
    def __init__(self, config: DataTransformationConfig, artifact: DataValidationArtifact):
        self.config = config
        self.artifact = artifact
        self.face_cropper = FaceCropper()

    def crop_and_save_folder(self, source_dir, split_name):
        """
        For each image in source_dir, detect and crop the face, then save
        the cropped image to the output directory under the same label folder.
        Images where no face is detected are dropped.
        Returns the output directory path.
        """
        output_split_dir = os.path.join(self.config.output_dir, split_name)
        total = saved = dropped = 0

        for label_folder in os.listdir(source_dir):
            label_dir = os.path.join(source_dir, label_folder)
            if not os.path.isdir(label_dir):
                continue

            out_label_dir = os.path.join(output_split_dir, label_folder)
            os.makedirs(out_label_dir, exist_ok=True)

            for filename in os.listdir(label_dir):
                if not filename.lower().endswith(constant.img_extention):
                    continue

                image_path = os.path.join(label_dir, filename)
                total += 1

                img = cv2.imread(image_path)
                if img is None:
                    dropped += 1
                    continue

                faces = self.face_cropper.detect_faces(img)
                if not faces:
                    logging.warning(f"No face detected, dropping: {split_name}/{label_folder}/{filename}")
                    dropped += 1
                    continue

                # Save the first (most confident) detected face crop
                cropped_list = self.face_cropper.crop_faces(img, faces)
                face_img = cropped_list[0]["face"]

                save_path = os.path.join(out_label_dir, filename)
                cv2.imwrite(save_path, face_img)
                saved += 1

        logging.info(f"{split_name} — Saved: {saved}, Dropped: {dropped}, Total: {total}")
        return output_split_dir

    def init_data_transformation(self) -> DataTransformationArtifact:
        try:
            train_out = os.path.join(self.config.output_dir, constant.TRAIN_DATA_DIR)
            test_out  = os.path.join(self.config.output_dir, constant.TEST_DATA_DIR)
            valid_out = os.path.join(self.config.output_dir, constant.VALID_DATA_DIR)

            # Skip if cropped directories already exist
            if os.path.exists(train_out) and os.path.exists(test_out) and os.path.exists(valid_out):
                logging.info("Cropped dataset already exists. Skipping data transformation step.")
                return DataTransformationArtifact(
                    train_dir_path=train_out,
                    test_dir_path=test_out,
                    valid_dir_path=valid_out,
                )

            logging.info("Starting Data Transformation (face cropping)")
            os.makedirs(self.config.output_dir, exist_ok=True)

            train_out = self.crop_and_save_folder(self.artifact.train_dir_path, constant.TRAIN_DATA_DIR)
            test_out  = self.crop_and_save_folder(self.artifact.test_dir_path,  constant.TEST_DATA_DIR)
            valid_out = self.crop_and_save_folder(self.artifact.valid_dir_path, constant.VALID_DATA_DIR)

            logging.info("Data Transformation Complete")
            return DataTransformationArtifact(
                train_dir_path=train_out,
                test_dir_path=test_out,
                valid_dir_path=valid_out,
            )

        except Exception as e:
            raise CustomException(e, sys)
