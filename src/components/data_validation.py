import os
import sys
import shutil
import cv2

from src.exception.exception import CustomException
from src.logger.logger import logging
from src import constant
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact


class DataValidation:
    def __init__(self, config: DataValidationConfig, artifact: DataIngestionArtifact):
        self.config = config
        self.artifact = artifact

    # ---- individual image checks ----

    def is_image_readable(self, image_path):
        """Try to read the image file with OpenCV."""
        img = cv2.imread(image_path)
        return img is not None, img

    def is_dimension_valid(self, img):
        """Check if the image has at least the minimum required dimensions."""
        h, w = img.shape[:2]
        return h >= self.config.min_height and w >= self.config.min_width

    def is_blurry(self, img):
        """Return True if the image is too blurry (Laplacian variance below threshold)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < self.config.blur_threshold, variance

    def is_label_valid(self, image_path):
        """Check that the parent folder name is a recognized class label."""
        label = os.path.basename(os.path.dirname(image_path))
        return label in self.config.valid_labels, label

    # ---- validate one folder (train / test / valid) ----

    def validate_and_copy_folder(self, source_dir, split_name, report_lines):
        """
        Walk through each image in source_dir, run all checks,
        and copy only the valid images to the output directory.
        Returns the output directory path for that split.
        """
        output_split_dir = os.path.join(self.config.output_dir, split_name)
        total = passed = skipped = 0

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
                issues = []

                # Check 1: readable
                readable, img = self.is_image_readable(image_path)
                if not readable:
                    issues.append("corrupt/unreadable")

                if readable:
                    # Check 2: dimensions
                    if not self.is_dimension_valid(img):
                        h, w = img.shape[:2]
                        issues.append(f"too small ({w}x{h})")

                    # Check 3: blur
                    blurry, variance = self.is_blurry(img)
                    if blurry:
                        issues.append(f"blurry (variance={variance:.1f})")

                    # Check 4: label
                    label_ok, label = self.is_label_valid(image_path)
                    if not label_ok:
                        issues.append(f"invalid label '{label}'")

                if issues:
                    skipped += 1
                    report_lines.append(f"  [SKIP] {split_name}/{label_folder}/{filename} — {', '.join(issues)}")
                    logging.warning(f"Skipped {filename}: {', '.join(issues)}")
                else:
                    passed += 1
                    shutil.copy2(image_path, os.path.join(out_label_dir, filename))

        report_lines.append(
            f"\n[{split_name.upper()}] Total: {total} | Passed: {passed} | Skipped: {skipped}"
        )
        logging.info(f"{split_name} — Passed: {passed}, Skipped: {skipped}, Total: {total}")
        return output_split_dir

    # ---- main entry point ----

    def init_data_validation(self) -> DataValidationArtifact:
        try:
            # Skip if report already exists
            if os.path.exists(self.config.report_file_path):
                logging.info("Data validation report already exists. Skipping validation step.")
                train_out = os.path.join(self.config.output_dir, constant.TRAIN_DATA_DIR)
                test_out  = os.path.join(self.config.output_dir, constant.TEST_DATA_DIR)
                valid_out = os.path.join(self.config.output_dir, constant.VALID_DATA_DIR)
                return DataValidationArtifact(
                    train_dir_path=train_out,
                    test_dir_path=test_out,
                    valid_dir_path=valid_out,
                    report_file_path=self.config.report_file_path,
                    is_valid=True,
                )

            logging.info("Starting Data Validation")
            os.makedirs(os.path.dirname(self.config.report_file_path), exist_ok=True)
            os.makedirs(self.config.output_dir, exist_ok=True)

            report_lines = ["==== DATA VALIDATION REPORT ====\n"]

            train_out = self.validate_and_copy_folder(
                self.artifact.train_dir_path, constant.TRAIN_DATA_DIR, report_lines
            )
            test_out = self.validate_and_copy_folder(
                self.artifact.test_dir_path, constant.TEST_DATA_DIR, report_lines
            )
            valid_out = self.validate_and_copy_folder(
                self.artifact.valid_dir_path, constant.VALID_DATA_DIR, report_lines
            )

            # Write report
            with open(self.config.report_file_path, "w") as f:
                f.write("\n".join(report_lines))

            logging.info(f"Data Validation Complete. Report saved at: {self.config.report_file_path}")

            return DataValidationArtifact(
                train_dir_path=train_out,
                test_dir_path=test_out,
                valid_dir_path=valid_out,
                report_file_path=self.config.report_file_path,
                is_valid=True,
            )

        except Exception as e:
            raise CustomException(e, sys)
