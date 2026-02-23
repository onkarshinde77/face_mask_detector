import os
import cv2
import xml.etree.ElementTree as ET
from src import constant

IMAGE_DIR = os.path.join("data2","images")
ANNOTATION_DIR = os.path.join("data2","annotations")

OUTPUT_DIR = "data2"
WITH_MASK_DIR = os.path.join(constant.ARTIFACT_DIR,OUTPUT_DIR, "with_mask")
WITHOUT_MASK_DIR = os.path.join(OUTPUT_DIR, "without_mask")

os.makedirs(WITH_MASK_DIR, exist_ok=True)
os.makedirs(WITHOUT_MASK_DIR, exist_ok=True)

for xml_file in os.listdir(ANNOTATION_DIR):
    tree = ET.parse(os.path.join(ANNOTATION_DIR, xml_file))
    root = tree.getroot()
    filename = root.find("filename").text
    image_path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(image_path)
    if img is None:
        continue
    for i, obj in enumerate(root.findall("object")):
        label = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        face = img[ymin:ymax, xmin:xmax]
        face = cv2.resize(face, (224, 224))
        if label == "with_mask":
            save_path = os.path.join(WITH_MASK_DIR, f"{filename}_{i}.jpg")
        else:
            save_path = os.path.join(WITHOUT_MASK_DIR, f"{filename}_{i}.jpg")
        cv2.imwrite(save_path, face)

print("Dataset prepared successfully.")