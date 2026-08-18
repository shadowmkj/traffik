# ==============================================================================
# Fine-Tuning YOLOv8 to Recognize "LORD_ALTO"
# ==============================================================================
# This script converts captured car images from `captures/` into a YOLO-formatted
# dataset, fine-tunes `yolov8n.pt` with class label "LORD_ALTO", and exports
# the trained weights as `yolov8_lord_alto.pt`.

import os
import glob
import shutil
import random
import yaml
from ultralytics import YOLO

CAPTURES_DIR = "captures"
DATASET_DIR = "dataset_alto"
MODEL_OUTPUT_PATH = "yolov8_lord_alto.pt"


def prepare_dataset():
    """Builds YOLO detection directory structure and generates bounding box labels."""
    images = glob.glob(os.path.join(CAPTURES_DIR, "*.jpg"))
    if not images:
        raise FileNotFoundError(
            f"No images found in '{CAPTURES_DIR}/'. Run `uv run main.py` first to collect car captures."
        )

    print(f"Found {len(images)} captured images in '{CAPTURES_DIR}/'.")

    # Clear and recreate dataset directory structure
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)

    for split in ["train", "val"]:
        os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)

    # Shuffle and split images 80% train / 20% val
    random.seed(42)
    random.shuffle(images)
    split_idx = int(len(images) * 0.8)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    def copy_and_label(image_list, split_name):
        for img_path in image_list:
            basename = os.path.basename(img_path)
            name_without_ext = os.path.splitext(basename)[0]

            # Copy image file to split images directory
            target_img_path = os.path.join(DATASET_DIR, "images", split_name, basename)
            shutil.copy(img_path, target_img_path)

            # Generate label file (Class 0: LORD_ALTO covering full crop: x_center=0.5, y_center=0.5, w=1.0, h=1.0)
            target_lbl_path = os.path.join(DATASET_DIR, "labels", split_name, f"{name_without_ext}.txt")
            with open(target_lbl_path, "w") as f:
                f.write("0 0.5 0.5 1.0 1.0\n")

    copy_and_label(train_imgs, "train")
    copy_and_label(val_imgs, "val")

    # Create dataset.yaml configuration file
    dataset_yaml = {
        "path": os.path.abspath(DATASET_DIR),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "LORD_ALTO"},
    }

    yaml_path = os.path.join(DATASET_DIR, "dataset_alto.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f)

    print(f"Dataset prepared successfully at '{DATASET_DIR}/'.")
    return yaml_path


def train_model(yaml_path):
    """Fine-tunes YOLOv8 on LORD_ALTO dataset."""
    print("Initializing YOLOv8 base model...")
    model = YOLO("yolov8n.pt")

    print("Starting fine-tuning for 'LORD_ALTO' class...")
    results = model.train(
        data=yaml_path,
        epochs=10,
        imgsz=320,
        batch=8,
        project="runs",
        name="lord_alto",
        exist_ok=True,
    )

    # Copy best trained model weights to root directory
    possible_weights_paths = [
        os.path.join("runs", "detect", "runs", "lord_alto", "weights", "best.pt"),
        os.path.join("runs", "detect", "lord_alto", "weights", "best.pt"),
        os.path.join("runs", "lord_alto", "weights", "best.pt"),
    ]

    saved_weights = None
    for p in possible_weights_paths:
        if os.path.exists(p):
            saved_weights = p
            break

    if saved_weights:
        shutil.copy(saved_weights, MODEL_OUTPUT_PATH)
        print(f"Fine-tuned model saved to '{MODEL_OUTPUT_PATH}'!")
    else:
        # Fallback search for any best.pt under runs/
        matches = glob.glob("runs/**/best.pt", recursive=True)
        if matches:
            shutil.copy(matches[0], MODEL_OUTPUT_PATH)
            print(f"Fine-tuned model saved to '{MODEL_OUTPUT_PATH}' from {matches[0]}!")
        else:
            print(f"Warning: Best weights file not found in runs/.")


if __name__ == "__main__":
    yaml_config = prepare_dataset()
    train_model(yaml_config)
