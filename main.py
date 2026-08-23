import os
import csv
import re
import cv2
import numpy as np
import torch
import easyocr
from ultralytics import YOLO
import supervision as sv
import supervision.detection.utils.internal
import supervision.detection.line_zone

# Detect Hardware Acceleration: NVIDIA CUDA > Apple Silicon MPS > CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# ==============================================================================
# 1. Compatibility Fixes
# ==============================================================================

# Fix for Supervision compatibility with NumPy 2.0+ 2D
# cross product calculation
def _patched_cross_product(anchors, vector):
    vector_at_zero = np.array([
        vector.end.x - vector.start.x,
        vector.end.y - vector.start.y,
    ])
    vector_start = np.array([vector.start.x, vector.start.y])
    diff = anchors - vector_start
    return vector_at_zero[0] * diff[..., 1] - vector_at_zero[1] * diff[..., 0]


supervision.detection.utils.internal.cross_product = _patched_cross_product
supervision.detection.line_zone.cross_product = _patched_cross_product


# ==============================================================================
# 2. Configuration & Model Setup
# ==============================================================================

# Paths configuration
SOURCE_VIDEO_PATH = "short.mp4" if os.path.exists(
    "traffic.mp4") else "short.mp4"
TARGET_VIDEO_PATH = "output.mp4"
CAPTURES_DIR = "captures"
PLATES_CSV_PATH = "number_plates.csv"

# Ensure output directories and files are initialized
os.makedirs(CAPTURES_DIR, exist_ok=True)

# Initialize CSV log file for detected license plates
if not os.path.exists(PLATES_CSV_PATH):
    with open(PLATES_CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "TrackID", "Class",
                        "PlateText", "Confidence", "CropFile"])

# Load YOLOv8 detection model & EasyOCR text recognition engine
model = YOLO('yolo26s.pt')
ocr_reader = easyocr.Reader(['en'], gpu=(DEVICE in ("cuda", "mps")))

# Track unique number plates to avoid printing duplicate log lines per frame
detected_plates_history = {}

# Map COCO 'car' class (index 2) to 'LORD_ALTO' for visual detection labels
if hasattr(model, "names") and 2 in model.names:
    model.names[2] = "LORD_ALTO"

# Extract video metadata for layout configuration
video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)

START = sv.Point(video_info.width // 3, 0)
END = sv.Point(video_info.width // 3, video_info.height)

tracker = sv.ByteTrack()
line_zone = sv.LineZone(start=START, end=END)
line_zone_annotator = sv.LineZoneAnnotator(
    thickness=3, text_thickness=2, text_scale=1.0
)
box_annotator = sv.BoxAnnotator(thickness=3)
label_annotator = sv.LabelAnnotator(
    text_scale=1.0, text_thickness=2, text_padding=8
)


def process_frame(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    results = model(frame, device=DEVICE)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    h, w, _ = frame.shape

    tracker_ids = (
        detections.tracker_id
        if detections.tracker_id is not None
        else [None] * len(detections)
    )

    for idx, (xyxy, class_id, tracker_id) in enumerate(
        zip(detections.xyxy, detections.class_id, tracker_ids)
    ):
        # Fetch human-readable class label name from model (e.g. 'car', 'bus', 'truck', 'person')
        class_name = (
            model.names[int(class_id)]
            if hasattr(model, "names") and int(class_id) in model.names
            else f"class_{class_id}"
        )

        # Only capture images of CARS for the dataset
        if class_name.lower() == "person":
            continue

        x1, y1, x2, y2 = xyxy

        # Bounding box clipping:
        # x1_c, y1_c, x2_c, y2_c are clipped integer pixel bounds.
        # max(0, ...) prevents negative indices on left/top.
        # min(w, ...) / min(h, ...) prevents out-of-bounds indices beyond image dimensions on right/bottom.
        x1_c, y1_c = max(0, int(x1)), max(0, int(y1))
        x2_c, y2_c = min(w, int(x2)), min(h, int(y2))

        # Check for valid positive crop area
        if x2_c > x1_c and y2_c > y1_c:
            crop = frame[y1_c:y2_c, x1_c:x2_c]

            # Format descriptive filename with frame index and track/detection ID
            if tracker_id is not None:
                filename = f"car_frame{frame_idx:05d}_track{tracker_id}.jpg"
            else:
                filename = f"car_frame{frame_idx:05d}_det{idx}.jpg"

            save_path = os.path.join(CAPTURES_DIR, filename)
            cv2.imwrite(save_path, crop)

            # Perform text detection and OCR on vehicle crop for license plates
            if crop.shape[0] >= 20 and crop.shape[1] >= 20:
                ocr_results = ocr_reader.readtext(crop)
                for bbox, raw_text, ocr_conf in ocr_results:
                    # Clean and format alphanumeric license plate characters
                    clean_plate = re.sub(r'[^A-Z0-9]', '', raw_text.upper())

                    # Filter for plausible license plate lengths (>= 4 chars) and confidence
                    if len(clean_plate) >= 4 and ocr_conf >= 0.35:
                        track_key = f"track_{
                            tracker_id}" if tracker_id is not None else f"det_{frame_idx}_{idx}"
                        prev_conf = detected_plates_history.get(
                            track_key, {}).get("confidence", 0.0)

                        if ocr_conf > prev_conf:
                            detected_plates_history[track_key] = {
                                "plate": clean_plate,
                                "raw": raw_text,
                                "confidence": ocr_conf,
                                "frame": frame_idx,
                                "class": class_name,
                                "file": filename,
                            }

                            # 1. Print plate immediately to console
                            print(f"[PLATE DETECTED] Frame {frame_idx:04d} | Track #{tracker_id} | Class: {
                                  class_name} | Plate: {clean_plate} (conf: {ocr_conf:.2%})")

                            # 2. Append detected plate to CSV file
                            with open(PLATES_CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
                                writer = csv.writer(f)
                                writer.writerow(
                                    [frame_idx, tracker_id, class_name, clean_plate, f"{ocr_conf:.4f}", filename])

    # Update line crossing logic
    line_zone.trigger(detections=detections)

    # Prepare class labels with track IDs & confidence for visual annotations
    labels = []
    for class_id, tracker_id, confidence in zip(
        detections.class_id, tracker_ids, detections.confidence
    ):
        class_name = (
            model.names[int(class_id)]
            if hasattr(model, "names") and int(class_id) in model.names
            else f"class_{class_id}"
        )
        if tracker_id is not None:
            labels.append(f"{class_name} #{tracker_id} {confidence:.2f}")
        else:
            labels.append(f"{class_name} {confidence:.2f}")

    # Annotate frame with bounding boxes, class labels, and line counting stats
    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )
    annotated_frame = line_zone_annotator.annotate(
        annotated_frame,
        line_counter=line_zone
    )

    return annotated_frame


# ==============================================================================
# 4. Execution Pipeline
# ==============================================================================

def main():
    print(f"Starting pipeline on '{SOURCE_VIDEO_PATH}'...")
    print(f"Number plates log file: '{PLATES_CSV_PATH}'\n")

    sv.process_video(
        source_path=SOURCE_VIDEO_PATH,
        target_path=TARGET_VIDEO_PATH,
        callback=process_frame
    )

    print("\n================ Pipeline Summary ================")
    print(f"Total vehicles crossed IN:  {line_zone.in_count}")
    print(f"Total vehicles crossed OUT: {line_zone.out_count}")
    print(f"Captured vehicle crops saved to: '{CAPTURES_DIR}/'")
    print(f"Unique vehicle tracks with plates: {len(detected_plates_history)}")
    print("--------------------------------------------------")
    for track_key, info in detected_plates_history.items():
        print(f" • {track_key} ({info['class']}): {info['plate']} (conf: {
              info['confidence']:.2%}, frame {info['frame']})")
    print(f"All number plate detections saved to: '{PLATES_CSV_PATH}'")
    print("==================================================\n")


if __name__ == "__main__":
    main()
