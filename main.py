import os
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import supervision.detection.utils.internal
import supervision.detection.line_zone


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
# 2. Configuration & Initialization
# ==============================================================================

# Paths configuration
SOURCE_VIDEO_PATH = "traffic.mp4"
TARGET_VIDEO_PATH = "output.mp4"
CAPTURES_DIR = "captures"

# Load YOLOv8 detection model
model = YOLO('yolo26s.pt')

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
    results = model(frame)[0]
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
                filename = f"car_frame{
                    frame_idx:05d}_track{tracker_id}.jpg"
            else:
                filename = f"car_frame{frame_idx:05d}_det{idx}.jpg"

            save_path = os.path.join(CAPTURES_DIR, filename)
            cv2.imwrite(save_path, crop)

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

sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=process_frame
)

print(f"Total vehicles crossed IN: {line_zone.in_count}")
print(f"Total vehicles crossed OUT: {line_zone.out_count}")
print(f"Captured detected images saved to '{CAPTURES_DIR}/'")
