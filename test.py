import torch
from ultralytics import YOLO
import numpy as np
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


model = YOLO('yolov8n.pt')
SOURCE_VIDEO_PATH = "clip.mp4"
TARGET_VIDEO_PATH = "out.mp4"
video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)

START = sv.Point(30, 964)
END = sv.Point(2643, 954)

tracker = sv.ByteTrack()
line_zone = sv.LineZone(start=START, end=END)
line_zone_annotator = sv.LineZoneAnnotator(
    thickness=2, text_thickness=1, text_scale=0.5
)
box_annotator = sv.BoxAnnotator(thickness=2)


def process_frame(frame: np.ndarray, _) -> np.ndarray:
    results = model(frame, device=DEVICE)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    line_zone.trigger(detections=detections)
    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )
    annotated_frame = line_zone_annotator.annotate(
        annotated_frame,
        line_counter=line_zone
    )

    return annotated_frame


sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=process_frame
)

# print(f"Total vehicles crossed IN: {line_zone.in_count}")
# print(f"Total vehicles crossed OUT: {line_zone.out_count}")
