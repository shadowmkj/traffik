import os
import cv2
import numpy as np
import torch
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
# 1. Compatibility Patch (Supervision + NumPy 2.0+)
# ==============================================================================

# Fix for Supervision cross product calculation when using NumPy 2.0+ 2D arrays
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

SOURCE_VIDEO_PATH = "short.mp4"
# Optional: Set to None if you only want live viewing without saving output video
TARGET_VIDEO_PATH = "stream_out.mp4"

# COCO vehicle class IDs: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
VEHICLE_CLASS_IDS = [2, 3, 5, 7]

# Load higher-accuracy model (yolo26s.pt or yolov8m.pt produce much fewer false negatives than yolov8n.pt)
MODEL_PATH = "yolov8n.pt" if os.path.exists("yolov8n.pt") else "yolov8n.pt"
model = YOLO(MODEL_PATH)
video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)

# Define counting line coordinates
START = sv.Point(30, 964)
END = sv.Point(2643, 954)

# Initialize tracking with tuned lost_track_buffer to prevent track ID drops during brief occlusions
tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=video_info.fps
)

# Use multiple triggering anchors (BOTTOM_CENTER & CENTER) to catch fast or partially cropped vehicles
line_zone = sv.LineZone(
    start=START,
    end=END,
    triggering_anchors=[sv.Position.BOTTOM_CENTER, sv.Position.CENTER]
)
line_zone_annotator = sv.LineZoneAnnotator(
    thickness=2, text_thickness=1, text_scale=0.5
)
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)


# ==============================================================================
# 3. Live Streaming & Processing Loop
# ==============================================================================

def main():
    print(f"Starting live video stream for '{SOURCE_VIDEO_PATH}'...")
    print(f"Hardware Acceleration Device: {DEVICE.upper()}")
    print("Press 'q' or 'ESC' in the window to stop streaming early.\n")

    window_name = "Traffik Live Tracking Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Use sv.get_video_frames_generator to stream frames on-the-fly
    frames_generator = sv.get_video_frames_generator(
        source_path=SOURCE_VIDEO_PATH)

    # Optional video recorder context if TARGET_VIDEO_PATH is specified
    if TARGET_VIDEO_PATH:
        sink = sv.VideoSink(target_path=TARGET_VIDEO_PATH,
                            video_info=video_info)
        sink.__enter__()
    else:
        sink = None

    try:
        for frame_idx, frame in enumerate(frames_generator):
            # Run YOLO detection on current frame using Metal (MPS) GPU acceleration
            results = model(frame, conf=0.25, imgsz=1280,
                            device=DEVICE, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)

            # Filter raw detections to vehicle classes (car, motorcycle, bus, truck) only
            detections = detections[np.isin(
                detections.class_id, VEHICLE_CLASS_IDS)]

            # Update ByteTrack tracker state
            detections = tracker.update_with_detections(detections)

            # Trigger line zone counter update
            line_zone.trigger(detections=detections)

            # Build label strings showing class and tracking ID
            labels = []
            if detections.tracker_id is not None:
                for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
                    class_name = model.names[int(class_id)] if hasattr(
                        model, "names") else f"class_{class_id}"
                    labels.append(f"{class_name} #{tracker_id}")

            # Annotate frame with boxes, labels, and line zone counter stats
            annotated_frame = box_annotator.annotate(
                scene=frame.copy(),
                detections=detections
            )
            if labels:
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections,
                    labels=labels
                )
            annotated_frame = line_zone_annotator.annotate(
                annotated_frame,
                line_counter=line_zone
            )

            # Stream immediately to screen
            cv2.imshow(window_name, annotated_frame)

            # Write frame to output video if sink is enabled
            if sink:
                sink.write_frame(annotated_frame)

            # Wait 1ms for keypress to allow real-time display and quit control
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                print(f"Streaming stopped early by user at frame {frame_idx}.")
                break

    finally:
        if sink:
            sink.__exit__(None, None, None)
        cv2.destroyAllWindows()

    print("\n================ Streaming Summary ================")
    print(f"Total vehicles crossed IN:  {line_zone.in_count}")
    print(f"Total vehicles crossed OUT: {line_zone.out_count}")
    print("=====================================================")


if __name__ == "__main__":
    main()
