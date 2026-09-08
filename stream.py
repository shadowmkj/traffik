import os
import argparse
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
# 2. CLI Arguments Parser
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream and track vehicles in video with real-time playback."
    )
    # Positional argument for source video file
    parser.add_argument(
        "source",
        nargs="?",
        default="clip.mp4",
        help="Path to source video file (e.g. clip.mp4). Default: clip.mp4"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Optional custom output path. Defaults to <source_name>_out.<ext>"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable recording output video to file."
    )
    return parser.parse_args()


# ==============================================================================
# 3. Live Streaming & Processing Pipeline
# ==============================================================================

def main():
    args = parse_args()
    source_video_path = args.source

    if not os.path.exists(source_video_path):
        print(f"Error: Source video file '{source_video_path}' not found.")
        return

    # Derive output filename with '_out' appended to the input filename stem
    base, ext = os.path.splitext(source_video_path)
    if args.no_save:
        target_video_path = None
    elif args.output:
        target_video_path = args.output
    else:
        target_video_path = f"{base}_out{ext}"

    print(f"Source video: '{source_video_path}'")
    if target_video_path:
        print(f"Target video: '{target_video_path}'")
    else:
        print("Recording:    Disabled (--no-save)")
    print(f"Hardware Acceleration Device: {DEVICE.upper()}")
    print("Press 'q' or 'ESC' in the playback window to stop streaming early.\n")

    # Load YOLO detection model
    model_path = "yolo26s.pt" if os.path.exists("yolo26s.pt") else ("yolov8m.pt" if os.path.exists("yolov8m.pt") else "yolov8n.pt")
    model = YOLO(model_path)

    # COCO vehicle class IDs: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
    vehicle_class_ids = [2, 3, 5, 7]

    # Extract video metadata
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)

    # Define counting line coordinates
    start_y = min(964, int(video_info.height * 0.75))
    end_y = min(954, int(video_info.height * 0.75))
    start_x = min(30, int(video_info.width * 0.05))
    end_x = min(2643, int(video_info.width * 0.95))

    start = sv.Point(start_x, start_y)
    end = sv.Point(end_x, end_y)

    # Initialize tracking with tuned lost_track_buffer
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=video_info.fps
    )

    # Use multiple triggering anchors (BOTTOM_CENTER & CENTER)
    line_zone = sv.LineZone(
        start=start,
        end=end,
        triggering_anchors=[sv.Position.BOTTOM_CENTER, sv.Position.CENTER]
    )
    line_zone_annotator = sv.LineZoneAnnotator(
        thickness=2, text_thickness=1, text_scale=0.5
    )
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    window_name = f"Traffik Stream - {os.path.basename(source_video_path)}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Stream frames on-the-fly
    frames_generator = sv.get_video_frames_generator(source_path=source_video_path)

    sink = None
    if target_video_path:
        sink = sv.VideoSink(target_path=target_video_path, video_info=video_info)
        sink.__enter__()

    try:
        for frame_idx, frame in enumerate(frames_generator):
            # Run YOLO detection on current frame using GPU acceleration
            results = model(frame, conf=0.25, imgsz=1280, device=DEVICE, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)

            # Filter raw detections to vehicle classes (car, motorcycle, bus, truck) only
            detections = detections[np.isin(detections.class_id, vehicle_class_ids)]

            # Update ByteTrack tracker state
            detections = tracker.update_with_detections(detections)

            # Trigger line zone counter update
            line_zone.trigger(detections=detections)

            # Build label strings showing class and tracking ID
            labels = []
            if detections.tracker_id is not None:
                for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
                    class_name = model.names[int(class_id)] if hasattr(model, "names") else f"class_{class_id}"
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
    if target_video_path and os.path.exists(target_video_path):
        print(f"Saved output video to: '{target_video_path}'")
    print("=====================================================")


if __name__ == "__main__":
    main()
