import os
import argparse
from collections import defaultdict, Counter
import cv2
import numpy as np
import torch
from tqdm import tqdm
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
        description="Stream or batch process vehicle tracking video."
    )
    # Required positional argument for source video file
    parser.add_argument(
        "source",
        help="Path to source video file (e.g. clip.mp4)."
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Optional custom output path. Defaults to <source_name>_out.<ext>"
    )
    parser.add_argument(
        "--no-stream", "--headless",
        dest="no_stream",
        action="store_true",
        help="Disable live GUI display for maximum batch processing speed."
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving output video to file."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference resolution (e.g. 640 for speed, 1280 for higher accuracy). Default: 640"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="YOLO detection confidence threshold. Default: 0.20"
    )
    return parser.parse_args()


# ==============================================================================
# 3. Processing Pipeline
# ==============================================================================

def main():
    args = parse_args()
    source_video_path = args.source

    if not os.path.exists(source_video_path):
        print(f"\nError: Source video file '{source_video_path}' not found.\n")
        return

    # Derive output filename with '_out' appended to the input filename stem
    base, ext = os.path.splitext(source_video_path)
    if args.no_save:
        target_video_path = None
    elif args.output:
        target_video_path = args.output
    else:
        target_video_path = f"{base}_out{ext}"

    print("================ Configuration ================")
    print(f"Source video:     '{source_video_path}'")
    if target_video_path:
        print(f"Target video:     '{target_video_path}'")
    else:
        print("Target video:     Disabled (--no-save)")
    print(f"Live stream GUI:  {
          'Disabled (Fast Headless Mode)' if args.no_stream else 'Enabled (Press q/ESC to stop)'}")
    print(f"Inference device: {DEVICE.upper()
                               } (imgsz={args.imgsz}, conf={args.conf})")
    print("================================================\n")

    # Load YOLO detection model
    model_path = "yolo26s.pt" if os.path.exists("yolo26s.pt") else (
        "yolov8m.pt" if os.path.exists("yolov8m.pt") else "yolov8n.pt")
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

    # Initialize tracking with tuned lost_track_buffer (1.5s memory) and lower activation threshold
    tracker = sv.ByteTrack(
        track_activation_threshold=0.20,
        lost_track_buffer=45,
        minimum_matching_threshold=0.75,
        frame_rate=video_info.fps
    )

    # Track-level class memory for majority voting (prevents auto/car/truck flickering)
    track_class_history = defaultdict(Counter)

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

    # Only open OpenCV window when streaming is enabled
    window_name = f"Traffik Stream - {os.path.basename(source_video_path)}"
    if not args.no_stream:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frames_generator = sv.get_video_frames_generator(
        source_path=source_video_path)

    # Wrap with tqdm progress bar in headless mode for clean progress tracking
    if args.no_stream:
        total_frames = getattr(video_info, "total_frames", None)
        frame_iterable = enumerate(tqdm(
            frames_generator, total=total_frames, desc="Processing Video", unit="frame"))
    else:
        frame_iterable = enumerate(frames_generator)

    sink = None
    if target_video_path:
        sink = sv.VideoSink(target_path=target_video_path,
                            video_info=video_info)
        sink.__enter__()

    try:
        # Disable autograd overhead with torch.inference_mode for fast execution
        with torch.inference_mode():
            for frame_idx, frame in frame_iterable:
                # Fast GPU model inference
                results = model(
                    frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    device=DEVICE,
                    verbose=False
                )[0]
                detections = sv.Detections.from_ultralytics(results)

                # Filter raw detections to vehicle classes only
                detections = detections[np.isin(
                    detections.class_id, vehicle_class_ids)]

                # Update ByteTrack tracker state (spatial tracking independent of class fluctuations)
                detections = tracker.update_with_detections(detections)

                # Trigger line zone counter update (prioritizes vehicle crossing count)
                line_zone.trigger(detections=detections)

                # If saving video or showing GUI, perform annotations
                if sink or not args.no_stream:
                    labels = []
                    if detections.tracker_id is not None:
                        confidences = (
                            detections.confidence
                            if detections.confidence is not None
                            else [1.0] * len(detections)
                        )
                        for class_id, tracker_id, conf in zip(detections.class_id, detections.tracker_id, confidences):
                            raw_class = model.names[int(class_id)] if hasattr(
                                model, "names") else f"class_{class_id}"
                            # Accumulate confidence-weighted votes across frames
                            track_class_history[tracker_id][raw_class] += float(conf)
                            # Majority-voted smoothed class
                            smoothed_class = track_class_history[tracker_id].most_common(1)[0][0]
                            labels.append(f"{smoothed_class} #{tracker_id}")

                    # In-place annotation on frame avoids redundant allocations
                    annotated_frame = box_annotator.annotate(
                        scene=frame,
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

                    # Write frame to video sink
                    if sink:
                        sink.write_frame(annotated_frame)

                    # Handle live GUI streaming if enabled
                    if not args.no_stream:
                        cv2.imshow(window_name, annotated_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord('q')):
                            print(f"\nStreaming stopped early by user at frame {
                                  frame_idx}.")
                            break

    finally:
        if sink:
            sink.__exit__(None, None, None)
        if not args.no_stream:
            cv2.destroyAllWindows()

    print("\n================ Execution Summary ================")
    print(f"Total vehicles crossed IN:  {line_zone.in_count}")
    print(f"Total vehicles crossed OUT: {line_zone.out_count}")
    if target_video_path and os.path.exists(target_video_path):
        print(f"Saved output video to: '{target_video_path}'")
    print("=====================================================")


if __name__ == "__main__":
    main()
