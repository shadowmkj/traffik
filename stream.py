import os
import argparse
from collections import defaultdict, Counter
from enum import Enum
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
# 2. Dual-Line Virtual Gate Architecture (Approach 1)
# ==============================================================================

class GateState(Enum):
    OUTSIDE = 0
    ENTERED_A = 1  # Crossed Line A first (moving towards Line B)
    ENTERED_B = 2  # Crossed Line B first (moving towards Line A)
    COUNTED = 3


class DualLineGate:
    """
    Two-line virtual gate (Line A & Line B) with a spatial buffer zone.
    Immune to single-frame detection drops or flickering track IDs.
    - Counts IN:  Crossed Line A -> then Line B
    - Counts OUT: Crossed Line B -> then Line A
    """
    def __init__(
        self,
        line_a_start: sv.Point,
        line_a_end: sv.Point,
        line_b_start: sv.Point,
        line_b_end: sv.Point,
        triggering_anchors=None
    ):
        self.line_a = sv.LineZone(
            start=line_a_start,
            end=line_a_end,
            triggering_anchors=triggering_anchors
        )
        self.line_b = sv.LineZone(
            start=line_b_start,
            end=line_b_end,
            triggering_anchors=triggering_anchors
        )
        self.track_states = defaultdict(lambda: GateState.OUTSIDE)
        self.in_count = 0
        self.out_count = 0

    def trigger(self, detections: sv.Detections):
        if detections.tracker_id is None or len(detections) == 0:
            return

        crossed_a_in, crossed_a_out = self.line_a.trigger(detections)
        crossed_b_in, crossed_b_out = self.line_b.trigger(detections)

        for idx, tracker_id in enumerate(detections.tracker_id):
            state = self.track_states[tracker_id]
            if state == GateState.COUNTED:
                continue

            # State transitions across the virtual gate
            if state == GateState.OUTSIDE:
                if crossed_a_in[idx]:
                    self.track_states[tracker_id] = GateState.ENTERED_A
                elif crossed_b_out[idx]:
                    self.track_states[tracker_id] = GateState.ENTERED_B
                elif crossed_b_in[idx]:
                    # Fallback for tracks starting inside gate moving IN
                    self.in_count += 1
                    self.track_states[tracker_id] = GateState.COUNTED
                elif crossed_a_out[idx]:
                    # Fallback for tracks starting inside gate moving OUT
                    self.out_count += 1
                    self.track_states[tracker_id] = GateState.COUNTED

            elif state == GateState.ENTERED_A:
                if crossed_b_in[idx] or crossed_b_out[idx]:
                    self.in_count += 1
                    self.track_states[tracker_id] = GateState.COUNTED

            elif state == GateState.ENTERED_B:
                if crossed_a_in[idx] or crossed_a_out[idx]:
                    self.out_count += 1
                    self.track_states[tracker_id] = GateState.COUNTED


def draw_counter_hud(frame: np.ndarray, in_count: int, out_count: int) -> np.ndarray:
    """Draw a clean counter HUD overlay at top of the frame."""
    banner_text = f"GATE COUNT | IN: {in_count}   OUT: {out_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(banner_text, font, font_scale, thickness)

    # Semi-transparent background box
    x1, y1 = 20, 20
    x2, y2 = x1 + text_w + 24, y1 + text_h + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 120), 2)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(
        frame,
        banner_text,
        (x1 + 12, y1 + text_h + 8),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )
    return frame


# ==============================================================================
# 3. CLI Arguments Parser
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream or batch process vehicle tracking video using Dual-Line Gate."
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
    parser.add_argument(
        "--gate-offset",
        type=int,
        default=60,
        help="Distance in pixels between Entry Line A and Exit Line B. Default: 60"
    )
    return parser.parse_args()


# ==============================================================================
# 4. Processing Pipeline
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
    print(f"Counting Mode:    Dual-Line Gate (Offset: {args.gate_offset}px)")
    print(f"Live stream GUI:  {'Disabled (Fast Headless Mode)' if args.no_stream else 'Enabled (Press q/ESC to stop)'}")
    print(f"Inference device: {DEVICE.upper()} (imgsz={args.imgsz}, conf={args.conf})")
    print("================================================\n")

    # Load YOLO detection model
    model_path = "yolo26s.pt" if os.path.exists("yolo26s.pt") else (
        "yolov8m.pt" if os.path.exists("yolov8m.pt") else "yolov8n.pt")
    model = YOLO(model_path)

    # COCO vehicle class IDs: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
    vehicle_class_ids = [2, 3, 5, 7]

    # Extract video metadata
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)

    # Custom Dual-Line Gate coordinates (calibrated for the road lanes)
    LINE_A_START = sv.Point(49, 1287)
    LINE_A_END   = sv.Point(1881, 816)
    LINE_B_START = sv.Point(101, 1458)
    LINE_B_END   = sv.Point(2381, 797)

    # Initialize Dual-Line Gate with your calibrated lines
    gate = DualLineGate(
        line_a_start=LINE_A_START,
        line_a_end=LINE_A_END,
        line_b_start=LINE_B_START,
        line_b_end=LINE_B_END,
        triggering_anchors=[sv.Position.BOTTOM_CENTER, sv.Position.CENTER]
    )

    # Initialize tracking with tuned lost_track_buffer (1.5s memory)
    tracker = sv.ByteTrack(
        track_activation_threshold=0.20,
        lost_track_buffer=45,
        minimum_matching_threshold=0.75,
        frame_rate=video_info.fps
    )

    # Track-level class memory for majority voting
    track_class_history = defaultdict(Counter)

    # Annotators
    line_a_annotator = sv.LineZoneAnnotator(
        thickness=2,
        color=sv.Color(r=0, g=220, b=255),   # Cyan for Line A
        text_thickness=1,
        text_scale=0.5
    )
    line_b_annotator = sv.LineZoneAnnotator(
        thickness=2,
        color=sv.Color(r=255, g=140, b=0),   # Orange for Line B
        text_thickness=1,
        text_scale=0.5
    )
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    # Open OpenCV window when streaming is enabled
    window_name = f"Traffik Stream - {os.path.basename(source_video_path)}"
    if not args.no_stream:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frames_generator = sv.get_video_frames_generator(
        source_path=source_video_path)

    # Wrap with tqdm progress bar in headless mode
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
        # Disable autograd overhead with torch.inference_mode
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

                # Update ByteTrack tracker state (spatial tracking)
                detections = tracker.update_with_detections(detections)

                # Trigger Dual-Line Gate counter update
                gate.trigger(detections=detections)

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
                            # Accumulate confidence-weighted votes
                            track_class_history[tracker_id][raw_class] += float(conf)
                            # Majority-voted smoothed class
                            smoothed_class = track_class_history[tracker_id].most_common(1)[0][0]
                            labels.append(f"{smoothed_class} #{tracker_id}")

                    # In-place box and label annotation
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

                    # Annotate Line A & Line B
                    annotated_frame = line_a_annotator.annotate(
                        annotated_frame,
                        line_counter=gate.line_a
                    )
                    annotated_frame = line_b_annotator.annotate(
                        annotated_frame,
                        line_counter=gate.line_b
                    )

                    # Overlay Combined Counter HUD
                    annotated_frame = draw_counter_hud(
                        annotated_frame,
                        gate.in_count,
                        gate.out_count
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

    print("\n================ Gate Counting Summary ================")
    print(f"Total vehicles crossed IN:  {gate.in_count}")
    print(f"Total vehicles crossed OUT: {gate.out_count}")
    print(f"Total Unique Vehicles Tracked: {len(track_class_history)}")
    if target_video_path and os.path.exists(target_video_path):
        print(f"Saved output video to: '{target_video_path}'")
    print("========================================================\n")


if __name__ == "__main__":
    main()
