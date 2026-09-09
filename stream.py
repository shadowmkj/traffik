"""Backward-compatible wrapper for streaming and batch vehicle tracking with Dual-Line Gate."""

import argparse
import os
import sys
from typing import List, Optional

import numpy as np

from traffik.config import Config, get_device
from traffik.counting.gate import DualLineGate, GateState
from traffik.pipeline.engine import VideoPipeline
from traffik.visualization.hud import draw_hud_banner as draw_counter_hud

# Expose device detection for backward compatibility
DEVICE = get_device("auto")


def _patched_cross_product(anchors, vector):
    """Compatibility function preserved for backward compatibility."""
    vector_at_zero = np.array([
        vector.end.x - vector.start.x,
        vector.end.y - vector.start.y,
    ])
    vector_start = np.array([vector.start.x, vector.start.y])
    diff = anchors - vector_start
    return vector_at_zero[0] * diff[..., 1] - vector_at_zero[1] * diff[..., 0]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments preserving original CLI expectations."""
    parser = argparse.ArgumentParser(
        description="Stream or batch process vehicle tracking video using Dual-Line Gate."
    )
    parser.add_argument(
        "source",
        help="Path to source video file (e.g. clip.mp4).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional custom output path. Defaults to <source_name>_out.<ext>",
    )
    parser.add_argument(
        "--no-stream",
        "--headless",
        dest="no_stream",
        action="store_true",
        help="Disable live GUI display for maximum batch processing speed.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving output video to file.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Override YOLO inference resolution (e.g. 640 or 1280). Default: from config",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Override YOLO detection confidence threshold. Default: from config",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Override YOLO model weights path (e.g. yolo11n.pt, yolov8m.pt). Default: from config",
    )
    parser.add_argument(
        "--gate-offset",
        type=int,
        default=None,
        help="Override distance in pixels between Line A and Line B. Default: from config",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/default.toml",
        help="Path to TOML configuration file. Default: configs/default.toml",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Execute video stream / batch processing delegating to VideoPipeline."""
    args = parse_args(argv)
    source_video_path = args.source

    if not os.path.exists(source_video_path):
        print(f"\nError: Source video file '{source_video_path}' not found.\n")
        return

    base, ext = os.path.splitext(source_video_path)
    if args.no_save:
        target_video_path = None
    elif args.output:
        target_video_path = args.output
    else:
        target_video_path = f"{base}_out{ext}"

    config_path = args.config if (args.config and os.path.exists(args.config)) else "configs/default.toml"
    cfg = Config.from_toml(config_path) if os.path.exists(config_path) else Config()

    # Apply explicit CLI argument overrides to config if provided
    if args.imgsz is not None:
        cfg.detector.imgsz = args.imgsz
    if args.conf is not None:
        cfg.detector.conf_threshold = args.conf
    if args.model is not None:
        cfg.detector.model_path = args.model
    if args.gate_offset is not None:
        cfg.gate.offset = args.gate_offset

    is_stream = not args.no_stream
    pipeline = VideoPipeline(cfg)

    print("================ Configuration ================")
    print(f"Config File:      '{config_path}'")
    print(f"Source Video:     '{source_video_path}'")
    print(f"Target Video:     {repr(target_video_path) if target_video_path else 'Disabled (--no-save)'}")
    print(f"Model Weights:    '{pipeline.detector.model_path}'")
    print(f"Compute Device:   {pipeline.detector.device.upper()} (imgsz={cfg.detector.imgsz}, conf={cfg.detector.conf_threshold})")
    print(f"Gate Line A:      {cfg.gate.line_a_start} -> {cfg.gate.line_a_end}")
    print(f"Gate Line B:      {cfg.gate.line_b_start} -> {cfg.gate.line_b_end}")
    print(f"Live Stream GUI:  {'Enabled (Press q/ESC to stop)' if is_stream else 'Disabled (Fast Headless Mode)'}")
    print("================================================\n")

    summary = pipeline.run(
        source_path=source_video_path,
        target_path=target_video_path,
        stream=is_stream,
    )

    print("\n================ Gate Counting Summary ================")
    print(f"Total vehicles crossed IN:  {summary.in_count}")
    print(f"Total vehicles crossed OUT: {summary.out_count}")
    print(f"Total Unique Vehicles Tracked: {summary.unique_tracks}")
    if summary.target and os.path.exists(summary.target):
        print(f"Saved output video to: '{summary.target}'")
    print("========================================================\n")


if __name__ == "__main__":
    main()
