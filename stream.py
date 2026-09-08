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
        default=640,
        help="YOLO inference resolution (e.g. 640 for speed, 1280 for higher accuracy). Default: 640",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="YOLO detection confidence threshold. Default: 0.20",
    )
    parser.add_argument(
        "--gate-offset",
        type=int,
        default=60,
        help="Distance in pixels between Entry Line A and Exit Line B. Default: 60",
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

    # Apply CLI argument overrides to config
    cfg.detector.imgsz = args.imgsz
    cfg.detector.conf_threshold = args.conf
    cfg.gate.offset = args.gate_offset

    is_stream = not args.no_stream

    print("================ Configuration ================")
    print(f"Source video:     '{source_video_path}'")
    if target_video_path:
        print(f"Target video:     '{target_video_path}'")
    else:
        print("Target video:     Disabled (--no-save)")
    print(f"Counting Mode:    Dual-Line Gate (Offset: {args.gate_offset}px)")
    print(f"Live stream GUI:  {'Enabled (Press q/ESC to stop)' if is_stream else 'Disabled (Fast Headless Mode)'}")
    print(f"Inference device: {cfg.general.device.upper()} (imgsz={args.imgsz}, conf={args.conf})")
    print("================================================\n")

    pipeline = VideoPipeline(cfg)
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
