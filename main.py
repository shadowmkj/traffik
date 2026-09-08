"""Backward-compatible wrapper for vehicle tracking and license plate OCR batch processor."""

import argparse
import os
import sys
from typing import List, Optional

import numpy as np
import supervision as sv

from traffik.config import Config, get_device
from traffik.counting.gate import DualLineGate, GateState
from traffik.detection.detector import VehicleDetector
from traffik.ocr.plate_reader import PlateReader, clean_plate_text
from traffik.pipeline.engine import VideoPipeline
from traffik.tracking.tracker import VehicleTracker
from traffik.visualization.hud import VisualAnnotator, draw_hud_banner

# Backward compatibility constants and aliases
DEVICE = get_device("auto")
SOURCE_VIDEO_PATH = "clip_1.mp4"
TARGET_VIDEO_PATH = "output.mp4"
CAPTURES_DIR = "captures"
PLATES_CSV_PATH = "number_plates.csv"


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
    """Parse command line arguments for the batch processor."""
    parser = argparse.ArgumentParser(
        description="Traffik vehicle counting and OCR batch processor."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=SOURCE_VIDEO_PATH if os.path.exists(SOURCE_VIDEO_PATH) else ("traffic.mp4" if os.path.exists("traffic.mp4") else "clip.mp4"),
        help="Path to source video file (e.g. clip_1.mp4).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=TARGET_VIDEO_PATH,
        help="Optional custom output path. Default: output.mp4",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/default.toml",
        help="Path to TOML configuration file",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving output video to file.",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable license plate OCR extraction.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Execute batch processing with OCR delegating to VideoPipeline."""
    args = parse_args(argv)
    source_video_path = args.source

    if not os.path.exists(source_video_path):
        print(f"Error: Source video file '{source_video_path}' not found.")
        return

    config_path = args.config if (args.config and os.path.exists(args.config)) else "configs/default.toml"
    cfg = Config.from_toml(config_path) if os.path.exists(config_path) else Config()
    if not args.no_ocr:
        cfg.ocr.enabled = True

    target_path = None if args.no_save else args.output

    print(f"Starting pipeline on '{source_video_path}'...")
    if cfg.ocr.enabled:
        print(f"Number plates log file: '{cfg.ocr.output_csv}'\n")

    pipeline = VideoPipeline(cfg)
    summary = pipeline.run(
        source_path=source_video_path,
        target_path=target_path,
        stream=False,
    )

    print("\n================ Pipeline Summary ================")
    print(f"Total vehicles crossed IN:  {summary.in_count}")
    print(f"Total vehicles crossed OUT: {summary.out_count}")
    print(f"Captured vehicle crops saved to: '{cfg.general.captures_dir}/'")
    print(f"Total unique tracks: {summary.unique_tracks}")
    print(f"Processed frames: {summary.total_frames} ({summary.fps:.1f} FPS)")
    if summary.target and os.path.exists(summary.target):
        print(f"Output video saved to: '{summary.target}'")
    print("==================================================\n")


if __name__ == "__main__":
    main()
