"""Standalone CLI tool for interactive 4-point speed ROI calibration."""

import argparse
import os
from typing import List, Optional

from traffik.speed.calibration import run_setup_speed_roi


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for speed ROI calibration."""
    parser = argparse.ArgumentParser(
        description="Interactive 4-Point Speed Estimation ROI Calibration Tool."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="clip_4.mp4" if os.path.exists("clip_4.mp4") else ("traffic.mp4" if os.path.exists("traffic.mp4") else "clip.mp4"),
        help="Path to source video file (default: clip_4.mp4 / traffic.mp4 / clip.mp4).",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=float,
        default=7.5,
        help="Real-world width of ROI in meters across the road (default: 7.5).",
    )
    parser.add_argument(
        "-l",
        "--length",
        type=float,
        default=25.0,
        help="Real-world length of ROI in meters along the road (default: 25.0).",
    )
    parser.add_argument(
        "-u",
        "--unit",
        type=str,
        default="km/h",
        choices=["km/h", "mph"],
        help="Speed measurement unit (default: km/h).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Main entrypoint for standalone speed ROI calibration script."""
    args = parse_args(argv)
    run_setup_speed_roi(
        source_video_path=args.source,
        target_width=args.width,
        target_length=args.length,
        unit=args.unit,
    )


if __name__ == "__main__":
    main()
