"""Traffik CLI - Unified Command-Line Interface for vehicle analytics, streaming, and calibration."""

import argparse
import os
import sys
from typing import List, Optional

from traffik.config import Config
from traffik.pipeline.engine import VideoPipeline


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the top-level argument parser for the Traffik CLI."""
    parser = argparse.ArgumentParser(
        prog="traffik",
        description="Traffik: Scalable Vehicle Analytics & Counting CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Process subcommand (Headless batch mode)
    p_proc = subparsers.add_parser(
        "process",
        help="Process video headlessly at maximum speed",
    )
    p_proc.add_argument("source", help="Path to input video file")
    p_proc.add_argument("-o", "--output", default=None, help="Output video path")
    p_proc.add_argument(
        "-c", "--config", default="configs/default.toml", help="Path to TOML config"
    )
    p_proc.add_argument(
        "--ocr", action="store_true", help="Enable OCR license plate extraction"
    )
    p_proc.add_argument(
        "--no-save", action="store_true", help="Do not write output video to disk"
    )

    # Stream subcommand (Interactive GUI playback)
    p_stream = subparsers.add_parser(
        "stream",
        help="Stream video with real-time OpenCV playback window",
    )
    p_stream.add_argument("source", help="Path to input video file")
    p_stream.add_argument("-o", "--output", default=None, help="Output video path")
    p_stream.add_argument(
        "-c", "--config", default="configs/default.toml", help="Path to TOML config"
    )
    p_stream.add_argument(
        "--no-save", action="store_true", help="Do not write output video to disk"
    )

    # Setup gate subcommand (Interactive calibration)
    p_gate = subparsers.add_parser(
        "setup-gate",
        help="Launch interactive 4-point gate calibration tool",
    )
    p_gate.add_argument("source", help="Path to input video file")

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entrypoint. Parses arguments and executes the appropriate pipeline or tool."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "setup-gate":
        try:
            from setup_line import run_setup_gate
            run_setup_gate(args.source)
        except ImportError:
            from setup_line import main as setup_main
            sys.argv = ["setup_line.py", args.source]
            setup_main()
        return

    config_path = args.config if (args.config and os.path.exists(args.config)) else "configs/default.toml"
    cfg = Config.from_toml(config_path) if os.path.exists(config_path) else Config()

    if getattr(args, "ocr", False):
        cfg.ocr.enabled = True

    base, ext = os.path.splitext(args.source)
    if args.no_save:
        target_path = None
    elif args.output:
        target_path = args.output
    else:
        target_path = f"{base}_out{ext}"

    pipeline = VideoPipeline(cfg)
    is_stream = args.command == "stream"

    print(f"Traffik running '{args.command}' on '{args.source}'...")
    summary = pipeline.run(
        source_path=args.source,
        target_path=target_path,
        stream=is_stream,
    )

    print("\n================ Execution Summary ================")
    print(f"Total Vehicles Crossed IN:  {summary.in_count}")
    print(f"Total Vehicles Crossed OUT: {summary.out_count}")
    print(f"Processed Frames:           {summary.total_frames} ({summary.fps:.1f} FPS)")
    if summary.target and os.path.exists(summary.target):
        print(f"Output Video Saved:         '{summary.target}'")
    print("====================================================\n")


if __name__ == "__main__":
    main()
