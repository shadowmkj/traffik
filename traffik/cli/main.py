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
        "-m", "--model", default=None, help="Override YOLO model weights path (e.g. yolo11n.pt, yolov8m.pt)"
    )
    p_proc.add_argument(
        "--ocr", action="store_true", help="Enable OCR license plate extraction"
    )
    p_proc.add_argument(
        "--speed", action="store_true", help="Enable vehicle speed estimation"
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
        "-m", "--model", default=None, help="Override YOLO model weights path (e.g. yolo11n.pt, yolov8m.pt)"
    )
    p_stream.add_argument(
        "--speed", action="store_true", help="Enable vehicle speed estimation"
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

    # Setup speed ROI subcommand (Interactive perspective calibration)
    p_speed_roi = subparsers.add_parser(
        "setup-speed-roi",
        help="Launch interactive 4-point speed ROI calibration tool",
    )
    p_speed_roi.add_argument("source", help="Path to input video file")
    p_speed_roi.add_argument(
        "-w", "--width", type=float, default=7.5, help="Road width in meters. Default: 7.5"
    )
    p_speed_roi.add_argument(
        "-l", "--length", type=float, default=25.0, help="Road section length in meters. Default: 25.0"
    )
    p_speed_roi.add_argument(
        "-u", "--unit", choices=["km/h", "mph"], default="km/h", help="Speed unit (km/h or mph). Default: km/h"
    )

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

    if args.command == "setup-speed-roi":
        from traffik.speed.calibration import run_setup_speed_roi
        run_setup_speed_roi(
            args.source,
            target_width=args.width,
            target_length=args.length,
            unit=args.unit,
        )
        return

    config_path = args.config if (args.config and os.path.exists(args.config)) else "configs/default.toml"
    cfg = Config.from_toml(config_path) if os.path.exists(config_path) else Config()

    if getattr(args, "model", None):
        cfg.detector.model_path = args.model

    if getattr(args, "ocr", False):
        cfg.ocr.enabled = True

    if getattr(args, "speed", False):
        cfg.speed.enabled = True

    base, ext = os.path.splitext(args.source)
    if args.no_save:
        target_path = None
    elif args.output:
        target_path = args.output
    else:
        target_path = f"{base}_out{ext}"

    pipeline = VideoPipeline(cfg)
    is_stream = args.command == "stream"

    speed_info = (
        f"Enabled (unit={cfg.speed.unit}, polygon={cfg.speed.source_polygon})"
        if cfg.speed.enabled
        else "Disabled"
    )

    print("================ Configuration ================")
    print(f"Config File:      '{config_path}'")
    print(f"Source Video:     '{args.source}'")
    print(f"Target Video:     {repr(target_path) if target_path else 'Disabled (--no-save)'}")
    print(f"Model Weights:    '{pipeline.detector.model_path}'")
    print(f"Compute Device:   {pipeline.detector.device.upper()} (imgsz={cfg.detector.imgsz}, conf={cfg.detector.conf_threshold})")
    print(f"Gate Line A:      {cfg.gate.line_a_start} -> {cfg.gate.line_a_end}")
    print(f"Gate Line B:      {cfg.gate.line_b_start} -> {cfg.gate.line_b_end}")
    print(f"License Plate OCR:{'Enabled' if cfg.ocr.enabled else 'Disabled'}")
    print(f"Speed Estimation: {speed_info}")
    print(f"Mode:             {'Interactive Stream GUI' if is_stream else 'Headless Batch Processing'}")
    print("================================================\n")

    summary = pipeline.run(
        source_path=args.source,
        target_path=target_path,
        stream=is_stream,
    )

    print("\n================ Execution Summary ================")
    print(f"Total Vehicles Crossed IN:  {summary.in_count}")
    print(f"Total Vehicles Crossed OUT: {summary.out_count}")
    print(f"Processed Frames:           {summary.total_frames} ({summary.fps:.1f} FPS)")
    if summary.run_number is not None:
        print(f"Logged to CSV:              '{cfg.general.outputs_csv}' (Run #{summary.run_number})")
    if summary.target and os.path.exists(summary.target):
        print(f"Output Video Saved:         '{summary.target}'")
    print("====================================================\n")


if __name__ == "__main__":
    main()
