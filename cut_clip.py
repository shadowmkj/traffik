"""
Standalone CLI tool to cut video segments using FFmpeg and record them in clips.txt.

Usage:
  # Cut a segment from a long video (auto-names clip_11.mp4 and updates clips.txt)
  python cut_clip.py full_recording.mp4 --start 12:30 --end 14:45

  # Cut with custom output name and accurate re-encoding
  python cut_clip.py full_recording.mp4 -s 01:15:00 -e 01:20:00 -o custom_clip.mp4 --accurate
"""

import argparse
import sys
from typing import List, Optional

from traffik.utils.video_cutter import cut_video_segment


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for cutting video segments."""
    parser = argparse.ArgumentParser(
        prog="cut_clip",
        description="Cut video segments using FFmpeg and record entries in clips.txt.",
    )
    parser.add_argument(
        "source",
        help="Path to source video file (e.g. traffic_long.mp4).",
    )
    parser.add_argument(
        "-s",
        "--start",
        required=True,
        help="Start timestamp (MM:SS, HH:MM:SS, or seconds).",
    )
    parser.add_argument(
        "-e",
        "--end",
        required=True,
        help="End timestamp (MM:SS, HH:MM:SS, or seconds).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output clip filename (e.g. clip_11.mp4). Default: next sequential name from clips.txt.",
    )
    parser.add_argument(
        "-d",
        "--dir",
        default=".",
        help="Directory to save cut clips (default: current directory).",
    )
    parser.add_argument(
        "-c",
        "--clips-file",
        default="clips.txt",
        help="Registry file to append the clip record to (default: clips.txt).",
    )
    parser.add_argument(
        "--accurate",
        action="store_true",
        help="Re-encode with libx264 for exact frame accuracy instead of fast stream-copy.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entry point for cutting a video segment."""
    args = parse_args(argv)
    try:
        out_path = cut_video_segment(
            source_video_path=args.source,
            start=args.start,
            end=args.end,
            output_name=args.output,
            output_dir=args.dir,
            clips_file=args.clips_file,
            accurate=args.accurate,
        )
        print(f"\n[Success] Clip created at: {out_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
