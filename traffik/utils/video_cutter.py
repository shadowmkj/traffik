"""
Video Segment Cutter and clips.txt Recording Module.

This module provides utilities to cut video segments using FFmpeg and automatically
record them into `clips.txt` in the standard formatted schema:
  <clip_name.mp4>   |  <start_time>  |  <end_time>

Features:
  - Fast stream-copy cutting (`-c copy`) for zero quality loss and near-instant processing.
  - Accurate frame-level re-encoding (`--accurate`) when cutting at exact intra-frame boundaries.
  - Automatic sequential naming (`clip_1.mp4`, `clip_2.mp4`, ...) based on existing records in `clips.txt`.
  - Flexible timestamp parsing (`MM:SS`, `HH:MM:SS`, or raw float/integer seconds).
"""

import os
import re
import subprocess
from typing import List, Optional, Tuple, Union


# ==============================================================================
# Timestamp Parsing & Validation
# ==============================================================================

def parse_timestamp(ts: Union[str, int, float]) -> Tuple[float, str]:
    """Parse a timestamp into total seconds and a clean string representation (MM:SS or HH:MM:SS).

    Supported formats:
      - "48:10" -> (2890.0, "48:10")
      - "01:15:30" -> (4530.0, "01:15:30")
      - 125 -> (125.0, "02:05")
    """
    if isinstance(ts, (int, float)):
        total_seconds = float(ts)
        if total_seconds < 0:
            raise ValueError(f"Timestamp cannot be negative: {ts}")
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        if hours > 0:
            formatted = f"{hours:02d}:{minutes:02d}:{seconds:05.2f}".rstrip("0").rstrip(".")
        else:
            if seconds.is_integer():
                formatted = f"{minutes:02d}:{int(seconds):02d}"
            else:
                formatted = f"{minutes:02d}:{seconds:05.2f}".rstrip("0").rstrip(".")
        return total_seconds, formatted

    ts_str = str(ts).strip()
    parts = ts_str.split(":")
    if len(parts) == 2:
        # MM:SS or MM:SS.sss
        try:
            m = int(parts[0])
            s = float(parts[1])
            if m < 0 or s < 0 or s >= 60:
                raise ValueError
            total_seconds = m * 60.0 + s
            # Format cleanly as MM:SS if integer seconds
            if s.is_integer():
                formatted = f"{m:02d}:{int(s):02d}"
            else:
                formatted = f"{m:02d}:{s:05.2f}"
            return total_seconds, formatted
        except ValueError:
            raise ValueError(f"Invalid MM:SS timestamp format: '{ts_str}'")

    elif len(parts) == 3:
        # HH:MM:SS or HH:MM:SS.sss
        try:
            h = int(parts[0])
            m = int(parts[1])
            s = float(parts[2])
            if h < 0 or m < 0 or m >= 60 or s < 0 or s >= 60:
                raise ValueError
            total_seconds = h * 3600.0 + m * 60.0 + s
            if s.is_integer():
                formatted = f"{h:02d}:{m:02d}:{int(s):02d}"
            else:
                formatted = f"{h:02d}:{m:02d}:{s:05.2f}"
            return total_seconds, formatted
        except ValueError:
            raise ValueError(f"Invalid HH:MM:SS timestamp format: '{ts_str}'")

    elif len(parts) == 1:
        try:
            sec = float(parts[0])
            return parse_timestamp(sec)
        except ValueError:
            raise ValueError(f"Invalid numeric timestamp format: '{ts_str}'")

    raise ValueError(f"Unrecognized timestamp format: '{ts_str}' (expected MM:SS or HH:MM:SS)")


def validate_time_range(
    start: Union[str, int, float],
    end: Union[str, int, float],
) -> Tuple[float, float, str, str]:
    """Validate that start time is strictly before end time.

    Returns (start_sec, end_sec, start_formatted, end_formatted).
    """
    s_sec, s_str = parse_timestamp(start)
    e_sec, e_str = parse_timestamp(end)

    if s_sec >= e_sec:
        raise ValueError(
            f"Start time must be strictly before end time. Got start={s_str} ({s_sec}s), end={e_str} ({e_sec}s)"
        )

    return s_sec, e_sec, s_str, e_str


# ==============================================================================
# clips.txt Registry & Auto-Naming
# ==============================================================================

def get_next_clip_name(clips_file_path: str = "clips.txt") -> str:
    """Determine the next sequential clip filename (e.g. clip_11.mp4) from clips.txt."""
    if not os.path.exists(clips_file_path):
        return "clip_1.mp4"

    max_num = 0
    pattern = re.compile(r"clip_(\d+)\.mp4", re.IGNORECASE)

    with open(clips_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = pattern.search(line)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num

    return f"clip_{max_num + 1}.mp4"


def append_clip_record(
    clips_file_path: str,
    clip_name: str,
    start_str: str,
    end_str: str,
    source_name: Optional[str] = None,
) -> None:
    """Append a newly cut clip entry to clips.txt with standard column alignment.

    Format:
      <clip_name>   |  <source_footage>   |  <start_time>  |  <end_time>
    """
    # Ensure directory exists if path contains directories
    os.makedirs(os.path.dirname(clips_file_path) if os.path.dirname(clips_file_path) else ".", exist_ok=True)

    if source_name:
        formatted_line = f"{clip_name:<12} |  {source_name:<16} |  {start_str:<5}  |  {end_str}\n"
    else:
        formatted_line = f"{clip_name:<12} |  {start_str:<5}  |  {end_str}\n"

    # Check if existing file has trailing newline
    has_trailing_newline = True
    if os.path.exists(clips_file_path) and os.path.getsize(clips_file_path) > 0:
        with open(clips_file_path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            has_trailing_newline = f.read(1) == b"\n"

    with open(clips_file_path, "a", encoding="utf-8") as f:
        if not has_trailing_newline:
            f.write("\n")
        f.write(formatted_line)


# ==============================================================================
# FFmpeg Command Builder & Execution
# ==============================================================================

def build_ffmpeg_command(
    source_path: str,
    start_str: str,
    end_str: str,
    output_path: str,
    accurate: bool = False,
) -> List[str]:
    """Construct FFmpeg CLI argument list for cutting a video segment."""
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_str),
        "-to",
        str(end_str),
        "-i",
        str(source_path),
    ]

    if accurate:
        # Re-encode for frame-accurate cuts at non-keyframe timestamps
        cmd.extend([
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "fast",
            "-c:a",
            "aac",
        ])
    else:
        # Stream copy: ultra fast, zero loss
        cmd.extend([
            "-c",
            "copy",
        ])

    cmd.append(str(output_path))
    return cmd


def cut_video_segment(
    source_video_path: str,
    start: Union[str, int, float],
    end: Union[str, int, float],
    output_name: Optional[str] = None,
    output_dir: str = ".",
    clips_file: str = "clips.txt",
    accurate: bool = False,
) -> str:
    """Cut a segment from source_video_path using FFmpeg and record it in clips_file.

    Parameters
    ----------
    source_video_path : str
        Path to input video file.
    start : str or numeric
        Start timestamp (e.g. '48:10', '01:15:00', or 120).
    end : str or numeric
        End timestamp (e.g. '50:40', '01:17:30', or 180).
    output_name : str, optional
        Name of output clip (default: auto-incremented clip_N.mp4 from clips.txt).
    output_dir : str
        Directory to save the cut clip (default: current directory).
    clips_file : str
        Path to clips record registry (default: clips.txt).
    accurate : bool
        If True, re-encodes for frame-accurate cuts; otherwise uses fast stream copy.

    Returns
    -------
    str
        Full absolute/relative path of the generated clip.
    """
    if not os.path.exists(source_video_path):
        raise FileNotFoundError(f"Source video file not found: '{source_video_path}'")

    s_sec, e_sec, s_str, e_str = validate_time_range(start, end)

    # Determine clip name if not provided
    if not output_name:
        output_name = get_next_clip_name(clips_file)
    elif not output_name.lower().endswith(".mp4"):
        output_name = f"{output_name}.mp4"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)

    # Build and execute FFmpeg command
    cmd = build_ffmpeg_command(
        source_path=source_video_path,
        start_str=s_str,
        end_str=e_str,
        output_path=output_path,
        accurate=accurate,
    )

    print(f"[VideoCutter] Cutting '{source_video_path}' [{s_str} -> {e_str}] to '{output_path}'...")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed with exit code {result.returncode}:\n{result.stderr}"
        )

    # Record in clips.txt with source footage name
    source_name = os.path.basename(source_video_path)
    append_clip_record(
        clips_file_path=clips_file,
        clip_name=output_name,
        start_str=s_str,
        end_str=e_str,
        source_name=source_name,
    )
    print(f"[VideoCutter] Recorded '{output_name} | {source_name} | {s_str} | {e_str}' in '{clips_file}'.")

    return output_path

