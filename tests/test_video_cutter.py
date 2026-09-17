"""Unit tests for the video cutter and clips.txt pipeline."""

import os
from unittest.mock import MagicMock, patch
import pytest

from traffik.utils.video_cutter import (
    append_clip_record,
    build_ffmpeg_command,
    cut_video_segment,
    get_next_clip_name,
    parse_timestamp,
    validate_time_range,
)


# ==============================================================================
# Timestamp Parsing & Validation Tests
# ==============================================================================

def test_parse_timestamp_mm_ss():
    seconds, formatted = parse_timestamp("48:10")
    assert seconds == 2890.0
    assert formatted == "48:10"


def test_parse_timestamp_hh_mm_ss():
    seconds, formatted = parse_timestamp("01:15:30")
    assert seconds == 4530.0
    assert formatted == "01:15:30"


def test_parse_timestamp_numeric_seconds():
    seconds, formatted = parse_timestamp(125)
    assert seconds == 125.0
    assert formatted == "02:05"


def test_parse_timestamp_invalid():
    with pytest.raises(ValueError):
        parse_timestamp("invalid_time")
    with pytest.raises(ValueError):
        parse_timestamp("12:34:56:78")


def test_validate_time_range():
    # Valid range
    validate_time_range("02:00", "03:30")

    # Start equal or greater than end
    with pytest.raises(ValueError, match="Start time must be strictly before end time"):
        validate_time_range("05:00", "04:00")
    with pytest.raises(ValueError, match="Start time must be strictly before end time"):
        validate_time_range("05:00", "05:00")


# ==============================================================================
# Clips.txt Management & Auto-Naming Tests
# ==============================================================================

def test_get_next_clip_name_empty(tmp_path):
    empty_file = tmp_path / "empty_clips.txt"
    empty_file.write_text("")
    name = get_next_clip_name(str(empty_file))
    assert name == "clip_1.mp4"


def test_get_next_clip_name_existing(tmp_path):
    clips_file = tmp_path / "clips.txt"
    content = (
        "clip_1.mp4   |  48:10  |  50:40\n"
        "clip_2.mp4   |  52:57  |  54:40\n"
        "clip_10.mp4  |  04:00  |  05:00\n"
    )
    clips_file.write_text(content)
    name = get_next_clip_name(str(clips_file))
    assert name == "clip_11.mp4"


def test_append_clip_record(tmp_path):
    clips_file = tmp_path / "clips.txt"
    clips_file.write_text("clip_1.mp4   |  source_a.mp4     |  00:10  |  01:20\n")

    append_clip_record(
        clips_file_path=str(clips_file),
        clip_name="clip_2.mp4",
        start_str="02:00",
        end_str="03:30",
        source_name="traffic_master.mp4",
    )

    lines = clips_file.read_text().strip().splitlines()
    assert len(lines) == 2
    assert "clip_2.mp4" in lines[1]
    assert "traffic_master.mp4" in lines[1]
    assert "02:00" in lines[1]
    assert "03:30" in lines[1]


# ==============================================================================
# FFmpeg Command Construction Tests
# ==============================================================================

def test_build_ffmpeg_command_fast_copy():
    cmd = build_ffmpeg_command(
        source_path="master.mp4",
        start_str="02:00",
        end_str="03:30",
        output_path="clips/clip_1.mp4",
        accurate=False,
    )
    assert cmd[0] == "ffmpeg"
    assert "-ss" in cmd
    assert "02:00" in cmd
    assert "-to" in cmd
    assert "03:30" in cmd
    assert "-c" in cmd and "copy" in cmd
    assert cmd[-1] == "clips/clip_1.mp4"


def test_build_ffmpeg_command_accurate_reencode():
    cmd = build_ffmpeg_command(
        source_path="master.mp4",
        start_str="02:00",
        end_str="03:30",
        output_path="clips/clip_1.mp4",
        accurate=True,
    )
    assert "-c:v" in cmd
    assert "libx264" in cmd
    assert "-c:a" in cmd


# ==============================================================================
# Pipeline Execution Tests
# ==============================================================================

def test_cut_video_segment_nonexistent_source():
    with pytest.raises(FileNotFoundError):
        cut_video_segment(
            source_video_path="non_existent_source.mp4",
            start="01:00",
            end="02:00",
        )


@patch("subprocess.run")
def test_cut_video_segment_success(mock_run, tmp_path):
    # Create real dummy source file
    src_file = tmp_path / "source_video.mp4"
    src_file.write_text("dummy video binary")

    clips_file = tmp_path / "clips.txt"
    clips_file.write_text("clip_1.mp4   |  00:10  |  01:20\n")

    mock_run.return_value = MagicMock(returncode=0)

    out_path = cut_video_segment(
        source_video_path=str(src_file),
        start="02:00",
        end="03:30",
        output_name="clip_2.mp4",
        output_dir=str(tmp_path),
        clips_file=str(clips_file),
    )

    assert out_path == str(tmp_path / "clip_2.mp4")
    mock_run.assert_called_once()
    assert "clip_2.mp4" in clips_file.read_text()
    assert "source_video.mp4" in clips_file.read_text()
