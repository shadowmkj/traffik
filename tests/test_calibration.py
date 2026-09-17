"""Unit tests for interactive speed ROI calibration tool and TOML configuration formatting."""

import os
import tomllib
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from setup_speed_roi import main as cli_main, parse_args
from traffik.speed.calibration import format_speed_toml_block, run_setup_speed_roi
import traffik
import traffik.speed


# ==============================================================================
# CLI Argument Parsing Tests
# ==============================================================================

def test_setup_speed_roi_arg_parsing_default():
    """Test default CLI arguments with a specified source video."""
    args = parse_args(["test_video.mp4"])
    assert args.source == "test_video.mp4"
    assert args.width == 7.5
    assert args.length == 25.0
    assert args.unit == "km/h"


def test_setup_speed_roi_arg_parsing_custom_flags():
    """Test custom flags for width, length, and unit."""
    args = parse_args([
        "traffic_cam.mp4",
        "--width", "10.5",
        "--length", "40.0",
        "--unit", "mph",
    ])
    assert args.source == "traffic_cam.mp4"
    assert args.width == 10.5
    assert args.length == 40.0
    assert args.unit == "mph"


def test_setup_speed_roi_arg_parsing_short_flags():
    """Test short flag options -w, -l, -u."""
    args = parse_args([
        "clip.mp4",
        "-w", "8.0",
        "-l", "30.0",
        "-u", "km/h",
    ])
    assert args.source == "clip.mp4"
    assert args.width == 8.0
    assert args.length == 30.0
    assert args.unit == "km/h"


def test_setup_speed_roi_arg_parsing_no_args(monkeypatch):
    """Test fallback default source video when no arguments are passed."""
    monkeypatch.setattr(os.path, "exists", lambda p: p == "clip_4.mp4")
    args = parse_args([])
    assert args.source == "clip_4.mp4"
    assert args.width == 7.5
    assert args.length == 25.0
    assert args.unit == "km/h"


# ==============================================================================
# TOML Formatting Helper Tests
# ==============================================================================

def test_format_speed_toml_block_default():
    """Test formatting standard TOML block and verify it parses cleanly with tomllib."""
    points = [[450, 600], [1450, 600], [2100, 1050], [100, 1050]]
    toml_str = format_speed_toml_block(
        points=points,
        target_width=7.5,
        target_length=25.0,
        unit="km/h",
    )

    assert "[speed]" in toml_str
    assert "enabled = true" in toml_str
    assert 'unit = "km/h"' in toml_str
    assert "source_polygon = [[450, 600], [1450, 600], [2100, 1050], [100, 1050]]" in toml_str
    assert "target_width = 7.5" in toml_str
    assert "target_length = 25.0" in toml_str
    assert "smoothing_window = 7" in toml_str

    parsed = tomllib.loads(toml_str)
    assert "speed" in parsed
    speed_cfg = parsed["speed"]
    assert speed_cfg["enabled"] is True
    assert speed_cfg["unit"] == "km/h"
    assert speed_cfg["source_polygon"] == points
    assert speed_cfg["target_width"] == 7.5
    assert speed_cfg["target_length"] == 25.0
    assert speed_cfg["smoothing_window"] == 7


def test_format_speed_toml_block_tuples_and_custom_unit():
    """Test formatting with tuple points and mph unit."""
    points = [(100, 200), (500, 210), (600, 800), (50, 820)]
    toml_str = format_speed_toml_block(
        points=points,
        target_width=12.0,
        target_length=50.0,
        unit="mph",
    )

    parsed = tomllib.loads(toml_str)
    assert parsed["speed"]["unit"] == "mph"
    assert parsed["speed"]["target_width"] == 12.0
    assert parsed["speed"]["target_length"] == 50.0
    assert parsed["speed"]["source_polygon"] == [[100, 200], [500, 210], [600, 800], [50, 820]]


# ==============================================================================
# Calibration Workflow & OpenCV Interaction Tests
# ==============================================================================

def test_run_setup_speed_roi_nonexistent_file(capsys):
    """Test graceful error message when video file does not exist."""
    result = run_setup_speed_roi("non_existent_video_path.mp4")
    assert result is None
    captured = capsys.readouterr()
    assert "Error: Video file" in captured.out


@patch("os.path.exists", return_value=True)
@patch("supervision.get_video_frames_generator")
def test_run_setup_speed_roi_empty_frames(mock_generator, mock_exists, capsys):
    """Test graceful handling when video has no frames (StopIteration)."""
    mock_generator.return_value = iter([])
    result = run_setup_speed_roi("empty_video.mp4")
    assert result is None
    captured = capsys.readouterr()
    assert "Error: Could not read frames" in captured.out


@patch("cv2.destroyAllWindows")
@patch("cv2.imwrite")
@patch("cv2.imshow")
@patch("cv2.setMouseCallback")
@patch("cv2.namedWindow")
@patch("supervision.get_video_frames_generator")
@patch("os.path.exists", return_value=True)
def test_run_setup_speed_roi_interactive_4_clicks(
    mock_exists,
    mock_generator,
    mock_named_win,
    mock_set_mouse,
    mock_imshow,
    mock_imwrite,
    mock_destroy,
    capsys,
):
    """Test full interactive 4-point click workflow, TOML output, and preview saving."""
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mock_generator.return_value = iter([dummy_frame])

    mouse_callback_holder = []

    def fake_set_mouse_callback(win_name, callback):
        mouse_callback_holder.append(callback)

    mock_set_mouse.side_effect = fake_set_mouse_callback

    # Simulate cv2.waitKey: return -1 during clicks, then 'q' (ord('q')) to exit
    wait_key_calls = 0

    def fake_wait_key(delay):
        nonlocal wait_key_calls
        wait_key_calls += 1
        if wait_key_calls == 1 and mouse_callback_holder:
            mouse_cb = mouse_callback_holder[0]
            # Simulate 4 mouse clicks (EVENT_LBUTTONDOWN = 1)
            pts = [(200, 300), (800, 300), (1000, 600), (100, 600)]
            for x, y in pts:
                mouse_cb(1, x, y, 0, None)
            return -1
        if wait_key_calls >= 2:
            return ord("q")
        return -1

    with patch("cv2.waitKey", side_effect=fake_wait_key):
        result = run_setup_speed_roi(
            source_video_path="test_clip.mp4",
            target_width=7.5,
            target_length=25.0,
            unit="km/h",
        )

    assert result is not None
    assert "[speed]" in result
    assert "source_polygon = [[200, 300], [800, 300], [1000, 600], [100, 600]]" in result

    mock_imwrite.assert_called_once()
    mock_destroy.assert_called_once()

    captured = capsys.readouterr()
    assert "Point 1 recorded: (200, 300)" in captured.out
    assert "Point 4 recorded: (100, 600)" in captured.out
    assert "COPY THIS CONFIGURATION INTO configs/default.toml:" in captured.out


@patch("cv2.destroyAllWindows")
@patch("cv2.imwrite")
@patch("cv2.imshow")
@patch("cv2.setMouseCallback")
@patch("cv2.namedWindow")
@patch("supervision.get_video_frames_generator")
@patch("os.path.exists", return_value=True)
def test_run_setup_speed_roi_reset_workflow(
    mock_exists,
    mock_generator,
    mock_named_win,
    mock_set_mouse,
    mock_imshow,
    mock_imwrite,
    mock_destroy,
    capsys,
):
    """Test resetting points with 'r' key and completing selection."""
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mock_generator.return_value = iter([dummy_frame])

    mouse_cb_holder = []
    mock_set_mouse.side_effect = lambda win, cb: mouse_cb_holder.append(cb)

    # Keypresses sequence: 'r' on call 1, 'q' on call 3
    key_sequence = [ord("r"), -1, ord("q")]
    call_idx = 0

    def fake_wait_key(delay):
        nonlocal call_idx
        if call_idx < len(key_sequence):
            k = key_sequence[call_idx]
            call_idx += 1
            return k
        return ord("q")

    with patch("cv2.waitKey", side_effect=fake_wait_key):
        run_setup_speed_roi("test_clip.mp4")

    captured = capsys.readouterr()
    assert "Points reset. Ready to redraw." in captured.out


# ==============================================================================
# CLI Main Wrapper Tests
# ==============================================================================

@patch("setup_speed_roi.run_setup_speed_roi")
def test_setup_speed_roi_main_dispatch(mock_run):
    """Test CLI main wrapper dispatches with parsed arguments."""
    cli_main(["video.mp4", "--width", "10.0", "--length", "30.0", "--unit", "mph"])
    mock_run.assert_called_once_with(
        source_video_path="video.mp4",
        target_width=10.0,
        target_length=30.0,
        unit="mph",
    )


# ==============================================================================
# Package Export Tests
# ==============================================================================

def test_package_exports():
    """Verify functions are exported from traffik.speed and traffik root."""
    assert hasattr(traffik.speed, "run_setup_speed_roi")
    assert hasattr(traffik.speed, "format_speed_toml_block")
    assert hasattr(traffik, "run_setup_speed_roi")
    assert hasattr(traffik, "format_speed_toml_block")
    assert "run_setup_speed_roi" in traffik.speed.__all__
    assert "format_speed_toml_block" in traffik.speed.__all__
    assert "run_setup_speed_roi" in traffik.__all__
    assert "format_speed_toml_block" in traffik.__all__
