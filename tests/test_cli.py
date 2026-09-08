"""Unit tests for Traffik unified CLI and backward-compatible wrappers."""

from unittest.mock import MagicMock, patch
import pytest

from traffik.cli.main import create_parser, main
from traffik.pipeline.engine import PipelineSummary
import main as main_wrapper
import setup_line as setup_line_wrapper
import stream as stream_wrapper


@pytest.fixture
def mock_summary():
    return PipelineSummary(
        source="test.mp4",
        target="test_out.mp4",
        in_count=12,
        out_count=7,
        unique_tracks=15,
        total_frames=100,
        fps=30.0,
    )


def test_cli_parser_creation():
    parser = create_parser()
    assert parser.prog == "traffik"


def test_cli_process_arg_parsing():
    parser = create_parser()
    args = parser.parse_args(["process", "video.mp4"])
    assert args.command == "process"
    assert args.source == "video.mp4"
    assert args.output is None
    assert args.config == "configs/default.toml"
    assert args.ocr is False
    assert args.no_save is False

    args_custom = parser.parse_args([
        "process", "clip.mp4",
        "-o", "custom_out.mp4",
        "-c", "custom_cfg.toml",
        "--ocr",
        "--no-save"
    ])
    assert args_custom.command == "process"
    assert args_custom.source == "clip.mp4"
    assert args_custom.output == "custom_out.mp4"
    assert args_custom.config == "custom_cfg.toml"
    assert args_custom.ocr is True
    assert args_custom.no_save is True


def test_cli_stream_arg_parsing():
    parser = create_parser()
    args = parser.parse_args(["stream", "stream.mp4"])
    assert args.command == "stream"
    assert args.source == "stream.mp4"
    assert args.output is None
    assert args.no_save is False

    args_custom = parser.parse_args([
        "stream", "stream.mp4",
        "-o", "stream_out.mp4",
        "-c", "custom.toml",
        "--no-save"
    ])
    assert args_custom.command == "stream"
    assert args_custom.output == "stream_out.mp4"
    assert args_custom.config == "custom.toml"
    assert args_custom.no_save is True


def test_cli_setup_gate_arg_parsing():
    parser = create_parser()
    args = parser.parse_args(["setup-gate", "calibrate.mp4"])
    assert args.command == "setup-gate"
    assert args.source == "calibrate.mp4"


def test_cli_help_flags(capsys):
    parser = create_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["process", "--help"])
    assert excinfo.value.code == 0

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["stream", "--help"])
    assert excinfo.value.code == 0

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["setup-gate", "--help"])
    assert excinfo.value.code == 0


@patch("traffik.cli.main.VideoPipeline")
def test_cli_main_process_dispatch(mock_pipeline_cls, mock_summary):
    mock_instance = MagicMock()
    mock_instance.run.return_value = mock_summary
    mock_pipeline_cls.return_value = mock_instance

    main(["process", "input.mp4", "-o", "output.mp4", "--ocr"])

    mock_pipeline_cls.assert_called_once()
    passed_cfg = mock_pipeline_cls.call_args[0][0]
    assert passed_cfg.ocr.enabled is True

    mock_instance.run.assert_called_once_with(
        source_path="input.mp4",
        target_path="output.mp4",
        stream=False,
    )


@patch("traffik.cli.main.VideoPipeline")
def test_cli_main_process_no_save(mock_pipeline_cls, mock_summary):
    mock_instance = MagicMock()
    mock_instance.run.return_value = mock_summary
    mock_pipeline_cls.return_value = mock_instance

    main(["process", "input.mp4", "--no-save"])

    mock_instance.run.assert_called_once_with(
        source_path="input.mp4",
        target_path=None,
        stream=False,
    )


@patch("traffik.cli.main.VideoPipeline")
def test_cli_main_stream_dispatch(mock_pipeline_cls, mock_summary):
    mock_instance = MagicMock()
    mock_instance.run.return_value = mock_summary
    mock_pipeline_cls.return_value = mock_instance

    main(["stream", "input.mp4"])

    mock_instance.run.assert_called_once_with(
        source_path="input.mp4",
        target_path="input_out.mp4",
        stream=True,
    )


@patch("setup_line.run_setup_gate")
def test_cli_main_setup_gate_dispatch(mock_run_setup_gate):
    main(["setup-gate", "clip_source.mp4"])
    mock_run_setup_gate.assert_called_once_with("clip_source.mp4")


@patch("stream.VideoPipeline")
@patch("os.path.exists", return_value=True)
def test_stream_wrapper(mock_exists, mock_pipeline_cls, mock_summary):
    mock_instance = MagicMock()
    mock_instance.run.return_value = mock_summary
    mock_pipeline_cls.return_value = mock_instance

    stream_wrapper.main(["test.mp4", "--no-stream", "--conf", "0.35", "--imgsz", "1280", "--gate-offset", "80"])

    mock_pipeline_cls.assert_called_once()
    cfg = mock_pipeline_cls.call_args[0][0]
    assert cfg.detector.conf_threshold == 0.35
    assert cfg.detector.imgsz == 1280
    assert cfg.gate.offset == 80

    mock_instance.run.assert_called_once_with(
        source_path="test.mp4",
        target_path="test_out.mp4",
        stream=False,
    )


@patch("main.VideoPipeline")
@patch("os.path.exists", return_value=True)
def test_main_wrapper(mock_exists, mock_pipeline_cls, mock_summary):
    mock_instance = MagicMock()
    mock_instance.run.return_value = mock_summary
    mock_pipeline_cls.return_value = mock_instance

    main_wrapper.main(["clip.mp4", "-o", "out_custom.mp4"])

    mock_pipeline_cls.assert_called_once()
    cfg = mock_pipeline_cls.call_args[0][0]
    assert cfg.ocr.enabled is True

    mock_instance.run.assert_called_once_with(
        source_path="clip.mp4",
        target_path="out_custom.mp4",
        stream=False,
    )


@patch("setup_line.run_setup_gate")
def test_setup_line_wrapper(mock_run_setup_gate):
    setup_line_wrapper.main(["calibrate.mp4"])
    mock_run_setup_gate.assert_called_once_with("calibrate.mp4")
