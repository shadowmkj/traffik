"""Unit tests for the VideoPipeline orchestration engine and PipelineSummary."""

import os
from pathlib import Path
from typing import Generator
import cv2
import numpy as np
import pytest
from traffik import Config, OCRConfig, PipelineSummary, VideoPipeline
from traffik.pipeline import PipelineSummary as PipelineSummaryFromMod
from traffik.pipeline import VideoPipeline as VideoPipelineFromMod


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Generator[str, None, None]:
    """Generate a 10-frame synthetic test video (640x480, 30 FPS)."""
    video_path = str(tmp_path / "test_synthetic.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    width, height = 640, 480
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame_idx in range(10):
        # Create a simple frame with a white moving box
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw a moving square simulating an object
        x_pos = 50 + frame_idx * 20
        cv2.rectangle(frame, (x_pos, 100), (x_pos + 80, 180), (255, 255, 255), -1)
        out.write(frame)

    out.release()
    yield video_path


def test_pipeline_exports():
    """Verify VideoPipeline and PipelineSummary are exported from package roots."""
    assert VideoPipeline is VideoPipelineFromMod
    assert PipelineSummary is PipelineSummaryFromMod


def test_pipeline_summary_fields():
    """Verify PipelineSummary dataclass fields and types."""
    summary = PipelineSummary(
        source="input.mp4",
        target="output.mp4",
        in_count=10,
        out_count=5,
        unique_tracks=15,
        total_frames=100,
        fps=30.0,
    )
    assert summary.source == "input.mp4"
    assert summary.target == "output.mp4"
    assert summary.in_count == 10
    assert summary.out_count == 5
    assert summary.unique_tracks == 15
    assert summary.total_frames == 100
    assert summary.fps == 30.0


def test_pipeline_instantiation_default():
    """Verify default VideoPipeline instantiation and component initialization."""
    cfg = Config()
    pipeline = VideoPipeline(cfg)

    assert pipeline.detector is not None
    assert pipeline.tracker is not None
    assert pipeline.gate is not None
    assert pipeline.annotator is not None
    assert pipeline.plate_reader is None


def test_pipeline_instantiation_with_ocr(tmp_path: Path):
    """Verify VideoPipeline initializes PlateReader when OCR is enabled."""
    captures_dir = str(tmp_path / "captures")
    csv_file = str(tmp_path / "plates.csv")
    cfg = Config(
        general=Config().general,
        ocr=OCRConfig(enabled=True, output_csv=csv_file),
    )
    cfg.general.captures_dir = captures_dir

    pipeline = VideoPipeline(cfg)
    assert pipeline.plate_reader is not None


def test_pipeline_run_nonexistent_source():
    """Verify VideoPipeline raises FileNotFoundError for missing source video."""
    cfg = Config()
    pipeline = VideoPipeline(cfg)

    with pytest.raises(FileNotFoundError, match="Source video '.*' not found"):
        pipeline.run(source_path="nonexistent_video_path_12345.mp4")


def test_pipeline_run_headless(synthetic_video: str):
    """Verify VideoPipeline processing loop on a synthetic video without video target."""
    cfg = Config()
    pipeline = VideoPipeline(cfg)

    summary = pipeline.run(source_path=synthetic_video)

    assert isinstance(summary, PipelineSummary)
    assert summary.source == synthetic_video
    assert summary.target is None
    assert summary.total_frames == 10
    assert summary.fps > 0.0
    assert summary.in_count >= 0
    assert summary.out_count >= 0
    assert summary.unique_tracks >= 0


def test_pipeline_run_with_target_sink(synthetic_video: str, tmp_path: Path):
    """Verify VideoPipeline writes annotated video output when target_path is specified."""
    target_path = str(tmp_path / "output_annotated.mp4")
    cfg = Config()
    pipeline = VideoPipeline(cfg)

    summary = pipeline.run(source_path=synthetic_video, target_path=target_path)

    assert isinstance(summary, PipelineSummary)
    assert summary.target == target_path
    assert summary.total_frames == 10
    assert os.path.exists(target_path)
    assert os.path.getsize(target_path) > 0


def test_pipeline_run_with_ocr_enabled(synthetic_video: str, tmp_path: Path):
    """Verify VideoPipeline processes frames with OCR enabled without errors."""
    captures_dir = str(tmp_path / "captures")
    csv_file = str(tmp_path / "plates.csv")
    cfg = Config(
        ocr=OCRConfig(enabled=True, output_csv=csv_file),
    )
    cfg.general.captures_dir = captures_dir

    pipeline = VideoPipeline(cfg)
    summary = pipeline.run(source_path=synthetic_video)

    assert isinstance(summary, PipelineSummary)
    assert summary.total_frames == 10


def test_record_run_to_csv(tmp_path: Path):
    """Verify record_run_to_csv creates and increments runs sequentially."""
    from traffik.pipeline.engine import record_run_to_csv
    csv_file = str(tmp_path / "outputs.csv")

    # Run 1 for clip_4.mp4
    r1 = record_run_to_csv(csv_file, "clips/clip_4.mp4", in_count=8, out_count=2)
    assert r1 == 1

    # Run 2 for clip_4.mp4
    r2 = record_run_to_csv(csv_file, "clips/clip_4.mp4", in_count=10, out_count=3)
    assert r2 == 2

    # Run 1 for clip_1.mp4
    r_other = record_run_to_csv(csv_file, "clips/clip_1.mp4", in_count=5, out_count=5)
    assert r_other == 1

    # Run 3 for clip_4.mp4
    r3 = record_run_to_csv(csv_file, "clips/clip_4.mp4", in_count=12, out_count=4)
    assert r3 == 3

    # Check file content
    with open(csv_file, mode="r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    assert lines[0] == "name,run,in,out"
    assert lines[1] == "clip_4.mp4,1,8,2"
    assert lines[2] == "clip_4.mp4,2,10,3"
    assert lines[3] == "clip_1.mp4,1,5,5"
    assert lines[4] == "clip_4.mp4,3,12,4"


def test_pipeline_outputs_csv_logging(synthetic_video: str, tmp_path: Path):
    """Verify VideoPipeline logs execution results to outputs_csv."""
    csv_file = str(tmp_path / "test_outputs.csv")
    cfg = Config()
    cfg.general.outputs_csv = csv_file

    pipeline = VideoPipeline(cfg)
    summary = pipeline.run(source_path=synthetic_video)

    assert summary.run_number == 1
    assert os.path.exists(csv_file)

    # Second run on same video
    summary2 = pipeline.run(source_path=synthetic_video)
    assert summary2.run_number == 2


def test_pipeline_skips_default_outputs_csv_during_pytest(synthetic_video: str):
    """Verify VideoPipeline does not write to the repository outputs.csv during test runs."""
    cfg = Config()
    cfg.general.outputs_csv = "outputs.csv"

    # Get current modification time and size of outputs.csv if it exists
    mtime_before = os.path.getmtime("outputs.csv") if os.path.exists("outputs.csv") else None
    size_before = os.path.getsize("outputs.csv") if os.path.exists("outputs.csv") else None

    pipeline = VideoPipeline(cfg)
    summary = pipeline.run(source_path=synthetic_video)

    assert summary.run_number is None
    if mtime_before is not None:
        assert os.path.getmtime("outputs.csv") == mtime_before
        assert os.path.getsize("outputs.csv") == size_before


