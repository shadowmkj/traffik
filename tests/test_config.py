import os
import tempfile
import pytest
from traffik.config import (
    Config,
    GeneralConfig,
    DetectorConfig,
    TrackerConfig,
    GateConfig,
    OCRConfig,
    get_device,
)


def test_get_device_auto():
    device = get_device("auto")
    assert device in ("cuda", "mps", "cpu")


def test_get_device_explicit():
    assert get_device("cpu") == "cpu"
    assert get_device("cuda") == "cuda"
    assert get_device("mps") == "mps"


def test_config_defaults():
    cfg = Config()
    assert isinstance(cfg.general, GeneralConfig)
    assert isinstance(cfg.detector, DetectorConfig)
    assert isinstance(cfg.tracker, TrackerConfig)
    assert isinstance(cfg.gate, GateConfig)
    assert isinstance(cfg.ocr, OCRConfig)
    assert cfg.general.device == "auto"
    assert cfg.detector.model_path == "yolo11n.pt"
    assert cfg.tracker.lost_track_buffer == 45
    assert cfg.gate.offset == 60
    assert cfg.ocr.enabled is False


def test_default_toml_file():
    default_toml_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.toml")
    assert os.path.exists(default_toml_path)
    cfg = Config.from_toml(default_toml_path)
    assert cfg.general.device == "auto"
    assert cfg.detector.model_path == "yolo11n.pt"
    assert cfg.detector.classes == [2, 3, 5, 7]
    assert cfg.tracker.lost_track_buffer == 45
    assert cfg.gate.line_a_start == [49, 1287]
    assert cfg.gate.line_b_end == [2381, 797]
    assert cfg.ocr.enabled is False


def test_load_config():
    toml_content = """
    [general]
    device = "cpu"
    captures_dir = "test_captures"

    [detector]
    model_path = "yolov8n.pt"
    conf_threshold = 0.25
    imgsz = 640
    classes = [2, 3, 5, 7]

    [tracker]
    track_activation_threshold = 0.20
    lost_track_buffer = 30
    minimum_matching_threshold = 0.75

    [gate]
    line_a_start = [0, 100]
    line_a_end   = [200, 100]
    line_b_start = [0, 200]
    line_b_end   = [200, 200]
    offset = 50

    [ocr]
    enabled = false
    conf_threshold = 0.35
    output_csv = "plates.csv"
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        f.flush()
        temp_path = f.name

    try:
        cfg = Config.from_toml(temp_path)
        assert cfg.general.device == "cpu"
        assert cfg.general.captures_dir == "test_captures"
        assert cfg.detector.model_path == "yolov8n.pt"
        assert cfg.detector.conf_threshold == 0.25
        assert cfg.detector.imgsz == 640
        assert cfg.detector.classes == [2, 3, 5, 7]
        assert cfg.tracker.track_activation_threshold == 0.20
        assert cfg.tracker.lost_track_buffer == 30
        assert cfg.tracker.minimum_matching_threshold == 0.75
        assert cfg.gate.line_a_start == [0, 100]
        assert cfg.gate.line_a_end == [200, 100]
        assert cfg.gate.line_b_start == [0, 200]
        assert cfg.gate.line_b_end == [200, 200]
        assert cfg.gate.offset == 50
        assert cfg.ocr.enabled is False
        assert cfg.ocr.conf_threshold == 0.35
        assert cfg.ocr.output_csv == "plates.csv"
    finally:
        os.remove(temp_path)
