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
    SpeedConfig,
    get_device,
)


def test_get_device_auto():
    device = get_device("auto")
    assert device in ("cuda", "mps", "cpu")


def test_get_device_explicit():
    assert get_device("cpu") == "cpu"
    assert get_device("cuda") == "cuda"
    assert get_device("mps") == "mps"


def test_speed_config_defaults():
    speed_cfg = SpeedConfig()
    assert speed_cfg.enabled is False
    assert speed_cfg.unit == "km/h"
    assert speed_cfg.source_polygon == [[450, 600], [1450, 600], [2100, 1050], [100, 1050]]
    assert speed_cfg.target_width == 7.5
    assert speed_cfg.target_length == 25.0
    assert speed_cfg.smoothing_window == 7


def test_config_defaults():
    cfg = Config()
    assert isinstance(cfg.general, GeneralConfig)
    assert isinstance(cfg.detector, DetectorConfig)
    assert isinstance(cfg.tracker, TrackerConfig)
    assert isinstance(cfg.gate, GateConfig)
    assert isinstance(cfg.ocr, OCRConfig)
    assert isinstance(cfg.speed, SpeedConfig)
    assert cfg.general.device == "auto"
    assert cfg.general.outputs_csv == "outputs.csv"
    assert cfg.detector.model_path == "yolo11n.pt"
    assert cfg.tracker.lost_track_buffer == 45
    assert cfg.gate.offset == 60
    assert cfg.ocr.enabled is False
    assert cfg.speed.enabled is False
    assert cfg.speed.unit == "km/h"


def test_default_toml_file():
    default_toml_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.toml")
    assert os.path.exists(default_toml_path)
    cfg = Config.from_toml(default_toml_path)
    assert cfg.general.device == "auto"
    assert cfg.general.outputs_csv == "outputs.csv"
    assert cfg.detector.model_path == "yolo11n.pt"
    assert cfg.detector.classes == [2, 3, 5, 7]
    assert cfg.tracker.lost_track_buffer == 45
    assert cfg.gate.line_a_start == [49, 1287]
    assert cfg.gate.line_b_end == [2381, 797]
    assert cfg.ocr.enabled is False
    assert cfg.speed.enabled is False
    assert cfg.speed.unit == "km/h"
    assert cfg.speed.source_polygon == [[450, 600], [1450, 600], [2100, 1050], [100, 1050]]
    assert cfg.speed.target_width == 7.5
    assert cfg.speed.target_length == 25.0
    assert cfg.speed.smoothing_window == 7


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

    [speed]
    enabled = true
    unit = "mph"
    source_polygon = [[100, 200], [300, 200], [350, 400], [50, 400]]
    target_width = 10.0
    target_length = 30.0
    smoothing_window = 5
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
        assert cfg.speed.enabled is True
        assert cfg.speed.unit == "mph"
        assert cfg.speed.source_polygon == [[100, 200], [300, 200], [350, 400], [50, 400]]
        assert cfg.speed.target_width == 10.0
        assert cfg.speed.target_length == 30.0
        assert cfg.speed.smoothing_window == 5
    finally:
        os.remove(temp_path)
