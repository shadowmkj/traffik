"""Traffik - Scalable Vision Pipeline for Vehicle Counting and License Plate OCR."""

from traffik.config import (
    Config,
    DetectorConfig,
    GateConfig,
    GeneralConfig,
    OCRConfig,
    SpeedConfig,
    TrackerConfig,
    get_device,
)
from traffik.counting import DualLineGate, GateState
from traffik.detection import VehicleDetector
from traffik.ocr import PlateReader, PlateResult, clean_plate_text
from traffik.pipeline import PipelineSummary, VideoPipeline, record_run_to_csv
from traffik.speed import (
    SpeedEstimator,
    ViewTransformer,
    format_speed_toml_block,
    run_setup_speed_roi,
)
from traffik.tracking import VehicleTracker
from traffik.visualization import VisualAnnotator, draw_hud_banner

__all__ = [
    "Config",
    "DetectorConfig",
    "DualLineGate",
    "GateConfig",
    "GateState",
    "GeneralConfig",
    "OCRConfig",
    "PipelineSummary",
    "PlateReader",
    "PlateResult",
    "SpeedConfig",
    "SpeedEstimator",
    "TrackerConfig",
    "VehicleDetector",
    "VehicleTracker",
    "VideoPipeline",
    "ViewTransformer",
    "VisualAnnotator",
    "clean_plate_text",
    "draw_hud_banner",
    "format_speed_toml_block",
    "get_device",
    "record_run_to_csv",
    "run_setup_speed_roi",
]


