"""Traffik - Scalable Vision Pipeline for Vehicle Counting and License Plate OCR."""

from traffik.config import (
    Config,
    DetectorConfig,
    GateConfig,
    GeneralConfig,
    OCRConfig,
    TrackerConfig,
    get_device,
)

__all__ = [
    "Config",
    "DetectorConfig",
    "GateConfig",
    "GeneralConfig",
    "OCRConfig",
    "TrackerConfig",
    "get_device",
]
