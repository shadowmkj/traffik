"""Configuration subsystem for Traffik.

Handles loading TOML configuration files into strongly-typed dataclasses
and automatic hardware acceleration device resolution (CUDA -> MPS -> CPU).
"""

import sys
from dataclasses import dataclass, field
from typing import List

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore

import torch


# ==============================================================================
# Hardware Device Resolution
# ==============================================================================

def get_device(requested: str = "auto") -> str:
    """Determine the optimal hardware compute device.

    Prioritizes NVIDIA CUDA, Apple Metal Performance Shaders (MPS), and falls
    back to CPU when 'auto' is selected. Explicit device requests ('cuda', 'mps',
    'cpu') are passed through directly.

    Args:
        requested: Hardware device string ("auto", "cuda", "mps", or "cpu").

    Returns:
        Resolved device string ("cuda", "mps", or "cpu").
    """
    if requested in ("cuda", "mps", "cpu"):
        return requested

    # Hardware detection precedence: CUDA -> MPS -> CPU
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ==============================================================================
# Configuration Dataclasses
# ==============================================================================

@dataclass
class GeneralConfig:
    """General pipeline execution options."""
    device: str = "auto"
    captures_dir: str = "captures"
    outputs_csv: str = "outputs.csv"


@dataclass
class DetectorConfig:
    """Object detector configuration parameters."""
    model_path: str = "yolo11n.pt"
    conf_threshold: float = 0.20
    imgsz: int = 640
    # COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
    classes: List[int] = field(default_factory=lambda: [2, 3, 5, 7])


@dataclass
class TrackerConfig:
    """Multi-object tracker (ByteTrack) configuration parameters."""
    track_activation_threshold: float = 0.20
    lost_track_buffer: int = 45
    minimum_matching_threshold: float = 0.75


@dataclass
class GateConfig:
    """Dual-line virtual counting gate configuration coordinates and offsets."""
    line_a_start: List[int] = field(default_factory=lambda: [49, 1287])
    line_a_end: List[int] = field(default_factory=lambda: [1881, 816])
    line_b_start: List[int] = field(default_factory=lambda: [101, 1458])
    line_b_end: List[int] = field(default_factory=lambda: [2381, 797])
    offset: int = 60


@dataclass
class OCRConfig:
    """License plate optical character recognition (OCR) configuration."""
    enabled: bool = False
    conf_threshold: float = 0.35
    output_csv: str = "number_plates.csv"


@dataclass
class SpeedConfig:
    """Speed estimation subsystem configuration."""
    enabled: bool = False
    unit: str = "km/h"
    source_polygon: List[List[int]] = field(
        default_factory=lambda: [[450, 600], [1450, 600], [2100, 1050], [100, 1050]]
    )
    target_width: float = 7.5
    target_length: float = 25.0
    smoothing_window: int = 7


@dataclass
class Config:
    """Root configuration object composing all subsystem configurations."""
    general: GeneralConfig = field(default_factory=GeneralConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)

    @classmethod
    def from_toml(cls, path: str) -> "Config":
        """Load and parse a TOML configuration file into a Config instance.

        Args:
            path: Path to the TOML configuration file.

        Returns:
            Instantiated and populated Config object.
        """
        with open(path, "rb") as f:
            data = tomllib.load(f)

        general_data = data.get("general", {})
        detector_data = data.get("detector", {})
        tracker_data = data.get("tracker", {})
        gate_data = data.get("gate", {})
        ocr_data = data.get("ocr", {})
        speed_data = data.get("speed", {})

        return cls(
            general=GeneralConfig(**general_data),
            detector=DetectorConfig(**detector_data),
            tracker=TrackerConfig(**tracker_data),
            gate=GateConfig(**gate_data),
            ocr=OCRConfig(**ocr_data),
            speed=SpeedConfig(**speed_data),
        )
