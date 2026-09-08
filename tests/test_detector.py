"""Unit tests for the VehicleDetector subsystem."""

import os
from types import SimpleNamespace
import numpy as np
import pytest
import supervision as sv
from traffik import VehicleDetector
from traffik.config import DetectorConfig
from traffik.detection import VehicleDetector as DetectionVehicleDetector
from traffik.detection.detector import _patched_cross_product


def test_detector_exports():
    """Verify VehicleDetector is correctly exported from package roots."""
    assert VehicleDetector is DetectionVehicleDetector


def test_patched_cross_product():
    """Verify patched cross product computes 2D scalar z-component properly."""
    # vector from (0, 0) to (1, 0) -> diff = (0, 1) - (0, 0) = (0, 1)
    # v_x * d_y - v_y * d_x = 1 * 1 - 0 * 0 = 1
    vector = SimpleNamespace(
        start=SimpleNamespace(x=0.0, y=0.0),
        end=SimpleNamespace(x=1.0, y=0.0)
    )
    anchors = np.array([[0.0, 1.0], [0.0, -1.0]])
    cross = _patched_cross_product(anchors, vector)
    assert cross[0] == 1.0
    assert cross[1] == -1.0


def test_detector_initialization_and_inference():
    """Test initializing detector and running inference on a dummy frame."""
    cfg = DetectorConfig(model_path="yolov8n.pt", conf_threshold=0.25, imgsz=320)
    detector = VehicleDetector(cfg, device="cpu")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(dummy_frame)

    assert isinstance(detections, sv.Detections)
    assert hasattr(detections, "xyxy")
    assert hasattr(detections, "class_id")
    assert hasattr(detections, "confidence")


def test_detector_fallback_model():
    """Verify detector falls back to yolov8n.pt when model_path does not exist."""
    cfg = DetectorConfig(model_path="nonexistent_model_file_12345.pt", conf_threshold=0.25, imgsz=320)
    detector = VehicleDetector(cfg, device="cpu")
    assert detector.model is not None


def test_detector_class_filtering_execution():
    """Test inference with specific class filter configured."""
    cfg = DetectorConfig(model_path="yolov8n.pt", conf_threshold=0.25, imgsz=320, classes=[2, 7])
    detector = VehicleDetector(cfg, device="cpu")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(dummy_frame)
    assert isinstance(detections, sv.Detections)
    if len(detections) > 0:
        assert all(cid in [2, 7] for cid in detections.class_id)
