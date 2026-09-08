"""Detection subsystem for Traffik.

Provides YOLO-based vehicle detection with automatic hardware device selection,
confidence thresholding, input resizing, and COCO class filtering.
"""

import os
from typing import Optional
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv
import supervision.detection.utils.internal
import supervision.detection.line_zone
from traffik.config import DetectorConfig, get_device


# ==============================================================================
# Supervision + NumPy 2.0+ Compatibility Patch
# ==============================================================================

# In NumPy 2.0+, np.cross requires 3D vectors or operates differently on 2D arrays.
# Supervision's internal cross_product helper expects 2D vectors and calculates
# the scalar z-component of the cross product: v1_x * v2_y - v1_y * v2_x.
# We monkey-patch the cross product calculation to ensure full compatibility.
def _patched_cross_product(anchors, vector):
    """Calculate 2D cross product compatible with NumPy 2.0+ array broadcasting."""
    vector_at_zero = np.array([
        vector.end.x - vector.start.x,
        vector.end.y - vector.start.y,
    ])
    vector_start = np.array([vector.start.x, vector.start.y])
    diff = anchors - vector_start
    return vector_at_zero[0] * diff[..., 1] - vector_at_zero[1] * diff[..., 0]


supervision.detection.utils.internal.cross_product = _patched_cross_product
supervision.detection.line_zone.cross_product = _patched_cross_product


# ==============================================================================
# Vehicle Detector Implementation
# ==============================================================================

class VehicleDetector:
    """YOLO-based object detector with automatic device acceleration and class filtering.

    Encapsulates model initialization, weights resolution with fallback,
    hardware device configuration, and per-frame inference returning
    Supervision `sv.Detections` objects.
    """

    def __init__(self, config: DetectorConfig, device: str = "auto") -> None:
        """Initialize the detector with configuration parameters and compute device.

        Args:
            config: DetectorConfig containing model path, confidence threshold,
                input resolution, and target vehicle class IDs.
            device: Requested compute device ("auto", "cuda", "mps", or "cpu").
        """
        self.config = config
        self.device = get_device(device)

        # Fallback to standard pretrained weights if custom model path does not exist
        model_path = config.model_path if os.path.exists(config.model_path) else "yolov8n.pt"
        self.model = YOLO(model_path)
        self.classes = config.classes

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run object detection inference on a single image or video frame.

        Args:
            frame: Input video frame as a NumPy array (H, W, C) in BGR format.

        Returns:
            sv.Detections object containing bounding boxes, confidences, and class IDs
            filtered to target vehicle classes.
        """
        # Disable autograd overhead for optimal inference throughput
        with torch.inference_mode():
            results = self.model(
                frame,
                conf=self.config.conf_threshold,
                imgsz=self.config.imgsz,
                device=self.device,
                verbose=False,
            )[0]

            detections = sv.Detections.from_ultralytics(results)

            # Filter detections to configured vehicle classes if specified
            if self.classes and len(detections) > 0:
                detections = detections[np.isin(detections.class_id, self.classes)]

            return detections
