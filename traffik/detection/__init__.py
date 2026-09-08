"""Detection subsystem for Traffik.

Exports the primary vehicle detector wrapping Ultralytics YOLO models.
"""

from traffik.detection.detector import VehicleDetector

__all__ = ["VehicleDetector"]
