"""Perspective view transformation module for mapping image coordinates to real-world ground plane."""

import cv2
import numpy as np


class ViewTransformer:
    """Transforms 2D image pixel coordinates to top-down metric ground coordinates using homography.

    Camera perspective causes objects further away to appear smaller and travel fewer pixels
    per unit time compared to objects close to the camera. ViewTransformer uses a 4-point
    planar homography (perspective transformation) matrix computed via cv2.getPerspectiveTransform
    to project camera coordinates onto an orthogonal bird's-eye metric plane where pixel distances
    map directly to real-world meters.
    """

    def __init__(
        self,
        source_polygon: np.ndarray,
        target_width: float,
        target_length: float,
    ) -> None:
        """Initialize perspective transformation matrix from source image polygon to metric rectangle.

        Args:
            source_polygon: 4x2 array or list of image pixel coordinates ordered as
                [top-left, top-right, bottom-right, bottom-left].
            target_width: Real-world width in meters between the top/bottom pairs.
            target_length: Real-world length in meters along the road section.
        """
        source = np.asarray(source_polygon, dtype=np.float32)
        if source.shape != (4, 2):
            raise ValueError(f"Source polygon must have shape (4, 2), got {source.shape}")

        # Construct target coordinates in metric space (meters)
        # Corresponding to: [top-left, top-right, bottom-right, bottom-left]
        target = np.array(
            [
                [0.0, 0.0],
                [float(target_width), 0.0],
                [float(target_width), float(target_length)],
                [0.0, float(target_length)],
            ],
            dtype=np.float32,
        )

        # Compute 3x3 homography matrix mapping image pixels -> real-world metric plane
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform 2D image pixel points into real-world metric coordinates.

        Args:
            points: (N, 2) array or list of (x, y) coordinates in pixel space.

        Returns:
            (N, 2) array of (x, y) coordinates in real-world metric space (meters).
        """
        pts = np.asarray(points, dtype=np.float32)
        if pts.size == 0 or len(pts) == 0:
            return np.empty((0, 2), dtype=np.float32)

        pts_reshaped = pts.reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts_reshaped, self.m)
        return transformed.reshape(-1, 2)
