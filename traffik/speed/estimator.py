"""Vehicle speed estimation subsystem for Traffik.

Tracks vehicle bottom-center bounding box anchors over consecutive video frames,
projects coordinates from 2D pixel space to an orthogonal metric ground plane
via perspective homography, and computes smoothed velocity vectors using rolling
window displacement and exponential moving average (EMA) smoothing.
"""

from collections import defaultdict, deque
import math
from typing import Deque, Dict, Optional, Tuple

import numpy as np
import supervision as sv

from traffik.config import SpeedConfig
from traffik.speed.transformer import ViewTransformer


# ==============================================================================
# Speed Estimation Engine
# ==============================================================================

class SpeedEstimator:
    """Estimates real-world speed of tracked vehicles across video frames.

    Perspective projection distorts apparent vehicle speeds across the camera's
    field of view: a vehicle moving at 100 km/h far from the camera traverses only
    a few pixels per second, whereas near the camera it traverses hundreds of pixels.

    SpeedEstimator resolves this through three distinct stages:
    1. **Bottom-Center Ground Anchoring**: Extracting [(x1 + x2)/2, y2] to estimate
       the contact patch where vehicle tires meet the road surface.
    2. **Homographic Metric Projection**: Transforming ground anchor points onto a
       top-down bird's-eye orthogonal metric plane (in meters).
    3. **Windowed Displacement & EMA Smoothing**: Measuring Euclidean displacement
       across a sliding temporal window of frames, converting to km/h or mph, and
       applying exponential moving average smoothing to eliminate frame-to-frame jitter.
    """

    def __init__(self, config: SpeedConfig, fps: float = 30.0) -> None:
        """Initialize the speed estimator with spatial configuration and frame rate.

        Args:
            config: SpeedConfig containing calibration polygon, metric dimensions,
                smoothing window size, and display unit ("km/h" or "mph").
            fps: Video capture frame rate in frames per second (must be > 0).
        """
        self.config = config
        self.fps = float(fps) if fps > 0 else 30.0
        self.unit = config.unit

        # Initialize perspective homography transformer from source polygon to target metric plane
        self.transformer = ViewTransformer(
            source_polygon=np.asarray(config.source_polygon, dtype=np.float32),
            target_width=config.target_width,
            target_length=config.target_length,
        )

        # Minimum window size of 2 frames is required to compute dt and displacement
        self.window = max(2, int(config.smoothing_window))

        # Track trajectory histories: map tracker_id -> deque of metric (xm, ym) coordinates
        # Double the capacity to allow flexible lookbacks without memory leaks
        self.track_positions: defaultdict[int, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.window * 2)
        )

        # Persistent smoothed speeds per tracked vehicle ID
        self.smoothed_speeds: Dict[int, float] = {}

    def update(self, detections: sv.Detections) -> Dict[int, float]:
        """Process detections for the current frame and update estimated vehicle speeds.

        Args:
            detections: supervision Detections object containing bounding boxes (.xyxy)
                and tracking IDs (.tracker_id).

        Returns:
            Dictionary mapping vehicle tracker_id to current smoothed speed (in km/h or mph).
        """
        if detections.tracker_id is None or len(detections) == 0:
            return self.smoothed_speeds

        xyxy = detections.xyxy
        tracker_ids = detections.tracker_id

        # Compute bottom-center anchor points for each bounding box: [(x1 + x2)/2, y2]
        # This approximates the vehicle's road contact point for ground-plane projection
        anchors = np.column_stack(
            [(xyxy[:, 0] + xyxy[:, 2]) / 2.0, xyxy[:, 3]]
        )

        # Transform pixel anchors into real-world metric coordinates (meters)
        metric_points = self.transformer.transform_points(anchors)

        for track_id_raw, (xm, ym) in zip(tracker_ids, metric_points):
            if track_id_raw is None:
                continue
            track_id = int(track_id_raw)
            history = self.track_positions[track_id]
            history.append((float(xm), float(ym)))

            # Only estimate speed when sufficient position history is accumulated
            if len(history) >= self.window:
                # Compare current position with position (window - 1) frames in the past
                x_prev, y_prev = history[-self.window]
                dist_meters = math.hypot(float(xm) - x_prev, float(ym) - y_prev)
                dt_seconds = (self.window - 1) / self.fps

                if dt_seconds > 0:
                    speed_mps = dist_meters / dt_seconds
                else:
                    speed_mps = 0.0

                # Convert metric m/s to chosen speed unit
                if self.unit == "mph":
                    # 1 m/s = 2.2369362920544 mph
                    raw_speed = speed_mps * 2.23694
                else:
                    # Default: 1 m/s = 3.6 km/h
                    raw_speed = speed_mps * 3.6

                # Filter out stationary noise (< 1.0) and unrealistic velocity spikes (> 250.0)
                if 1.0 <= raw_speed <= 250.0:
                    if track_id in self.smoothed_speeds:
                        # Exponential Moving Average: 40% current observation + 60% historical speed
                        self.smoothed_speeds[track_id] = (
                            0.4 * raw_speed + 0.6 * self.smoothed_speeds[track_id]
                        )
                    else:
                        self.smoothed_speeds[track_id] = raw_speed

        return self.smoothed_speeds

    def get_speed(self, tracker_id: int) -> Optional[float]:
        """Retrieve the latest smoothed speed for a tracked vehicle.

        Args:
            tracker_id: Unique integer identifier for the tracked vehicle.

        Returns:
            Smoothed speed as a float, or None if the vehicle has not accumulated
            sufficient tracking history or is not active.
        """
        return self.smoothed_speeds.get(tracker_id)
