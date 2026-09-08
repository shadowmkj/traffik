"""Multi-object tracking subsystem with confidence-weighted class smoothing."""

from collections import Counter, defaultdict
from typing import Dict, List, Optional
import supervision as sv
from traffik.config import TrackerConfig


# ==============================================================================
# Vehicle Tracker & Class Smoothing
# ==============================================================================

class VehicleTracker:
    """Multi-object tracker wrapping ByteTrack with majority-vote class smoothing.

    Per-frame detector misclassifications (e.g., flickering between car and truck)
    are smoothed across track lifetime using cumulative confidence accumulation.
    """

    def __init__(self, config: TrackerConfig, fps: int = 30):
        """Initialize the ByteTrack tracker and track history state.

        Args:
            config: Tracker configuration containing activation and matching thresholds.
            fps: Video frame rate for internal Kalman filter motion prediction.
        """
        self.config = config
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.track_activation_threshold,
            lost_track_buffer=config.lost_track_buffer,
            minimum_matching_threshold=config.minimum_matching_threshold,
            frame_rate=fps,
        )
        # Maps tracker_id -> Counter({class_name: cumulative_confidence})
        self.track_class_history: defaultdict[int, Counter[str]] = defaultdict(Counter)

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker state with new frame detections.

        Args:
            detections: Frame detections from object detector.

        Returns:
            Detections object populated with persistent tracker_id attributes.
        """
        return self.tracker.update_with_detections(detections)

    def record_class_vote(
        self, tracker_id: int, raw_class_name: str, conf: float = 1.0
    ) -> None:
        """Accumulate class detection confidence for a tracked vehicle.

        Args:
            tracker_id: Unique persistent identifier for the tracked vehicle.
            raw_class_name: Class name assigned by current frame detection.
            conf: Confidence score of the current detection (weights the vote).
        """
        self.track_class_history[tracker_id][raw_class_name] += float(conf)

    def get_smoothed_class(self, tracker_id: int, fallback: str = "vehicle") -> str:
        """Retrieve majority-voted class name for a tracked vehicle.

        Args:
            tracker_id: Tracked vehicle identifier.
            fallback: Default class name if no history exists for tracker_id.

        Returns:
            Smoothed vehicle class name with highest cumulative confidence.
        """
        if tracker_id in self.track_class_history and self.track_class_history[tracker_id]:
            return self.track_class_history[tracker_id].most_common(1)[0][0]
        return fallback

    def get_labels(
        self,
        detections: sv.Detections,
        class_names: Optional[Dict[int, str]] = None,
    ) -> List[str]:
        """Record class votes and generate formatted visualization labels.

        Args:
            detections: Tracked detections (must contain tracker_id).
            class_names: Mapping of numeric class IDs to string class names.

        Returns:
            List of formatted labels like '<smoothed_class> #<tracker_id>' for each detection.
        """
        labels: List[str] = []
        if detections.tracker_id is None:
            return labels

        confidences = (
            detections.confidence
            if detections.confidence is not None
            else [1.0] * len(detections)
        )

        class_ids = (
            detections.class_id
            if detections.class_id is not None
            else [0] * len(detections)
        )

        for class_id, tracker_id, conf in zip(
            class_ids, detections.tracker_id, confidences
        ):
            tracker_id_int = int(tracker_id)
            class_id_int = int(class_id)
            raw_class = (
                class_names.get(class_id_int, f"class_{class_id_int}")
                if class_names
                else f"class_{class_id_int}"
            )
            self.record_class_vote(tracker_id_int, raw_class, conf)
            smoothed_class = self.get_smoothed_class(
                tracker_id_int, fallback=raw_class
            )
            labels.append(f"{smoothed_class} #{tracker_id_int}")

        return labels
