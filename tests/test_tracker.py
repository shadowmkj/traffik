"""Unit tests for the VehicleTracker subsystem and majority-vote class smoothing."""

import numpy as np
import pytest
import supervision as sv
from traffik import VehicleTracker
from traffik.config import TrackerConfig
from traffik.tracking import VehicleTracker as TrackingVehicleTracker


def test_tracker_exports():
    """Verify VehicleTracker is correctly exported from package roots."""
    assert VehicleTracker is TrackingVehicleTracker


def test_tracker_class_smoothing():
    """Verify confidence-weighted majority vote determines smoothed vehicle class."""
    cfg = TrackerConfig()
    tracker = VehicleTracker(cfg, fps=30)

    # Simulate votes for tracker 1
    tracker.record_class_vote(tracker_id=1, raw_class_name="car", conf=0.9)
    tracker.record_class_vote(tracker_id=1, raw_class_name="truck", conf=0.4)

    assert tracker.get_smoothed_class(tracker_id=1, fallback="car") == "car"
    assert tracker.get_smoothed_class(tracker_id=999, fallback="vehicle") == "vehicle"


def test_tracker_majority_vote_shift():
    """Verify smoothed class updates when competing class gains higher cumulative confidence."""
    cfg = TrackerConfig()
    tracker = VehicleTracker(cfg, fps=30)

    # Initial vote: car leads (0.7 vs 0.0)
    tracker.record_class_vote(tracker_id=1, raw_class_name="car", conf=0.7)
    assert tracker.get_smoothed_class(tracker_id=1) == "car"

    # Subsequent votes: truck overtakes (0.5 + 0.5 = 1.0 vs 0.7)
    tracker.record_class_vote(tracker_id=1, raw_class_name="truck", conf=0.5)
    tracker.record_class_vote(tracker_id=1, raw_class_name="truck", conf=0.5)
    assert tracker.get_smoothed_class(tracker_id=1) == "truck"


def test_tracker_get_labels():
    """Verify label generation and formatting '<smoothed_class> #<tracker_id>'."""
    cfg = TrackerConfig()
    tracker = VehicleTracker(cfg, fps=30)

    xyxy = np.array([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=np.float32)
    confidence = np.array([0.92, 0.85], dtype=np.float32)
    class_id = np.array([2, 7], dtype=int)
    tracker_id = np.array([1, 2], dtype=int)

    detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        tracker_id=tracker_id,
    )
    class_names = {2: "car", 7: "truck"}

    labels = tracker.get_labels(detections, class_names=class_names)
    assert labels == ["car #1", "truck #2"]


def test_tracker_get_labels_without_tracker_id():
    """Verify get_labels safely handles detections without tracker IDs."""
    cfg = TrackerConfig()
    tracker = VehicleTracker(cfg, fps=30)

    xyxy = np.array([[10, 10, 50, 50]], dtype=np.float32)
    detections = sv.Detections(xyxy=xyxy)

    labels = tracker.get_labels(detections)
    assert labels == []


def test_tracker_update():
    """Verify update returns an sv.Detections instance."""
    cfg = TrackerConfig()
    tracker = VehicleTracker(cfg, fps=30)

    xyxy = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
    confidence = np.array([0.9], dtype=np.float32)
    class_id = np.array([2], dtype=int)
    detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

    tracked = tracker.update(detections)
    assert isinstance(tracked, sv.Detections)
