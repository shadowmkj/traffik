"""Unit tests for vehicle speed estimation components and perspective homography transformation."""

import numpy as np
import pytest
import supervision as sv

from traffik.config import SpeedConfig
from traffik.speed.estimator import SpeedEstimator
from traffik.speed.transformer import ViewTransformer


# ==============================================================================
# Perspective ViewTransformer Tests
# ==============================================================================

def test_view_transformer_square():
    """Test standard orthogonal mapping from pixel rectangle to metric dimensions."""
    # Define a 100x100 square in pixels mapping to 10m x 20m in real world
    source = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    transformer = ViewTransformer(source_polygon=source, target_width=10.0, target_length=20.0)

    # Test transforming all 4 corner points
    test_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    transformed = transformer.transform_points(test_pts)

    assert transformed.shape == (4, 2)
    np.testing.assert_allclose(transformed[0], [0.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(transformed[1], [10.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(transformed[2], [10.0, 20.0], atol=1e-3)
    np.testing.assert_allclose(transformed[3], [0.0, 20.0], atol=1e-3)


def test_view_transformer_perspective_trapezoid():
    """Test perspective trapezoid homography mapping to orthogonal metric ground plane."""
    # Perspective trapezoid simulating a road receding into the distance
    source = np.array(
        [[450, 600], [1450, 600], [2100, 1050], [100, 1050]],
        dtype=np.float32,
    )
    target_width = 7.5
    target_length = 25.0
    transformer = ViewTransformer(
        source_polygon=source,
        target_width=target_width,
        target_length=target_length,
    )

    # Corners must map precisely to [[0, 0], [W, 0], [W, L], [0, L]]
    transformed_corners = transformer.transform_points(source)
    np.testing.assert_allclose(transformed_corners[0], [0.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(transformed_corners[1], [target_width, 0.0], atol=1e-3)
    np.testing.assert_allclose(transformed_corners[2], [target_width, target_length], atol=1e-3)
    np.testing.assert_allclose(transformed_corners[3], [0.0, target_length], atol=1e-3)


def test_view_transformer_empty_points():
    """Test handling of empty point arrays returns a valid (0, 2) array."""
    source = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    transformer = ViewTransformer(source_polygon=source, target_width=10.0, target_length=20.0)

    empty_pts = np.empty((0, 2), dtype=np.float32)
    transformed = transformer.transform_points(empty_pts)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == (0, 2)


def test_view_transformer_shape_validation():
    """Test that source_polygon must have shape (4, 2), raising ValueError otherwise."""
    # Invalid: 3 points (triangle)
    with pytest.raises(ValueError, match=r"shape \(4, 2\)"):
        ViewTransformer(
            source_polygon=np.array([[0, 0], [100, 0], [100, 100]], dtype=np.float32),
            target_width=10.0,
            target_length=20.0,
        )

    # Invalid: 5 points (pentagon)
    with pytest.raises(ValueError, match=r"shape \(4, 2\)"):
        ViewTransformer(
            source_polygon=np.array([[0, 0], [100, 0], [100, 100], [50, 150], [0, 100]], dtype=np.float32),
            target_width=10.0,
            target_length=20.0,
        )

    # Invalid: 4 points of 3D coordinates (4, 3)
    with pytest.raises(ValueError, match=r"shape \(4, 2\)"):
        ViewTransformer(
            source_polygon=np.ones((4, 3), dtype=np.float32),
            target_width=10.0,
            target_length=20.0,
        )


def test_view_transformer_list_input():
    """Test accepting Python lists for source_polygon as well as np.ndarray."""
    source_list = [[0, 0], [100, 0], [100, 100], [0, 100]]
    transformer = ViewTransformer(source_polygon=source_list, target_width=5.0, target_length=15.0)

    transformed = transformer.transform_points([[50, 50]])
    assert transformed.shape == (1, 2)
    np.testing.assert_allclose(transformed[0], [2.5, 7.5], atol=1e-3)


# ==============================================================================
# SpeedEstimator Tests
# ==============================================================================

def test_speed_estimator_constant_speed_kmh():
    """Test synthetic trajectory at constant simulated speed converges to ~108 km/h."""
    # 1:1 mapping between pixels and meters
    config = SpeedConfig(
        source_polygon=[[0, 0], [10, 0], [10, 100], [0, 100]],
        target_width=10.0,
        target_length=100.0,
        smoothing_window=7,
        unit="km/h",
    )
    fps = 30.0
    estimator = SpeedEstimator(config=config, fps=fps)

    # Simulate vehicle 1 moving at 1 meter/frame.
    # At 30 FPS, 1 m/frame = 30 m/s = 108 km/h.
    # Vehicle bottom-center anchor: [(x1 + x2)/2, y2] = [5.0, y]
    for frame_idx in range(25):
        y = float(frame_idx)
        # Bounding box with bottom center at (5.0, y)
        xyxy = np.array([[4.0, max(0.0, y - 2.0), 6.0, y]], dtype=np.float32)
        detections = sv.Detections(
            xyxy=xyxy,
            tracker_id=np.array([1], dtype=int),
        )
        speeds = estimator.update(detections)

        # Before reaching window size (7 frames), speed should not be estimated
        if frame_idx < 6:
            assert 1 not in speeds
        else:
            assert 1 in speeds

    estimated_speed = estimator.get_speed(1)
    assert estimated_speed is not None
    assert isinstance(estimated_speed, float)
    # Expected speed: 108 km/h (+- 5 km/h)
    assert abs(estimated_speed - 108.0) <= 5.0


def test_speed_estimator_unit_conversion_mph():
    """Test unit conversion to mph: 30 m/s = ~67.1 mph."""
    config = SpeedConfig(
        source_polygon=[[0, 0], [10, 0], [10, 100], [0, 100]],
        target_width=10.0,
        target_length=100.0,
        smoothing_window=7,
        unit="mph",
    )
    fps = 30.0
    estimator = SpeedEstimator(config=config, fps=fps)

    # Simulate 1 m/frame at 30 fps = 30 m/s = 30 * 2.23694 = 67.1082 mph
    for frame_idx in range(25):
        y = float(frame_idx)
        xyxy = np.array([[4.0, max(0.0, y - 2.0), 6.0, y]], dtype=np.float32)
        detections = sv.Detections(
            xyxy=xyxy,
            tracker_id=np.array([1], dtype=int),
        )
        estimator.update(detections)

    estimated_speed = estimator.get_speed(1)
    assert estimated_speed is not None
    assert abs(estimated_speed - 67.1) <= 3.0


def test_speed_estimator_empty_and_no_tracker_id():
    """Test graceful handling of empty detections and missing tracker IDs."""
    config = SpeedConfig(
        source_polygon=[[0, 0], [10, 0], [10, 100], [0, 100]],
        target_width=10.0,
        target_length=100.0,
        smoothing_window=5,
        unit="km/h",
    )
    estimator = SpeedEstimator(config=config, fps=30.0)

    # 1. Empty detections
    res1 = estimator.update(sv.Detections.empty())
    assert res1 == {}

    # 2. Detections with tracker_id=None
    xyxy = np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32)
    res2 = estimator.update(sv.Detections(xyxy=xyxy, tracker_id=None))
    assert res2 == {}

    # 3. Empty numpy array detections
    res3 = estimator.update(sv.Detections(xyxy=np.zeros((0, 4)), tracker_id=np.array([])))
    assert res3 == {}


def test_speed_estimator_get_speed_unknown_track():
    """Test get_speed returns None for untracked vehicle ID."""
    config = SpeedConfig()
    estimator = SpeedEstimator(config=config, fps=30.0)

    assert estimator.get_speed(999) is None
    assert estimator.get_speed(-1) is None


def test_speed_estimator_multiple_tracks():
    """Test simultaneous speed estimation for multiple distinct vehicle tracks."""
    config = SpeedConfig(
        source_polygon=[[0, 0], [20, 0], [20, 100], [0, 100]],
        target_width=20.0,
        target_length=100.0,
        smoothing_window=5,
        unit="km/h",
    )
    estimator = SpeedEstimator(config=config, fps=30.0)

    # Track 10: 1.0 m/frame (108 km/h) on lane X=5
    # Track 20: 0.5 m/frame (54 km/h) on lane X=15
    for frame_idx in range(20):
        y1 = float(frame_idx) * 1.0
        y2 = float(frame_idx) * 0.5
        xyxy = np.array([
            [4.0, max(0.0, y1 - 2.0), 6.0, y1],
            [14.0, max(0.0, y2 - 2.0), 16.0, y2],
        ], dtype=np.float32)
        detections = sv.Detections(
            xyxy=xyxy,
            tracker_id=np.array([10, 20], dtype=int),
        )
        estimator.update(detections)

    speed_10 = estimator.get_speed(10)
    speed_20 = estimator.get_speed(20)

    assert speed_10 is not None
    assert speed_20 is not None
    assert abs(speed_10 - 108.0) <= 5.0
    assert abs(speed_20 - 54.0) <= 5.0


def test_speed_estimator_out_of_range_filtering():
    """Test stationary vehicles (< 1.0) and unrealistic velocity spikes (> 250.0) are filtered."""
    config = SpeedConfig(
        source_polygon=[[0, 0], [10, 0], [10, 100], [0, 100]],
        target_width=10.0,
        target_length=100.0,
        smoothing_window=3,
        unit="km/h",
    )
    estimator = SpeedEstimator(config=config, fps=30.0)

    # Stationary vehicle (0 movement): 0 km/h < 1.0 -> not added to smoothed_speeds
    for _ in range(10):
        xyxy = np.array([[4.0, 10.0, 6.0, 12.0]], dtype=np.float32)
        estimator.update(sv.Detections(xyxy=xyxy, tracker_id=np.array([1], dtype=int)))

    assert estimator.get_speed(1) is None
