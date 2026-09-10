"""Unit tests for vehicle speed estimation components and perspective homography transformation."""

import numpy as np
import pytest

from traffik.speed.transformer import ViewTransformer


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
