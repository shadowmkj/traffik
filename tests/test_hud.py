import numpy as np
import supervision as sv
from traffik.config import GateConfig
from traffik.counting.gate import DualLineGate
from traffik.visualization.hud import VisualAnnotator, draw_hud_banner


def test_draw_hud_banner_shape_and_type():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated = draw_hud_banner(frame, in_count=5, out_count=3)
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == (480, 640, 3)
    assert annotated.dtype == np.uint8


def test_draw_hud_banner_draws_content():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    # Check that pixels are modified (overlay drawn)
    assert np.all(frame == 0)
    annotated = draw_hud_banner(frame, in_count=10, out_count=20)
    assert np.any(annotated > 0)


def test_visual_annotator_init():
    annotator = VisualAnnotator()
    assert isinstance(annotator.box_annotator, sv.BoxAnnotator)
    assert isinstance(annotator.label_annotator, sv.LabelAnnotator)
    assert isinstance(annotator.line_a_annotator, sv.LineZoneAnnotator)
    assert isinstance(annotator.line_b_annotator, sv.LineZoneAnnotator)


def test_visual_annotator_annotate_minimal():
    annotator = VisualAnnotator()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = sv.Detections.empty()

    # With no labels and no gate
    out = annotator.annotate(frame, detections)
    assert out.shape == (480, 640, 3)
    assert isinstance(out, np.ndarray)


def test_visual_annotator_annotate_with_labels():
    annotator = VisualAnnotator()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 50, 50], [60, 60, 100, 100]]),
        confidence=np.array([0.9, 0.8]),
        class_id=np.array([2, 7]),
        tracker_id=np.array([1, 2]),
    )
    labels = ["car #1", "truck #2"]

    out = annotator.annotate(frame, detections, labels=labels)
    assert out.shape == (480, 640, 3)
    assert isinstance(out, np.ndarray)


def test_visual_annotator_annotate_with_gate_and_labels():
    annotator = VisualAnnotator()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    gate_cfg = GateConfig(
        line_a_start=[10, 50],
        line_a_end=[200, 50],
        line_b_start=[10, 100],
        line_b_end=[200, 100],
    )
    gate = DualLineGate.from_config(gate_cfg)

    detections = sv.Detections(
        xyxy=np.array([[20, 20, 40, 40]]),
        confidence=np.array([0.95]),
        class_id=np.array([2]),
        tracker_id=np.array([1]),
    )
    labels = ["car #1"]

    out = annotator.annotate(frame, detections, labels=labels, gate=gate)
    assert out.shape == (480, 640, 3)
    assert isinstance(out, np.ndarray)
    # Ensure drawing happened on the frame
    assert np.any(out > 0)
