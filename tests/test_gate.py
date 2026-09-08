"""Unit tests for DualLineGate counting engine and state machine transitions."""

from unittest.mock import MagicMock
import numpy as np
import pytest
import supervision as sv
from traffik import DualLineGate, GateState
from traffik.config import GateConfig
from traffik.counting import DualLineGate as CountingDualLineGate, GateState as CountingGateState


def test_gate_exports():
    """Verify DualLineGate and GateState are exported from package roots."""
    assert DualLineGate is CountingDualLineGate
    assert GateState is CountingGateState


def test_gate_initialization_from_config():
    """Verify DualLineGate.from_config creates proper LineZone instances and initial states."""
    cfg = GateConfig(
        line_a_start=[10, 100],
        line_a_end=[200, 100],
        line_b_start=[10, 200],
        line_b_end=[200, 200],
        offset=60,
    )
    gate = DualLineGate.from_config(cfg)

    assert gate.in_count == 0
    assert gate.out_count == 0
    assert len(gate.track_states) == 0
    assert isinstance(gate.line_a, sv.LineZone)
    assert isinstance(gate.line_b, sv.LineZone)


def test_gate_initialization_direct():
    """Verify direct initialization with sv.Point coordinates and custom anchors."""
    start_a = sv.Point(0, 50)
    end_a = sv.Point(100, 50)
    start_b = sv.Point(0, 150)
    end_b = sv.Point(100, 150)
    anchors = [sv.Position.CENTER]

    gate = DualLineGate(
        line_a_start=start_a,
        line_a_end=end_a,
        line_b_start=start_b,
        line_b_end=end_b,
        triggering_anchors=anchors,
    )

    assert gate.in_count == 0
    assert gate.out_count == 0
    assert gate.line_a.triggering_anchors == anchors
    assert gate.line_b.triggering_anchors == anchors


def test_gate_trigger_empty_detections():
    """Verify trigger handles empty detections gracefully without modifying counts."""
    cfg = GateConfig()
    gate = DualLineGate.from_config(cfg)

    empty_detections = sv.Detections.empty()
    gate.trigger(empty_detections)

    assert gate.in_count == 0
    assert gate.out_count == 0
    assert len(gate.track_states) == 0


def test_gate_trigger_no_tracker_id():
    """Verify trigger handles detections without tracker_id gracefully."""
    cfg = GateConfig()
    gate = DualLineGate.from_config(cfg)

    detections = sv.Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        tracker_id=None,
    )
    gate.trigger(detections)

    assert gate.in_count == 0
    assert gate.out_count == 0
    assert len(gate.track_states) == 0


def test_gate_state_machine_inward_crossing():
    """Verify full state transition: OUTSIDE -> ENTERED_A -> COUNTED (in_count + 1)."""
    cfg = GateConfig()
    gate = DualLineGate.from_config(cfg)

    # Mock line zones to isolate state machine logic
    gate.line_a = MagicMock()
    gate.line_b = MagicMock()

    detections = sv.Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        tracker_id=np.array([1], dtype=int),
    )

    # Step 1: Vehicle crosses Line A inward
    gate.line_a.trigger.return_value = (np.array([True]), np.array([False]))
    gate.line_b.trigger.return_value = (np.array([False]), np.array([False]))
    gate.trigger(detections)

    assert gate.track_states[1] == GateState.ENTERED_A
    assert gate.in_count == 0
    assert gate.out_count == 0

    # Step 2: Vehicle crosses Line B inward -> COUNTED, in_count += 1
    gate.line_a.trigger.return_value = (np.array([False]), np.array([False]))
    gate.line_b.trigger.return_value = (np.array([True]), np.array([False]))
    gate.trigger(detections)

    assert gate.track_states[1] == GateState.COUNTED
    assert gate.in_count == 1
    assert gate.out_count == 0

    # Step 3: Subsequent frames for already counted track ID do not re-increment
    gate.line_a.trigger.return_value = (np.array([True]), np.array([False]))
    gate.line_b.trigger.return_value = (np.array([True]), np.array([False]))
    gate.trigger(detections)

    assert gate.in_count == 1
    assert gate.out_count == 0


def test_gate_state_machine_outward_crossing():
    """Verify full state transition: OUTSIDE -> ENTERED_B -> COUNTED (out_count + 1)."""
    cfg = GateConfig()
    gate = DualLineGate.from_config(cfg)

    gate.line_a = MagicMock()
    gate.line_b = MagicMock()

    detections = sv.Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        tracker_id=np.array([2], dtype=int),
    )

    # Step 1: Vehicle crosses Line B outward
    gate.line_a.trigger.return_value = (np.array([False]), np.array([False]))
    gate.line_b.trigger.return_value = (np.array([False]), np.array([True]))
    gate.trigger(detections)

    assert gate.track_states[2] == GateState.ENTERED_B
    assert gate.in_count == 0
    assert gate.out_count == 0

    # Step 2: Vehicle crosses Line A outward -> COUNTED, out_count += 1
    gate.line_a.trigger.return_value = (np.array([False]), np.array([True]))
    gate.line_b.trigger.return_value = (np.array([False]), np.array([False]))
    gate.trigger(detections)

    assert gate.track_states[2] == GateState.COUNTED
    assert gate.in_count == 0
    assert gate.out_count == 1

    # Step 3: Subsequent frames for already counted track ID do not re-increment
    gate.line_a.trigger.return_value = (np.array([False]), np.array([True]))
    gate.line_b.trigger.return_value = (np.array([False]), np.array([True]))
    gate.trigger(detections)

    assert gate.in_count == 0
    assert gate.out_count == 1


def test_gate_fallback_inside_start_inward():
    """Verify fallback for track starting inside gate moving inward directly crossing Line B."""
    cfg = GateConfig()
    gate = DualLineGate.from_config(cfg)

    gate.line_a = MagicMock()
    gate.line_b = MagicMock()

    detections = sv.Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        tracker_id=np.array([3], dtype=int),
    )

    # Crosses Line B in directly while in OUTSIDE state
    gate.line_a.trigger.return_value = (np.array([False]), np.array([False]))
    gate.line_b.trigger.return_value = (np.array([True]), np.array([False]))
    gate.trigger(detections)

    assert gate.track_states[3] == GateState.COUNTED
    assert gate.in_count == 1
    assert gate.out_count == 0


def test_gate_fallback_inside_start_outward():
    """Verify fallback for track starting inside gate moving outward directly crossing Line A."""
    cfg = GateConfig()
    gate = DualLineGate.from_config(cfg)

    gate.line_a = MagicMock()
    gate.line_b = MagicMock()

    detections = sv.Detections(
        xyxy=np.array([[10, 10, 50, 50]], dtype=np.float32),
        tracker_id=np.array([4], dtype=int),
    )

    # Crosses Line A out directly while in OUTSIDE state
    gate.line_a.trigger.return_value = (np.array([False]), np.array([True]))
    gate.line_b.trigger.return_value = (np.array([False]), np.array([False]))
    gate.trigger(detections)

    assert gate.track_states[4] == GateState.COUNTED
    assert gate.in_count == 0
    assert gate.out_count == 1


def test_gate_multiple_concurrent_tracks():
    """Verify multiple vehicle tracks updating concurrently with different states."""
    cfg = GateConfig()
    gate = DualLineGate.from_config(cfg)

    gate.line_a = MagicMock()
    gate.line_b = MagicMock()

    # Track 10 starts crossing A (inward), Track 20 starts crossing B (outward)
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=np.float32),
        tracker_id=np.array([10, 20], dtype=int),
    )

    gate.line_a.trigger.return_value = (np.array([True, False]), np.array([False, False]))
    gate.line_b.trigger.return_value = (np.array([False, False]), np.array([False, True]))
    gate.trigger(detections)

    assert gate.track_states[10] == GateState.ENTERED_A
    assert gate.track_states[20] == GateState.ENTERED_B
    assert gate.in_count == 0
    assert gate.out_count == 0

    # Next frame: Track 10 completes crossing B (inward), Track 20 completes crossing A (outward)
    gate.line_a.trigger.return_value = (np.array([False, True]), np.array([False, False]))
    gate.line_b.trigger.return_value = (np.array([True, False]), np.array([False, False]))
    gate.trigger(detections)

    assert gate.track_states[10] == GateState.COUNTED
    assert gate.track_states[20] == GateState.COUNTED
    assert gate.in_count == 1
    assert gate.out_count == 1


def test_gate_end_to_end_crossing_geometry():
    """Verify end-to-end crossing detection with real LineZone geometric calculations."""
    # Line A at y=100 (from 200,100 to 0,100 -> moving top-to-bottom crosses inward)
    # Line B at y=200 (from 200,200 to 0,200 -> moving top-to-bottom crosses inward)
    cfg = GateConfig(
        line_a_start=[200, 100],
        line_a_end=[0, 100],
        line_b_start=[200, 200],
        line_b_end=[0, 200],
    )
    gate = DualLineGate.from_config(cfg)

    # Frame 1: Vehicle above Line A (center at y=50)
    d1 = sv.Detections(
        xyxy=np.array([[40, 40, 60, 60]], dtype=np.float32),
        tracker_id=np.array([1], dtype=int),
    )
    gate.trigger(d1)
    assert gate.track_states[1] == GateState.OUTSIDE
    assert gate.in_count == 0

    # Frame 2: Vehicle crosses Line A into buffer zone (center at y=150)
    d2 = sv.Detections(
        xyxy=np.array([[40, 140, 60, 160]], dtype=np.float32),
        tracker_id=np.array([1], dtype=int),
    )
    gate.trigger(d2)
    assert gate.track_states[1] == GateState.ENTERED_A
    assert gate.in_count == 0

    # Frame 3: Vehicle crosses Line B out of gate zone (center at y=250)
    d3 = sv.Detections(
        xyxy=np.array([[40, 240, 60, 260]], dtype=np.float32),
        tracker_id=np.array([1], dtype=int),
    )
    gate.trigger(d3)
    assert gate.track_states[1] == GateState.COUNTED
    assert gate.in_count == 1
    assert gate.out_count == 0

