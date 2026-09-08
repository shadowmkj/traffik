"""Virtual gate counting subsystem for Traffik.

Implements a robust two-line virtual counting gate (Line A and Line B) with a
spatial buffer zone. State machine tracking prevents duplicate counts caused by
detection flickering, partial occlusions, or boundary jitter.
"""

from collections import defaultdict
from enum import Enum
from typing import List, Optional
import numpy as np
import supervision as sv
import supervision.detection.utils.internal
import supervision.detection.line_zone
from traffik.config import GateConfig


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
# Virtual Gate State Machine
# ==============================================================================

class GateState(Enum):
    """Lifecycle state of a tracked vehicle relative to the virtual gate."""
    OUTSIDE = 0    # Vehicle has not entered the gate zone
    ENTERED_A = 1  # Crossed Line A first (moving towards Line B / inward)
    ENTERED_B = 2  # Crossed Line B first (moving towards Line A / outward)
    COUNTED = 3    # Vehicle has completed crossing and count was registered


# ==============================================================================
# Dual-Line Virtual Gate Engine
# ==============================================================================

class DualLineGate:
    """Two-line virtual gate (Line A & Line B) with directional state tracking.

    Unlike single-line counters that can count oscillating vehicles multiple times,
    the dual-line architecture creates a spatial hysteresis buffer:
    - Counts IN:  Crossed Line A -> then Line B
    - Counts OUT: Crossed Line B -> then Line A
    - Fallback:   Tracks originating within the buffer zone that cross an exit line
                  are counted once and transitioned to COUNTED.
    """

    def __init__(
        self,
        line_a_start: sv.Point,
        line_a_end: sv.Point,
        line_b_start: sv.Point,
        line_b_end: sv.Point,
        triggering_anchors: Optional[List[sv.Position]] = None,
    ) -> None:
        """Initialize dual line zones and tracking state registry.

        Args:
            line_a_start: Starting coordinate for boundary Line A.
            line_a_end: Ending coordinate for boundary Line A.
            line_b_start: Starting coordinate for boundary Line B.
            line_b_end: Ending coordinate for boundary Line B.
            triggering_anchors: Bounding box anchor points used for crossing
                detection. Defaults to [BOTTOM_CENTER, CENTER].
        """
        anchors = (
            triggering_anchors
            if triggering_anchors is not None
            else [sv.Position.BOTTOM_CENTER, sv.Position.CENTER]
        )
        self.line_a = sv.LineZone(
            start=line_a_start,
            end=line_a_end,
            triggering_anchors=anchors,
        )
        self.line_b = sv.LineZone(
            start=line_b_start,
            end=line_b_end,
            triggering_anchors=anchors,
        )
        self.track_states: defaultdict[int, GateState] = defaultdict(lambda: GateState.OUTSIDE)
        self.in_count: int = 0
        self.out_count: int = 0

    @classmethod
    def from_config(cls, config: GateConfig) -> "DualLineGate":
        """Instantiate a DualLineGate from a GateConfig object.

        Args:
            config: GateConfig containing coordinate pairs for Line A and Line B.

        Returns:
            Configured DualLineGate instance.
        """
        return cls(
            line_a_start=sv.Point(*config.line_a_start),
            line_a_end=sv.Point(*config.line_a_end),
            line_b_start=sv.Point(*config.line_b_start),
            line_b_end=sv.Point(*config.line_b_end),
        )

    def trigger(self, detections: sv.Detections) -> None:
        """Update virtual line crossings and execute state machine transitions.

        Processes detection movements across Line A and Line B, updating internal
        directional counts and vehicle lifecycle states.

        Args:
            detections: Current frame detections with assigned tracker_ids.
        """
        if detections.tracker_id is None or len(detections) == 0:
            return

        crossed_a_in, crossed_a_out = self.line_a.trigger(detections)
        crossed_b_in, crossed_b_out = self.line_b.trigger(detections)

        for idx, tracker_id in enumerate(detections.tracker_id):
            state = self.track_states[tracker_id]
            if state == GateState.COUNTED:
                continue

            # State transitions across the dual-line virtual gate
            if state == GateState.OUTSIDE:
                if crossed_a_in[idx]:
                    self.track_states[tracker_id] = GateState.ENTERED_A
                elif crossed_b_out[idx]:
                    self.track_states[tracker_id] = GateState.ENTERED_B
                elif crossed_b_in[idx]:
                    # Fallback for tracks originating inside gate moving IN
                    self.in_count += 1
                    self.track_states[tracker_id] = GateState.COUNTED
                elif crossed_a_out[idx]:
                    # Fallback for tracks originating inside gate moving OUT
                    self.out_count += 1
                    self.track_states[tracker_id] = GateState.COUNTED

            elif state == GateState.ENTERED_A:
                if crossed_b_in[idx] or crossed_b_out[idx]:
                    self.in_count += 1
                    self.track_states[tracker_id] = GateState.COUNTED

            elif state == GateState.ENTERED_B:
                if crossed_a_in[idx] or crossed_a_out[idx]:
                    self.out_count += 1
                    self.track_states[tracker_id] = GateState.COUNTED
