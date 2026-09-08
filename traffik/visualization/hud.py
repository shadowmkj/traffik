"""HUD banner overlay and visual annotation components for video streams."""

from typing import List, Optional
import cv2
import numpy as np
import supervision as sv
from traffik.counting.gate import DualLineGate


# ==============================================================================
# HUD Banner Rendering
# ==============================================================================

def draw_hud_banner(frame: np.ndarray, in_count: int, out_count: int) -> np.ndarray:
    """Draw a high-visibility translucent HUD banner at the top of the frame.

    Renders a rounded or bordered dark box with vibrant text indicating cumulative
    IN and OUT counts for clear situational awareness during stream playback.

    Args:
        frame: BGR image frame as a NumPy array.
        in_count: Number of vehicles detected entering the monitored zone.
        out_count: Number of vehicles detected exiting the monitored zone.

    Returns:
        Frame with overlaid HUD banner.
    """
    banner_text = f"GATE COUNT | IN: {in_count}   OUT: {out_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(banner_text, font, font_scale, thickness)

    # Calculate bounding box coordinates with padding
    x1, y1 = 20, 20
    x2, y2 = x1 + text_w + 24, y1 + text_h + 20

    # Create translucent dark overlay behind text to maintain readability on busy scenes
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 120), 2)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Draw crisp anti-aliased text
    cv2.putText(
        frame,
        banner_text,
        (x1 + 12, y1 + text_h + 8),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return frame


# ==============================================================================
# Multi-layer Visual Annotator
# ==============================================================================

class VisualAnnotator:
    """Coordinates bounding box, label, gate line, and HUD banner rendering.

    Combines Supervision's BoxAnnotator, LabelAnnotator, and LineZoneAnnotator
    with custom color schemes:
    - Line A: Cyan (0, 220, 255)
    - Line B: Orange (255, 140, 0)
    """

    def __init__(self):
        """Initialize the annotators with calibrated line colors and styling."""
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        self.line_a_annotator = sv.LineZoneAnnotator(
            thickness=2,
            color=sv.Color(r=0, g=220, b=255),
            text_thickness=1,
            text_scale=0.5,
        )
        self.line_b_annotator = sv.LineZoneAnnotator(
            thickness=2,
            color=sv.Color(r=255, g=140, b=0),
            text_thickness=1,
            text_scale=0.5,
        )

    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        labels: Optional[List[str]] = None,
        gate: Optional[DualLineGate] = None,
    ) -> np.ndarray:
        """Annotate a frame with detections, labels, gate lines, and HUD.

        Args:
            frame: Input video frame (BGR).
            detections: Supervision detections containing bounding boxes.
            labels: Optional formatted strings for vehicle class and track ID.
            gate: Optional DualLineGate to draw Line A, Line B, and counter HUD.

        Returns:
            Annotated frame.
        """
        annotated = self.box_annotator.annotate(scene=frame, detections=detections)
        if labels:
            annotated = self.label_annotator.annotate(
                scene=annotated, detections=detections, labels=labels
            )
        if gate:
            annotated = self.line_a_annotator.annotate(
                annotated, line_counter=gate.line_a
            )
            annotated = self.line_b_annotator.annotate(
                annotated, line_counter=gate.line_b
            )
            annotated = draw_hud_banner(annotated, gate.in_count, gate.out_count)
        return annotated
