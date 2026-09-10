"""Interactive 4-point speed ROI calibration module for perspective homography."""

import os
from typing import List, Optional, Tuple
import cv2
import numpy as np
import supervision as sv


# ==============================================================================
# TOML Block Formatting
# ==============================================================================

def format_speed_toml_block(
    points: List[List[int]],
    target_width: float = 7.5,
    target_length: float = 25.0,
    unit: str = "km/h",
) -> str:
    """Format a copy-pasteable TOML configuration block for speed estimation.

    Parameters
    ----------
    points : List[List[int]]
        4 corner pixel coordinates in order: [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
    target_width : float
        Physical width of the ROI in meters across the road.
    target_length : float
        Physical length of the ROI in meters along the road.
    unit : str
        Measurement unit ('km/h' or 'mph').

    Returns
    -------
    str
        Formatted TOML block suitable for pasting into configs/default.toml.
    """
    pts_int = [[int(pt[0]), int(pt[1])] for pt in points]
    formatted_pts = ", ".join(f"[{p[0]}, {p[1]}]" for p in pts_int)
    return (
        f"[speed]\n"
        f"enabled = true\n"
        f'unit = "{unit}"\n'
        f"source_polygon = [{formatted_pts}]\n"
        f"target_width = {float(target_width)}\n"
        f"target_length = {float(target_length)}\n"
        f"smoothing_window = 7"
    )


# ==============================================================================
# Interactive OpenCV Calibration GUI
# ==============================================================================

def run_setup_speed_roi(
    source_video_path: str,
    target_width: float = 7.5,
    target_length: float = 25.0,
    unit: str = "km/h",
    preview_output_path: str = "speed_roi_preview.jpg",
) -> Optional[str]:
    """Run interactive 4-point speed ROI calibration on the first frame of a video.

    Parameters
    ----------
    source_video_path : str
        Path to source video file.
    target_width : float
        Real-world metric width in meters (default: 7.5m).
    target_length : float
        Real-world metric length in meters (default: 25.0m).
    unit : str
        Speed unit ('km/h' or 'mph', default: 'km/h').
    preview_output_path : str
        Filepath to save the calibrated ROI preview image.

    Returns
    -------
    Optional[str]
        Formatted TOML string if 4 points were selected, None otherwise.
    """
    # Verify file existence before opening video streams
    if not os.path.exists(source_video_path):
        print(f"Error: Video file '{source_video_path}' not found.")
        return None

    # Load first frame of source video
    generator = sv.get_video_frames_generator(source_path=source_video_path)
    try:
        base_frame = next(generator)
    except StopIteration:
        print(f"Error: Could not read frames from '{source_video_path}'.")
        return None

    h, w, _ = base_frame.shape

    # Display console instructions and metadata
    print("==================================================")
    print(f"Speed ROI Calibration: '{source_video_path}' ({w}x{h})")
    print(f"Target Dimensions: {target_width}m (width) x {target_length}m (length), Unit: {unit}")
    print("--------------------------------------------------")
    print("INSTRUCTIONS:")
    print(" Click 4 points in clockwise order on the road surface:")
    print("  1. Top-Left     (Far Left road boundary / lane line)")
    print("  2. Top-Right    (Far Right road boundary / lane line)")
    print("  3. Bottom-Right (Near Right road boundary / lane line)")
    print("  4. Bottom-Left  (Near Left road boundary / lane line)")
    print(" Controls:")
    print("  • Press 'r' to reset points")
    print("  • Press 'q' or 'ESC' to exit when done")
    print("==================================================\n")

    point_labels = [
        ("P1: Top-Left (Far Left)", (255, 255, 0)),        # Cyan / Yellow
        ("P2: Top-Right (Far Right)", (255, 255, 0)),      # Cyan / Yellow
        ("P3: Bottom-Right (Near Right)", (0, 165, 255)),  # Orange
        ("P4: Bottom-Left (Near Left)", (0, 165, 255)),    # Orange
    ]

    points: List[Tuple[int, int]] = []
    display_frame = base_frame.copy()
    generated_toml: Optional[str] = None

    def redraw() -> None:
        nonlocal display_frame
        display_frame = base_frame.copy()

        # Step instruction text for top banner
        if len(points) == 0:
            step_text = "Step 1/4: Click Top-Left (Far Left)"
        elif len(points) == 1:
            step_text = "Step 2/4: Click Top-Right (Far Right)"
        elif len(points) == 2:
            step_text = "Step 3/4: Click Bottom-Right (Near Right)"
        elif len(points) == 3:
            step_text = "Step 4/4: Click Bottom-Left (Near Left)"
        else:
            step_text = "Speed ROI Defined! Press 'q' or ESC to finish (or 'r' to reset)"

        # Draw top banner HUD
        cv2.rectangle(display_frame, (0, 0), (w, 60), (30, 30, 30), -1)
        cv2.putText(
            display_frame,
            step_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Draw points and connecting lines
        for idx, pt in enumerate(points):
            label_text, color = point_labels[idx]
            cv2.circle(display_frame, pt, 7, color, -1)
            cv2.circle(display_frame, pt, 9, (0, 0, 0), 2)
            cv2.putText(
                display_frame,
                label_text,
                (pt[0] + 12, pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        # Draw line P1 -> P2 (Far road boundary)
        if len(points) >= 2:
            cv2.line(display_frame, points[0], points[1], (255, 255, 0), 2, cv2.LINE_AA)

        # Draw line P2 -> P3 (Right road boundary)
        if len(points) >= 3:
            cv2.line(display_frame, points[1], points[2], (0, 200, 255), 2, cv2.LINE_AA)

        # Draw line P3 -> P4 and P4 -> P1 (Near road boundary and Left road boundary)
        if len(points) == 4:
            cv2.line(display_frame, points[2], points[3], (0, 165, 255), 2, cv2.LINE_AA)
            cv2.line(display_frame, points[3], points[0], (0, 200, 255), 2, cv2.LINE_AA)

            # Draw translucent green ROI fill overlay
            poly = np.array(points, dtype=np.int32)
            overlay = display_frame.copy()
            cv2.fillPoly(overlay, [poly], (0, 220, 100))
            cv2.addWeighted(overlay, 0.25, display_frame, 0.75, 0, display_frame)

    def mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal generated_toml
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))
                print(f"Point {len(points)} recorded: ({x}, {y})")
                redraw()

                if len(points) == 4:
                    pts_list = [[p[0], p[1]] for p in points]
                    generated_toml = format_speed_toml_block(
                        points=pts_list,
                        target_width=target_width,
                        target_length=target_length,
                        unit=unit,
                    )
                    print("\n=======================================================")
                    print("COPY THIS CONFIGURATION INTO configs/default.toml:")
                    print("-------------------------------------------------------")
                    print(generated_toml)
                    print("=======================================================\n")

    window_name = f"Speed ROI Setup - {os.path.basename(source_video_path)}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    redraw()

    while True:
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(50) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("r"):
            points.clear()
            generated_toml = None
            print("Points reset. Ready to redraw.")
            redraw()

    # Save preview image if 4 points were selected
    if len(points) == 4:
        cv2.imwrite(preview_output_path, display_frame)
        print(f"Saved preview image to '{preview_output_path}'.")

    cv2.destroyAllWindows()
    return generated_toml
