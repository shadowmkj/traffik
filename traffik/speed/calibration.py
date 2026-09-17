import contextlib
import os
import select
import subprocess
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv


# ==============================================================================
# Terminal & Window Focus Helpers
# ==============================================================================

def activate_window_focus() -> None:
    """Bring the OpenCV GUI window / Python process to the front on macOS."""
    if sys.platform == "darwin":
        try:
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    f'tell application "System Events" to set frontmost of first process whose unix id is {os.getpid()} to true',
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1,
            )
        except Exception:
            pass


@contextlib.contextmanager
def raw_terminal_input():
    """Context manager to enable non-blocking single-keypress reads from the terminal."""
    is_tty = False
    old_settings = None
    fd = None

    try:
        if hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            is_tty = True
    except Exception:
        is_tty = False

    def poll_terminal_char() -> Optional[str]:
        if not is_tty or fd is None:
            return None
        try:
            rlist, _, _ = select.select([sys.stdin], [], [], 0)
            if rlist:
                return sys.stdin.read(1)
        except Exception:
            return None
        return None

    try:
        yield poll_terminal_char
    finally:
        if is_tty and fd is not None and old_settings is not None:
            try:
                import termios
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass


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
    fullscreen: bool = True,
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
    fullscreen : bool
        Whether to start in fullscreen mode (default: True).

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
    print(" CONTROLS:")
    print("  • 'f' / 'F'        : Toggle Fullscreen / Windowed")
    print("  • 'u' / Backspace  : Undo last point")
    print("  • 'r' / 'R'        : Reset points")
    print("  • 'q' / ESC / Enter: Save and exit")
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
    is_fullscreen = bool(fullscreen)

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
            step_text = "Speed ROI Defined! Press [Q / ESC / Enter] to Save & Exit (or 'r' to reset)"

        # Draw top banner HUD
        cv2.rectangle(display_frame, (0, 0), (w, 55), (20, 20, 20), -1)
        cv2.putText(
            display_frame,
            step_text,
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Draw bottom controls HUD
        cv2.rectangle(display_frame, (0, h - 40), (w, h), (20, 20, 20), -1)
        controls_text = "[F] Fullscreen  |  [U/Bksp] Undo  |  [R] Reset  |  [Q/ESC/Enter] Save & Exit"
        cv2.putText(
            display_frame,
            controls_text,
            (20, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
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

    def set_win_fullscreen(enabled: bool) -> None:
        try:
            prop = cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)
            if not enabled:
                cv2.resizeWindow(window_name, min(1280, w), min(720, h))
        except Exception:
            pass

    set_win_fullscreen(is_fullscreen)
    cv2.setMouseCallback(window_name, mouse_callback)

    redraw()
    cv2.imshow(window_name, display_frame)

    # Bring window to front on macOS
    activate_window_focus()

    with raw_terminal_input() as get_term_char:
        while True:
            cv2.imshow(window_name, display_frame)

            # Detect if window was closed with OS window close button
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Setup window closed.")
                    break
            except Exception:
                pass

            raw_key = cv2.waitKey(20)
            term_char = get_term_char()

            char = ""
            key = -1

            if raw_key != -1:
                key = raw_key & 0xFF
                char = chr(key).lower() if (0 <= key < 128) else ""
            elif term_char is not None:
                key = ord(term_char)
                char = term_char.lower()

            if key == -1 and not char:
                continue

            # Exit / Save: 'q', 'Q', ESC (27), Enter (13 or 10), Space (32)
            if key in (27, 13, 10, 32) or char == "q":
                break

            # Reset: 'r', 'R'
            elif char == "r":
                points.clear()
                generated_toml = None
                print("Points reset. Ready to redraw.")
                redraw()

            # Undo: 'u', 'U', 'z', 'Z', Backspace (8 or 127)
            elif char in ("u", "z") or key in (8, 127):
                if points:
                    undone = points.pop()
                    generated_toml = None
                    print(f"Undid point {len(points) + 1} {undone}. Remaining: {len(points)}/4")
                    redraw()
                else:
                    print("No points to undo.")

            # Toggle Fullscreen: 'f', 'F'
            elif char == "f":
                is_fullscreen = not is_fullscreen
                set_win_fullscreen(is_fullscreen)
                if is_fullscreen:
                    print("Switched to FULLSCREEN mode.")
                else:
                    print("Switched to WINDOWED mode.")
                redraw()

    # Save preview image if 4 points were selected
    if len(points) == 4:
        cv2.imwrite(preview_output_path, display_frame)
        print(f"Saved preview image to '{preview_output_path}'.")

    cv2.destroyAllWindows()
    return generated_toml


