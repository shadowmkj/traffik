"""Interactive Dual-Line Gate Setup Tool.

Click 4 points to define entry line (Line A) and exit line (Line B) on a video frame.

Controls (works both in GUI window and Terminal):
  • Left Mouse Click : Select point coordinates (A1 -> A2 -> B1 -> B2)
  • 'f' / 'F'        : Toggle Fullscreen / Windowed mode
  • 'u' / Backspace  : Undo last clicked point
  • 'r' / 'R'        : Reset all points
  • 'q' / ESC / Enter: Save coordinates and exit
"""

import argparse
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
    """Context manager to enable non-blocking single-keypress reads from the terminal.

    Allows keys pressed in terminal to register even if the GUI window lost focus.
    """
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
# CLI Argument Parsing
# ==============================================================================

def parse_args(argv=None) -> argparse.Namespace:
    """Parse command line arguments for gate calibration."""
    parser = argparse.ArgumentParser(
        description="Interactive Dual-Line Gate Setup Tool (Click 4 points to define Line A and Line B)."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="clip_4.mp4"
        if os.path.exists("clip_4.mp4")
        else ("traffic.mp4" if os.path.exists("traffic.mp4") else "clip.mp4"),
        help="Path to source video file (e.g. clip_4.mp4). Default: clip_4.mp4",
    )
    parser.add_argument(
        "--no-fullscreen",
        action="store_true",
        help="Start in windowed mode instead of fullscreen (can still be toggled with 'f')",
    )
    return parser.parse_args(argv)


# ==============================================================================
# Gate Calibration Engine
# ==============================================================================

def run_setup_gate(
    source_video_path: str,
    fullscreen: bool = True,
    preview_output_path: str = "gate_preview.jpg",
) -> Optional[dict]:
    """Run interactive 4-point gate calibration on the first frame of a video.

    Parameters
    ----------
    source_video_path : str
        Path to source video file.
    fullscreen : bool
        Whether to open the window in fullscreen mode by default (default: True).
    preview_output_path : str
        Filepath to save the calibrated gate preview image.

    Returns
    -------
    Optional[dict]
        Dictionary with coordinates if 4 points were selected, None otherwise.
    """
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

    print("==================================================")
    print(f"Dual-Line Gate Calibration: '{source_video_path}' ({w}x{h})")
    print("--------------------------------------------------")
    print("INSTRUCTIONS:")
    print(" 1. Click 2 points for Line A (Entry Line - Cyan: A1 -> A2)")
    print(" 2. Click 2 points for Line B (Exit Line  - Orange: B1 -> B2)")
    print(" CONTROLS (GUI or Terminal):")
    print("  • 'f' / 'F'        : Toggle Fullscreen / Windowed")
    print("  • 'u' / Backspace  : Undo last point")
    print("  • 'r' / 'R'        : Reset points")
    print("  • 'q' / ESC / Enter: Save and exit")
    print("==================================================\n")

    points: List[Tuple[int, int]] = []
    display_frame = base_frame.copy()
    is_fullscreen = bool(fullscreen)

    def redraw() -> None:
        nonlocal display_frame
        display_frame = base_frame.copy()

        # Step instruction text for top banner
        if len(points) == 0:
            step_text = "Step 1/4: Click START for Line A (Entry Line - A1)"
            step_color = (255, 255, 0)
        elif len(points) == 1:
            step_text = "Step 2/4: Click END for Line A (Entry Line - A2)"
            step_color = (255, 255, 0)
        elif len(points) == 2:
            step_text = "Step 3/4: Click START for Line B (Exit Line - B1)"
            step_color = (0, 165, 255)
        elif len(points) == 3:
            step_text = "Step 4/4: Click END for Line B (Exit Line - B2)"
            step_color = (0, 165, 255)
        else:
            step_text = "Gate Defined! Press [Q / ESC / Enter] to Save & Exit (or 'r' to reset)"
            step_color = (0, 255, 120)

        # Top Banner HUD
        cv2.rectangle(display_frame, (0, 0), (w, 55), (20, 20, 20), -1)
        cv2.putText(
            display_frame,
            step_text,
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            step_color,
            2,
            cv2.LINE_AA,
        )

        # Bottom Controls HUD
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

        # Draw Line A (Points 0 & 1)
        if len(points) >= 1:
            cv2.circle(display_frame, points[0], 7, (255, 255, 0), -1)
            cv2.circle(display_frame, points[0], 9, (0, 0, 0), 2)
            cv2.putText(
                display_frame,
                "A1 (Entry Start)",
                (points[0][0] + 12, points[0][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
        if len(points) >= 2:
            cv2.circle(display_frame, points[1], 7, (255, 255, 0), -1)
            cv2.circle(display_frame, points[1], 9, (0, 0, 0), 2)
            cv2.putText(
                display_frame,
                "A2 (Entry End)",
                (points[1][0] + 12, points[1][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.line(display_frame, points[0], points[1], (255, 255, 0), 3, cv2.LINE_AA)

        # Draw Line B (Points 2 & 3)
        if len(points) >= 3:
            cv2.circle(display_frame, points[2], 7, (0, 165, 255), -1)
            cv2.circle(display_frame, points[2], 9, (0, 0, 0), 2)
            cv2.putText(
                display_frame,
                "B1 (Exit Start)",
                (points[2][0] + 12, points[2][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
        if len(points) >= 4:
            cv2.circle(display_frame, points[3], 7, (0, 165, 255), -1)
            cv2.circle(display_frame, points[3], 9, (0, 0, 0), 2)
            cv2.putText(
                display_frame,
                "B2 (Exit End)",
                (points[3][0] + 12, points[3][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.line(display_frame, points[2], points[3], (0, 165, 255), 3, cv2.LINE_AA)

            # Shade the buffer zone between Line A and Line B
            poly = np.array([points[0], points[1], points[3], points[2]], dtype=np.int32)
            overlay = display_frame.copy()
            cv2.fillPoly(overlay, [poly], (0, 220, 100))
            cv2.addWeighted(overlay, 0.28, display_frame, 0.72, 0, display_frame)

    def mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))
                print(f"Point {len(points)} recorded: ({x}, {y})")
                redraw()

                if len(points) == 4:
                    print("\n=======================================================")
                    print("COPY THESE COORDINATES INTO configs/default.toml:")
                    print("-------------------------------------------------------")
                    print(f"line_a_start = [{points[0][0]}, {points[0][1]}]")
                    print(f"line_a_end   = [{points[1][0]}, {points[1][1]}]")
                    print(f"line_b_start = [{points[2][0]}, {points[2][1]}]")
                    print(f"line_b_end   = [{points[3][0]}, {points[3][1]}]")
                    print("=======================================================\n")

    window_name = f"Dual-Line Gate Setup - {os.path.basename(source_video_path)}"
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
                print("Points reset. Ready to redraw from Step 1.")
                redraw()

            # Undo: 'u', 'U', 'z', 'Z', Backspace (8 or 127)
            elif char in ("u", "z") or key in (8, 127):
                if points:
                    undone = points.pop()
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

    result = None
    if len(points) == 4:
        cv2.imwrite(preview_output_path, display_frame)
        print(f"Saved gate preview image to '{preview_output_path}'.")
        result = {
            "line_a_start": [points[0][0], points[0][1]],
            "line_a_end": [points[1][0], points[1][1]],
            "line_b_start": [points[2][0], points[2][1]],
            "line_b_end": [points[3][0], points[3][1]],
        }

    cv2.destroyAllWindows()
    return result


def main(argv=None) -> None:
    """CLI entrypoint for standalone gate setup."""
    args = parse_args(argv)
    if args.no_fullscreen:
        run_setup_gate(args.source, fullscreen=False)
    else:
        run_setup_gate(args.source)


if __name__ == "__main__":
    main()
