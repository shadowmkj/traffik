import os
import sys
import argparse
import cv2
import numpy as np
import supervision as sv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive Dual-Line Gate Setup Tool (Click 4 points to define Line A and Line B)."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="clip_4.mp4" if os.path.exists("clip_4.mp4") else ("traffic.mp4" if os.path.exists("traffic.mp4") else "clip.mp4"),
        help="Path to source video file (e.g. clip_4.mp4). Default: clip_4.mp4"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_video_path = args.source

    if not os.path.exists(source_video_path):
        print(f"Error: Video file '{source_video_path}' not found.")
        return

    # Load first frame of source video
    generator = sv.get_video_frames_generator(source_path=source_video_path)
    base_frame = next(generator)
    h, w, _ = base_frame.shape

    print("==================================================")
    print(f"Loaded frame from '{source_video_path}' ({w}x{h})")
    print("--------------------------------------------------")
    print("INSTRUCTIONS:")
    print(" 1. Click 2 points for Line A (Entry Line - Cyan)")
    print(" 2. Click 2 points for Line B (Exit Line  - Orange)")
    print(" • Press 'r' to reset and redraw points")
    print(" • Press 'q' or 'ESC' to exit when done")
    print("==================================================\n")

    points = []
    display_frame = base_frame.copy()

    def redraw():
        nonlocal display_frame
        display_frame = base_frame.copy()

        # Instructions banner at top of window
        step_text = ""
        if len(points) == 0:
            step_text = "Step 1/4: Click START for Line A (Entry Line)"
        elif len(points) == 1:
            step_text = "Step 2/4: Click END for Line A (Entry Line)"
        elif len(points) == 2:
            step_text = "Step 3/4: Click START for Line B (Exit Line)"
        elif len(points) == 3:
            step_text = "Step 4/4: Click END for Line B (Exit Line)"
        else:
            step_text = "Gate Defined! Press 'q' or ESC to finish (or 'r' to reset)"

        # Draw banner HUD
        cv2.rectangle(display_frame, (0, 0), (w, 60), (30, 30, 30), -1)
        cv2.putText(display_frame, step_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw Line A if points available
        if len(points) >= 1:
            cv2.circle(display_frame, points[0], 6, (255, 255, 0), -1)
            cv2.putText(display_frame, "A1", (points[0][0] + 10, points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if len(points) >= 2:
            cv2.circle(display_frame, points[1], 6, (255, 255, 0), -1)
            cv2.putText(display_frame, "A2", (points[1][0] + 10, points[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.line(display_frame, points[0], points[1], (255, 255, 0), 3)

        # Draw Line B if points available
        if len(points) >= 3:
            cv2.circle(display_frame, points[2], 6, (0, 140, 255), -1)
            cv2.putText(display_frame, "B1", (points[2][0] + 10, points[2][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
        if len(points) >= 4:
            cv2.circle(display_frame, points[3], 6, (0, 140, 255), -1)
            cv2.putText(display_frame, "B2", (points[3][0] + 10, points[3][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
            cv2.line(display_frame, points[2], points[3], (0, 140, 255), 3)

            # Shade the buffer zone between Line A and Line B
            poly = np.array([points[0], points[1], points[3], points[2]], dtype=np.int32)
            overlay = display_frame.copy()
            cv2.fillPoly(overlay, [poly], (0, 200, 100))
            cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))
                print(f"Point {len(points)} recorded: ({x}, {y})")
                redraw()

                if len(points) == 4:
                    print("\n=======================================================")
                    print("COPY THESE COORDINATES INTO stream.py OR main.py:")
                    print("-------------------------------------------------------")
                    print(f"LINE_A_START = sv.Point({points[0][0]}, {points[0][1]})")
                    print(f"LINE_A_END   = sv.Point({points[1][0]}, {points[1][1]})")
                    print(f"LINE_B_START = sv.Point({points[2][0]}, {points[2][1]})")
                    print(f"LINE_B_END   = sv.Point({points[3][0]}, {points[3][1]})")
                    print("=======================================================\n")

    window_name = f"Dual-Line Gate Setup - {os.path.basename(source_video_path)}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    redraw()

    while True:
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(50) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('r'):
            points.clear()
            print("Points reset. Ready to redraw.")
            redraw()

    # Save a static preview image
    if len(points) == 4:
        cv2.imwrite("gate_preview.jpg", display_frame)
        print("Saved preview image to 'gate_preview.jpg'.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
