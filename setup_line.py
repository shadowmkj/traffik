import cv2
import supervision as sv

SOURCE_VIDEO_PATH = "clip.mp4"

def main():
    # Load first frame of source video
    generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)
    frame = next(generator)
    h, w, _ = frame.shape

    print(f"Loaded frame from {SOURCE_VIDEO_PATH} (Resolution: {w}x{h})")

    points = []
    frame_copy = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal frame_copy
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Point {len(points)} selected: ({x}, {y})")

            # Mark selected point
            cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)

            # Once 2 points are selected, draw the line and output python snippet
            if len(points) == 2:
                cv2.line(frame_copy, points[0], points[1], (0, 255, 0), 2)
                print("\n==============================================")
                print("Copy these lines into test.py or main.py:")
                print(f"START = sv.Point({points[0][0]}, {points[0][1]})")
                print(f"END   = sv.Point({points[1][0]}, {points[1][1]})")
                print("==============================================\n")

    window_name = "Select 2 Points for Counting Line (Click Start, then End)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        cv2.imshow(window_name, frame_copy)
        key = cv2.waitKey(100) & 0xFF
        # Exit on 'q', ESC, or once 2 points are set
        if key in (27, ord('q')) or len(points) >= 2:
            cv2.imshow(window_name, frame_copy)
            cv2.waitKey(1500)
            break

    # Also save a static preview image
    if len(points) == 2:
        line_zone = sv.LineZone(
            start=sv.Point(points[0][0], points[0][1]),
            end=sv.Point(points[1][0], points[1][1])
        )
    else:
        line_zone = sv.LineZone(
            start=sv.Point(0, h // 4),
            end=sv.Point(w, h // 4)
        )
    annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    annotated = annotator.annotate(frame.copy(), line_counter=line_zone)
    cv2.imwrite("line_preview.jpg", annotated)
    print("Saved preview image to 'line_preview.jpg'.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
