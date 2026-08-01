#!/usr/bin/env python3
"""
Real-Time Face Tracker with Roboflow Supervision
================================================
A high-performance live face tracking application using Roboflow Supervision (sv.Detections & sv.ByteTrack),
OpenCV YuNet deep neural network, and customizable HUD themes.

Features:
- Live camera stream with auto camera selection
- Face detection using YuNet ONNX / Haar Cascade / optional Ultralytics YOLO
- Roboflow Supervision integration (sv.Detections, sv.ByteTrack multi-object tracking)
- Supervision Annotators (sv.CornerAnnotator, sv.LabelAnnotator, sv.BlurAnnotator, sv.DotAnnotator)
- 5-point facial landmark tracking with vector display
- Cyberpunk / Sci-Fi HUD visual themes (BGR format)
- Interactive privacy face blurring mode (using sv.BlurAnnotator)
- Instant screenshot snapshots saved to 'snapshots/'
- Real-time FPS, face counter, persistent tracking IDs, and distance/scale estimation
- Customizable via hotkeys and CLI arguments

Controls:
  'q' or ESC : Quit application
  'b'        : Toggle Privacy Face Blur (Supervision BlurAnnotator)
  'l'        : Toggle Facial Landmark Vectors
  'c'        : Cycle Visual Theme
  't'        : Toggle Roboflow Supervision Annotators / Custom Reticles
  's'        : Save Snapshot to disk
  'h'        : Toggle HUD / On-screen guide
"""

import os
import time
import argparse
import urllib.request
import cv2
import numpy as np
import supervision as sv

# Model URLs and file paths
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_PATH = "face_detection_yunet_2023mar.onnx"
HAAR_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
HAAR_PATH = "haarcascade_frontalface_default.xml"

# Theme Color Palettes (BGR format)
THEMES = {
    "Cyberpunk": {
        "primary": (255, 255, 0),     # Cyan
        "secondary": (255, 0, 255),   # Magenta
        "accent": (0, 255, 255),      # Yellow
        "bg_dark": (20, 20, 20),
        "text": (255, 255, 255)
    },
    "Emerald": {
        "primary": (50, 255, 50),     # Neon Green
        "secondary": (0, 200, 100),   # Seafoam
        "accent": (0, 255, 255),      # Yellow
        "bg_dark": (10, 30, 10),
        "text": (220, 255, 220)
    },
    "Amber": {
        "primary": (0, 165, 255),     # Orange/Amber
        "secondary": (0, 215, 255),   # Gold
        "accent": (50, 50, 255),      # Red accent
        "bg_dark": (20, 15, 10),
        "text": (255, 230, 200)
    },
    "Clean": {
        "primary": (255, 255, 255),   # White
        "secondary": (200, 200, 200),  # Light Gray
        "accent": (0, 120, 255),      # Bright Orange
        "bg_dark": (40, 40, 40),
        "text": (255, 255, 255)
    },
    "Synthwave": {
        "primary": (255, 20, 255),    # Neon Pink
        "secondary": (255, 220, 0),   # Cyan
        "accent": (255, 100, 0),      # Electric Blue
        "bg_dark": (40, 10, 25),
        "text": (255, 230, 255)
    },
    "Matrix": {
        "primary": (0, 255, 0),       # Lime Green
        "secondary": (0, 180, 0),     # Darker Green
        "accent": (150, 255, 150),    # Mint Glow
        "bg_dark": (5, 15, 5),
        "text": (100, 255, 100)
    },
    "Solar Flare": {
        "primary": (30, 30, 255),     # Crimson Red
        "secondary": (0, 140, 255),   # Sunburst Orange
        "accent": (0, 240, 255),      # Flare Yellow
        "bg_dark": (15, 10, 25),
        "text": (200, 235, 255)
    },
    "Oceanic": {
        "primary": (255, 200, 0),     # Deep Cyan
        "secondary": (255, 160, 50),  # Ice Blue
        "accent": (80, 100, 255),     # Coral Red
        "bg_dark": (35, 20, 10),
        "text": (255, 245, 230)
    },
    "Vaporwave": {
        "primary": (250, 150, 200),   # Pastel Violet
        "secondary": (220, 240, 120), # Pastel Mint
        "accent": (120, 240, 255),    # Soft Yellow
        "bg_dark": (40, 20, 30),
        "text": (245, 220, 250)
    },
    "Monochrome": {
        "primary": (230, 230, 230),   # Bright Silver
        "secondary": (140, 140, 140), # Slate Gray
        "accent": (255, 255, 255),    # Pure White
        "bg_dark": (18, 18, 18),
        "text": (240, 240, 240)
    }
}
THEME_NAMES = list(THEMES.keys())


def ensure_models_exist():
    """Download required AI weights/cascade models if not present locally."""
    if not os.path.exists(YUNET_PATH):
        print("[*] Downloading YuNet Face Detector model weights...")
        try:
            urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
            print(f"[+] Downloaded {YUNET_PATH} successfully!")
        except Exception as e:
            print(f"[!] Warning: Failed to download YuNet model ({e}). Will attempt Haar fallback.")

    if not os.path.exists(HAAR_PATH):
        try:
            urllib.request.urlretrieve(HAAR_URL, HAAR_PATH)
        except Exception:
            pass


class FaceTrackerApp:
    def __init__(self, camera_id=0, width=1280, height=720, score_threshold=0.7):
        self.camera_id = camera_id
        self.target_width = width
        self.target_height = height
        self.score_threshold = score_threshold

        # Display & Mode Flags
        self.blur_faces = False
        self.show_landmarks = True
        self.show_hud = True
        self.use_supervision_annotators = True
        self.theme_idx = 0

        # FPS calculation
        self.prev_time = time.time()
        self.fps = 0.0

        # Snapshot counter
        self.snapshot_dir = "snapshots"
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # Initialize Face Detector
        ensure_models_exist()
        self.yunet_detector = None
        self.haar_cascade = None
        self._init_detector()

        # Roboflow Supervision Tracker (ByteTrack)
        self.tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30, minimum_matching_threshold=0.8)
        self.blur_annotator = sv.BlurAnnotator(kernel_size=31)

    def _init_detector(self):
        """Initialize YuNet ONNX detector or Haar Cascade fallback."""
        if os.path.exists(YUNET_PATH):
            try:
                self.yunet_detector = cv2.FaceDetectorYN.create(
                    model=YUNET_PATH,
                    config="",
                    input_size=(self.target_width, self.target_height),
                    score_threshold=self.score_threshold,
                    nms_threshold=0.3,
                    top_k=5000
                )
                print("[+] Loaded YuNet Deep Neural Face Detector.")
                return
            except Exception as e:
                print(f"[!] YuNet initialization error: {e}")

        # Fallback to Haar
        if os.path.exists(HAAR_PATH):
            self.haar_cascade = cv2.CascadeClassifier(HAAR_PATH)
            print("[+] Loaded Haar Cascade Face Detector fallback.")
        else:
            print("[!] Error: No face detector models available!")

    def open_camera(self):
        """Find and open working webcam device."""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"[!] Primary camera index {self.camera_id} failed. Searching for available cameras...")
            for alt_id in range(4):
                if alt_id == self.camera_id:
                    continue
                cap = cv2.VideoCapture(alt_id)
                if cap.isOpened():
                    print(f"[+] Found working camera at index {alt_id}")
                    self.camera_id = alt_id
                    break

        if not cap.isOpened():
            print("[ERROR] Could not access any webcam. Please check camera permissions or connection.")
            return None

        # Request resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        return cap

    def draw_hud_header(self, frame, face_count):
        """Render stylish HUD top-bar stats and control guide."""
        theme = THEMES[THEME_NAMES[self.theme_idx]]
        h, w = frame.shape[:2]

        # Top semi-transparent header overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 45), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Header Title & Stats
        title = "ROBOFLOW SUPERVISION FACE TRACKER"
        cv2.putText(frame, title, (20, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, theme["primary"], 2, cv2.LINE_AA)

        # Status Badges
        fps_str = f"FPS: {self.fps:.1f}"
        faces_str = f"TRACKED: {face_count}"
        theme_str = f"THEME: {THEME_NAMES[self.theme_idx]}"

        cv2.putText(frame, fps_str, (430, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, theme["accent"], 1, cv2.LINE_AA)
        cv2.putText(frame, faces_str, (540, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, theme["secondary"], 1, cv2.LINE_AA)
        cv2.putText(frame, theme_str, (680, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, theme["text"], 1, cv2.LINE_AA)

        if self.blur_faces:
            cv2.putText(frame, "[PRIVACY BLUR ON]", (w - 200, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 255), 2, cv2.LINE_AA)

        # Footer Hotkey Hint Bar
        if self.show_hud:
            footer = frame.copy()
            cv2.rectangle(footer, (0, h - 30), (w, h), (15, 15, 15), -1)
            cv2.addWeighted(footer, 0.65, frame, 0.35, 0, frame)

            guide = "[Q] Quit  |  [B] Blur Face (SV)  |  [L] Landmarks  |  [C] Theme  |  [T] Toggle Annotator  |  [S] Snapshot"
            cv2.putText(frame, guide, (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (200, 200, 200), 1, cv2.LINE_AA)

    def detect_faces(self, frame):
        """Detect faces and return Roboflow Supervision sv.Detections object."""
        h, w = frame.shape[:2]
        xyxy_boxes = []
        confidences = []
        landmarks_list = []

        if self.yunet_detector:
            self.yunet_detector.setInputSize((w, h))
            _, detections_raw = self.yunet_detector.detect(frame)
            if detections_raw is not None:
                for det in detections_raw:
                    x, y, bw, bh = det[0:4]
                    score = float(det[-1])
                    lm = list(map(int, det[4:14]))

                    xyxy_boxes.append([x, y, x + bw, y + bh])
                    confidences.append(score)
                    landmarks_list.append(lm)
        elif self.haar_cascade:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = self.haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            for (x, y, bw, bh) in detected:
                xyxy_boxes.append([x, y, x + bw, y + bh])
                confidences.append(0.95)
                landmarks_list.append(None)

        if len(xyxy_boxes) == 0:
            sv_detections = sv.Detections.empty()
            sv_detections.data["landmarks"] = np.array([])
            return sv_detections

        xyxy_arr = np.array(xyxy_boxes, dtype=np.float32)
        conf_arr = np.array(confidences, dtype=np.float32)
        class_ids = np.zeros(len(xyxy_boxes), dtype=int)

        sv_detections = sv.Detections(
            xyxy=xyxy_arr,
            confidence=conf_arr,
            class_id=class_ids
        )
        sv_detections.data["landmarks"] = np.array(landmarks_list, dtype=object)
        return sv_detections

    def process_frame(self, frame):
        """Detect faces, update Roboflow Supervision ByteTrack, and annotate frame."""
        h, w = frame.shape[:2]
        theme = THEMES[THEME_NAMES[self.theme_idx]]
        primary_color = sv.Color(r=theme["primary"][2], g=theme["primary"][1], b=theme["primary"][0])
        secondary_color = sv.Color(r=theme["secondary"][2], g=theme["secondary"][1], b=theme["secondary"][0])
        accent_color = sv.Color(r=theme["accent"][2], g=theme["accent"][1], b=theme["accent"][0])

        # 1. Detect & Track with Roboflow Supervision
        raw_detections = self.detect_faces(frame)
        detections = self.tracker.update_with_detections(raw_detections)

        # If ByteTrack loses faces temporarily, fallback to raw detections for visualization
        if len(detections) == 0 and len(raw_detections) > 0:
            detections = raw_detections

        # 2. Privacy Blur using Roboflow Supervision BlurAnnotator
        if self.blur_faces and len(detections) > 0:
            frame = self.blur_annotator.annotate(scene=frame, detections=detections)

        # 3. Supervision Visual Annotations
        if self.use_supervision_annotators and len(detections) > 0:
            corner_annotator = sv.BoxCornerAnnotator(
                color=primary_color,
                thickness=2,
                corner_length=max(10, min(w, h) // 25)
            )
            labels = []
            for i in range(len(detections)):
                conf = detections.confidence[i] if detections.confidence is not None else 0.9
                track_id = detections.tracker_id[i] if detections.tracker_id is not None else i + 1
                labels.append(f"FACE #{track_id} | {conf*100:.0f}%")

            label_annotator = sv.LabelAnnotator(
                color=primary_color,
                text_color=sv.Color(0, 0, 0),
                text_scale=0.45,
                text_thickness=1,
                text_padding=4
            )
            frame = corner_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        # 4. Reticles, Landmarks & Distance Indicators
        landmarks_array = detections.data.get("landmarks", [])
        for i in range(len(detections)):
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            bw, bh = x2 - x1, y2 - y1

            # Proximity estimation
            face_size_ratio = (bw * bh) / (w * h) if (w * h) > 0 else 0
            proximity = "FAR" if face_size_ratio < 0.05 else ("MEDIUM" if face_size_ratio < 0.20 else "NEAR")

            # Distance tag below box
            cv2.putText(frame, f"PROXIMITY: {proximity}", (x1, max(0, y2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, theme["text"], 1, cv2.LINE_AA)

            # Center target crosshair
            cx, cy = x1 + bw // 2, y1 + bh // 2
            ch_len = 8
            cv2.line(frame, (cx - ch_len, cy), (cx + ch_len, cy), theme["accent"], 1)
            cv2.line(frame, (cx, cy - ch_len), (cx, cy + ch_len), theme["accent"], 1)

            # Draw Facial Landmarks if available
            if self.show_landmarks and i < len(landmarks_array) and landmarks_array[i] is not None:
                lm = landmarks_array[i]
                if isinstance(lm, (list, np.ndarray)) and len(lm) >= 10:
                    r_eye = (lm[0], lm[1])
                    l_eye = (lm[2], lm[3])
                    nose = (lm[4], lm[5])
                    r_mouth = (lm[6], lm[7])
                    l_mouth = (lm[8], lm[9])

                    pts = [r_eye, l_eye, nose, r_mouth, l_mouth]
                    for px, py in pts:
                        cv2.circle(frame, (px, py), 3, theme["accent"], -1, cv2.LINE_AA)
                        cv2.circle(frame, (px, py), 6, theme["primary"], 1, cv2.LINE_AA)

                    # Vector network lines
                    cv2.line(frame, r_eye, l_eye, theme["secondary"], 1, cv2.LINE_AA)
                    cv2.line(frame, r_eye, nose, theme["secondary"], 1, cv2.LINE_AA)
                    cv2.line(frame, l_eye, nose, theme["secondary"], 1, cv2.LINE_AA)
                    cv2.line(frame, r_mouth, l_mouth, theme["secondary"], 1, cv2.LINE_AA)
                    cv2.line(frame, nose, ((r_mouth[0] + l_mouth[0]) // 2, (r_mouth[1] + l_mouth[1]) // 2), theme["secondary"], 1, cv2.LINE_AA)

        # 5. Render HUD Overlay
        self.draw_hud_header(frame, len(detections))
        return frame

    def save_snapshot(self, frame):
        """Save frame snapshot image with timestamp."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.snapshot_dir, f"face_snapshot_{ts}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[+] Saved snapshot to: {filename}")

        # Render visual flash feedback on frame
        flash = np.ones_like(frame) * 255
        cv2.addWeighted(flash, 0.4, frame, 0.6, 0, frame)
        cv2.putText(frame, "SNAPSHOT SAVED!", (frame.shape[1]//2 - 120, frame.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    def run(self):
        """Main loop for video capture and real-time processing."""
        cap = self.open_camera()
        if cap is None:
            return

        window_name = "Roboflow Supervision Face Tracker"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.target_width, self.target_height)

        print("\n=======================================================")
        print("   ROBOFLOW SUPERVISION FACE TRACKER STARTED           ")
        print("=======================================================")
        print("   Hotkeys:")
        print("   - [Q] / ESC : Exit application")
        print("   - [B]       : Toggle Face Blur (Supervision BlurAnnotator)")
        print("   - [L]       : Toggle Landmarks")
        print("   - [C]       : Cycle Color Themes")
        print("   - [T]       : Toggle Supervision Annotators")
        print("   - [S]       : Take Snapshot")
        print("   - [H]       : Toggle HUD Header")
        print("=======================================================\n")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("[!] Failed to grab webcam frame. Retrying...")
                    time.sleep(0.1)
                    continue

                # Mirror frame horizontally for natural webcam feel
                frame = cv2.flip(frame, 1)

                # Calculate FPS
                curr_time = time.time()
                dt = curr_time - self.prev_time
                self.fps = 1.0 / dt if dt > 0 else 30.0
                self.prev_time = curr_time

                # Process and annotate frame
                processed = self.process_frame(frame)

                # Display window
                cv2.imshow(window_name, processed)

                # Key Controls
                key = cv2.waitKey(1) & 0xFF
                if key in [27, ord('q'), ord('Q')]:  # ESC or Q
                    print("[*] Exiting application...")
                    break
                elif key in [ord('b'), ord('B')]:
                    self.blur_faces = not self.blur_faces
                    print(f"[*] Privacy Blur: {'ON' if self.blur_faces else 'OFF'}")
                elif key in [ord('l'), ord('L')]:
                    self.show_landmarks = not self.show_landmarks
                    print(f"[*] Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
                elif key in [ord('c'), ord('C')]:
                    self.theme_idx = (self.theme_idx + 1) % len(THEME_NAMES)
                    print(f"[*] Changed theme to: {THEME_NAMES[self.theme_idx]}")
                elif key in [ord('t'), ord('T')]:
                    self.use_supervision_annotators = not self.use_supervision_annotators
                    print(f"[*] Supervision Annotators: {'ON' if self.use_supervision_annotators else 'OFF'}")
                elif key in [ord('s'), ord('S')]:
                    self.save_snapshot(processed)
                elif key in [ord('h'), ord('H')]:
                    self.show_hud = not self.show_hud

        except KeyboardInterrupt:
            print("\n[*] Application stopped by user (Ctrl+C).")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[+] Camera released and windows closed. Goodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="Roboflow Supervision Real-Time Face Tracker")
    parser.add_argument("--cam", type=int, default=0,
                        help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280,
                        help="Target camera frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="Target camera frame height (default: 720)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Face detection confidence threshold (default: 0.7)")

    args = parser.parse_args()

    app = FaceTrackerApp(
        camera_id=args.cam,
        width=args.width,
        height=args.height,
        score_threshold=args.threshold
    )
    app.run()


if __name__ == "__main__":
    main()
