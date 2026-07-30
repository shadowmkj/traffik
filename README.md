# Traffik — Traffic Detection & Vehicle Counting Research
A lightweight computer vision research project for real-time traffic detection, vehicle tracking, and line-crossing counting.
Built with **YOLOv8**, **ByteTrack**, and **Roboflow Supervision**.

---

## 🎯 Features

- **Object Detection**: Detects vehicles using YOLOv8 (`yolov8n.pt`).
- **Multi-Object Tracking**: Tracks individual vehicle trajectories across frames with **ByteTrack**.
- **Line Crossing Counting**: Counts incoming and outgoing vehicles crossing a defined Virtual Line Zone.
- **Video Annotation**: Draws bounding boxes, tracking labels, and line crossing stats directly onto output video frames.

---

## 🛠️ Requirements & Setup

### Prerequisites
- Python **3.12+**
- [uv](https://github.com/astral-sh/uv) (recommended package manager)

### Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repo-url>
   cd traffik
   ```

2. Install dependencies using `uv`:
   ```bash
   uv sync
   ```

---

## 🚀 How to Run

1. Place your target video file (`traffic.mp4`) in the project root directory.
2. Run the detection and counting script:
   ```bash
   uv run main.py
   ```
3. Processed output with visual annotations will be saved to `output_counted.mp4`. Total vehicle counts (`IN` / `OUT`) will be printed to the terminal upon completion.

---

## 📁 Project Structure

```text
traffik/
├── main.py              # Main execution script (YOLOv8 + ByteTrack + LineZone)
├── pyproject.toml       # Dependencies and project config
├── uv.lock              # Lockfile for reproducible environment
├── traffic.mp4          # Input source video
└── output_counted.mp4   # Annotated output video
```

---

## 🔬 Research & Tech Stack

- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** — Real-time object detection model.
- **[Supervision](https://github.com/roboflow/supervision)** — Computer vision utilities for tracking, line counting, and annotations.
- **[OpenCV](https://opencv.org/)** — Video frame processing and rendering.
