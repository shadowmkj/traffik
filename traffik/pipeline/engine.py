"""Video pipeline execution engine for Traffik.

This module orchestrates the end-to-end computer vision workflow:
1. Video ingest and frame decoding
2. Object detection with YOLOv8/v26
3. Multi-object tracking with ByteTrack and majority-vote smoothing
4. Virtual gate crossing and bidirectional counting
5. Optional license plate OCR extraction and logging
6. Multi-layer visual annotation, stream preview, and video rendering
7. Processing metrics aggregation and summary reporting
"""

import csv
import os
import time
from dataclasses import dataclass
from typing import Optional
import cv2
import supervision as sv
from tqdm import tqdm

from traffik.config import Config
from traffik.counting.gate import DualLineGate
from traffik.detection.detector import VehicleDetector
from traffik.ocr.plate_reader import PlateReader
from traffik.tracking.tracker import VehicleTracker
from traffik.visualization.hud import VisualAnnotator


def record_run_to_csv(
    csv_path: str,
    source_path: str,
    in_count: int,
    out_count: int,
) -> int:
    """Record execution summary (name, run, in, out) to an outputs CSV log file.

    Args:
        csv_path: Path to the outputs CSV file.
        source_path: Path or filename of the processed input video.
        in_count: Number of vehicles crossed inward.
        out_count: Number of vehicles crossed outward.

    Returns:
        The incremented run number for this video.
    """
    file_name = os.path.basename(source_path)
    runs = []

    if os.path.exists(csv_path):
        with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0] == file_name:
                    try:
                        runs.append(int(row[1]))
                    except ValueError:
                        pass
    else:
        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "run", "in", "out"])

    run_number = (max(runs) + 1) if runs else 1

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([file_name, run_number, in_count, out_count])

    return run_number


# ==============================================================================
# Pipeline Execution Summary
# ==============================================================================

@dataclass
class PipelineSummary:
    """Summary metrics produced after processing a video file.

    Attributes:
        source: Path to the input source video.
        target: Path to the annotated output video, or None if not rendered.
        in_count: Total vehicles detected crossing into the monitored zone.
        out_count: Total vehicles detected crossing out of the monitored zone.
        unique_tracks: Number of unique vehicle tracks registered by the tracker.
        total_frames: Total number of video frames processed.
        fps: Effective end-to-end processing throughput in frames per second.
        run_number: Incremental execution run index logged for this video file.
    """
    source: str
    target: Optional[str]
    in_count: int
    out_count: int
    unique_tracks: int
    total_frames: int
    fps: float
    run_number: Optional[int] = None


# ==============================================================================
# Video Pipeline Orchestrator Engine
# ==============================================================================

class VideoPipeline:
    """End-to-end video processing engine orchestrating all vision subsystems.

    Initializes detection, tracking, counting, OCR, and visualization components
    from a unified `Config` specification and executes the frame processing loop.
    """

    def __init__(self, config: Config) -> None:
        """Initialize pipeline sub-components using the provided configuration.

        Args:
            config: Complete application configuration containing general, detector,
                tracker, gate, and OCR settings.
        """
        self.config = config
        self.detector = VehicleDetector(
            config.detector,
            device=config.general.device,
        )
        self.tracker = VehicleTracker(config.tracker)
        self.gate = DualLineGate.from_config(config.gate)
        self.annotator = VisualAnnotator()
        self.plate_reader: Optional[PlateReader] = (
            PlateReader(
                config.ocr,
                device=config.general.device,
                captures_dir=config.general.captures_dir,
            )
            if config.ocr.enabled
            else None
        )

    def run(
        self,
        source_path: str,
        target_path: Optional[str] = None,
        stream: bool = False,
    ) -> PipelineSummary:
        """Execute the video processing pipeline on an input video file.

        Args:
            source_path: File path to the input video.
            target_path: Optional destination path to write the annotated video file.
            stream: If True, renders live preview in an OpenCV GUI window.

        Returns:
            A `PipelineSummary` containing cumulative counts, track statistics,
            frame totals, and processing FPS.

        Raises:
            FileNotFoundError: If `source_path` does not exist on disk.
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source video '{source_path}' not found.")

        # Extract video metadata and calibrate tracker Kalman filter frame rate
        video_info = sv.VideoInfo.from_video_path(video_path=source_path)
        self.tracker = VehicleTracker(self.config.tracker, fps=video_info.fps)

        # Initialize live stream window if requested
        window_name = f"Traffik Stream - {os.path.basename(source_path)}"
        if stream:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Prepare frame generator with optional CLI progress bar
        frames_gen = sv.get_video_frames_generator(source_path=source_path)
        if not stream:
            frame_iter = enumerate(
                tqdm(
                    frames_gen,
                    total=video_info.total_frames,
                    desc="Processing Video",
                    unit="frame",
                )
            )
        else:
            frame_iter = enumerate(frames_gen)

        # Initialize video writer sink if target output is requested
        sink: Optional[sv.VideoSink] = None
        if target_path:
            target_dir = os.path.dirname(os.path.abspath(target_path))
            os.makedirs(target_dir, exist_ok=True)
            sink = sv.VideoSink(target_path=target_path, video_info=video_info)
            sink.__enter__()

        t0 = time.time()
        processed_frames = 0

        try:
            for frame_idx, frame in frame_iter:
                processed_frames += 1

                # 1. Vehicle Detection
                detections = self.detector.detect(frame)

                # 2. Multi-object Tracking
                detections = self.tracker.update(detections)

                # 3. Virtual Gate State Machine & Directional Counting
                self.gate.trigger(detections)

                # 4. Optional License Plate OCR on Tracked Vehicle Crops
                if (
                    self.plate_reader is not None
                    and detections.tracker_id is not None
                    and len(detections) > 0
                ):
                    h, w, _ = frame.shape
                    class_ids = (
                        detections.class_id
                        if detections.class_id is not None
                        else [0] * len(detections)
                    )
                    for xyxy, class_id, tracker_id in zip(
                        detections.xyxy, class_ids, detections.tracker_id
                    ):
                        x1, y1 = max(0, int(xyxy[0])), max(0, int(xyxy[1]))
                        x2, y2 = min(w, int(xyxy[2])), min(h, int(xyxy[3]))
                        if x2 > x1 and y2 > y1:
                            crop = frame[y1:y2, x1:x2]
                            class_name = self.detector.model.names.get(
                                int(class_id), f"class_{class_id}"
                            )
                            self.plate_reader.process_crop(
                                crop=crop,
                                frame_idx=frame_idx,
                                tracker_id=int(tracker_id),
                                class_name=class_name,
                            )

                # 5. Visual Annotation & Stream Display / Video Sink Output
                if sink or stream:
                    labels = self.tracker.get_labels(
                        detections,
                        class_names=self.detector.model.names,
                    )
                    annotated_frame = self.annotator.annotate(
                        frame=frame,
                        detections=detections,
                        labels=labels,
                        gate=self.gate,
                    )

                    if sink is not None:
                        sink.write_frame(annotated_frame)

                    if stream:
                        cv2.imshow(window_name, annotated_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):  # ESC or 'q' to break
                            break

        finally:
            # Graceful resource cleanup
            if sink is not None:
                sink.__exit__(None, None, None)
            if stream:
                cv2.destroyAllWindows()

        elapsed = max(0.001, time.time() - t0)

        # Log run results to outputs CSV if configured
        run_number: Optional[int] = None
        if self.config.general.outputs_csv:
            is_pytest = "PYTEST_CURRENT_TEST" in os.environ
            is_default_output = os.path.abspath(self.config.general.outputs_csv) == os.path.abspath("outputs.csv")
            if not (is_pytest and is_default_output):
                try:
                    run_number = record_run_to_csv(
                        csv_path=self.config.general.outputs_csv,
                        source_path=source_path,
                        in_count=self.gate.in_count,
                        out_count=self.gate.out_count,
                    )
                except Exception as e:
                    print(
                        f"[Traffik Pipeline] Warning: Could not write run metrics to '{self.config.general.outputs_csv}': {e}"
                    )

        return PipelineSummary(
            source=source_path,
            target=target_path,
            in_count=self.gate.in_count,
            out_count=self.gate.out_count,
            unique_tracks=len(self.tracker.track_class_history),
            total_frames=processed_frames,
            fps=processed_frames / elapsed,
            run_number=run_number,
        )
