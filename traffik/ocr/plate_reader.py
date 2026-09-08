"""Optical Character Recognition (OCR) module for license plate extraction.

This module provides the `PlateReader` engine which wraps EasyOCR to perform
text extraction on vehicle crops, cleans alphanumeric plate text, filters out
low-confidence or invalid-length strings, and logs high-confidence plates to CSV.
"""

import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import easyocr

from traffik.config import OCRConfig, get_device


# ==============================================================================
# Text Sanitization & Normalization
# ==============================================================================

def clean_plate_text(text: str) -> str:
    """Sanitize and normalize license plate text.

    Removes all non-alphanumeric characters (spaces, hyphens, punctuation)
    and converts text to uppercase to produce standard plate identifiers.

    Args:
        text: Raw OCR string candidate.

    Returns:
        Uppercase alphanumeric string.
    """
    return re.sub(r"[^A-Z0-9]", "", text.upper())


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class PlateResult:
    """Structured record of a detected license plate.

    Attributes:
        frame: Video frame index where the plate was read.
        track_id: Tracking ID of the vehicle (if tracking is enabled), else None.
        class_name: Classified vehicle type (e.g. 'car', 'truck', 'bus').
        plate_text: Sanitized uppercase license plate text.
        confidence: OCR recognition confidence score (between 0.0 and 1.0).
        crop_filename: Filename of the vehicle/plate image crop saved on disk.
    """
    frame: int
    track_id: Optional[int]
    class_name: str
    plate_text: str
    confidence: float
    crop_filename: str


# ==============================================================================
# Plate Reader Engine
# ==============================================================================

class PlateReader:
    """Manages EasyOCR inference, per-track confidence tracking, and CSV export.

    The PlateReader processes image crops of detected vehicles. To avoid spamming
    the CSV log and disk storage with multiple redundant reads of the same vehicle,
    it tracks the highest-confidence reading for each unique track ID.
    """

    def __init__(
        self,
        config: OCRConfig,
        device: str = "auto",
        captures_dir: str = "captures"
    ):
        """Initialize the PlateReader with hardware acceleration and logging paths.

        Args:
            config: OCR configuration containing confidence threshold and CSV path.
            device: Requested hardware acceleration device ('auto', 'cuda', 'mps', 'cpu').
            captures_dir: Target directory where vehicle crops are stored.
        """
        self.config = config
        self.device = get_device(device)
        self.captures_dir = captures_dir

        # Ensure captures directory exists
        os.makedirs(self.captures_dir, exist_ok=True)

        # EasyOCR supports CUDA and MPS on supported platforms
        gpu_enabled = self.device in ("cuda", "mps")
        self.reader = easyocr.Reader(["en"], gpu=gpu_enabled)

        # In-memory dictionary tracking best detection per track or frame
        self.best_plates: Dict[str, PlateResult] = {}

        # Initialize CSV log with standard header if specified and not yet created
        if config.output_csv:
            csv_dir = os.path.dirname(config.output_csv)
            if csv_dir:
                os.makedirs(csv_dir, exist_ok=True)
            if not os.path.exists(config.output_csv):
                with open(config.output_csv, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Frame", "TrackID", "Class", "PlateText", "Confidence", "CropFile"])

    def process_crop(
        self,
        crop: np.ndarray,
        frame_idx: int,
        tracker_id: Optional[int],
        class_name: str
    ) -> Optional[PlateResult]:
        """Run OCR on a vehicle image crop and record high-confidence license plates.

        Args:
            crop: BGR image crop of the vehicle.
            frame_idx: Current video frame number.
            tracker_id: ByteTrack ID assigned to this vehicle, or None.
            class_name: Name of the detected vehicle class.

        Returns:
            A `PlateResult` if a new best-confidence plate was recognized, else `None`.
        """
        # Discard tiny crops that are too degraded for reliable OCR
        if crop.shape[0] < 20 or crop.shape[1] < 20:
            return None

        ocr_results = self.reader.readtext(crop)

        for result_tuple in ocr_results:
            # EasyOCR returns (bbox, text, prob)
            if len(result_tuple) == 3:
                _bbox, raw_text, ocr_conf = result_tuple
            elif len(result_tuple) == 2:
                _bbox, raw_text = result_tuple
                ocr_conf = 1.0
            else:
                continue

            cleaned = clean_plate_text(raw_text)

            # Enforce minimum character length (standard plates are >= 4 chars) and confidence
            if len(cleaned) >= 4 and ocr_conf >= self.config.conf_threshold:
                track_key = f"track_{tracker_id}" if tracker_id is not None else f"frame_{frame_idx}"
                prev_conf = self.best_plates[track_key].confidence if track_key in self.best_plates else 0.0

                # Only persist and log if this read improves upon our previous confidence
                if ocr_conf > prev_conf:
                    crop_filename = (
                        f"car_frame{frame_idx:05d}_track{tracker_id}.jpg"
                        if tracker_id is not None
                        else f"car_frame{frame_idx:05d}_det.jpg"
                    )
                    crop_path = os.path.join(self.captures_dir, crop_filename)
                    cv2.imwrite(crop_path, crop)

                    plate_result = PlateResult(
                        frame=frame_idx,
                        track_id=tracker_id,
                        class_name=class_name,
                        plate_text=cleaned,
                        confidence=float(ocr_conf),
                        crop_filename=crop_filename
                    )
                    self.best_plates[track_key] = plate_result

                    # Append record to CSV log
                    if self.config.output_csv:
                        with open(self.config.output_csv, mode="a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            track_id_str = tracker_id if tracker_id is not None else ""
                            writer.writerow([
                                frame_idx,
                                track_id_str,
                                class_name,
                                cleaned,
                                f"{ocr_conf:.4f}",
                                crop_filename
                            ])

                    return plate_result

        return None
