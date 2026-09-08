"""OCR package for vehicle license plate recognition and logging."""

from traffik.ocr.plate_reader import (
    PlateReader,
    PlateResult,
    clean_plate_text,
)

__all__ = [
    "PlateReader",
    "PlateResult",
    "clean_plate_text",
]
