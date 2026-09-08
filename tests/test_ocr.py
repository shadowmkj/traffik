import os
import tempfile
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from traffik.config import OCRConfig
from traffik.ocr.plate_reader import PlateReader, PlateResult, clean_plate_text


def test_clean_plate_text():
    assert clean_plate_text("DL 10 CF-6010") == "DL10CF6010"
    assert clean_plate_text("abc-123!") == "ABC123"
    assert clean_plate_text("  MH 12  AB 1234 ") == "MH12AB1234"
    assert clean_plate_text("ka-01-ea-1111") == "KA01EA1111"
    assert clean_plate_text("@#$%^&*()") == ""
    assert clean_plate_text("") == ""


def test_package_exports():
    import traffik
    import traffik.ocr

    assert hasattr(traffik, "PlateReader")
    assert hasattr(traffik, "PlateResult")
    assert hasattr(traffik, "clean_plate_text")
    assert hasattr(traffik.ocr, "PlateReader")
    assert hasattr(traffik.ocr, "PlateResult")
    assert hasattr(traffik.ocr, "clean_plate_text")


@patch("traffik.ocr.plate_reader.easyocr.Reader")
def test_plate_reader_initialization(mock_reader_cls):
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "test_plates.csv")
        captures_dir = os.path.join(tmp_dir, "test_captures")

        cfg = OCRConfig(enabled=True, conf_threshold=0.35, output_csv=csv_path)
        reader = PlateReader(cfg, device="cpu", captures_dir=captures_dir)

        assert os.path.isdir(captures_dir)
        assert os.path.isfile(csv_path)

        with open(csv_path, mode="r", encoding="utf-8") as f:
            header = f.readline().strip()
            assert header == "Frame,TrackID,Class,PlateText,Confidence,CropFile"


@patch("traffik.ocr.plate_reader.easyocr.Reader")
def test_plate_reader_filters_small_crops(mock_reader_cls):
    mock_reader = MagicMock()
    mock_reader_cls.return_value = mock_reader

    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = OCRConfig(enabled=True, conf_threshold=0.35, output_csv="")
        reader = PlateReader(cfg, device="cpu", captures_dir=tmp_dir)

        small_crop = np.zeros((15, 15, 3), dtype=np.uint8)
        result = reader.process_crop(small_crop, frame_idx=1, tracker_id=1, class_name="car")

        assert result is None
        mock_reader.readtext.assert_not_called()


@patch("traffik.ocr.plate_reader.easyocr.Reader")
def test_plate_reader_filters_low_confidence_or_short_text(mock_reader_cls):
    mock_reader = MagicMock()
    mock_reader_cls.return_value = mock_reader

    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = OCRConfig(enabled=True, conf_threshold=0.50, output_csv="")
        reader = PlateReader(cfg, device="cpu", captures_dir=tmp_dir)

        valid_size_crop = np.zeros((50, 100, 3), dtype=np.uint8)

        # 1. Short text (< 4 chars) with high confidence
        mock_reader.readtext.return_value = [([[0, 0], [10, 0], [10, 10], [0, 10]], "ABC", 0.90)]
        result_short = reader.process_crop(valid_size_crop, frame_idx=1, tracker_id=1, class_name="car")
        assert result_short is None

        # 2. Valid text length but below confidence threshold (0.40 < 0.50)
        mock_reader.readtext.return_value = [([[0, 0], [10, 0], [10, 10], [0, 10]], "DL10CF6010", 0.40)]
        result_low_conf = reader.process_crop(valid_size_crop, frame_idx=1, tracker_id=1, class_name="car")
        assert result_low_conf is None


@patch("traffik.ocr.plate_reader.easyocr.Reader")
def test_plate_reader_process_crop_success_and_confidence_update(mock_reader_cls):
    mock_reader = MagicMock()
    mock_reader_cls.return_value = mock_reader

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "plates.csv")
        captures_dir = os.path.join(tmp_dir, "captures")
        cfg = OCRConfig(enabled=True, conf_threshold=0.35, output_csv=csv_path)
        reader = PlateReader(cfg, device="cpu", captures_dir=captures_dir)

        crop = np.zeros((60, 120, 3), dtype=np.uint8)

        # 1. First valid detection with confidence 0.70
        mock_reader.readtext.return_value = [([[0, 0], [10, 0], [10, 10], [0, 10]], "DL 10 CF 6010", 0.70)]
        res1 = reader.process_crop(crop, frame_idx=10, tracker_id=42, class_name="car")

        assert res1 is not None
        assert res1.frame == 10
        assert res1.track_id == 42
        assert res1.class_name == "car"
        assert res1.plate_text == "DL10CF6010"
        assert pytest.approx(res1.confidence) == 0.70
        assert res1.crop_filename == "car_frame00010_track42.jpg"
        assert os.path.exists(os.path.join(captures_dir, res1.crop_filename))

        # 2. Second detection with lower confidence (0.60 < 0.70) for same track
        mock_reader.readtext.return_value = [([[0, 0], [10, 0], [10, 10], [0, 10]], "DL 10 CF 6010", 0.60)]
        res2 = reader.process_crop(crop, frame_idx=11, tracker_id=42, class_name="car")
        assert res2 is None
        assert reader.best_plates["track_42"].confidence == 0.70

        # 3. Third detection with higher confidence (0.95 > 0.70) for same track
        mock_reader.readtext.return_value = [([[0, 0], [10, 0], [10, 10], [0, 10]], "DL 10 CF 6010", 0.95)]
        res3 = reader.process_crop(crop, frame_idx=15, tracker_id=42, class_name="car")
        assert res3 is not None
        assert pytest.approx(res3.confidence) == 0.95
        assert reader.best_plates["track_42"].confidence == 0.95

        # Check CSV rows: 1 header + 2 appended rows (first and third detection)
        with open(csv_path, mode="r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
            assert len(lines) == 3
            assert lines[0] == "Frame,TrackID,Class,PlateText,Confidence,CropFile"
            assert lines[1] == "10,42,car,DL10CF6010,0.7000,car_frame00010_track42.jpg"
            assert lines[2] == "15,42,car,DL10CF6010,0.9500,car_frame00015_track42.jpg"


@patch("traffik.ocr.plate_reader.easyocr.Reader")
def test_plate_reader_without_tracker_id(mock_reader_cls):
    mock_reader = MagicMock()
    mock_reader_cls.return_value = mock_reader

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "plates.csv")
        captures_dir = os.path.join(tmp_dir, "captures")
        cfg = OCRConfig(enabled=True, conf_threshold=0.35, output_csv=csv_path)
        reader = PlateReader(cfg, device="cpu", captures_dir=captures_dir)

        crop = np.zeros((60, 120, 3), dtype=np.uint8)
        mock_reader.readtext.return_value = [([[0, 0], [10, 0], [10, 10], [0, 10]], "MH 02 BB 1234", 0.85)]

        res = reader.process_crop(crop, frame_idx=5, tracker_id=None, class_name="truck")
        assert res is not None
        assert res.frame == 5
        assert res.track_id is None
        assert res.class_name == "truck"
        assert res.plate_text == "MH02BB1234"
        assert "frame_5" in reader.best_plates
        assert os.path.exists(os.path.join(captures_dir, res.crop_filename))
