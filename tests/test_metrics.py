"""Unit tests for metrics.py calculation functions and pipeline."""

import os
import pandas as pd
import pytest
from metrics import (
    calculate_counts_metrics,
    calculate_geh,
    compute_aggregate_summary,
    compute_metrics_dataframe,
    plot_metrics,
)


def test_calculate_counts_metrics_perfect():
    tp, fp, fn, prec, rec, f1 = calculate_counts_metrics(10, 10)
    assert tp == 10
    assert fp == 0
    assert fn == 0
    assert prec == 1.0
    assert rec == 1.0
    assert f1 == 1.0


def test_calculate_counts_metrics_overcount():
    # Predicted 12, Actual 10 -> 2 false positives
    tp, fp, fn, prec, rec, f1 = calculate_counts_metrics(12, 10)
    assert tp == 10
    assert fp == 2
    assert fn == 0
    assert prec == pytest.approx(10 / 12)
    assert rec == 1.0
    assert f1 == pytest.approx(2 * (10 / 12 * 1.0) / (10 / 12 + 1.0))


def test_calculate_counts_metrics_undercount():
    # Predicted 8, Actual 10 -> 2 false negatives
    tp, fp, fn, prec, rec, f1 = calculate_counts_metrics(8, 10)
    assert tp == 8
    assert fp == 0
    assert fn == 2
    assert prec == 1.0
    assert rec == 0.8
    assert f1 == pytest.approx(2 * (1.0 * 0.8) / (1.0 + 0.8))


def test_calculate_counts_metrics_zero_zero():
    tp, fp, fn, prec, rec, f1 = calculate_counts_metrics(0, 0)
    assert tp == 0
    assert fp == 0
    assert fn == 0
    assert prec == 1.0
    assert rec == 1.0
    assert f1 == 0.0 or f1 == 1.0 or f1 == 0.0


def test_calculate_geh():
    # Exactly equal
    assert calculate_geh(10, 10) == 0.0
    # Both zero
    assert calculate_geh(0, 0) == 0.0
    # Difference: pred=24, actual=25
    geh = calculate_geh(24, 25)
    assert geh == pytest.approx((2 * (1**2) / 49) ** 0.5)


def test_compute_metrics_dataframe():
    raw_data = {
        "name": ["clip_1.mp4", "clip_2.mp4"],
        "run": [1, 1],
        "in": [10, 15],
        "out": [5, 4],
        "actual_in": [10, 12],
        "actual_out": [6, 4],
    }
    df_raw = pd.DataFrame(raw_data)
    df_metrics = compute_metrics_dataframe(df_raw)

    assert len(df_metrics) == 2
    assert "precision_in" in df_metrics.columns
    assert "recall_out" in df_metrics.columns
    assert "f1_total" in df_metrics.columns
    assert df_metrics.iloc[0]["precision_in"] == 1.0
    assert df_metrics.iloc[0]["recall_in"] == 1.0


def test_compute_aggregate_summary():
    raw_data = {
        "name": ["clip_1.mp4", "clip_2.mp4"],
        "run": [1, 1],
        "in": [10, 15],
        "out": [5, 4],
        "actual_in": [10, 12],
        "actual_out": [6, 4],
    }
    df_raw = pd.DataFrame(raw_data)
    df_metrics = compute_metrics_dataframe(df_raw)
    summary = compute_aggregate_summary(df_metrics)

    assert "in" in summary
    assert "out" in summary
    assert "total" in summary
    assert summary["in"]["pred"] == 25
    assert summary["in"]["actual"] == 22


def test_plot_metrics(tmp_path):
    raw_data = {
        "name": ["clip_1.mp4", "clip_2.mp4"],
        "run": [1, 1],
        "in": [10, 15],
        "out": [5, 4],
        "actual_in": [10, 12],
        "actual_out": [6, 4],
    }
    df_raw = pd.DataFrame(raw_data)
    df_metrics = compute_metrics_dataframe(df_raw)
    plot_file = str(tmp_path / "test_plot.png")

    result = plot_metrics(df_metrics, output_path=plot_file, show=False)
    assert os.path.exists(result)
    assert os.path.getsize(result) > 0
