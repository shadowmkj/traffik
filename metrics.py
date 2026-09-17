"""
Vehicle Counting & Detection Evaluation Metrics Pipeline.

This module computes computer vision and traffic counting evaluation metrics
from detection outputs (predictions vs. ground truth annotations).

Calculated Metrics:
  - Precision: P = TP / (TP + FP) (identifies over-counting / phantom detections)
  - Recall: R = TP / (TP + FN) (identifies under-counting / missed vehicles)
  - F1-Score: 2 * (P * R) / (P + R) (harmonic balance of precision & recall)
  - Counting Accuracy: 1 - (|Pred - Actual| / Actual)
  - GEH Statistic: sqrt(2 * (Pred - Actual)^2 / (Pred + Actual))

Visualizations:
  - Aggregate Directional Precision, Recall & F1-Score (IN vs OUT vs TOTAL)
  - Predicted vs. Actual Counts per Video Clip
  - Directional Error Breakdown (True Positives, False Positives, False Negatives)
  - Per-Clip Performance Profile across Benchmark Clips
"""

import argparse
import math
import os
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==============================================================================
# Metric Computation Core
# ==============================================================================

def calculate_counts_metrics(pred: int, actual: int) -> Tuple[int, int, int, float, float, float]:
    """Compute True Positives, False Positives, False Negatives, Precision, Recall,
    and F1-Score for count-based evaluations.

    Math Rationale:
      - True Positives (TP): Vehicles correctly detected and counted: min(pred, actual)
      - False Positives (FP): Overcounted / phantom counts: max(0, pred - actual)
      - False Negatives (FN): Missed vehicles: max(0, actual - pred)
    """
    pred_val = int(max(0, pred))
    act_val = int(max(0, actual))

    tp = min(pred_val, act_val)
    fp = max(0, pred_val - act_val)
    fn = max(0, act_val - pred_val)

    # Calculate Precision with zero-division safeguard
    if (tp + fp) == 0:
        precision = 1.0 if act_val == 0 else 0.0
    else:
        precision = tp / (tp + fp)

    # Calculate Recall with zero-division safeguard
    if (tp + fn) == 0:
        recall = 1.0 if pred_val == 0 else 0.0
    else:
        recall = tp / (tp + fn)

    # Calculate F1-Score
    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)

    return tp, fp, fn, precision, recall, f1


def calculate_geh(pred: float, actual: float) -> float:
    """Calculate the Geoffrey E. Havers (GEH) statistic.

    GEH is the standard traffic engineering metric:
      GEH = sqrt( 2 * (pred - actual)^2 / (pred + actual) )

    Values < 5.0 indicate acceptable alignment with real-world counts.
    """
    total = pred + actual
    if total <= 0:
        return 0.0
    return math.sqrt((2.0 * ((pred - actual) ** 2)) / total)


# ==============================================================================
# Dataset Processing & Aggregation
# ==============================================================================

def compute_metrics_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Process raw prediction and actual counts into a full metrics DataFrame.

    Supports both 'in'/'out'/'actual_in'/'actual_out' or already-named columns.
    Total True Positives, False Positives, and False Negatives are summed across
    directions to avoid offsetting error cancellation.
    """
    col_map = {
        "in": "pred_in",
        "out": "pred_out",
        "actual_in": "actual_in",
        "actual_out": "actual_out",
    }
    df = df_raw.rename(columns=col_map).copy()

    records = []
    for idx, row in df.iterrows():
        name = row.get("name", f"clip_{idx+1}")
        run = row.get("run", 1)

        p_in = int(row.get("pred_in", 0))
        a_in = int(row.get("actual_in", 0))
        p_out = int(row.get("pred_out", 0))
        a_out = int(row.get("actual_out", 0))

        p_tot = p_in + p_out
        a_tot = a_in + a_out

        # Direction IN
        tp_in, fp_in, fn_in, prec_in, rec_in, f1_in = calculate_counts_metrics(p_in, a_in)
        geh_in = calculate_geh(p_in, a_in)

        # Direction OUT
        tp_out, fp_out, fn_out, prec_out, rec_out, f1_out = calculate_counts_metrics(p_out, a_out)
        geh_out = calculate_geh(p_out, a_out)

        # TOTAL: Direction-aware summing to preserve error isolation
        tp_tot = tp_in + tp_out
        fp_tot = fp_in + fp_out
        fn_tot = fn_in + fn_out

        prec_tot = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else (1.0 if a_tot == 0 else 0.0)
        rec_tot = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else (1.0 if p_tot == 0 else 0.0)
        f1_tot = (2.0 * prec_tot * rec_tot) / (prec_tot + rec_tot) if (prec_tot + rec_tot) > 0 else 0.0

        acc_tot = 1.0 - (abs(p_tot - a_tot) / a_tot) if a_tot > 0 else 1.0
        geh_tot = calculate_geh(p_tot, a_tot)

        records.append({
            "name": name,
            "run": run,
            # IN
            "pred_in": p_in,
            "actual_in": a_in,
            "tp_in": tp_in,
            "fp_in": fp_in,
            "fn_in": fn_in,
            "precision_in": prec_in,
            "recall_in": rec_in,
            "f1_in": f1_in,
            "geh_in": geh_in,
            # OUT
            "pred_out": p_out,
            "actual_out": a_out,
            "tp_out": tp_out,
            "fp_out": fp_out,
            "fn_out": fn_out,
            "precision_out": prec_out,
            "recall_out": rec_out,
            "f1_out": f1_out,
            "geh_out": geh_out,
            # TOTAL
            "pred_total": p_tot,
            "actual_total": a_tot,
            "tp_total": tp_tot,
            "fp_total": fp_tot,
            "fn_total": fn_tot,
            "precision_total": prec_tot,
            "recall_total": rec_tot,
            "f1_total": f1_tot,
            "accuracy_total": acc_tot,
            "geh_total": geh_tot,
        })

    return pd.DataFrame(records)


def compute_aggregate_summary(df_metrics: pd.DataFrame) -> Dict[str, Any]:
    """Compute overall micro- and macro-aggregated Precision, Recall, F1, and WAPE."""
    # IN Direction Aggregate
    tot_tp_in = int(df_metrics["tp_in"].sum())
    tot_fp_in = int(df_metrics["fp_in"].sum())
    tot_fn_in = int(df_metrics["fn_in"].sum())
    tot_p_in = int(df_metrics["pred_in"].sum())
    tot_a_in = int(df_metrics["actual_in"].sum())
    agg_prec_in = tot_tp_in / (tot_tp_in + tot_fp_in) if (tot_tp_in + tot_fp_in) > 0 else 0.0
    agg_rec_in = tot_tp_in / (tot_tp_in + tot_fn_in) if (tot_tp_in + tot_fn_in) > 0 else 0.0
    agg_f1_in = (
        2.0 * (agg_prec_in * agg_rec_in) / (agg_prec_in + agg_rec_in)
        if (agg_prec_in + agg_rec_in) > 0
        else 0.0
    )

    # OUT Direction Aggregate
    tot_tp_out = int(df_metrics["tp_out"].sum())
    tot_fp_out = int(df_metrics["fp_out"].sum())
    tot_fn_out = int(df_metrics["fn_out"].sum())
    tot_p_out = int(df_metrics["pred_out"].sum())
    tot_a_out = int(df_metrics["actual_out"].sum())
    agg_prec_out = tot_tp_out / (tot_tp_out + tot_fp_out) if (tot_tp_out + tot_fp_out) > 0 else 0.0
    agg_rec_out = tot_tp_out / (tot_tp_out + tot_fn_out) if (tot_tp_out + tot_fn_out) > 0 else 0.0
    agg_f1_out = (
        2.0 * (agg_prec_out * agg_rec_out) / (agg_prec_out + agg_rec_out)
        if (agg_prec_out + agg_rec_out) > 0
        else 0.0
    )

    # TOTAL Aggregate
    tot_tp = int(df_metrics["tp_total"].sum())
    tot_fp = int(df_metrics["fp_total"].sum())
    tot_fn = int(df_metrics["fn_total"].sum())
    tot_p = int(df_metrics["pred_total"].sum())
    tot_a = int(df_metrics["actual_total"].sum())
    agg_prec_tot = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    agg_rec_tot = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    agg_f1_tot = (
        2.0 * (agg_prec_tot * agg_rec_tot) / (agg_prec_tot + agg_rec_tot)
        if (agg_prec_tot + agg_rec_tot) > 0
        else 0.0
    )

    wape = (abs(df_metrics["pred_total"] - df_metrics["actual_total"]).sum()) / tot_a if tot_a > 0 else 0.0
    overall_acc = 1.0 - wape

    return {
        "in": {
            "pred": tot_p_in,
            "actual": tot_a_in,
            "tp": tot_tp_in,
            "fp": tot_fp_in,
            "fn": tot_fn_in,
            "precision": agg_prec_in,
            "recall": agg_rec_in,
            "f1": agg_f1_in,
        },
        "out": {
            "pred": tot_p_out,
            "actual": tot_a_out,
            "tp": tot_tp_out,
            "fp": tot_fp_out,
            "fn": tot_fn_out,
            "precision": agg_prec_out,
            "recall": agg_rec_out,
            "f1": agg_f1_out,
        },
        "total": {
            "pred": tot_p,
            "actual": tot_a,
            "tp": tot_tp,
            "fp": tot_fp,
            "fn": tot_fn,
            "precision": agg_prec_tot,
            "recall": agg_rec_tot,
            "f1": agg_f1_tot,
            "accuracy": overall_acc,
            "wape": wape,
        },
    }


# ==============================================================================
# Visualization Suite (Matplotlib)
# ==============================================================================

def plot_metrics(
    df_metrics: pd.DataFrame,
    output_path: str = "accuracy_metrics_plot.png",
    show: bool = False,
) -> str:
    """Generate and save a comprehensive multi-panel evaluation visualization.

    Panels:
      1. Aggregate Precision, Recall & F1-Score (IN vs OUT vs TOTAL)
      2. Predicted vs. Actual Counts per Clip
      3. Error Breakdown (TP, FP, FN counts per clip)
      4. Per-Clip F1-Score & Accuracy Performance
    """
    summary = compute_aggregate_summary(df_metrics)

    # Set style and figure layout
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Vehicle Detection & Counting Accuracy Evaluation", fontsize=18, fontweight="bold", y=0.98)

    colors = {
        "precision": "#2b5c8f",  # Deep Blue
        "recall": "#34a853",     # Green
        "f1": "#fbbc05",         # Amber/Gold
        "actual": "#5f6368",     # Slate Gray
        "pred": "#1a73e8",       # Bright Blue
        "tp": "#34a853",         # Green
        "fp": "#ea4335",         # Red
        "fn": "#ff9900",         # Orange
    }

    # --------------------------------------------------------------------------
    # Panel 1: Aggregate Precision, Recall, F1 (IN vs OUT vs TOTAL)
    # --------------------------------------------------------------------------
    ax1 = axes[0, 0]
    categories = ["IN Flow", "OUT Flow", "TOTAL Combined"]
    precisions = [summary["in"]["precision"] * 100, summary["out"]["precision"] * 100, summary["total"]["precision"] * 100]
    recalls = [summary["in"]["recall"] * 100, summary["out"]["recall"] * 100, summary["total"]["recall"] * 100]
    f1_scores = [summary["in"]["f1"] * 100, summary["out"]["f1"] * 100, summary["total"]["f1"] * 100]

    x = np.arange(len(categories))
    width = 0.25

    rects1 = ax1.bar(x - width, precisions, width, label="Precision", color=colors["precision"])
    rects2 = ax1.bar(x, recalls, width, label="Recall", color=colors["recall"])
    rects3 = ax1.bar(x + width, f1_scores, width, label="F1-Score", color=colors["f1"])

    ax1.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Directional Precision, Recall & F1-Score", fontsize=14, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Add data labels
    for rects in (rects1, rects2, rects3):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # --------------------------------------------------------------------------
    # Panel 2: Predicted vs Actual Vehicle Counts per Clip
    # --------------------------------------------------------------------------
    ax2 = axes[0, 1]
    clip_labels = [f"{row['name'].replace('.mp4', '')} (r{row['run']})" for _, row in df_metrics.iterrows()]
    x_clips = np.arange(len(clip_labels))
    clip_bar_width = 0.38

    ax2.bar(x_clips - clip_bar_width / 2, df_metrics["actual_total"], clip_bar_width, label="Actual Count", color=colors["actual"], alpha=0.85)
    ax2.bar(x_clips + clip_bar_width / 2, df_metrics["pred_total"], clip_bar_width, label="Predicted Count", color=colors["pred"], alpha=0.85)

    ax2.set_ylabel("Vehicle Count", fontsize=12, fontweight="bold")
    ax2.set_title("Predicted vs. Actual Vehicle Counts per Clip", fontsize=14, fontweight="bold", pad=10)
    ax2.set_xticks(x_clips)
    ax2.set_xticklabels(clip_labels, rotation=45, ha="right", fontsize=9)
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    # --------------------------------------------------------------------------
    # Panel 3: Error Breakdown per Clip (TP, FP, FN)
    # --------------------------------------------------------------------------
    ax3 = axes[1, 0]
    error_width = 0.26

    ax3.bar(x_clips - error_width, df_metrics["tp_total"], error_width, label="True Positives (TP)", color=colors["tp"])
    ax3.bar(x_clips, df_metrics["fp_total"], error_width, label="False Positives (FP)", color=colors["fp"])
    ax3.bar(x_clips + error_width, df_metrics["fn_total"], error_width, label="False Negatives (FN)", color=colors["fn"])

    ax3.set_ylabel("Number of Vehicles", fontsize=12, fontweight="bold")
    ax3.set_title("Detection Breakdown (TP / FP / FN) per Clip", fontsize=14, fontweight="bold", pad=10)
    ax3.set_xticks(x_clips)
    ax3.set_xticklabels(clip_labels, rotation=45, ha="right", fontsize=9)
    ax3.legend(loc="upper right", frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.5, axis="y")

    # --------------------------------------------------------------------------
    # Panel 4: Per-Clip F1-Score & Counting Accuracy Profile
    # --------------------------------------------------------------------------
    ax4 = axes[1, 1]
    f1_vals = df_metrics["f1_total"] * 100
    acc_vals = df_metrics["accuracy_total"] * 100

    ax4.plot(x_clips, f1_vals, marker="o", linewidth=2.5, markersize=7, label="F1-Score (%)", color="#1a73e8")
    ax4.plot(x_clips, acc_vals, marker="s", linewidth=2.5, markersize=7, label="Count Accuracy (%)", color="#34a853")
    ax4.axhline(90, color="gray", linestyle=":", linewidth=1.5, label="90% Target Benchmark")

    ax4.set_ylabel("Metric (%)", fontsize=12, fontweight="bold")
    ax4.set_title("F1-Score and Counting Accuracy Across Clips", fontsize=14, fontweight="bold", pad=10)
    ax4.set_xticks(x_clips)
    ax4.set_xticklabels(clip_labels, rotation=45, ha="right", fontsize=9)
    ax4.set_ylim(50, 105)
    ax4.legend(loc="lower right", frameon=True)
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[Metrics] Visualization plot saved successfully to: {output_path}")

    if show:
        plt.show()

    plt.close()
    return output_path


# ==============================================================================
# CLI Reporting & Main
# ==============================================================================

def print_summary_table(df_metrics: pd.DataFrame, summary: Dict[str, Any]) -> None:
    """Print a clean, formatted terminal summary of the evaluation results."""
    print("\n" + "=" * 80)
    print(" VEHICLE COUNTING & DETECTION EVALUATION REPORT ".center(80, "="))
    print("=" * 80)

    print("\n[PER-CLIP BREAKDOWN]")
    print(f"{'Clip Name':<16} {'Run':<4} {'IN (P/A)':<10} {'IN Prec/Rec/F1':<18} {'OUT (P/A)':<10} {'OUT Prec/Rec/F1':<18} {'Total F1':<9}")
    print("-" * 88)
    for _, r in df_metrics.iterrows():
        name = str(r["name"])
        run = int(r["run"])
        in_pa = f"{r['pred_in']}/{r['actual_in']}"
        in_prf = f"{r['precision_in']*100:.0f}% / {r['recall_in']*100:.0f}% / {r['f1_in']*100:.0f}%"
        out_pa = f"{r['pred_out']}/{r['actual_out']}"
        out_prf = f"{r['precision_out']*100:.0f}% / {r['recall_out']*100:.0f}% / {r['f1_out']*100:.0f}%"
        tot_f1 = f"{r['f1_total']*100:.1f}%"
        print(f"{name:<16} {run:<4} {in_pa:<10} {in_prf:<18} {out_pa:<10} {out_prf:<18} {tot_f1:<9}")

    print("\n" + "-" * 88)
    print("[AGGREGATE PERFORMANCE SUMMARY]")
    print(f"{'Direction':<18} {'Pred':<6} {'Actual':<8} {'TP':<5} {'FP':<5} {'FN':<5} {'Precision':<11} {'Recall':<11} {'F1-Score':<10}")
    print("-" * 88)
    for key, label in [("in", "IN Direction"), ("out", "OUT Direction"), ("total", "TOTAL Overall")]:
        s = summary[key]
        print(
            f"{label:<18} {s['pred']:<6} {s['actual']:<8} {s['tp']:<5} {s['fp']:<5} {s['fn']:<5} "
            f"{s['precision']*100:>6.2f}%     {s['recall']*100:>6.2f}%     {s['f1']*100:>6.2f}%"
        )
    print("=" * 88)
    print(f"Overall Counting Accuracy (1 - WAPE): {summary['total']['accuracy']*100:.2f}%\n")


def main() -> None:
    """CLI entry point for metrics computation and visualization."""
    parser = argparse.ArgumentParser(description="Evaluate vehicle detection & counting metrics (Precision, Recall, F1).")
    parser.add_argument("--input", "-i", default="outputs.csv", help="Path to raw output CSV (default: outputs.csv)")
    parser.add_argument("--output-csv", "-o", default="accuracy_metrics.csv", help="Path to save computed metrics CSV")
    parser.add_argument("--output-plot", "-p", default="accuracy_metrics_plot.png", help="Path to save visualization plot")
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    args = parser.parse_args()

    # Load dataset
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df_raw = pd.read_csv(args.input)

    # Compute metrics
    df_metrics = compute_metrics_dataframe(df_raw)
    summary = compute_aggregate_summary(df_metrics)

    # Save CSV
    df_metrics.to_csv(args.output_csv, index=False)
    print(f"[Metrics] Computed metrics saved to: {args.output_csv}")

    # Generate plots
    plot_metrics(df_metrics, output_path=args.output_plot, show=args.show)

    # Print summary
    print_summary_table(df_metrics, summary)


if __name__ == "__main__":
    main()
