"""
Evaluation, Reporting, and Modular Publication-Quality Visualizations for Project 1.
Computes validation rates, classification metrics (Acc, Prec, Rec, F1),
preserves comprehensive .txt error report, and exports individual minimalist,
publication-quality charts in English.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_FILE = "project_1_structured_extraction/data/processed/extracted_results.csv"
PLOTS_DIR = "project_1_structured_extraction/reports/plots"
TEXT_REPORT_PATH = "project_1_structured_extraction/reports/error_analysis.txt"


def load_and_validate_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Results file '{filepath}' not found. Please run pipeline.py first."
        )
    df = pd.read_csv(filepath)
    print(f"\n[INFO] Loaded {len(df)} samples from: {filepath}")
    return df


def generate_text_report(df: pd.DataFrame, report_path: str):
    """
    Computes all evaluation metrics, extracts all neutral texts,
    False Positive and False Negative cases, and writes them to a .txt file.
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    total = len(df)
    valid_count = df["is_valid_json"].sum()
    valid_rate = (valid_count / total) * 100

    lat = df["latency_ms"]
    lat_mean = lat.mean()
    lat_median = lat.median()
    lat_warm_mean = lat.iloc[1:].mean() if len(lat) > 1 else lat_mean

    y_true = df["ground_truth_sentiment"].astype(str).str.lower().str.strip()
    y_pred = df["sentiment"].fillna("none").astype(str).str.lower().str.strip()

    strict_acc = (y_true == y_pred).mean() * 100

    # Binary metrics
    tp_neg = ((y_true == "negative") & (y_pred == "negative")).sum()
    fp_neg = ((y_true == "positive") & (y_pred == "negative")).sum()
    fn_neg = ((y_true == "negative") & (y_pred != "negative")).sum()
    prec_neg = (tp_neg / (tp_neg + fp_neg)) * 100 if (tp_neg + fp_neg) > 0 else 0.0
    rec_neg = (tp_neg / (tp_neg + fn_neg)) * 100 if (tp_neg + fn_neg) > 0 else 0.0
    f1_neg = (2 * prec_neg * rec_neg / (prec_neg + rec_neg)) if (prec_neg + rec_neg) > 0 else 0.0

    tp_pos = ((y_true == "positive") & (y_pred == "positive")).sum()
    fp_pos = ((y_true == "negative") & (y_pred == "positive")).sum()
    fn_pos = ((y_true == "positive") & (y_pred != "positive")).sum()
    prec_pos = (tp_pos / (tp_pos + fp_pos)) * 100 if (tp_pos + fp_pos) > 0 else 0.0
    rec_pos = (tp_pos / (tp_pos + fn_pos)) * 100 if (tp_pos + fn_pos) > 0 else 0.0
    f1_pos = (2 * prec_pos * rec_pos / (prec_pos + rec_pos)) if (prec_pos + rec_pos) > 0 else 0.0

    macro_f1 = (f1_neg + f1_pos) / 2
    ct = pd.crosstab(y_true, y_pred, margins=True)

    lines = []
    lines.append("=" * 80)
    lines.append("CEN471 - PROJECT 1: DETAILED EVALUATION & QUALITATIVE ERROR REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append("--- 1. OVERALL PIPELINE & SCHEMA VALIDATION ---")
    lines.append(f"Total Samples           : {total}")
    lines.append(f"Valid JSON Output Rate  : {valid_count}/{total} (%{valid_rate:.2f})")
    lines.append(f"Invalid JSON Output     : {total - valid_count}")
    lines.append(f"Mean Latency            : {lat_mean:.2f} ms")
    lines.append(f"Warm Mean Latency       : {lat_warm_mean:.2f} ms (Excluding 1st sample cold-start)")
    lines.append(f"Median Latency          : {lat_median:.2f} ms")
    lines.append("")
    lines.append("--- 2. CLASSIFICATION PERFORMANCE METRICS ---")
    lines.append(f"Strict Accuracy (Pred == True) : %{strict_acc:.2f}")
    lines.append(f"Macro F1-Score (Pos/Neg)       : %{macro_f1:.2f}")
    lines.append(f"Negative Class -> Precision: %{prec_neg:.2f} | Recall: %{rec_neg:.2f} | F1: %{f1_neg:.2f}")
    lines.append(f"Positive Class -> Precision: %{prec_pos:.2f} | Recall: %{rec_pos:.2f} | F1: %{f1_pos:.2f}")
    lines.append("")
    lines.append("Confusion Matrix (Cross-tabulation):")
    lines.append(ct.to_string())
    lines.append("\n" + "=" * 80)

    # 3. Neutral predictions
    neutral_df = df[y_pred == "neutral"]
    lines.append(f"3. SAMPLES PREDICTED AS 'NEUTRAL' (TOTAL: {len(neutral_df)})")
    lines.append("=" * 80)
    if len(neutral_df) == 0:
        lines.append("No samples were predicted as neutral.\n")
    else:
        for _, row in neutral_df.iterrows():
            lines.append(f"[ID #{row['id']}] Ground Truth: {row['ground_truth_sentiment'].upper()} | Category: {row.get('category', 'N/A')}")
            lines.append(f"Text Content:\n{row['text']}")
            lines.append("-" * 80)

    # 4. False Positive
    fp_df = df[(y_true == "negative") & (y_pred == "positive")]
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"4. FALSE POSITIVE SAMPLES (True: NEGATIVE -> Pred: POSITIVE) [Top 3 of {len(fp_df)}]")
    lines.append("=" * 80)
    for _, row in fp_df.head(3).iterrows():
        lines.append(f"[ID #{row['id']}] Category: {row.get('category', 'N/A')}")
        lines.append(f"Text Content:\n{row['text']}")
        lines.append("-" * 80)

    # 5. False Negative
    fn_df = df[(y_true == "positive") & (y_pred == "negative")]
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"5. FALSE NEGATIVE SAMPLES (True: POSITIVE -> Pred: NEGATIVE) [Top 3 of {len(fn_df)}]")
    lines.append("=" * 80)
    for _, row in fn_df.head(3).iterrows():
        lines.append(f"[ID #{row['id']}] Category: {row.get('category', 'N/A')}")
        lines.append(f"Text Content:\n{row['text']}")
        lines.append("-" * 80)

    report_content = "\n".join(lines)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n[INFO] Full text report preserved at: '{report_path}'")


def generate_modular_visuals(df: pd.DataFrame, output_dir: str):
    """
    Generates high-aesthetic, publication-grade individual figures in English.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Modern minimalist matplotlib configuration
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"
    plt.rcParams["axes.linewidth"] = 0.8

    y_true = df["ground_truth_sentiment"].astype(str).str.lower().str.strip()
    y_pred = df["sentiment"].fillna("none").astype(str).str.lower().str.strip()
    labels = ["negative", "neutral", "positive"]

    # -------------------------------------------------------------
    # 1. Performance Metrics
    # -------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(7, 5), dpi=300)
    valid_rate = (df["is_valid_json"].sum() / len(df)) * 100
    acc = (y_true == y_pred).mean() * 100
    rec_neg = (((y_true == "negative") & (y_pred == "negative")).sum() / (y_true == "negative").sum()) * 100
    rec_pos = (((y_true == "positive") & (y_pred == "positive")).sum() / (y_true == "positive").sum()) * 100

    metrics = ["Schema\nValidity", "Overall\nAccuracy", "Negative\nRecall", "Positive\nRecall"]
    scores = [valid_rate, acc, rec_neg, rec_pos]
    colors = ["#1D3557", "#457B9D", "#6C9A8B", "#E76F51"]

    bars1 = ax1.bar(metrics, scores, color=colors, width=0.45, edgecolor="none", zorder=3)
    ax1.set_ylim(0, 115)
    ax1.set_ylabel("Score (%)", fontsize=11, fontweight="bold", color="#333333")
    ax1.set_title("Model Extraction & Classification Performance", fontsize=13, fontweight="bold", pad=15, color="#111111")
    ax1.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax1.tick_params(axis="x", labelsize=10, colors="#222222")
    ax1.tick_params(axis="y", labelsize=10, colors="#222222")
    sns.despine(top=True, right=True)

    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                     fontsize=10.5, fontweight="bold", color="#222222")

    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "1_performance_metrics.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved: '{chart1_path}'")

    # -------------------------------------------------------------
    # 2. Confusion Matrix
    # -------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(6.5, 5), dpi=300)
    cm = pd.crosstab(y_true, y_pred).reindex(index=["negative", "positive"], columns=labels, fill_value=0).values

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2, cbar=False,
                annot_kws={"fontsize": 13, "fontweight": "bold"},
                linewidths=1.2, linecolor="white",
                xticklabels=["Negative", "Neutral", "Positive"],
                yticklabels=["Negative", "Positive"])
    ax2.set_title("Sentiment Classification Confusion Matrix", fontsize=13, fontweight="bold", pad=15, color="#111111")
    ax2.set_xlabel("Predicted Label", fontsize=11, fontweight="bold", labelpad=10, color="#333333")
    ax2.set_ylabel("Ground Truth Label", fontsize=11, fontweight="bold", labelpad=10, color="#333333")
    ax2.tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "2_confusion_matrix.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved: '{chart2_path}'")

    # -------------------------------------------------------------
    # 3. Prediction Distribution
    # -------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(6.5, 5), dpi=300)
    counts = y_pred.value_counts().reindex(labels, fill_value=0)
    palette_dist = ["#264653", "#E9C46A", "#2A9D8F"]
    bars3 = ax3.bar(["Negative", "Neutral", "Positive"], counts.values, color=palette_dist, width=0.45, zorder=3)
    ax3.set_ylabel("Number of Predictions", fontsize=11, fontweight="bold", color="#333333")
    ax3.set_title("Distribution of Predicted Sentiments", fontsize=13, fontweight="bold", pad=15, color="#111111")
    ax3.set_ylim(0, max(counts.values) + 20)
    ax3.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    sns.despine(top=True, right=True)

    for b in bars3:
        h = b.get_height()
        ax3.annotate(f"{int(h)}", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                     fontsize=11, fontweight="bold", color="#222222")

    plt.tight_layout()
    chart3_path = os.path.join(output_dir, "3_prediction_distribution.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved: '{chart3_path}'")

    # -------------------------------------------------------------
    # 4. Latency Distribution
    # -------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(5.5, 5), dpi=300)
    warm_lat = df["latency_ms"].iloc[1:] if len(df) > 1 else df["latency_ms"]
    
    # Fully backwards/forward compatible boxplot
    ax4.boxplot(warm_lat, patch_artist=True, widths=0.35,
                boxprops=dict(facecolor="#8ECAE6", color="#1D3557", linewidth=1.5),
                medianprops=dict(color="#1D3557", linewidth=2.5),
                whiskerprops=dict(color="#1D3557", linewidth=1.2),
                capprops=dict(color="#1D3557", linewidth=1.2),
                flierprops=dict(marker="o", markersize=4, markerfacecolor="#457B9D", markeredgecolor="none", alpha=0.6))
    
    ax4.set_ylabel("Inference Latency (ms)", fontsize=11, fontweight="bold", color="#333333")
    ax4.set_title("Local Inference Latency Distribution", fontsize=13, fontweight="bold", pad=15, color="#111111")
    ax4.set_xticklabels(["Qwen 2.5:7B (Warm)"], fontsize=10, fontweight="bold")
    ax4.grid(axis="y", linestyle="--", alpha=0.5)
    sns.despine(top=True, right=True)

    plt.tight_layout()
    chart4_path = os.path.join(output_dir, "4_latency_distribution.png")
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved: '{chart4_path}'\n")


def main():
    df = load_and_validate_data(RESULTS_FILE)
    generate_text_report(df, TEXT_REPORT_PATH)
    generate_modular_visuals(df, PLOTS_DIR)


if __name__ == "__main__":
    main()