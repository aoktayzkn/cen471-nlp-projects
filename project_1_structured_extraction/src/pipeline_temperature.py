"""
Inference pipeline for Project 1.
Reads the IMDB sample dataset, calls local LLM via Ollama, parses/validates JSON,
and logs latency and schema compliance metrics.

Examples:
    python pipeline.py
    python pipeline.py --temperature 0.3
    python pipeline.py --all-temperatures
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
import ollama

# Üst dizindeki modülleri görebilmek için sys.path ayarı
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from project_1_structured_extraction.prompts.extraction_prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
)
from project_1_structured_extraction.src.parser import parse_and_validate_json

MODEL_NAME = "qwen2.5:7b"
INPUT_CSV = "project_1_structured_extraction/data/raw/imdb_balanced_200.csv"
PROCESSED_DIR = "project_1_structured_extraction/data/processed"
COMPARISON_CSV = os.path.join(PROCESSED_DIR, "temperature_comparison.csv")

# Temperature = 0.0 is retained as the reference/baseline setting.
# It is a common starting point for classification experiments where consistency
# is desirable, but it is NOT a universal requirement or guarantee of determinism.
DEFAULT_TEMPERATURE = 0.0
TEMPERATURES_TO_COMPARE = [0.0, 0.3, 0.7, 1.0]


def safe_json_loads(raw_output):
    """Return True only if the raw response is directly parseable as JSON."""
    try:
        parsed = json.loads(raw_output)
        return isinstance(parsed, dict)
    except (json.JSONDecodeError, TypeError):
        return False


def calculate_metrics(results_df, temperature):
    """Calculate output-format, sentiment and latency metrics for one run."""
    valid_count = int(results_df["is_valid_json"].fillna(False).astype(bool).sum())
    raw_json_count = int(results_df["raw_json_parse_first_try"].fillna(False).astype(bool).sum())
    total = len(results_df)

    # The evaluation dataset contains positive/negative ground-truth labels.
    # Predictions such as "neutral" are counted as incorrect for accuracy.
    evaluation_rows = results_df[results_df["ground_truth_sentiment"].notna()].copy()
    true_labels = evaluation_rows["ground_truth_sentiment"].astype(str).str.lower()
    predicted_labels = evaluation_rows["sentiment"].fillna("").astype(str).str.lower()

    if total:
        accuracy = (predicted_labels == true_labels).sum() / total
        raw_json_rate = raw_json_count / total
        parser_valid_rate = valid_count / total
        average_latency = results_df["latency_ms"].mean()
    else:
        accuracy = raw_json_rate = parser_valid_rate = average_latency = 0.0

    # Macro-F1 over the two ground-truth classes (positive and negative).
    f1_scores = []
    for label in ("positive", "negative"):
        tp = int((
            (true_labels == label) & (predicted_labels == label)
        ).sum())
        fp = int((
            (true_labels != label) & (predicted_labels == label)
        ).sum())
        fn = int((
            (true_labels == label) & (predicted_labels != label)
        ).sum())
        denominator = (2 * tp) + fp + fn
        f1_scores.append((2 * tp / denominator) if denominator else 0.0)

    return {
        "temperature": float(temperature),
        "samples": total,
        "raw_json_parse_first_try_rate": round(raw_json_rate, 4),
        "parser_validation_success_rate": round(parser_valid_rate, 4),
        "accuracy": round(float(accuracy), 4),
        "macro_f1_positive_negative": round(sum(f1_scores) / len(f1_scores), 4),
        "average_latency_ms": round(float(average_latency), 2),
    }


def update_comparison_file(metrics):
    """Update only this temperature's row; retain previous experiment rows."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    new_row = pd.DataFrame([metrics])

    if os.path.exists(COMPARISON_CSV):
        previous = pd.read_csv(COMPARISON_CSV)
        if "temperature" in previous.columns:
            previous = previous[previous["temperature"] != metrics["temperature"]]
        combined = pd.concat([previous, new_row], ignore_index=True)
    else:
        combined = new_row

    combined = combined.sort_values("temperature").reset_index(drop=True)
    combined.to_csv(COMPARISON_CSV, index=False, encoding="utf-8")
    print(f"Temperature comparison updated: '{COMPARISON_CSV}'")


def run_extraction_pipeline(temperature):
    if not os.path.exists(INPUT_CSV):
        print(
            f"Error: Input file '{INPUT_CSV}' not found. "
            "Run data_loader.py first, and execute this script from the project root."
        )
        return

    print(f"Loading input samples from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    results = []
    print(
        f"Starting inference with '{MODEL_NAME}' across {len(df)} samples "
        f"(temperature={temperature})...\n"
    )

    for _, row in df.iterrows():
        sample_id = row["id"]
        review_text = row["text"]
        ground_truth = row["ground_truth_sentiment"]
        user_prompt = build_user_prompt(review_text)

        start_time = time.perf_counter()
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                format="json",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    # Temperature = 0.0 is a common baseline for classification
                    # tasks that prioritize consistency; it is not a universal standard.
                    "temperature": float(temperature)
                },
            )
            raw_output = response["message"]["content"]
        except Exception as exc:
            raw_output = f"ERROR: {exc}"

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Measure raw JSON parsing separately from the project's parser/validation.
        raw_json_parse_first_try = safe_json_loads(raw_output)

        # Parser and schema-validation check (may include cleanup/normalization).
        is_valid, parsed_data, error_msg = parse_and_validate_json(raw_output)

        extracted_category = parsed_data.get("category", "") if is_valid else None
        extracted_sentiment = parsed_data.get("sentiment", "") if is_valid else None
        extracted_entities = parsed_data.get("entities", []) if is_valid else []

        results.append(
            {
                "id": sample_id,
                "text": review_text,
                "ground_truth_sentiment": ground_truth,
                "raw_output": raw_output,
                "raw_json_parse_first_try": raw_json_parse_first_try,
                "is_valid_json": is_valid,
                "validation_error": error_msg,
                "category": extracted_category,
                "sentiment": extracted_sentiment,
                "entities": str(extracted_entities),
                "latency_ms": round(elapsed_ms, 2),
                "temperature": float(temperature),
            }
        )

        status_flag = "VALID" if is_valid else "INVALID"
        print(
            f"[{sample_id}/{len(df)}] {status_flag} | "
            f"Latency: {elapsed_ms:.1f}ms | Pred: {extracted_sentiment} "
            f"(True: {ground_truth})"
        )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_df = pd.DataFrame(results)

    # Use a temperature-specific filename so the old extracted_results.csv
    # and results from other temperature runs are never overwritten.
    temp_label = f"{float(temperature):.1f}"
    output_csv = os.path.join(
        PROCESSED_DIR, f"extracted_results_temp_{temp_label}.csv"
    )
    out_df.to_csv(output_csv, index=False, encoding="utf-8")

    metrics = calculate_metrics(out_df, temperature)
    update_comparison_file(metrics)

    print(f"\nRun completed. Detailed results saved to: '{output_csv}'")
    print("Run metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Project 1 extraction at a selected temperature."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature for this run (default: 0.0).",
    )
    parser.add_argument(
        "--all-temperatures",
        action="store_true",
        help="Run the comparison values: 0.0, 0.3, 0.7, and 1.0.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    temperatures = (
        TEMPERATURES_TO_COMPARE if args.all_temperatures else [args.temperature]
    )
    for temp in temperatures:
        run_extraction_pipeline(temp)
