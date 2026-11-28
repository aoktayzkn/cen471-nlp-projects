"""Create a blind rating sheet: 30 random test translations without references or scores.

Run from the repository root:
    python project_2_translation/src/make_review_sheet.py
"""
import csv
import json
import random
from pathlib import Path

# The run to be judged, fixed before looking at any sentence score
RUN = Path("project_2_translation/data/outputs/test_T0.1_seed42_4636c8.jsonl")
SHEET = Path("project_2_translation/results/human_review.csv")
SAMPLE_SIZE, SEED = 30, 42


def main():
    if SHEET.exists():
        raise SystemExit(f"{SHEET} already exists. It is not overwritten, so ratings are safe.")
    with RUN.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    sample = random.Random(SEED).sample(rows, SAMPLE_SIZE)  # random order as well

    SHEET.parent.mkdir(parents=True, exist_ok=True)
    # ";" and utf-8-sig so that Turkish-locale Excel opens the file correctly
    with SHEET.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["id", "source", "translation", "score (1-5)", "note"])
        for row in sample:
            writer.writerow([row["id"], row["source"], row["output"], "", ""])
    print(f"Saved {len(sample)} translations to {SHEET}")


if __name__ == "__main__":
    main()
