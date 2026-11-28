"""Plot metric scores and output stability against temperature.

Run from the repository root, after evaluate.py --split test:
    python project_2_translation/src/plot.py
"""
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")  # save to file, no window
import matplotlib.pyplot as plt

OUT_DIR = Path("project_2_translation/data/outputs")
RESULT_DIR = Path("project_2_translation/results")
MODEL, PROMPT = "gemma3:4b", "v1"
# metric -> (colour, marker); different markers keep the lines readable in greyscale
STYLE = {"BLEU": ("#257de0", "o"), "chrF": ("#f47230", "s"), "METEOR": ("#00b894", "^")}


def load_runs():
    """Group the test runs of the chosen model and prompt by temperature."""
    with (RESULT_DIR / "test_summary.csv").open(encoding="utf-8-sig") as f:
        runs = [r for r in csv.DictReader(f, delimiter=";")
                if r["model"] == MODEL and r["prompt"] == PROMPT]
    if not runs:
        raise SystemExit("No matching runs. Run evaluate.py --split test first.")
    groups = {}
    for run in sorted(runs, key=lambda r: float(r["temperature"])):
        groups.setdefault(float(run["temperature"]), []).append(run)
    return groups


def identical_share(runs):
    """Percentage of sentences translated identically under every seed."""
    outputs = {}
    for run in runs:
        with (OUT_DIR / run["file"]).open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                outputs.setdefault(row["id"], []).append(row["output"])
    same = sum(len(set(texts)) == 1 for texts in outputs.values())
    return 100 * same / len(outputs)


def summarise(groups):
    table = []
    for temperature, runs in groups.items():
        row = {"temperature": temperature, "seeds": len(runs)}
        for metric in STYLE:
            values = [float(r[metric]) for r in runs]
            row[f"{metric}_mean"] = round(mean(values), 2)
            row[f"{metric}_min"] = min(values)
            row[f"{metric}_max"] = max(values)
        row["identical_percent"] = round(identical_share(runs), 1)
        table.append(row)
    return table


def draw(table, path):
    temps = [row["temperature"] for row in table]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.6), width_ratios=[3, 2])

    for metric, (colour, marker) in STYLE.items():
        means = [row[f"{metric}_mean"] for row in table]
        # whiskers show the lowest and highest score among the seeds
        spread = [[row[f"{metric}_mean"] - row[f"{metric}_min"] for row in table],
                  [row[f"{metric}_max"] - row[f"{metric}_mean"] for row in table]]
        left.errorbar(temps, means, yerr=spread, color=colour, marker=marker,
                      linewidth=2, markersize=7, capsize=4, label=metric)
        left.annotate(f"{metric} {means[-1]:.1f}", (temps[-1], means[-1]),
                      xytext=(10, 0), textcoords="offset points", va="center", fontsize=9)
    left.set_ylim(0, 60)  # starts at zero so small differences are not exaggerated
    left.set_xlim(0, 1.55)
    left.set_title("Corpus score (mean, min-max over seeds)", fontsize=10)

    shares = [row["identical_percent"] for row in table]
    right.plot(temps, shares, color="#444444", marker="o", linewidth=2, markersize=7)
    for t, share in zip(temps, shares):
        right.annotate(f"{share:.0f}%", (t, share), xytext=(0, 9),
                       textcoords="offset points", ha="center", fontsize=9)
    right.set_ylim(0, 100)
    right.set_xlim(-0.05, 1.35)
    right.set_title("Sentences identical across all seeds", fontsize=10)

    for axis in (left, right):
        axis.set_xticks(temps)
        axis.set_xlabel("Temperature")
        axis.grid(axis="y", color="#e7e7e7", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    left.legend(frameon=False, ncols=3, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main():
    table = summarise(load_runs())
    with (RESULT_DIR / "temperature_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(table)
    draw(table, RESULT_DIR / "temperature_plot.png")
    for row in table:
        print(row)
    print(f"Saved the table and the plot to {RESULT_DIR}")


if __name__ == "__main__":
    main()
