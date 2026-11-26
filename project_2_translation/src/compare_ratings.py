"""Compare the blind human ratings with sentence-level BLEU, chrF and METEOR.

Run from the repository root:
    python project_2_translation/src/compare_ratings.py
"""
import csv
import statistics
from pathlib import Path

RESULTS = Path("project_2_translation/results")
HUMAN = RESULTS / "human_review.csv"
# Scores of the same run that was rated
SENTENCES = RESULTS / "test_T0.1_seed42_4636c8_sentences.csv"
OUTPUT = RESULTS / "human_vs_metrics.csv"
METRICS = ["BLEU", "chrF", "METEOR"]
GOOD, BAD = 4, 2  # a human score >= GOOD is a good translation, <= BAD a bad one
GROUP = 10  # size of the lowest / highest group: one third of the 30 sentences


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def ranks(values):
    """Rank from 1 (lowest); tied values share the average rank."""
    ordered = sorted(values)
    return [ordered.index(v) + (ordered.count(v) + 1) / 2 for v in values]


def disagreement(score, rank, n):
    """underrated: good translation, low metric. overrated: bad translation, high metric."""
    if score >= GOOD and rank <= GROUP:
        return "underrated"
    if score <= BAD and rank >= n + 1 - GROUP:
        return "overrated"
    return ""


def main():
    human = read_csv(HUMAN)
    scored = {row["id"]: row for row in read_csv(SENTENCES)}
    for row in human:
        if row["translation"] != scored[row["id"]]["output"]:
            raise SystemExit(f"id {row['id']}: the rated text differs from the scored text")

    scores = [int(row["score (1-5)"]) for row in human]
    table = [{"id": row["id"], "human": score, "note": row["note"]}
             for row, score in zip(human, scores)]

    print(f"{'metric':<8}{'Spearman':>9}  disagreements")
    for metric in METRICS:
        values = [float(scored[row["id"]][metric]) for row in human]
        metric_ranks = ranks(values)
        flags = [disagreement(s, r, len(human)) for s, r in zip(scores, metric_ranks)]
        for out, value, rank, flag in zip(table, values, metric_ranks, flags):
            out.update({metric: value, f"{metric}_rank": rank, f"{metric}_flag": flag})
        # Spearman correlation = Pearson correlation of the ranks
        rho = statistics.correlation(ranks(scores), metric_ranks)
        print(f"{metric:<8}{rho:>9.2f}  underrated {flags.count('underrated')}, "
              f"overrated {flags.count('overrated')}")
        for name in ("underrated", "overrated"):
            ids = [out["id"] for out, flag in zip(table, flags) if flag == name]
            print(f"{'':<19}{name}: {', '.join(ids)}")

    for out, row in zip(table, human):
        out.update({"source": row["source"], "reference": scored[row["id"]]["reference"],
                    "translation": row["translation"]})
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(table)
    print(f"Saved {len(table)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
