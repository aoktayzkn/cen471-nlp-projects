"""Show how the metrics react when one thing in a saved sentence is repaired by hand.

Run from the repository root:
    python project_2_translation/src/defense_checks.py
Only saved outputs are read; no model is called and nothing is saved.
"""
import nltk

from evaluate import score
from translate import OUT_DIR, read_jsonl

RUNS = {"dev": "dev_T0.1_seed42_4636c8.jsonl", "test": "test_T0.1_seed42_4636c8.jsonl"}
# (title, split, sentence id, text to repair, replacements made by hand)
CHECKS = [
    ("Omission: 'I said' restored", "dev", 2323, "output",
     [("değilim.", "değilim dedim.")]),
    ("Omission: 'imprisonment' restored", "test", 2952, "output",
     [("cezayı", "hapis cezasını")]),
    ("Merged words in the reference separated", "test", 1153, "reference",
     [("fırsatıverdikleri", "fırsatı verdikleri"), ("teşekkürederim", "teşekkür ederim")]),
]


def repair(text, replacements):
    for old, new in replacements:
        if old not in text:
            raise SystemExit(f"'{old}' not found: the saved text has changed.")
        text = text.replace(old, new)
    return text


def main():
    nltk.download("wordnet", quiet=True)  # METEOR needs it
    for title, split, number, field, replacements in CHECKS:
        row = next(r for r in read_jsonl(OUT_DIR / RUNS[split]) if r["id"] == number)
        fixed = {**row, field: repair(row[field], replacements)}
        before, after = (score([r["output"]], [r["reference"]])[1][0] for r in (row, fixed))
        print(f"\n===== {title} ({split} id {number}) =====")
        print(f"source:    {row['source']}\nreference: {row['reference']}\noutput:    {row['output']}")
        print(f"repaired {field}: {fixed[field]}")
        for metric in before:
            print(f"  {metric:<7}{before[metric]:>7.2f} -> {after[metric]:>6.2f}  "
                  f"({after[metric] - before[metric]:+.2f})")


if __name__ == "__main__":
    main()
