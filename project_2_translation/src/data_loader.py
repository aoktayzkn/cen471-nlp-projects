"""Stream WMT18 tr-en and save seeded dev and test samples (English -> Turkish).

dev.jsonl  <- validation split: prompt tuning
test.jsonl <- test split: final experiments only

Run from the repository root:
    python project_2_translation/src/data_loader.py
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

REVISION = "ceae0f932f739f66498b7695b1807878f82d9b22"  # fixed dataset commit
SEED = 42
SIZES = {"dev": ("validation", 40), "test": ("test", 150)}
MIN_WORDS, MAX_WORDS = 5, 50  # English source length filter (state this in the report)
OUTPUT_DIR = Path("project_2_translation/data/raw")


def select(rows, n, seed=SEED):
    """Filter by length, drop duplicate sources, then take a seeded random sample."""
    seen, pool = set(), []
    for index, row in enumerate(rows):
        source = row["translation"]["en"].strip()
        reference = row["translation"]["tr"].strip()
        if not reference or source in seen:
            continue
        if not MIN_WORDS <= len(source.split()) <= MAX_WORDS:
            continue
        seen.add(source)
        pool.append({"id": index, "source": source, "reference": reference})
    if len(pool) < n:
        raise ValueError(f"Only {len(pool)} usable pairs, {n} requested")
    return sorted(random.Random(seed).sample(pool, n), key=lambda r: r["id"])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (split, n) in SIZES.items():
        print(f"Streaming the {split} split from Hugging Face...")
        rows = load_dataset(
            "wmt/wmt18", "tr-en", split=split, revision=REVISION, streaming=True
        )
        sample = select(rows, n)
        path = OUTPUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for item in sample:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(sample)} pairs to: {path}")


if __name__ == "__main__":
    main()
