"""Count tokens of the English sources and Turkish references with the model's own tokenizer,
and relate the number of generated tokens to translation time in a saved run.

Run from the repository root:
    python project_2_translation/src/token_counts.py
"""
import csv
import statistics
from pathlib import Path

import ollama

from translate import DEFAULT_MODEL, NUM_CTX, OUT_DIR, read_jsonl

RUN = OUT_DIR / "test_T0.1_seed42_4636c8.jsonl"
OUTPUT = Path("project_2_translation/results/token_counts.csv")


def count_tokens(text):
    """raw=True sends the text alone, without the system prompt or the chat template."""
    response = ollama.generate(model=DEFAULT_MODEL, prompt=text, raw=True,
                               options={"num_predict": 1, "num_ctx": NUM_CTX})
    return response["prompt_eval_count"]


def main():
    run = read_jsonl(RUN)
    if not run:
        raise SystemExit(f"{RUN} not found. Run from the repository root.")
    # If Ollama reported only uncached tokens, a repeated text would give a smaller count
    if count_tokens(run[0]["source"]) != count_tokens(run[0]["source"]):
        raise SystemExit("Token counts differ for the same text, so they cannot be trusted.")

    rows = [{
        "id": item["id"],
        "en_words": len(item["source"].split()),
        "tr_words": len(item["reference"].split()),
        "en_tokens": count_tokens(item["source"]),
        "tr_tokens": count_tokens(item["reference"]),
        "output_tokens": item["output_tokens"],
        "seconds": item["seconds"],
    } for item in run]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    total = {key: sum(row[key] for row in rows) for key in rows[0] if key != "id"}
    more = sum(row["tr_tokens"] > row["en_tokens"] for row in rows)
    print(f"{len(rows)} sentence pairs, tokenizer of {DEFAULT_MODEL}")
    for language, name in (("en", "English source"), ("tr", "Turkish reference")):
        tokens, words = total[f"{language}_tokens"], total[f"{language}_words"]
        print(f"{name:<18} words {words:>5}  tokens {tokens:>5}  tokens per word {tokens / words:.2f}")
    print(f"Turkish / English tokens: {total['tr_tokens'] / total['en_tokens']:.2f}  "
          f"(more tokens in Turkish for {more} of {len(rows)} pairs)")
    correlation = statistics.correlation([row["output_tokens"] for row in rows],
                                         [row["seconds"] for row in rows])
    print(f"Translation run: {total['output_tokens']} generated tokens in {total['seconds']:.1f} s "
          f"= {total['output_tokens'] / total['seconds']:.1f} tokens per second (total time)")
    print(f"Correlation between generated tokens and seconds: {correlation:.2f}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
