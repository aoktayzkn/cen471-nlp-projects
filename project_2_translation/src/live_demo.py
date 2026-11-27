"""Live tests for the video: translate on screen and compute the metrics at once.

Run from the repository root:
    python project_2_translation/src/live_demo.py          # 5 test sentences, two temperatures
    python project_2_translation/src/live_demo.py --new    # type new sentences
Nothing is saved, so the stored results are never changed.
"""
import argparse
import random

import nltk

from evaluate import score
from translate import (DATA_DIR, DEFAULT_MODEL, NUM_CTX, NUM_PREDICT, read_jsonl, run_metadata,
                       translate)

PROMPT = "v1"
TEMPERATURES = (0.1, 1.2)
SEED, SAMPLE_SIZE = 42, 5


def options(temperature):
    return {"temperature": temperature, "seed": SEED,
            "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT}


def run(source, temperature):
    return translate(DEFAULT_MODEL, PROMPT, source, options(temperature))["output"]


def format_scores(scores):
    return "  ".join(f"{name} {value:.2f}" for name, value in scores.items())


def compare_temperatures():
    """The same randomly chosen test sentences at a low and a high temperature."""
    items = random.Random(SEED).sample(read_jsonl(DATA_DIR / "test.jsonl"), SAMPLE_SIZE)
    references = [item["reference"] for item in items]
    outputs, totals = {}, {}
    for temperature in TEMPERATURES:
        print(f"\n===== temperature {temperature} =====")
        outputs[temperature] = []
        for item in items:
            output = run(item["source"], temperature)
            outputs[temperature].append(output)
            sentence = score([output], [item["reference"]])[1][0]
            print(f"\n[{item['id']}] {item['source']}\n  reference: {item['reference']}"
                  f"\n  output:    {output}\n  {format_scores(sentence)}")
        totals[temperature] = score(outputs[temperature], references)[0]

    print(f"\n===== corpus scores on these {SAMPLE_SIZE} sentences =====")
    for temperature, corpus in totals.items():
        print(f"T={temperature}: {format_scores(corpus)}")
    first, second = (outputs[t] for t in TEMPERATURES)
    same = sum(a == b for a, b in zip(first, second))
    print(f"Identical translations at both temperatures: {same} of {SAMPLE_SIZE}")


def new_sentences(temperature):
    """The reference is asked before translating, so the model output cannot influence it."""
    while True:
        source = input("\nEnglish sentence (empty to quit): ").strip()
        if not source:
            return
        reference = input("Your Turkish reference (Enter to skip): ").strip()
        output = run(source, temperature)
        print(f"  output: {output}")
        if reference:
            print(f"  sentence scores: {format_scores(score([output], [reference])[1][0])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", action="store_true", help="translate typed sentences")
    parser.add_argument("--temperature", type=float, default=0.1, help="used with --new")
    args = parser.parse_args()

    nltk.download("wordnet", quiet=True)  # METEOR needs it
    meta = run_metadata(DEFAULT_MODEL, PROMPT, options(args.temperature))
    print(f"{meta['model']} {meta['quantization']} | {meta['ollama_version']} | prompt {PROMPT} | "
          f"seed {SEED} | num_ctx {NUM_CTX} | num_predict {NUM_PREDICT}")
    if args.new:
        print(f"temperature {args.temperature}")
        new_sentences(args.temperature)
    else:
        compare_temperatures()


if __name__ == "__main__":
    main()
