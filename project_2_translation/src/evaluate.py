"""Score saved translations with corpus BLEU, chrF and METEOR.

Run from the repository root:
    python project_2_translation/src/evaluate.py --split dev
"""
import argparse
import csv
import json
import re
from pathlib import Path

import nltk
from nltk.translate.meteor_score import meteor_score
from sacrebleu.metrics import BLEU, CHRF

DATA_DIR = Path("project_2_translation/data/raw")
OUT_DIR = Path("project_2_translation/data/outputs")
RESULT_DIR = Path("project_2_translation/results")


def turkish_lower(token):
    """Python's lower() turns I into i; Turkish needs I -> ı and İ -> i."""
    return token.replace("I", "ı").replace("İ", "i").lower()


def tokenize(text):
    """Split into words and punctuation marks."""
    return re.findall(r"\w+|[^\w\s]", text)


def meteor_sentence(hypothesis, reference):
    return meteor_score(
        [tokenize(reference)],
        tokenize(hypothesis),
        preprocess=turkish_lower,
    )


def score(hypotheses, references):
    """Return corpus-level scores, sentence-level scores and sacrebleu signatures."""
    bleu, chrf = BLEU(), CHRF()
    sentence_bleu = BLEU(effective_order=True)  # meant for single sentences
    sentences = [
        {
            "BLEU": sentence_bleu.sentence_score(h, [r]).score,
            "chrF": chrf.sentence_score(h, [r]).score,
            "METEOR": 100 * meteor_sentence(h, r),
        }
        for h, r in zip(hypotheses, references)
    ]
    corpus = {
        "BLEU": bleu.corpus_score(hypotheses, [references]).score,
        "chrF": chrf.corpus_score(hypotheses, [references]).score,
        # NLTK has no corpus-level METEOR: report the mean of the sentence scores
        "METEOR": sum(s["METEOR"] for s in sentences) / len(sentences),
    }
    signatures = {"BLEU": str(bleu.get_signature()), "chrF": str(chrf.get_signature())}
    return corpus, sentences, signatures


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path, rows):
    # ";" and utf-8-sig so that Turkish-locale Excel opens the file correctly
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()

    nltk.download("wordnet", quiet=True)  # METEOR needs it
    expected = {item["id"] for item in read_jsonl(DATA_DIR / f"{args.split}.jsonl")}
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    summary, signatures = [], {}
    for path in sorted(OUT_DIR.glob(f"{args.split}_*.jsonl")):
        rows = read_jsonl(path)
        if {r["id"] for r in rows} != expected:
            # scores of different sentence sets are not comparable
            print(f"skipped (incomplete, {len(rows)}/{len(expected)}): {path.name}")
            continue
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        corpus, sentences, signatures = score(
            [r["output"] for r in rows], [r["reference"] for r in rows]
        )
        summary.append({
            "model": meta["model"],
            "temperature": meta["options"]["temperature"],
            "seed": meta["options"]["seed"],
            "n": len(rows),
            **{name: round(value, 2) for name, value in corpus.items()},
            "file": path.name,
        })
        write_csv(RESULT_DIR / f"{path.stem}_sentences.csv", [
            {"id": r["id"], **{k: round(v, 2) for k, v in s.items()},
             "source": r["source"], "reference": r["reference"], "output": r["output"]}
            for r, s in zip(rows, sentences)
        ])

    if not summary:
        raise SystemExit("No complete output files. Run translate.py first, from the repository root.")
    write_csv(RESULT_DIR / f"{args.split}_summary.csv", summary)
    settings = {
        "signatures": signatures,
        "meteor": {"library": f"nltk {nltk.__version__}", "tokenizer": r"\w+|[^\w\s]",
                   "lowercase": "Turkish-aware", "corpus": "mean of sentence scores"},
    }
    (RESULT_DIR / f"{args.split}_metric_settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'model':<14}{'T':>5}{'seed':>6}{'n':>5}{'BLEU':>8}{'chrF':>8}{'METEOR':>8}")
    for s in summary:
        print(f"{s['model']:<14}{s['temperature']:>5}{s['seed']:>6}{s['n']:>5}"
              f"{s['BLEU']:>8.2f}{s['chrF']:>8.2f}{s['METEOR']:>8.2f}")
    print("\n".join(signatures.values()))


if __name__ == "__main__":
    main()
