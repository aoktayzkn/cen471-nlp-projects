"""
Compares the prompt-only and schema-constrained settings on the four required
metrics, plus entity F1 against the manually annotated gold file.

Run from the repository root (after pipeline_settings.py):
    python project_1_structured_extraction/src/evaluate_settings.py
    python project_1_structured_extraction/src/evaluate_settings.py --temperature 1.0

Metrics (denominator = ALL samples; an unparsable / invalid output counts as wrong):
  raw_json_rate      json.loads on the raw model output, before any cleaning
  cleaned_json_rate  parse success after cleaning (strip ``` fences, extract {...})
  schema_valid_rate  all required fields present, correct types, allowed values
  accuracy / macro_f1  sentiment vs. IMDB label
  entity P/R/F1      micro-averaged over the annotated subset, two matching rules:
                     strict  = exact match after normalisation
                     lenient = one name's tokens are contained in the other's
                               (e.g. "Van Damme" ~ "Jean-Claude Van Damme")
"""

import argparse
import json
import os
import re
import string

import pandas as pd

RESULT_DIR = "project_1_structured_extraction/data/processed/settings"
GOLD_CSV = "project_1_structured_extraction/data/annotation/entity_gold.csv"
REPORT_MD = "project_1_structured_extraction/reports/settings_comparison.md"
SETTINGS = ["prompt_only", "schema"]


# ----------------------------- entity matching ------------------------------
def norm_entity(s):
    """Lowercase, drop possessive 's, strip punctuation, collapse spaces.
    (English data. For Turkish text map 'İ'->'i' and 'I'->'ı' BEFORE lower().)"""
    s = str(s).strip().replace("’", "'")
    s = re.sub(r"'s\b", "", s)
    s = s.translate(str.maketrans({c: " " for c in string.punctuation if c != "-"}))
    s = s.replace("-", " ")
    return re.sub(r"\s+", " ", s).strip().lower()


def lenient_match(a, b):
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return False
    sa, sb = set(ta), set(tb)
    return sa <= sb or sb <= sa


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def entity_counts(gold, pred, lenient):
    gold = {g for g in map(norm_entity, gold) if g}
    pred = {p for p in map(norm_entity, pred) if p}
    if not lenient:
        tp = len(gold & pred)
        return tp, len(pred) - tp, len(gold) - tp
    unmatched = set(pred)
    tp = 0
    for g in sorted(gold):
        hit = next((p for p in sorted(unmatched) if p == g or lenient_match(g, p)), None)
        if hit is not None:
            unmatched.discard(hit)
            tp += 1
    return tp, len(unmatched), len(gold) - tp


def load_gold(path):
    if not os.path.exists(path):
        return None
    g = pd.read_csv(path, encoding="utf-8-sig")
    g["gold_entities"] = g["gold_entities"].fillna("").astype(str).str.strip()
    g = g[g["gold_entities"] != ""].copy()  # annotated rows only
    g["gold_list"] = g["gold_entities"].apply(
        lambda s: [] if s.upper() == "NONE" else [x.strip() for x in s.split(";") if x.strip()]
    )
    return g.set_index("id")["gold_list"].to_dict()


# ------------------------------- metrics ------------------------------------
def sentiment_metrics(df):
    y = df["ground_truth_sentiment"].astype(str).str.lower()
    p = df["sentiment"].fillna("").astype(str).str.lower()
    f1s = []
    for lab in ("positive", "negative"):
        tp = ((y == lab) & (p == lab)).sum()
        fp = ((y != lab) & (p == lab)).sum()
        fn = ((y == lab) & (p != lab)).sum()
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float((y == p).mean()), float(sum(f1s) / 2)


def evaluate_one(df, gold):
    acc, macro = sentiment_metrics(df)
    out = {
        "n": len(df),
        "raw_json_rate": df["raw_json_ok"].astype(bool).mean(),
        "cleaned_json_rate": df["cleaned_json_ok"].astype(bool).mean(),
        "schema_valid_rate": df["schema_valid"].astype(bool).mean(),
        "accuracy": acc,
        "macro_f1": macro,
        "truncated_outputs": int((df["done_reason"] == "length").sum()),
        "mean_latency_ms": df["latency_ms"].mean(),
    }
    if gold:
        sub = df[df["id"].isin(gold)]
        out["entity_n"] = len(sub)
        for name, lenient in (("strict", False), ("lenient", True)):
            tp = fp = fn = 0
            for _, r in sub.iterrows():
                a, b, c = entity_counts(gold[r["id"]], json.loads(r["entities_json"]), lenient)
                tp, fp, fn = tp + a, fp + b, fn + c
            p, rec, f = prf(tp, fp, fn)
            out.update({f"ent_{name}_P": p, f"ent_{name}_R": rec, f"ent_{name}_F1": f})
    return out


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test on discordant pairs (b, c)."""
    from scipy.stats import binomtest

    n = b + c
    return 1.0 if n == 0 else float(binomtest(min(b, c), n, 0.5).pvalue)


def paired_report(a, b):
    m = a.merge(b, on="id", suffixes=("_po", "_sc"))
    ok_po = m["sentiment_po"].fillna("") == m["ground_truth_sentiment_po"]
    ok_sc = m["sentiment_sc"].fillna("") == m["ground_truth_sentiment_sc"]
    only_po, only_sc = int((ok_po & ~ok_sc).sum()), int((~ok_po & ok_sc).sum())
    same_pred = int((m["sentiment_po"].fillna("") == m["sentiment_sc"].fillna("")).sum())
    same_raw = int((m["raw_output_po"] == m["raw_output_sc"]).sum())
    return {
        "paired_n": len(m),
        "identical_raw_output": same_raw,
        "identical_sentiment": same_pred,
        "correct_only_prompt_only": only_po,
        "correct_only_schema": only_sc,
        "mcnemar_p": mcnemar_exact(only_po, only_sc),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-style", choices=["hardened", "minimal"], default="hardened")
    args = ap.parse_args()

    frames = {}
    for s in SETTINGS:
        suffix = "" if args.prompt_style == "hardened" else f"_{args.prompt_style}"
        path = os.path.join(RESULT_DIR, f"{s}{suffix}_T{args.temperature:.1f}_seed{args.seed}.csv")
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}. Run pipeline_settings.py first.")
        frames[s] = pd.read_csv(path, keep_default_na=False, na_values=[""])
        frames[s]["entities_json"] = frames[s]["entities_json"].fillna("[]")
        frames[s]["done_reason"] = frames[s]["done_reason"].fillna("")

    gold = load_gold(GOLD_CSV)
    if gold is None:
        print("[note] No gold file yet -> entity F1 skipped (run build_annotation_template.py).")
    elif len(gold) < 100:
        print(f"[warn] Only {len(gold)} annotated rows; the assignment needs >= 100.")

    table = pd.DataFrame({s: evaluate_one(frames[s], gold) for s in SETTINGS}).T
    paired = paired_report(frames["prompt_only"], frames["schema"])

    pd.set_option("display.width", 200)
    print(f"\n=== prompt={args.prompt_style}, T={args.temperature}, seed={args.seed} ===")
    print(table.T.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nPaired comparison:", json.dumps(paired, indent=2))

    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    with open(REPORT_MD, "a", encoding="utf-8") as f:
        f.write(f"\n## prompt={args.prompt_style}, T={args.temperature}, seed={args.seed}\n\n")
        try:
            f.write(table.T.to_markdown(floatfmt=".4f"))
        except ImportError:  # `tabulate` not installed
            f.write(table.T.to_string(float_format=lambda x: f"{x:.4f}"))
        f.write("\n\nPaired comparison: " + json.dumps(paired) + "\n")
    print(f"\nAppended to {REPORT_MD}")


if __name__ == "__main__":
    main()
