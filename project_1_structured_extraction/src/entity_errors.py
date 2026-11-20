"""
Entity error analysis for the report: what does the model add (false positives) and
what does it miss (false negatives) compared with the gold annotation?

Run from the repository root:
    python project_1_structured_extraction/src/entity_errors.py
    python project_1_structured_extraction/src/entity_errors.py --setting prompt_only --temperature 1.0

Strict errors are split into:
  boundary  : the strict miss has a lenient partial match on the other side
              (e.g. gold "Jean-Claude Van Damme" vs predicted "Van Damme")
  pure      : no counterpart at all -> real extra / missed entity
Saves the text to reports/entity_errors*.txt.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from project_1_structured_extraction.src.evaluate_settings import (  # noqa: E402
    GOLD_CSV,
    RESULT_DIR,
    lenient_match,
    load_gold,
    norm_entity,
    salvaged_entities,
)

REPORT_DIR = "project_1_structured_extraction/reports"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["prompt_only", "schema"], default="schema")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-style", choices=["hardened", "minimal"], default="hardened")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    suffix = "" if args.prompt_style == "hardened" else f"_{args.prompt_style}"
    path = os.path.join(RESULT_DIR, f"{args.setting}{suffix}_T{args.temperature:.1f}_seed{args.seed}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. Run pipeline_settings.py first.")
    gold = load_gold(GOLD_CSV)
    if not gold:
        raise SystemExit("No annotated gold rows found.")
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    df = df[df["id"].isin(gold)]

    fp_pure, fp_boundary = defaultdict(list), defaultdict(list)
    fn_pure, fn_boundary = defaultdict(list), defaultdict(list)
    for _, r in df.iterrows():
        i = int(r["id"])
        # salvaged entities so schema-invalid rows are still analysed
        pred = {norm_entity(e): e for e in salvaged_entities(r["raw_output"]) if norm_entity(e)}
        gold_map = {norm_entity(e): e for e in gold[i] if norm_entity(e)}
        for k, e in pred.items():
            if k in gold_map:
                continue
            partial = any(lenient_match(k, g) for g in gold_map)
            (fp_boundary if partial else fp_pure)[e].append(i)
        for k, e in gold_map.items():
            if k in pred:
                continue
            partial = any(lenient_match(k, p) for p in pred)
            (fn_boundary if partial else fn_pure)[e].append(i)

    def block(title, d):
        total = sum(len(v) for v in d.values())
        lines = [f"{title}: {total} instances"]
        for e, ids in sorted(d.items(), key=lambda kv: -len(kv[1]))[: args.top]:
            lines.append(f"  {e!r:45} x{len(ids)}  ids={ids[:6]}")
        return lines

    out = [
        "=" * 90,
        f"setting={args.setting} prompt={args.prompt_style} T={args.temperature} | annotated rows={len(df)}",
        "(entities taken from the raw output whenever the JSON parses)",
        "=" * 90,
    ]
    out += block("FALSE POSITIVES, pure (model added something the gold does not have)", fp_pure)
    out += [""] + block("FALSE POSITIVES, boundary (partial match to a gold entity)", fp_boundary)
    out += [""] + block("FALSE NEGATIVES, pure (gold entity the model never produced)", fn_pure)
    out += [""] + block("FALSE NEGATIVES, boundary (partial match to a predicted entity)", fn_boundary)
    text = "\n".join(out)
    print(text)
    os.makedirs(REPORT_DIR, exist_ok=True)
    name = f"entity_errors_{args.setting}{suffix}_T{args.temperature:.1f}.txt"
    with open(os.path.join(REPORT_DIR, name), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nSaved to {REPORT_DIR}/{name}")


if __name__ == "__main__":
    main()
