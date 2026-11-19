"""
Lists every output that violated the schema (or was unparsable) in the prompt-only
setting, grouped by failure type, for the report and the defense video.

Run from the repository root (after pipeline_settings.py):
    python project_1_structured_extraction/src/show_failures.py
    python project_1_structured_extraction/src/show_failures.py --temperatures 0.0 1.0
    python project_1_structured_extraction/src/show_failures.py --prompt-style minimal
    python project_1_structured_extraction/src/show_failures.py --prompt-style minimal --rescued

Prints to the console and saves the same text to reports/format_failures*.txt.
"""

import argparse
import os
import sys
from collections import Counter

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from project_1_structured_extraction.src.parser import parse_json_stages  # noqa: E402

RESULT_DIR = "project_1_structured_extraction/data/processed/settings"
REPORT_DIR = "project_1_structured_extraction/reports"


def failure_kind(error):
    e = str(error)
    if e.startswith("JSON decode failed") or e.startswith("No JSON") or "not an object" in e:
        return "unparsable_json"
    if e.startswith("category not allowed"):
        return "invalid_category"
    if e.startswith("sentiment not allowed"):
        return "invalid_sentiment"
    if e.startswith("missing keys"):
        return "missing_keys"
    if e.startswith("entities"):
        return "invalid_entities"
    return "other"


def rescue_kind(raw):
    text = raw.lstrip() if isinstance(raw, str) else ""
    if text.startswith("```"):
        return "markdown_fence"
    if not text.startswith("{"):
        return "text_before_json"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-style", choices=["hardened", "minimal"], default="hardened")
    ap.add_argument("--setting", choices=["prompt_only", "schema"], default="prompt_only")
    ap.add_argument(
        "--rescued", action="store_true",
        help="Also list outputs that json.loads rejected but the cleaning code recovered.",
    )
    ap.add_argument("--examples", type=int, default=3, help="Rescued examples to print per type.")
    args = ap.parse_args()

    suffix = "" if args.prompt_style == "hardened" else f"_{args.prompt_style}"
    lines, failing_ids = [], {}
    for t in args.temperatures:
        path = os.path.join(RESULT_DIR, f"{args.setting}{suffix}_T{t:.1f}_seed{args.seed}.csv")
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}. Run pipeline_settings.py first.")
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
        bad = df[~df["schema_valid"].astype(bool)].copy()
        failing_ids[t] = set(int(i) for i in bad["id"])

        lines.append("=" * 90)
        lines.append(
            f"setting={args.setting} prompt={args.prompt_style} T={t} seed={args.seed}: "
            f"{len(bad)}/{len(df)} outputs invalid"
        )
        lines.append("=" * 90)
        kinds = Counter(failure_kind(e) for e in bad["schema_error"])
        lines.append("By type: " + (", ".join(f"{k}={v}" for k, v in sorted(kinds.items())) or "none"))
        lines.append("")
        for _, r in bad.iterrows():
            raw = r["raw_output"] if isinstance(r["raw_output"], str) else ""
            data = parse_json_stages(raw)["data"]
            said = data.get("sentiment") if isinstance(data, dict) else None
            lines.append(f"[id {int(r['id'])}] type={failure_kind(r['schema_error'])}")
            lines.append(f"  error     : {r['schema_error']}")
            lines.append(f"  true label: {r['ground_truth_sentiment']} | sentiment in output: {said}")
            lines.append(f"  raw output: {raw[:400]!r}")
            lines.append("")

    if args.rescued:
        for t in args.temperatures:
            path = os.path.join(RESULT_DIR, f"{args.setting}{suffix}_T{t:.1f}_seed{args.seed}.csv")
            df = pd.read_csv(path, keep_default_na=False, na_values=[""])
            resc = df[(~df["raw_json_ok"].astype(bool)) & df["cleaned_json_ok"].astype(bool)]
            lines.append("=" * 90)
            lines.append(
                f"RESCUED BY CLEANING: setting={args.setting} prompt={args.prompt_style} T={t}: "
                f"{len(resc)}/{len(df)} outputs failed json.loads but parsed after cleaning"
            )
            lines.append("=" * 90)
            by_type = {}
            for _, r in resc.iterrows():
                by_type.setdefault(rescue_kind(r["raw_output"]), []).append(r)
            lines.append("By type: " + (", ".join(f"{k}={len(v)}" for k, v in sorted(by_type.items())) or "none"))
            lines.append("")
            for kind, rows in sorted(by_type.items()):
                for r in rows[: args.examples]:
                    lines.append(f"[id {int(r['id'])}] type={kind}")
                    lines.append(f"  raw output: {str(r['raw_output'])[:300]!r}")
                    lines.append("")

    if len(args.temperatures) > 1:
        common = set.intersection(*failing_ids.values())
        lines.append("-" * 90)
        lines.append(f"Ids invalid at ALL listed temperatures: {sorted(common) or 'none'}")
        for t, ids in failing_ids.items():
            lines.append(f"  invalid only at T={t}: {sorted(ids - common) or 'none'}")

    text = "\n".join(lines)
    print(text)
    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, f"format_failures{suffix}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
