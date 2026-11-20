"""
Live demo for the defense video: runs the SAME review(s) through both settings
(prompt_only vs schema) at one or more temperatures and prints a side-by-side summary.

Run from the repository root:

    # your own new reviews (not in the dataset), blocks separated by a line with ---
    python project_1_structured_extraction/src/live_demo.py --file project_1_structured_extraction/data/live_demo/new_reviews.txt

    # one review typed on the command line
    python project_1_structured_extraction/src/live_demo.py --text "Great cast, but the plot made no sense."

    # reviews from the project dataset, by id (the "same examples in both settings" part)
    python project_1_structured_extraction/src/live_demo.py --ids 2 136 156

    # paste reviews one at a time while recording (empty line = finish the review)
    python project_1_structured_extraction/src/live_demo.py --interactive

Useful options:
    --temperatures 0.0 1.0     (default: both)
    --prompt-style minimal     (shows why the cleaning code is needed: raw json.loads often fails)
    --show-raw                 (always print the raw model output)

Optional label per review in the --file: put `# label: positive` (or negative) as the
first line of the block and the demo marks the sentiment as correct/wrong.

Model, seed, num_ctx, num_predict and prompts are the ones used in the experiments, so the
results are comparable. A warm-up call is made first so the first latency is not inflated.
Every run is appended to reports/live_demo_log.csv.
"""

import argparse
import csv
import datetime as dt
import os
import re
import sys
import time

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from project_1_structured_extraction.src.parser import (  # noqa: E402
    parse_json_stages,
    strict_schema_check,
)
from project_1_structured_extraction.src.pipeline_settings import (  # noqa: E402
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    DEFAULT_SEED,
    INPUT_CSV,
    MODEL_NAME,
    call_model,
    get_system_prompt_for,
)

LOG_CSV = "project_1_structured_extraction/reports/live_demo_log.csv"
SETTINGS = ["prompt_only", "schema"]


def read_blocks(path):
    raw = open(path, encoding="utf-8").read()
    out = []
    for block in re.split(r"^---\s*$", raw, flags=re.M):
        block = block.strip()
        if not block:
            continue
        label = None
        m = re.match(r"#\s*label:\s*(positive|negative)\s*\n", block, flags=re.I)
        if m:
            label, block = m.group(1).lower(), block[m.end():].strip()
        if block.upper().startswith("PASTE "):
            print(f"[skip] placeholder block not filled in: {block[:40]!r}")
            continue
        out.append((label, block))
    return out


def read_interactive():
    items = []
    print("Paste a review, then press Enter on an empty line. Type 'done' on its own line to start.")
    while True:
        lines = []
        while True:
            try:
                line = input("> ")
            except EOFError:
                line = "done"
            if line.strip().lower() == "done" and not lines:
                return items
            if line.strip() == "":
                break
            lines.append(line)
        if lines:
            items.append((None, " ".join(lines)))


def run_one(text, setting, temperature, args, system_prompt):
    start = time.perf_counter()
    try:
        raw, done = call_model(text, setting, temperature, args.seed, args.num_ctx, args.num_predict, system_prompt)
        error = ""
    except Exception as exc:  # noqa: BLE001
        raw, done, error = "", "error", str(exc)
    ms = (time.perf_counter() - start) * 1000
    stages = parse_json_stages(raw)
    ok, reason = strict_schema_check(stages["data"])
    data = stages["data"] if isinstance(stages["data"], dict) else {}
    return {
        "raw": raw, "done": done, "error": error, "ms": ms,
        "raw_ok": stages["raw_ok"], "cleaned_ok": stages["cleaned_ok"],
        "schema_ok": ok, "reason": (reason if stages["cleaned_ok"] else stages["error"]) or reason,
        "sentiment": data.get("sentiment") if ok else None,
        "category": data.get("category") if ok else None,
        "entities": data.get("entities") if ok else None,
    }


def yn(flag):
    return "OK " if flag else "FAIL"


def main():
    ap = argparse.ArgumentParser(description="Side-by-side live demo of prompt-only vs schema.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file")
    src.add_argument("--text")
    src.add_argument("--ids", type=int, nargs="+")
    src.add_argument("--interactive", action="store_true")
    ap.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--prompt-style", choices=["hardened", "minimal"], default="hardened")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    ap.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT)
    ap.add_argument("--show-raw", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    if args.file:
        items = read_blocks(args.file)
    elif args.text:
        items = [(None, args.text)]
    elif args.ids:
        df = pd.read_csv(INPUT_CSV).set_index("id")
        items = [(df.loc[i, "ground_truth_sentiment"], df.loc[i, "text"]) for i in args.ids]
    else:
        items = read_interactive()
    if not items:
        sys.exit("No reviews to run.")

    system_prompt = get_system_prompt_for(args.prompt_style)
    print(f"Model: {MODEL_NAME} | prompt: {args.prompt_style} | seed={args.seed} | num_ctx={args.num_ctx}")
    print("Warming up the model...")
    try:
        call_model("Warm-up.", "prompt_only", 0.0, args.seed, args.num_ctx, 8, system_prompt)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] warm-up failed: {exc}")

    new_log = not os.path.exists(LOG_CSV)
    os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)
    logf = open(LOG_CSV, "a", newline="", encoding="utf-8")
    log = csv.writer(logf)
    if new_log:
        log.writerow(["time", "prompt_style", "temperature", "setting", "review", "label", "raw_ok",
                      "cleaned_ok", "schema_ok", "sentiment", "category", "entities", "latency_ms"])

    for n, (label, text) in enumerate(items, 1):
        print("\n" + "=" * 100)
        preview = (text[:150] + "...") if len(text) > 150 else text
        print(f"Review {n}/{len(items)}" + (f"  (true label: {label})" if label else ""))
        print(preview)
        print("=" * 100)
        for t in args.temperatures:
            results = {}
            for s in SETTINGS:
                r = run_one(text, s, t, args, system_prompt)
                results[s] = r
                verdict = ""
                if label and r["sentiment"]:
                    verdict = "  [correct]" if r["sentiment"] == label else "  [WRONG]"
                print(
                    f"T={t:<3} {s:<12} {r['ms']:5.0f} ms | json.loads {yn(r['raw_ok'])} | "
                    f"after cleaning {yn(r['cleaned_ok'])} | schema {yn(r['schema_ok'])}"
                )
                if r["schema_ok"]:
                    print(f"      sentiment={r['sentiment']}{verdict}  category={r['category']}")
                    print(f"      entities={r['entities']}")
                else:
                    print(f"      INVALID: {r['reason']}")
                if args.show_raw or not r["schema_ok"] or not r["raw_ok"]:
                    print(f"      raw: {r['raw'][:300]!r}")
                log.writerow([dt.datetime.now().isoformat(timespec="seconds"), args.prompt_style, t, s,
                              text[:80], label or "", r["raw_ok"], r["cleaned_ok"], r["schema_ok"],
                              r["sentiment"], r["category"], r["entities"], round(r["ms"])])
            a, b = results["prompt_only"], results["schema"]
            if a["sentiment"] != b["sentiment"] or a["entities"] != b["entities"]:
                print(f"  -> T={t}: the two settings DIFFER on this review")
            else:
                print(f"  -> T={t}: same content in both settings")
        logf.flush()

    logf.close()
    print(f"\nLogged to {LOG_CSV}")


if __name__ == "__main__":
    main()
