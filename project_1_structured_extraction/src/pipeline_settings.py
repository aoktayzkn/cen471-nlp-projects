"""
Pipeline for the two experimental settings required by the assignment.

  prompt_only : JSON is requested ONLY through the prompt. The Ollama `format`
                parameter is NOT passed.
  schema      : the Ollama `format` parameter is passed a JSON schema
                (constrained decoding).

Everything else (model, system prompt, temperature, seed, context length, data)
is identical, so any difference comes from the `format` parameter alone.

Run from the repository root:
    python project_1_structured_extraction/src/pipeline_settings.py --setting both
    python project_1_structured_extraction/src/pipeline_settings.py --setting both --temperature 1.0
    python project_1_structured_extraction/src/pipeline_settings.py --setting prompt_only --limit 5
    python project_1_structured_extraction/src/pipeline_settings.py --setting both --prompt-style minimal
"""

import argparse
import json
import os
import subprocess
import sys
import time

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from project_1_structured_extraction.prompts.extraction_prompt import (
    ALLOWED_CATEGORIES,
    ALLOWED_SENTIMENTS,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from project_1_structured_extraction.src.parser import (
    parse_json_stages,
    strict_schema_check,
)

MODEL_NAME = "qwen2.5:7b"  # Q4_K_M, see `ollama show qwen2.5:7b`
INPUT_CSV = "project_1_structured_extraction/data/raw/imdb_balanced_200.csv"
OUT_DIR = "project_1_structured_extraction/data/processed/settings"

DEFAULT_SEED = 42
DEFAULT_NUM_CTX = 4096
# Upper bound on generated tokens. Guards against runaway generation; a row whose
# done_reason == "length" was cut off and is logged as such.
DEFAULT_NUM_PREDICT = 512

# Prompt styles (2x2 design: prompt style x {prompt_only, schema}).
#   hardened : the active v3 prompt, which explicitly forbids fences / filler text.
#   minimal  : the SAME classification/extraction rules, but the output-format
#              instructions are relaxed to a plain "return JSON" request. This is how
#              a typical first-attempt prompt looks; it isolates the effect of the
#              format instructions without touching the task rules.
_MINIMAL_EDITS = [
    (
        "You must return ONLY a single valid JSON object strictly matching this schema:",
        "Return the result as a JSON object with this schema:",
    ),
    (
        "4. Return raw JSON only. Absolutely NO markdown backticks (no ``` or ```json) and NO conversational filler.\n",
        "",
    ),
]


def get_system_prompt_for(style):
    if style == "hardened":
        return SYSTEM_PROMPT
    prompt = SYSTEM_PROMPT
    for old, new in _MINIMAL_EDITS:
        assert old in prompt, "Prompt text changed; update _MINIMAL_EDITS."
        prompt = prompt.replace(old, new)
    return prompt


JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ALLOWED_CATEGORIES},
        "sentiment": {"type": "string", "enum": ALLOWED_SENTIMENTS},
        "entities": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["category", "sentiment", "entities"],
    "additionalProperties": False,
}


def call_model(review_text, setting, temperature, seed, num_ctx, num_predict, system_prompt=SYSTEM_PROMPT):
    """One Ollama call. Returns (raw_output, done_reason)."""
    import ollama  # imported lazily so the module can be tested without Ollama

    kwargs = dict(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(review_text)},
        ],
        options={
            "temperature": float(temperature),
            "seed": int(seed),
            "num_ctx": int(num_ctx),
            "num_predict": int(num_predict),
        },
    )
    if setting == "schema":
        kwargs["format"] = JSON_SCHEMA  # constrained decoding
    # prompt_only: NO `format` key at all.

    response = ollama.chat(**kwargs)
    return response["message"]["content"], response.get("done_reason", "")


def collect_run_metadata(args):
    """Everything the report's reproducibility paragraph needs."""
    meta = {
        "model": MODEL_NAME,
        "temperature": args.temperature,
        "seed": args.seed,
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
        "prompt_style": args.prompt_style,
        "system_prompt_chars": len(get_system_prompt_for(args.prompt_style)),
        "system_prompt": get_system_prompt_for(args.prompt_style),
        "input_csv": INPUT_CSV,
    }
    try:
        out = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=15)
        meta["ollama_version"] = (out.stdout + out.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        meta["ollama_version"] = f"unavailable: {exc}"
    try:
        import ollama

        info = ollama.show(MODEL_NAME)
        details = info["details"] if "details" in info else {}
        meta["quantization_level"] = getattr(details, "quantization_level", None) or details.get("quantization_level")
        meta["parameter_size"] = getattr(details, "parameter_size", None) or details.get("parameter_size")
        meta["model_family"] = getattr(details, "family", None) or details.get("family")
    except Exception as exc:  # noqa: BLE001
        meta["model_info_error"] = str(exc)
    return meta


def style_suffix(style):
    """Hardened (main experiment) keeps the original file names; minimal adds a suffix."""
    return "" if style == "hardened" else f"_{style}"


def run_setting(df, setting, args):
    rows = []
    system_prompt = get_system_prompt_for(args.prompt_style)
    print(
        f"\n=== setting={setting} | prompt={args.prompt_style} | "
        f"T={args.temperature} | seed={args.seed} | n={len(df)} ==="
    )

    # Warm-up call so the first row's latency does not include model loading.
    try:
        call_model("Warm-up.", setting, args.temperature, args.seed, args.num_ctx, 16, system_prompt)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] warm-up failed: {exc}")

    for _, row in df.iterrows():
        start = time.perf_counter()
        try:
            raw, done_reason = call_model(
                row["text"], setting, args.temperature, args.seed, args.num_ctx, args.num_predict,
                system_prompt,
            )
            call_error = ""
        except Exception as exc:  # noqa: BLE001
            raw, done_reason, call_error = "", "error", str(exc)
        latency_ms = (time.perf_counter() - start) * 1000

        stages = parse_json_stages(raw)
        schema_ok, schema_reason = strict_schema_check(stages["data"])
        data = stages["data"] if schema_ok else None

        rows.append(
            {
                "id": row["id"],
                "setting": setting,
                "prompt_style": args.prompt_style,
                "temperature": args.temperature,
                "seed": args.seed,
                "ground_truth_sentiment": row["ground_truth_sentiment"],
                "raw_output": raw,
                "done_reason": done_reason,
                "call_error": call_error,
                "raw_json_ok": stages["raw_ok"],
                "cleaned_json_ok": stages["cleaned_ok"],
                "schema_valid": schema_ok,
                "schema_error": schema_reason if stages["cleaned_ok"] else stages["error"],
                "category": data["category"] if data else None,
                "sentiment": data["sentiment"] if data else None,
                "entities_json": json.dumps(data["entities"], ensure_ascii=False) if data else "[]",
                "latency_ms": round(latency_ms, 2),
            }
        )
        flag = "OK " if schema_ok else "BAD"
        print(
            f"[{int(row['id']):03d}] {flag} raw_ok={stages['raw_ok']} cleaned_ok={stages['cleaned_ok']} "
            f"{latency_ms:6.0f}ms pred={rows[-1]['sentiment']} true={row['ground_truth_sentiment']}"
        )

    out = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{setting}{style_suffix(args.prompt_style)}_T{args.temperature:.1f}_seed{args.seed}.csv")
    out.to_csv(path, index=False, encoding="utf-8")
    print(f"Saved: {path}")
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Run prompt-only and/or schema-constrained extraction.")
    p.add_argument("--setting", choices=["prompt_only", "schema", "both"], default="both")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    p.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT)
    p.add_argument(
        "--prompt-style", choices=["hardened", "minimal"], default="hardened",
        help="hardened = main experiment; minimal = format-instruction ablation.",
    )
    p.add_argument("--limit", type=int, default=None, help="Only the first N rows (smoke test).")
    p.add_argument("--ids", type=str, default=None, help="Text file with one sample id per line.")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(INPUT_CSV):
        sys.exit(f"Input file '{INPUT_CSV}' not found. Run from the repository root.")
    df = pd.read_csv(INPUT_CSV)
    if args.ids:
        wanted = {int(x) for x in open(args.ids).read().split()}
        df = df[df["id"].isin(wanted)]
    if args.limit:
        # --limit takes a balanced head: half from each class.
        half = max(args.limit // 2, 1)
        df = pd.concat([g.head(half) for _, g in df.groupby("ground_truth_sentiment")])

    os.makedirs(OUT_DIR, exist_ok=True)
    meta = collect_run_metadata(args)
    meta_path = os.path.join(OUT_DIR, f"run_meta{style_suffix(args.prompt_style)}_T{args.temperature:.1f}_seed{args.seed}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("Run metadata:", json.dumps(meta, indent=2, ensure_ascii=False))

    settings = ["prompt_only", "schema"] if args.setting == "both" else [args.setting]
    for s in settings:
        run_setting(df, s, args)


if __name__ == "__main__":
    main()
