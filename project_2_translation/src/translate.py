"""Translate English sentences into Turkish with a local LLM (Ollama).

Run from the repository root:
    python project_2_translation/src/translate.py --split dev --limit 10
    python project_2_translation/src/translate.py --split test --temperature 0.7 --seed 42
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import ollama

MODEL = "qwen2.5:7b"
DATA_DIR = Path("project_2_translation/data/raw")
OUT_DIR = Path("project_2_translation/data/outputs")
NUM_CTX, NUM_PREDICT = 2048, 512

SYSTEM_PROMPT = (
    "You are a professional English-to-Turkish translator. "
    "Translate the user's text into Turkish. "
    "Output only the Turkish translation, with no explanations, notes or quotation marks."
)

# A label at the very start of the answer, e.g. "Çeviri:" or "Here is the translation:"
PREFIX = re.compile(
    r"^\s*(here(?:'s| is) the (?:turkish )?translation|turkish translation|translation|"
    r"işte çeviri(?:si)?|türkçe çeviri(?:si)?|çeviri)\s*:\s*",
    re.IGNORECASE,
)
# Cyrillic, Arabic, Thai, Japanese kana, Chinese and Korean characters
FOREIGN = re.compile("[Ѐ-ӿ؀-ۿ฀-๿぀-ヿ㐀-鿿가-힯]")


def build_messages(source):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": source},
    ]


def clean_output(raw, source):
    """Return (cleaned_text, flags). The raw answer is always saved as well."""
    flags = []
    text = raw.strip()
    if PREFIX.match(text):
        text = PREFIX.sub("", text).strip()
        flags.append("prefix_removed")
    # remove wrapping quotes, unless the source itself starts with a quote
    if len(text) > 1 and text[0] in '"“' and text[-1] in '"”' and source.lstrip()[0] not in '"“':
        text = text[1:-1].strip()
        flags.append("quotes_removed")
    if "\n" in text:
        text = " ".join(text.split())
        flags.append("multiline")
    if FOREIGN.search(text):
        flags.append("foreign_script")  # only counted, never deleted
    if not text:
        flags.append("empty")
    return text, flags


def translate(source, options):
    response = ollama.chat(model=MODEL, messages=build_messages(source), options=options)
    raw = response["message"]["content"]
    output, flags = clean_output(raw, source)
    if response.get("done_reason") == "length":
        flags.append("truncated")
    return {
        "raw": raw,
        "output": output,
        "flags": flags,
        "seconds": (response.get("total_duration") or 0) / 1e9,  # reported in nanoseconds
        "prompt_tokens": response.get("prompt_eval_count"),
        "output_tokens": response.get("eval_count"),
    }


def run_metadata(options):
    """Details needed to reproduce the run."""
    version = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=15)
    details = ollama.show(MODEL)["details"]
    return {
        "model": MODEL,
        "quantization": details["quantization_level"],
        "ollama_version": (version.stdout + version.stderr).strip(),
        "options": options,
        "system_prompt": SYSTEM_PROMPT,
    }


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="only the first N sentences (quick trial)")
    args = parser.parse_args()

    options = {"temperature": args.temperature, "seed": args.seed,
               "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT}
    data = read_jsonl(DATA_DIR / f"{args.split}.jsonl")
    if not data:
        raise SystemExit("Data not found. Run from the repository root after data_loader.py.")
    if args.limit:
        data = data[: args.limit]

    # A changed model, prompt or fixed setting gets a new file, so old outputs are never mixed in
    config = json.dumps([MODEL, SYSTEM_PROMPT, NUM_CTX, NUM_PREDICT])
    tag = hashlib.sha1(config.encode()).hexdigest()[:6]
    name = f"{args.split}_T{args.temperature}_seed{args.seed}_{tag}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.jsonl"
    done = {row["id"] for row in read_jsonl(path)}
    todo = [item for item in data if item["id"] not in done]
    print(f"{len(done)} already saved, {len(todo)} to translate -> {path}")
    if not todo:
        return

    meta = json.dumps(run_metadata(options), indent=2, ensure_ascii=False)
    (OUT_DIR / f"{name}.meta.json").write_text(meta, encoding="utf-8")
    translate("Hello.", options)  # warm-up: model loading is kept out of the timing

    for number, item in enumerate(todo, 1):
        row = {**item, **translate(item["source"], options)}
        with path.open("a", encoding="utf-8") as f:  # saved at once, so a run can be resumed
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{number}/{len(todo)}] {row['seconds']:.1f}s {row['flags']}\n  {row['output']}")


if __name__ == "__main__":
    main()
