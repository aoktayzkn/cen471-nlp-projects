"""
Defense experiment: how does response time change as few-shot examples are added
(= the prompt gets longer)?

The same N dataset reviews are sent with k = 0, 2, 4, 8 few-shot examples
(prompt_only setting, T=0.0, seed 42, same model and system prompt as the main experiments).
Few-shot examples are invented short reviews (NOT from the dataset), added as earlier
user/assistant turns.

Per call Ollama reports:
    prompt_eval_count / prompt_eval_duration : tokens processed in the prompt (prefill) / time
    eval_count / eval_duration               : generated tokens / time (decode)
    total wall-clock latency is measured here with perf_counter.

Caching caveat: the few-shot block is an identical prefix for every review, so Ollama may
re-use its KV cache after the first call of a condition. Therefore the first call of each
condition ("cold") is reported separately from the median of the remaining calls ("warm").
If the whole prompt is cached, prompt_eval_count can be small; the prompt size is therefore
also reported as characters.

Run from the repository root:
    python project_1_structured_extraction/src/fewshot_latency.py
    python project_1_structured_extraction/src/fewshot_latency.py --n 20 --ks 0 2 4 8 16
"""

import argparse
import json
import os
import statistics as st
import sys
import time

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from project_1_structured_extraction.prompts.extraction_prompt import (  # noqa: E402
    SYSTEM_PROMPT,
    build_user_prompt,
)
from project_1_structured_extraction.src.pipeline_settings import (  # noqa: E402
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    DEFAULT_SEED,
    INPUT_CSV,
    MODEL_NAME,
)

OUT_CSV = "project_1_structured_extraction/reports/fewshot_latency.csv"
OUT_MD = "project_1_structured_extraction/reports/fewshot_latency.md"

# Invented examples (not in the dataset). Balanced positive / negative.
EXAMPLES = [
    ("A warm, funny comedy. Emma Stone is brilliant in Sunny Days and the script never drags.",
     {"category": "Comedy", "sentiment": "positive", "entities": ["Emma Stone", "Sunny Days"]}),
    ("Dull and badly paced. Dark Harbor wastes its cast and the ending makes no sense.",
     {"category": "Thriller", "sentiment": "negative", "entities": ["Dark Harbor"]}),
    ("Stunning visuals and a moving story. Director Lena Hart delivers her best drama yet in Quiet River.",
     {"category": "Drama", "sentiment": "positive", "entities": ["Lena Hart", "Quiet River"]}),
    ("I wanted to like Space Raiders, but the effects look cheap and the acting by Tom Brandt is wooden.",
     {"category": "Sci-Fi", "sentiment": "negative", "entities": ["Space Raiders", "Tom Brandt"]}),
    ("A tense, well-made horror film. The Cellar kept me on the edge of my seat until the last scene.",
     {"category": "Horror", "sentiment": "positive", "entities": ["The Cellar"]}),
    ("Painfully predictable romance. Paris in May has no chemistry between its leads, Ava Lund and Marc Roy.",
     {"category": "Romance", "sentiment": "negative", "entities": ["Paris in May", "Ava Lund", "Marc Roy"]}),
    ("An eye-opening documentary. Ocean Voices explains the problem clearly and respectfully.",
     {"category": "Documentary", "sentiment": "positive", "entities": ["Ocean Voices"]}),
    ("The animation of Tiny Giants is lovely but the story is thin and the jokes fall flat.",
     {"category": "Animation", "sentiment": "negative", "entities": ["Tiny Giants"]}),
    ("Non-stop action! Jake Rowan carries Iron Run with real charm and the stunts are excellent.",
     {"category": "Action", "sentiment": "positive", "entities": ["Jake Rowan", "Iron Run"]}),
    ("A forgettable mess. Nothing in The Long Night works, from the script to the editing.",
     {"category": "Drama", "sentiment": "negative", "entities": ["The Long Night"]}),
    ("Charming and clever, with a great performance by Nora Fell in Paper Boats.",
     {"category": "Drama", "sentiment": "positive", "entities": ["Nora Fell", "Paper Boats"]}),
    ("Two hours of my life I will not get back. Cold Case Files is boring and the plot is a mess.",
     {"category": "Thriller", "sentiment": "negative", "entities": ["Cold Case Files"]}),
    ("A joyful family movie; Sam Ortiz is hilarious as the clumsy hero of Bumble Brothers.",
     {"category": "Comedy", "sentiment": "positive", "entities": ["Sam Ortiz", "Bumble Brothers"]}),
    ("Terrible dialogue and no suspense at all in Night Shift. Even Kate Doyle cannot save it.",
     {"category": "Horror", "sentiment": "negative", "entities": ["Night Shift", "Kate Doyle"]}),
    ("Beautifully shot sci-fi with real ideas. Orbit Zero is the best of the year.",
     {"category": "Sci-Fi", "sentiment": "positive", "entities": ["Orbit Zero"]}),
    ("Messy, loud and exhausting. Steel Rain is an action movie with nothing new to say.",
     {"category": "Action", "sentiment": "negative", "entities": ["Steel Rain"]}),
]


def build_messages(review, k):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for text, answer in EXAMPLES[:k]:
        msgs.append({"role": "user", "content": build_user_prompt(text)})
        msgs.append({"role": "assistant", "content": json.dumps(answer)})
    msgs.append({"role": "user", "content": build_user_prompt(review)})
    return msgs


def call(review, k, args):
    import ollama

    msgs = build_messages(review, k)
    start = time.perf_counter()
    resp = ollama.chat(
        model=MODEL_NAME,
        messages=msgs,
        options={"temperature": 0.0, "seed": args.seed, "num_ctx": args.num_ctx,
                 "num_predict": args.num_predict},
    )
    wall_ms = (time.perf_counter() - start) * 1000
    g = lambda key: (resp[key] if key in resp and resp[key] is not None else 0)  # noqa: E731
    return {
        "prompt_chars": sum(len(m["content"]) for m in msgs),
        "prompt_tokens_evaluated": g("prompt_eval_count"),
        "prefill_ms": g("prompt_eval_duration") / 1e6,
        "gen_tokens": g("eval_count"),
        "decode_ms": g("eval_duration") / 1e6,
        "wall_ms": wall_ms,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="reviews per condition")
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 2, 4, 8])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    ap.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT)
    args = ap.parse_args()
    if max(args.ks) > len(EXAMPLES):
        sys.exit(f"At most {len(EXAMPLES)} examples are defined.")

    df = pd.read_csv(INPUT_CSV).head(args.n)
    print(f"Model {MODEL_NAME} | T=0.0 | seed={args.seed} | n={len(df)} reviews | k={args.ks}")
    print("Warming up the model (model loading must not count)...")
    call("Warm-up.", 0, args)

    rows = []
    for k in args.ks:
        print(f"\n--- k={k} few-shot examples ---")
        for i, (_, r) in enumerate(df.iterrows()):
            m = call(r["text"], k, args)
            m.update({"k": k, "call_index": i, "kind": "cold" if i == 0 else "warm", "id": int(r["id"])})
            rows.append(m)
            print(f"  [{int(r['id']):03d}] {m['kind']:<4} wall={m['wall_ms']:6.0f} ms  "
                  f"prefill={m['prefill_ms']:6.0f} ms ({m['prompt_tokens_evaluated']} tok)  "
                  f"decode={m['decode_ms']:6.0f} ms ({m['gen_tokens']} tok)  chars={m['prompt_chars']}")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    lines = ["| k | prompt chars | cold wall ms | warm median wall ms | cold prefill ms | warm median prefill ms | warm median decode ms |",
             "|---|---|---|---|---|---|---|"]
    for k in args.ks:
        sub = out[out["k"] == k]
        cold, warm = sub[sub["kind"] == "cold"], sub[sub["kind"] == "warm"]
        med = lambda col: st.median(warm[col]) if len(warm) else float("nan")  # noqa: E731
        lines.append(
            f"| {k} | {int(sub['prompt_chars'].median())} | {cold['wall_ms'].iloc[0]:.0f} | {med('wall_ms'):.0f} | "
            f"{cold['prefill_ms'].iloc[0]:.0f} | {med('prefill_ms'):.0f} | {med('decode_ms'):.0f} |"
        )
    table = "\n".join(lines)
    print("\n" + table)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(f"# Few-shot latency ({MODEL_NAME}, T=0.0, seed={args.seed}, n={len(df)})\n\n{table}\n")
    print(f"\nSaved: {OUT_CSV} and {OUT_MD}")


if __name__ == "__main__":
    main()
