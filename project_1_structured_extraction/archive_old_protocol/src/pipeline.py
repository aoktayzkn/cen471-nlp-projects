"""
Inference pipeline for Project 1.
Reads the IMDB sample dataset, calls local LLM via Ollama, parses/validates JSON,
and logs latency and schema compliance metrics.
"""

import os
import sys
import time
import pandas as pd
import ollama

# Üst dizindeki modülleri görebilmek için sys.path ayarı
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from project_1_structured_extraction.prompts.extraction_prompt import SYSTEM_PROMPT, build_user_prompt
from project_1_structured_extraction.src.parser import parse_and_validate_json

MODEL_NAME = "qwen2.5:7b"  # Sisteminde yüklü olan modele göre değiştirebilirsin (örn: "llama2:13b", "qwen2.5:7b", "mistral:7b", vb.)
INPUT_CSV = "project_1_structured_extraction/data/raw/imdb_balanced_200.csv"
OUTPUT_CSV = "project_1_structured_extraction/data/processed/extracted_results.csv"

def run_extraction_pipeline():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file '{INPUT_CSV}' not found. Run data_loader.py first.")
        return

    print(f"Loading input samples from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    results = []
    print(f"Starting inference with local model '{MODEL_NAME}' across {len(df)} samples...\n")

    for idx, row in df.iterrows():
        sample_id = row["id"]
        review_text = row["text"]
        ground_truth = row["ground_truth_sentiment"]

        user_prompt = build_user_prompt(review_text)

        start_time = time.perf_counter()
        try:
            # Ollama yerel çağrısı (format='json' ile model katı JSON moduna zorlanır)
            response = ollama.chat(
                model=MODEL_NAME,
                format="json",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                options={"temperature": 0.0}  # Deterministik ve tutarlı çıktı için
            )
            raw_output = response["message"]["content"]
        except Exception as e:
            raw_output = f"ERROR: {str(e)}"

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Parser ve validasyon kontrolü
        is_valid, parsed_data, error_msg = parse_and_validate_json(raw_output)

        extracted_category = parsed_data.get("category", "") if is_valid else None
        extracted_sentiment = parsed_data.get("sentiment", "") if is_valid else None
        extracted_entities = parsed_data.get("entities", []) if is_valid else []

        results.append({
            "id": sample_id,
            "text": review_text,
            "ground_truth_sentiment": ground_truth,
            "raw_output": raw_output,
            "is_valid_json": is_valid,
            "validation_error": error_msg,
            "category": extracted_category,
            "sentiment": extracted_sentiment,
            "entities": str(extracted_entities),
            "latency_ms": round(elapsed_ms, 2)
        })

        status_flag = "VALID" if is_valid else "INVALID"
        print(f"[{sample_id:03d}/{len(df)}] Status: {status_flag} | Latency: {elapsed_ms:.1f}ms | Pred: {extracted_sentiment} (True: {ground_truth})")

    # Çıktıyı kaydet
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nPipeline completed! Extracted dataset saved to: '{OUTPUT_CSV}'")

if __name__ == "__main__":
    run_extraction_pipeline()