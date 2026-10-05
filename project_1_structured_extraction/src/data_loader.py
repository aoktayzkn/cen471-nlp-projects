import os
import pandas as pd
from datasets import load_dataset


def download_imdb_sample(
    output_path="project_1_structured_extraction/data/raw/imdb_test_100.csv",
    sample_count=100,
):
  print("Streaming IMDB test dataset from Hugging Face...")
  # 'imdb' yerine resmi adresi 'stanfordnlp/imdb' kullanıyoruz:
  dataset = load_dataset("stanfordnlp/imdb", split="test", streaming=True)

  samples = []
  for i, item in enumerate(dataset):
    if i >= sample_count:
      break
    samples.append({
        "id": i + 1,
        "text": item["text"][:1200],  # Balanced context window for local LLMs
        "ground_truth_sentiment": (
            "positive" if item["label"] == 1 else "negative"
        ),
    })

  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  df = pd.DataFrame(samples)
  df.to_csv(output_path, index=False, encoding="utf-8")
  print(
      f"Successfully saved {sample_count} test samples to: '{output_path}'\n"
  )


if __name__ == "__main__":
  download_imdb_sample()