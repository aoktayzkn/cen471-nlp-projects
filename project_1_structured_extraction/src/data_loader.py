import os
import pandas as pd
from datasets import load_dataset


def download_balanced_imdb_sample(
    output_path="project_1_structured_extraction/data/raw/imdb_balanced_200.csv",
    samples_per_class=100,
):
  print("Streaming balanced IMDB test dataset from Hugging Face...")
  dataset = load_dataset(
      "stanfordnlp/imdb", split="test", streaming=True
  )

  neg_samples = []
  pos_samples = []

  for item in dataset:
    label = item["label"]
    if label == 0 and len(neg_samples) < samples_per_class:
      neg_samples.append({
          "text": item["text"][:1200],
          "ground_truth_sentiment": "negative",
      })
    elif label == 1 and len(pos_samples) < samples_per_class:
      pos_samples.append({
          "text": item["text"][:1200],
          "ground_truth_sentiment": "positive",
      })

    if (
        len(neg_samples) >= samples_per_class
        and len(pos_samples) >= samples_per_class
    ):
      break

  # Birleştirip dengeli dağıtıyoruz
  combined = neg_samples + pos_samples
  samples = []
  for i, row in enumerate(combined):
    row["id"] = i + 1
    samples.append(row)

  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  df = pd.DataFrame(samples)
  # Sütun sırasını düzenle
  df = df[["id", "text", "ground_truth_sentiment"]]
  df.to_csv(output_path, index=False, encoding="utf-8")
  print(
      f"Successfully saved balanced dataset ({len(neg_samples)} Neg, {len(pos_samples)} Pos) to: '{output_path}'\n"
  )


if __name__ == "__main__":
  download_balanced_imdb_sample()