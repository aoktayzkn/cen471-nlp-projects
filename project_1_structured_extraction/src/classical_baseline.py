"""
Defense experiment (section G): classical baselines on the same 200 reviews.

  1. TF-IDF + logistic regression vs bag-of-words (counts) + logistic regression,
     5-fold stratified cross-validation on the 200 reviews (each fold trains on 160).
  2. Only 10 labelled examples (5 positive + 5 negative): train on them, test on the other
     190 reviews, repeated for 30 random draws.
  3. The most influential words of each model.

Sentiment only (the classical models do not extract entities). The LLM numbers to compare
with are in reports/settings_comparison.md (accuracy 0.955-0.970 with the schema setting).

Run from the repository root:
    python project_1_structured_extraction/src/classical_baseline.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

INPUT_CSV = "project_1_structured_extraction/data/raw/imdb_balanced_200.csv"
OUT_MD = "project_1_structured_extraction/reports/classical_baseline.md"
SEED = 42

MODELS = {
    "TF-IDF + LR": lambda: make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000)),
    "BoW counts + LR": lambda: make_pipeline(CountVectorizer(), LogisticRegression(max_iter=1000)),
}


def main():
    df = pd.read_csv(INPUT_CSV)
    X, y = df["text"], (df["ground_truth_sentiment"] == "positive").astype(int)
    lines = ["# Classical baselines (sentiment only, 200 reviews)\n"]

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    lines += ["## 5-fold CV (160 training reviews per fold)\n", "| model | mean accuracy | per fold |", "|---|---|---|"]
    for name, make in MODELS.items():
        s = cross_val_score(make(), X, y, cv=cv)
        lines.append(f"| {name} | {s.mean():.3f} | {', '.join(f'{v:.3f}' for v in s)} |")

    rng = np.random.default_rng(SEED)
    lines += ["\n## Only 10 labelled examples (5+5), test on the other 190, 30 random draws\n",
              "| model | mean accuracy | min | max |", "|---|---|---|---|"]
    for name, make in MODELS.items():
        accs = []
        for _ in range(30):
            tr = np.concatenate([rng.choice(np.where(y == c)[0], 5, replace=False) for c in (0, 1)])
            te = np.setdiff1d(np.arange(len(df)), tr)
            m = make().fit(X.iloc[tr], y.iloc[tr])
            accs.append(float((m.predict(X.iloc[te]) == y.iloc[te]).mean()))
        lines.append(f"| {name} | {np.mean(accs):.3f} | {np.min(accs):.3f} | {np.max(accs):.3f} |")

    lines += ["\n## Most influential words (fit on all 200 reviews)\n"]
    for name, make in MODELS.items():
        p = make().fit(X, y)
        vocab, coef = p[0].get_feature_names_out(), p[1].coef_[0]
        order = np.argsort(coef)
        lines.append(f"- {name}: negative = {list(vocab[order[:8]])}; positive = {list(vocab[order[-8:]])}")

    text = "\n".join(lines) + "\n"
    print(text)
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
