# CEN471 NLP Project II: English to Turkish Translation with a Local LLM and Metric Evaluation

Author: Ali Oktay Özkan, Çukurova University, Department of Computer Engineering

English sentences from WMT18 are translated into Turkish by a local LLM (`gemma3:4b` via Ollama). The translations are scored with corpus BLEU, chrF and METEOR, and the metrics are compared with the author's own ratings. No cloud API is used.

## Main findings

- Headline run (`gemma3:4b`, temperature 0.1, seed 42, 150 test sentences): **BLEU 15.71, chrF 52.38, METEOR 46.14**.
- Raising the temperature from 0.1 to 1.2 changes the wording much more than the scores. The share of sentences translated identically under three seeds falls from 58.0% to 7.3%, while mean BLEU moves from 15.82 to 15.22 with overlapping seed ranges.
- On 30 sentences rated before seeing any score, no metric showed a clear relation to the human judgement (Spearman between -0.10 and 0.05).
- `gemma3:1b` is about 1.9 times faster but scores much lower (BLEU 6.18, chrF 39.13). Three-shot prompting shows no clear gain (BLEU 15.73, chrF 52.41, METEOR 46.66).
- The Turkish references need 1.26 times as many tokens as the English sources.

Full details: the technical report ([CEN471_NLP_Project2_Technical_Report.pdf](CEN471_NLP_Project2_Technical_Report.pdf)) and the tables in `results/`.

## Reproducibility details

| Item | Value |
|---|---|
| Model | `gemma3:4b`, quantization `Q4_K_M` |
| Comparison models | `qwen2.5:7b` (dev set), `gemma3:1b` (test set), both `Q4_K_M` |
| Ollama version | 0.35.1 (Python client `ollama` 0.6.3) |
| Temperature | 0.1 (headline); 0.1, 0.7 and 1.2 (experiment) |
| Seed | 42; 43 and 44 added for the temperature experiment |
| Context length (`num_ctx`) | 2048 |
| Max generated tokens (`num_predict`) | 512 |
| Data | [WMT18 tr-en](https://huggingface.co/datasets/wmt/wmt18), revision `ceae0f932f739f66498b7695b1807878f82d9b22`, used English to Turkish |
| Sampling | seed 42; English source of 5-50 words; duplicate sources and empty references removed |
| Sets | 40 dev pairs (validation split), 150 test pairs (test split) |

System prompt (the English sentence is sent alone as the user message):

> You are a professional English-to-Turkish translator. Translate the user's text into Turkish. Output only the Turkish translation, with no explanations, notes or quotation marks.

Each run writes a `.meta.json` file (model, quantization, Ollama version, options and the full messages) next to its outputs. With the same seed, a rerun on the same machine reproduced five checked test sentences exactly at temperature 0.1 and 1.2; no claim is made for other hardware or Ollama versions.

## Setup

```bash
ollama pull gemma3:4b
ollama pull qwen2.5:7b    # only for the dev-set model comparison
ollama pull gemma3:1b     # only for the smaller-model experiment
pip install -r requirements.txt
```

Ollama must be running. All commands are run from the repository root. An internet connection is needed to download the data (step 1) and, the first time, the NLTK WordNet files used by METEOR.

## How to run

All paths are relative to the repository root; the project lives in `project_2_translation/`.

The repository already contains all model outputs, and `translate.py` skips sentences that are already saved. To translate everything again, first move `project_2_translation/data/outputs/` somewhere else.

**1. Download and sample the data** (writes `dev.jsonl` and `test.jsonl` to `data/raw/`):

```bash
python project_2_translation/src/data_loader.py
```

**2. Choose the model and the prompt** on the dev set:

```bash
python project_2_translation/src/translate.py --split dev
python project_2_translation/src/translate.py --split dev --model qwen2.5:7b
python project_2_translation/src/translate.py --split dev --prompt v2
python project_2_translation/src/evaluate.py --split dev
```

Quick trial on a few sentences: `python project_2_translation/src/translate.py --split dev --limit 10`. Incomplete runs are skipped by `evaluate.py`.

**3. Temperature experiment** on the test set (3 temperatures x 3 seeds, PowerShell):

```powershell
foreach ($t in "0.1", "0.7", "1.2") {
  foreach ($s in "42", "43", "44") {
    python project_2_translation/src/translate.py --split test --temperature $t --seed $s
  }
}
```

**4. Additional experiments** (smaller model, three-shot prompting):

```bash
python project_2_translation/src/translate.py --split test --model gemma3:1b
python project_2_translation/src/translate.py --split test --prompt fewshot
```

**5. Score all test runs and draw the temperature figure:**

```bash
python project_2_translation/src/evaluate.py --split test
python project_2_translation/src/plot.py
```

**6. Human evaluation:**

```bash
python project_2_translation/src/make_review_sheet.py   # refuses to overwrite an existing sheet
python project_2_translation/src/compare_ratings.py
```

The ratings belong to the saved headline run. If new outputs differ from the rated text, `compare_ratings.py` stops with an error instead of mixing them.

**7. Token counts and speed:**

```bash
python project_2_translation/src/token_counts.py
```

**8. Live demo used in the video** (nothing is saved):

```bash
python project_2_translation/src/live_demo.py          # 5 test sentences at temperature 0.1 and 1.2
python project_2_translation/src/live_demo.py --new    # type new sentences, optionally with a reference
```

## Repository structure

```
project_2_translation/
  README.md
  CEN471_NLP_Project2_Technical_Report.pdf   technical report (4 pages)
  src/
    data_loader.py          streams WMT18 tr-en and saves the seeded dev and test samples
    translate.py            prompts, Ollama calls, output cleaning and flags, run metadata
    evaluate.py             corpus and sentence BLEU, chrF and METEOR, metric settings
    plot.py                 temperature table and figure
    make_review_sheet.py    blind rating sheet with 30 random test translations
    compare_ratings.py      Spearman correlation and disagreement cases
    token_counts.py         token counts of sources and references, generation speed
    live_demo.py            live tests with immediate scores
  data/
    raw/                    dev.jsonl (40 pairs) and test.jsonl (150 pairs)
    outputs/                raw and cleaned translations (.jsonl) and metadata (.meta.json)
  results/
    dev_summary.csv, test_summary.csv      corpus scores of every run
    *_sentences.csv                        sentence scores with source, reference and output
    *_metric_settings.json                 sacrebleu signatures and METEOR settings
    temperature_summary.csv, temperature_plot.png
    human_review.csv                       the author's ratings and notes
    human_vs_metrics.csv                   ratings next to metric scores and ranks
    token_counts.csv
```

Output files are named `<split>_T<temperature>_seed<seed>_<tag>.jsonl`. The tag changes with the model, the messages or the fixed settings:

| Tag | Model | Prompt |
|---|---|---|
| `4636c8` | `gemma3:4b` | v1 (main) |
| `a43008` | `qwen2.5:7b` | v1 |
| `f3d71d` | `gemma3:4b` | v2 |
| `951532` | `gemma3:1b` | v1 |
| `dfca5c` | `gemma3:4b` | three-shot (v1 instruction and 3 dev pairs) |

## Metrics

All scores use one reference per sentence and a 0-100 scale.

| Metric | Library and settings | Corpus score |
|---|---|---|
| BLEU | sacrebleu 2.6.0: `nrefs:1\|case:mixed\|eff:no\|tok:13a\|smooth:exp\|version:2.6.0` | corpus-level |
| chrF | sacrebleu 2.6.0: `nrefs:1\|case:mixed\|eff:yes\|nc:6\|nw:0\|space:no\|version:2.6.0` | corpus-level |
| METEOR | NLTK 3.10.4, tokens `\w+\|[^\w\s]`, Turkish-aware lowercasing | mean of sentence scores |

Sentence-level BLEU uses `effective_order=True`. NLTK's default METEOR stemmer and WordNet are English, so for Turkish it mostly counts surface-form matches.

## Model and prompt choice (dev set)

Temperature 0.1, seed 42, 40 sentences.

| Model | Prompt | BLEU | chrF | METEOR |
|---|---|---|---|---|
| `gemma3:4b` | v1 | 14.31 | 50.14 | 47.23 |
| `qwen2.5:7b` | v1 | 11.03 | 43.92 | 39.96 |
| `gemma3:4b` | v2 | 16.88 | 50.17 | 45.66 |

`gemma3:4b` was chosen over `qwen2.5:7b`. The stricter prompt v2 raised BLEU but left chrF flat, lowered METEOR and fixed few of the targeted meaning errors, so v1 was kept.

## Human evaluation

Thirty random outputs of the headline run were rated from 1 to 5, seeing only the source and the translation: 5 = complete and natural, 4 = small language error, 3 = main idea right but a noticeable error or omission, 2 = important information wrong, 1 = unintelligible. No score was changed after the metrics were computed.

## Limitations

- One rater and 30 sentences; weak correlations cannot be told apart from zero.
- One reference per sentence; some references and sources contain errors.
- WMT18 is public and may be in the model's training data; this was not tested.
- Three seeds per temperature and one run per additional experiment; no BERTScore.
- The output check flags foreign alphabets but cannot detect a Latin-script answer in the wrong language.
