# CEN471 NLP Project I: Structured Classification and Entity Extraction with a Local LLM

Author: Ali Oktay Özkan, Çukurova University, Department of Computer Engineering

Reviews from IMDB are sent to a local LLM (`qwen2.5:7b` via Ollama), which must return one JSON object with a genre `category`, a binary `sentiment` and a list of named `entities`. The project compares two ways of getting valid JSON and measures what each one changes:

| Setting | How JSON is requested |
|---|---|
| `prompt_only` | Only the prompt asks for JSON (no `format` parameter). |
| `schema` | Ollama's `format` parameter receives a JSON schema (constrained decoding). |

Everything else (model, system prompt, data, seed, context length) is identical in both settings.

## Main findings

- Schema-constrained decoding gives 100% raw parsing, 100% parsing after cleaning and 100% strict schema validity in all four conditions (prompt-only: 97.0-98.0% schema validity; with a minimal prompt only 54-64% raw `json.loads` success, because the model wraps JSON in markdown fences).
- End-to-end sentiment accuracy is 1.5-2.5 points higher with the schema, but this gain comes only from the 4-6 outputs per 200 that prompt-only loses to format violations. Salvaged accuracy (sentiment read from the output whenever it is valid) is identical in every condition, and the paired McNemar tests are not significant.
- Entity F1 (100 manually verified reviews, 268 entities) is about 0.66-0.70 strict and 0.72-0.73 lenient, with no consistent difference between the settings.

Full details: the technical report in `report/` (PDF) and `reports/settings_comparison.md`.

## Reproducibility details

| Item | Value |
|---|---|
| Model | `qwen2.5:7b`, 7.6B parameters, quantization `Q4_K_M` |
| Ollama version | 0.35.1 |
| Temperature | 0.0 and 1.0 |
| Seed | 42 |
| Context length (`num_ctx`) | 4096 |
| Max generated tokens (`num_predict`) | 512 |
| Data | `data/raw/imdb_balanced_200.csv` (100 positive, 100 negative) |

Each run also writes a `run_meta*.json` file (model, quantization, Ollama version, options and the full system prompt) next to its results. The full T=0.0 experiment was repeated from scratch on the same machine and produced identical output files; no claim is made for other hardware or Ollama versions.

## Setup

```bash
ollama pull qwen2.5:7b
pip install pandas numpy scipy scikit-learn matplotlib tabulate ollama
# optional, only for the tokenizer demo: pip install transformers
```

Ollama must be running. All commands below are run from the repository root.

## How to run

All paths are relative to the repository root; the project lives in `project_1_structured_extraction/`.

**1. Run both settings** (writes CSVs and `run_meta` to `data/processed/settings/`):

```bash
python project_1_structured_extraction/src/pipeline_settings.py --setting both --temperature 0.0
python project_1_structured_extraction/src/pipeline_settings.py --setting both --temperature 1.0
# minimal-prompt ablation
python project_1_structured_extraction/src/pipeline_settings.py --setting both --temperature 0.0 --prompt-style minimal
python project_1_structured_extraction/src/pipeline_settings.py --setting both --temperature 1.0 --prompt-style minimal
```

Quick smoke test (results go to `settings/partial/` and never overwrite the real results): add `--limit 6`.

**2. Evaluate** (appends tables to `reports/settings_comparison.md`):

```bash
python project_1_structured_extraction/src/evaluate_settings.py --temperature 0.0
python project_1_structured_extraction/src/evaluate_settings.py --temperature 1.0
python project_1_structured_extraction/src/evaluate_settings.py --temperature 0.0 --prompt-style minimal
python project_1_structured_extraction/src/evaluate_settings.py --temperature 1.0 --prompt-style minimal
```

**3. Failure and entity-error analysis, figures:**

```bash
python project_1_structured_extraction/src/show_failures.py
python project_1_structured_extraction/src/entity_errors.py
python project_1_structured_extraction/src/make_plots.py
```

**4. Live demo** (same review through both settings at two temperatures):

```bash
python project_1_structured_extraction/src/live_demo.py --file project_1_structured_extraction/data/live_demo/new_reviews.txt
python project_1_structured_extraction/src/live_demo.py --ids 2 --prompt-style minimal --show-raw
```

**5. Additional experiments used in the video:**

```bash
python project_1_structured_extraction/src/fewshot_latency.py      # latency vs. number of few-shot examples
python project_1_structured_extraction/src/classical_baseline.py   # TF-IDF / BoW + logistic regression
```

## Repository structure

```
project_1_structured_extraction/
  prompts/extraction_prompt.py     system prompt (v3) and user prompt
  src/
    pipeline_settings.py           runs prompt_only and schema, saves raw outputs and run_meta
    parser.py                      cleaning, stage-wise parsing, strict schema check
    evaluate_settings.py           the four metrics, entity P/R/F1, paired McNemar test
    build_annotation_template.py   creates the gold-annotation template
    review_gold.py                 terminal tool used to verify and correct the gold entities
    show_failures.py, entity_errors.py   error analyses
    make_plots.py                  report figures
    live_demo.py                   side-by-side live demo
    fewshot_latency.py, classical_baseline.py   extra experiments
  data/
    raw/imdb_balanced_200.csv      evaluation set
    annotation/entity_gold.csv     manually verified entity gold (100 reviews, 268 entities)
    live_demo/new_reviews.txt      five new reviews for the live test
    processed/settings/            result CSVs and run_meta files
  reports/                         comparison tables, error analyses, plots
  archive_old_protocol/            first experiments, kept for transparency (see below)
report/                            technical report (PDF)
```

## Metrics

1. Raw JSON parsing: `json.loads` on the raw output, before any cleaning.
2. JSON parsing after cleaning: strip markdown fences, extract the first `{...}` block.
3. Schema validity: required fields present, correct types, allowed values; nothing is silently repaired (an unknown category is a violation, not coerced to "Other").
4. Task accuracy: sentiment accuracy and macro-F1 against the IMDB label; entity precision/recall/F1 against the manual gold, with strict (exact after normalization) and lenient (token containment) matching.

An output that cannot be parsed or fails the schema counts as wrong. "Salvaged" accuracy additionally reads the sentiment field whenever it is valid, to separate format failures from classification quality.

## Entity gold annotation

Entities are named people (actors, directors, writers, characters) and titles of films, TV shows and books. Genres, studios, generic roles and pronouns are not annotated. The first draft of the annotation was produced with LLM assistance; every review was then checked and corrected manually (`src/review_gold.py` records which rows were verified and changed).

## Limitations

- One seed, one model, 200 reviews (100 for entities); differences of 1-3 points are within noise.
- The v3 system prompt was refined by inspecting errors on the same 200 reviews that are reported, so absolute accuracy is probably optimistic.
- The cleaning regex `\{.*\}` is greedy and fails if a response contains two separate JSON objects.
- The gold set was drafted with LLM assistance and verified by a single annotator.

## Archived early experiments

`archive_old_protocol/` contains the first version of the project, made before the final requirements. Those runs used Ollama's generic `format="json"` mode, i.e. neither of the two required settings, and their scripts no longer run from that location. Prompt v1 is in the git history; the text of prompt v2 was not saved, so the v1 to v2 to v3 ablation cannot be rerun exactly.

## Prompt development history (early phase, `format="json"` mode)

Before the final two-setting protocol, the system prompt was refined in three steps on the same 200 reviews (T=0.0, Ollama's generic JSON mode, not one of the two final settings). Because the same reviews were used for tuning and for the final numbers, final accuracy is probably slightly optimistic. Plots are in `archive_old_protocol/reports/` (figures 7-9).

| Prompt | Main change | Errors / 200 | Accuracy |
|---|---|---|---|
| v1 | three labels (pos/neg/neutral), open entity extraction | 19 | 90.5% |
| v2 | neutral removed, proper-noun entity filter | 15 | 92.5% |
| v3 | explicit rating cues, "sole redeeming factor" and concluding-stance rules | 8 | 96.0% |