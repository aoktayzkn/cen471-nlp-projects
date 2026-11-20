# Archived: early experiments (not the final protocol)

This folder keeps the first version of Project 1, made before the final assignment
requirements. It is kept for transparency only.

- Runs here used Ollama's `format="json"` mode, i.e. neither the prompt-only nor the
  JSON-schema setting required by the assignment.
- The final protocol is in `../src/pipeline_settings.py` and `../src/evaluate_settings.py`;
  its results are in `../data/processed/settings/` and `../reports/`.
- The scripts in `src/` no longer run from this location (their paths point to the old layout).
- Prompt v1 is in the git history; the text of prompt v2 was not saved, so the v1 -> v2 -> v3
  ablation cannot be rerun exactly.