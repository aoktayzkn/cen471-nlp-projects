# CEN471 - Natural Language Processing Course Projects

Three NLP course projects built on local LLMs (Ollama).

| Project | Topic | Status |
|---|---|---|
| **Project 1** | Structured JSON classification and entity extraction on IMDB reviews: prompt-only vs. schema-constrained decoding | **Complete**, see [project_1_structured_extraction](project_1_structured_extraction/README.md) |
| **Project 2** | English to Turkish machine translation with `gemma3:4b` and metric evaluation (BLEU, chrF, METEOR) | **Complete**, see [project_2_translation](project_2_translation/README.md) |
| **Project 3** | Long-document summarization | Pending |

## Setup

```bash
git clone https://github.com/aoktayzkn/cen471-nlp-projects.git
cd cen471-nlp-projects

python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows
# source venv/bin/activate        # Linux / macOS

pip install -r requirements.txt
ollama pull qwen2.5:7b    # Project 1
ollama pull gemma3:4b     # Project 2
```

Ollama must be installed and running. Run all commands from the repository root.

## Project 1 in short

Prompt-only JSON generation is compared with schema-constrained decoding (Ollama's `format` parameter) on 200 IMDB reviews. Schema-constrained decoding makes every output parseable and schema-valid (100% in all conditions) but does not change the model's sentiment judgment. Report (PDF), code, results and run instructions are in [project_1_structured_extraction](project_1_structured_extraction/README.md).

## Project 2 in short

150 English sentences from WMT18 are translated into Turkish with `gemma3:4b` (BLEU 15.71, chrF 52.38, METEOR 46.14 at temperature 0.1). Higher temperatures change the wording much more than the scores, and on 30 human-rated sentences none of the three metrics showed a clear relation to the ratings. Report (PDF), code, results and run instructions are in [project_2_translation](project_2_translation/README.md).
