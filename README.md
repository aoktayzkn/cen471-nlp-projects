# CEN471 - Natural Language Processing Course Projects

This repository hosts three modular NLP course projects developed under strict local execution constraints—**strictly zero cloud API dependencies (e.g., OpenAI, Anthropic)**—powered entirely by local open-source LLMs via the **Ollama** ecosystem.

---

## 📌 Projects Matrix

| Project | Domain | Architecture / Local Engine | Status |
| :--- | :--- | :--- | :---: |
| **Project 1** | Structured Information Extraction & Temperature Ablation | `qwen2.5:7b` + JSON Schema Guardrails | **Completed** |
| **Project 2** | Machine Translation & Paraphrasing | Local Seq2Seq / LLM In-Context Learning | Planned |
| **Project 3** | Long-Document Summarization | Hierarchical MapReduce via Local LLM | Planned |

---

## 🛠️ Global Environment & Setup

Clone the repository and initialize the Python virtual environment:

```bash
git clone https://github.com/aoktayzkn/cen471-nlp-projects.git
cd cen471-nlp-projects

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # On Windows
# source venv/bin/activate    # On Linux / macOS

# Install dependencies across all projects
pip install -r requirements.txt
```

Ensure Ollama is installed and the base model is pulled locally:
```bash
ollama run qwen2.5:7b
```

---

## 📁 Repository Architecture

```text
cen471-nlp-projects/
├── requirements.txt
├── README.md
├── project_1_structured_extraction/
│   ├── README.md
│   ├── data/
│   │   ├── raw/
│   │   │   ├── imdb_test_100.csv
│   │   │   └── imdb_balanced_200.csv
│   │   └── processed/
│   │       ├── extracted_results.csv
│   │       ├── extracted_results_temp_0.3.csv
│   │       ├── extracted_results_temp_0.7.csv
│   │       ├── extracted_results_temp_1.0.csv
│   │       └── temperature_comparison.csv
│   ├── prompts/
│   │   └── extraction_prompt.py
│   ├── src/
│   │   ├── data_loader.py
│   │   ├── parser.py
│   │   ├── pipeline.py
│   │   ├── pipeline_temperature.py
│   │   └── evaluator.py
│   └── reports/
│       ├── error_analysis.txt
│       ├── result.txt
│       ├── plots/
│       │   ├── 1_performance_metrics.png
│       │   ├── 2_confusion_matrix.png
│       │   ├── 3_prediction_distribution.png
│       │   ├── 4_latency_distribution.png
│       │   ├── 5_temperature_divergence_analysis.png
│       │   └── 6_temperature_evaluation_benchmark.png
│       └── imbalanced dataset results/
│           ├── baseline_imbalanced_evaluation_report.txt
│           └── imbalanced_report.txt
├── project_2_translation_paraphrase/
└── project_3_summarization_mapreduce/
```

---

## 🔬 Project 1 Highlights: Temperature Ablation Study

An empirical ablation study was conducted across **200 balanced IMDB reviews** evaluating model behavior across varying sampling temperatures ($T \in \{0.0, 0.3, 0.7, 1.0\}$):

| Temperature ($T$) | Schema Validity | First-Try JSON Parse | Accuracy | Macro-F1 | Steady-State Latency |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0 (Baseline)** | **100.0%** | **100.0%** | **90.50%** | **0.9255** | ~780 ms |
| **0.3** | **100.0%** | **100.0%** | 90.00% | 0.9203 | ~781 ms |
| **0.7** | **100.0%** | **100.0%** | 89.50% | 0.9174 | ~775 ms |
| **1.0** | **100.0%** | **100.0%** | **90.50%** | **0.9257** | ~777 ms |

### Key Findings
1. **Schema Invariance:** Ollama's constrained JSON mode maintained a strict **100% schema compliance rate** with zero validation failures even at $T=1.0$.
2. **Entropy Hierarchy across Tasks:** While closed-domain classification remained deterministic (sentiment prediction flipped in $\le 2.0\%$ of samples), open-domain extraction exhibited substantial divergence: category assignments shifted by **13.5%**, and extracted named entity sets diverged by **53.0%** at $T=1.0$ compared to $T=0.0$.
3. **Engineering Standard:** For structured information extraction pipelines in production, **$T=0.0$** is retained as the optimal configuration to guarantee determinism, minimize qualitative entity noise, and eliminate stochastic variation.

For detailed methodology, prompt templates, and step-by-step pipeline execution, please refer to [`project_1_structured_extraction/README.md`](project_1_structured_extraction/README.md).
