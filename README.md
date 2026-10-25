# CEN471 - Natural Language Processing Course Projects

This repository hosts three modular NLP course projects developed under strict local constraints.

---

## 📌 Projects Matrix

| Project | Domain | Architecture / Local Engine | Status |
| :--- | :--- | :--- | :---: |
| **Project 1** | Structured Information Extraction, Temperature & Prompt Ablation | Local Seq2Seq / LLM In-Context | Complete |
| **Project 2** | Machine Translation & Paraphrasing | Local Seq2Seq / LLM In-Context | Complete |
| **Project 3** | Long-Document Summarization | Hierarchical MapReduce via Local LLM | Complete |

---

## 🛠️ Global Environment & Setup

Clone the repository and initialize the Python virtual environment:

```bash
git clone https://github.com/aoktayzkn/cen471-nlp-projects.git
cd cen471-nlp-projects

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1    # On Windows
# source venv/bin/activate      # On Linux / macOS

# Install dependencies across all projects
pip install -r requirements.txt
```

Ensure Ollama is installed and the base model is pulled locally:

```bash
ollama run qwen2.5:7b
```

---

## 📂 Repository Architecture

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
│   │       ├── extracted_results_v2_temp_0.0.csv
│   │       ├── extracted_results_v3_temp_0.0.csv
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
│       │   ├── 6_temperature_evaluation_benchmark.png
│       │   ├── prompt_ablation_performance_progression.png
│       │   ├── confusion_matrices_v2_vs_v3.png
│       │   └── precision_recall_comparison_v2_v3.png
│       └── imbalanced dataset results/
│           ├── baseline_imbalanced_evaluation_report.txt
│           └── imbalanced_report.txt
├── project_2_translation_paraphrase/
└── project_3_summarization_mapreduce/
```

---

## 🔨 Project 1 Highlights: Dual Ablation Framework

Project 1 was rigorously benchmarked across **200 balanced IMDB reviews** under two empirical axes: **Sampling Temperature Dynamics** and **System Prompt Optimization**.

### 1. Temperature Ablation Study ($T \in \{0.0, 0.3, 0.7, 1.0\}$)

Evaluated model behavior and determinism using the baseline prompt:

| Temperature ($T$) | Schema Validity | First-Try JSON Parse | Accuracy | Macro-F1 | Steady-State Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.0 (Optimal)** | 100.0% | 100.0% | 90.50% | 0.9255 | ~780 ms |
| **0.3** | 100.0% | 100.0% | 90.00% | 0.9203 | ~781 ms |
| **0.7** | 100.0% | 100.0% | 89.50% | 0.9174 | ~775 ms |
| **1.0** | 100.0% | 100.0% | 90.50% | 0.9257 | ~777 ms |

- **Schema Invariance:** Ollama's constrained JSON mode maintained a strict **100% schema compliance rate** with zero validation failures even at $T = 1.0$.
- **Entropy Hierarchy:** While closed-domain classification remained deterministic (sentiment flipped in $\le 2.0\%$ of samples), open-domain extraction exhibited significant divergence: category assignments shifted by **13.5%**, and extracted named entity sets diverged by **53.0%** at $T = 1.0$ compared to $T = 0.0$.
- **Engineering Standard:** $T = 0.0$ is retained as the production configuration to guarantee determinism, eliminate hallucinated entities, and enforce strict reproducibility.

---

### 2. Prompt Optimization Ablation Study ($v1 \to v2 \to v3$ at $T = 0.0$)

Following error analysis on the baseline outputs, the system prompt underwent two iterative refinements:

| Prompt Version | Architectural Interventions | Total Errors | Accuracy | Macro-F1 | Schema Validity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v1 (Baseline)** | Ternary label domain (`pos` / `neg` / `neu`), open entity extraction | 19 / 200 | 90.50% | 0.9255 | 100.0% |
| **v2 (Binary Alignment)** | Polarity resolution (neutral removed), proper noun entity filter | 15 / 200 | 92.50% | 0.9250 | 100.0% |
| **v3 (Heuristic-Guided)** | Explicit numerical rating cues, sole redeeming factor rule, concluding stance dominance | 8 / 200 | 96.00% | 0.9600 | 100.0% |

```text
PROMPT ABLATION PROGRESSION (200 SAMPLES)
v1 (Baseline)       [========================================> ] 90.50% (19 errors)
v2 (Binary Alignment) [==========================================> ] 92.50% (15 errors)
v3 (Heuristic-Guided) [==============================================] 96.00% (8 errors)
```

### Key Architectural Contributions in Prompt v3

1. **Zero-Regression Optimization:** Transitioning from v2 to v3 corrected **7 critical edge-case errors** while preserving **100% of previously correct classifications** (`newly_broken: 0`).
2. **Symmetric Class Balance:** Prompt v3 achieved completely balanced classification across both classes:
   - **Negative Sentiment:** 96.00% Precision | 96.00% Recall | 0.9600 F1 (96 TN, 4 FP)
   - **Positive Sentiment:** 96.00% Precision | 96.00% Recall | 0.9600 F1 (96 TP, 4 FN)
3. **Semantic Boundary Resolution:** Explicit heuristics resolved subtle classification traps, including sarcastic/ironic reviews (e.g., *"give it a 3/10"*), nostalgic guilty pleasures, and reviews praising an actor's appearance while condemning the overall production.
