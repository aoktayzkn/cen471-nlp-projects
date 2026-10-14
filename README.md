# CEN471 - Natural Language Processing Course Projects

This repository hosts three modular NLP course projects developed under strict local execution constraints—**strictly zero cloud API dependencies (e.g., OpenAI, Anthropic)**—powered entirely by local open-source LLMs via the **Ollama** ecosystem.

---

## 📌 Projects Matrix

| Project | Domain | Architecture / Local Engine | Status |
| :--- | :--- | :--- | :---: |
| **Project 1** | Structured Information Extraction | `qwen2.5:7b` + JSON Schema Guardrails | **Completed** |
| **Project 2** | Machine Translation & Paraphrasing | Local Seq2Seq / LLM In-Context Learning | Planned |
| **Project 3** | Long-Document Summarization | Hierarchical MapReduce via Local LLM | Planned |

---

## 🛠️ Global Environment & Setup

Clone the repository and initialize the Python virtual environment:

```bash
git clone [https://github.com/aoktayzkn/cen471-nlp-projects.git](https://github.com/aoktayzkn/cen471-nlp-projects.git)
cd cen471-nlp-projects

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # On Windows
# source venv/bin/activate    # On Linux / macOS

# Install dependencies across all projects
pip install -r requirements.txt
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
│   │   └── processed/
│   ├── prompts/
│   ├── src/
│   └── reports/
│       ├── error_analysis.txt
│       └── plots/
├── project_2_translation_paraphrase/
└── project_3_summarization_mapreduce/
```

For detailed methodology, benchmark figures, and empirical findings for each project, please refer to their respective directory READMEs.
