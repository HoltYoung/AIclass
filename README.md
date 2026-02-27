# QAC387 AI Data Analysis Pipeline

Data analysis pipeline and interactive LLM assistant for QAC387.

## Repository Structure

```
.
├── builds/
│   ├── build0_data_analysis_pipeline_assignment_1.py   # Build 0: Data analysis pipeline
│   └── build1_llm_assistant_Assignment_2.py            # Build 1: Interactive LLM assistant
├── src/                          # Refactored modules from Build 0
│   ├── __init__.py               # Re-exports all functions
│   ├── utilities.py              # File I/O and directory helpers
│   ├── profiling.py              # Dataset profiling and column splitting
│   ├── summaries.py              # Numeric and categorical summaries
│   ├── analysis.py               # Missingness, regression, correlations
│   ├── plots.py                  # Visualization functions
│   └── checks.py                 # JSON validation and target checks
├── data/
│   └── penguins.csv              # Palmer Penguins dataset
├── test_models.py                # Module import and functionality tests
├── requirements.txt
├── .env.example                  # Template for API key configuration
└── ASSIGNMENT_README.md          # Original Build 0 assignment instructions
```

## Setup

1. Clone the repository and create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # macOS/Linux
   .venv\Scripts\activate           # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your OpenAI API key:
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

4. Verify modules work:
   ```bash
   python test_models.py
   ```

## Build 0: Data Analysis Pipeline

Automated pipeline that loads a CSV dataset, generates profiling reports, summary statistics, correlation analysis, regression output, and visualizations.

```bash
python builds/build0_data_analysis_pipeline_assignment_1.py --data data/penguins.csv
```

## Build 1: Interactive LLM Assistant

Interactive CLI assistant powered by LangChain LCEL that answers questions about dataset schema. Supports three modes:

**Run 1 — No memory:**
```bash
python builds/build1_llm_assistant_Assignment_2.py --data data/penguins.csv
```

**Run 2 — With memory** (retains conversation context):
```bash
python builds/build1_llm_assistant_Assignment_2.py --data data/penguins.csv --memory
```

**Run 3 — Memory + streaming** (streams responses token-by-token):
```bash
python builds/build1_llm_assistant_Assignment_2.py --data data/penguins.csv --memory --stream
```

### Additional flags

| Flag | Description |
|------|-------------|
| `--model` | LLM model name (default: `gpt-4o-mini`) |
| `--temperature` | Sampling temperature (default: `0.2`) |
| `--quiet_schema` | Suppress schema display at startup |
| `--report_dir` | Output directory (default: `reports`) |
