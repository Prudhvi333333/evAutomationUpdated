# EV LLM Comparison Pipeline

This repository is now organized around one production use case: compare multiple LLMs on Excel-based EV supply-chain questions, with and without RAG, and export a single evaluation workbook with responses, retrieval evidence, reference answers, and RAGAS scores.

## Supported runs

- Qwen with RAG
- Qwen without RAG
- TinyLlama with RAG
- TinyLlama without RAG
- Gemini 2.5 Flash with RAG
- Gemini 2.5 Flash without RAG

## What changed

- Replaced the generic document-QA prototype with an Excel-first pipeline under `src/ev_llm_compare/`
- Archived experimental scripts, UI code, old tests, and generated outputs under `unused_files/`
- Added structured row-aware chunking plus hybrid dense + lexical retrieval
- Added a single CLI entrypoint for repeatable comparisons on changing workbooks
- Added reference-answer generation and RAGAS scoring

## Project layout

- `src/ev_llm_compare/`: active production code
- `unused_files/`: archived legacy files and generated artifacts
- `artifacts/qdrant/`: local vector index storage
- `artifacts/results/`: generated comparison workbooks
- `main.py`: CLI entrypoint

## Setup

```bash
uv sync
source .venv/bin/activate
```

Environment variables:

```bash
export GEMINI_API_KEY=your_key_here
export OLLAMA_BASE_URL=http://localhost:11434
export QWEN_MODEL=qwen2.5:14b
export TINYLLAMA_MODEL=tinyllama
```

Make sure the local Ollama models are pulled:

```bash
ollama pull qwen2.5:14b
ollama pull tinyllama
```

## Run

```bash
python main.py \
  --data-workbook "GNEM updated excel (1).xlsx" \
  --question-workbook "Sample questions.xlsx"
```

To compare only the four Ollama runs and save clean per-run response files:

```bash
python main.py \
  --data-workbook "GNEM updated excel (1).xlsx" \
  --question-workbook "Sample questions.xlsx" \
  --run-name qwen_rag \
  --run-name qwen_no_rag \
  --run-name tinyllama_rag \
  --run-name tinyllama_no_rag \
  --output-dir "artifacts/results/sample_run" \
  --response-dir "artifacts/correct_responses/sample_run" \
  --skip-ragas
```

Optional runtime overrides:

```bash
export MODEL_MAX_TOKENS=1600
export MODEL_TEMPERATURE=0.1
```

To skip RAGAS while validating model access:

```bash
python main.py --skip-ragas
```

## Retrieval and chunking strategy

- Every tabular Excel row becomes multiple chunk views:
  - full row snapshot
  - company profile
  - identity theme
  - location theme
  - supply-chain theme
  - product theme
- Single-column sheets such as methodology or definitions are indexed as note/reference chunks.
- Retrieval uses sentence-transformer dense search plus lexical overlap scoring and reciprocal-rank fusion.
- Exact entity mentions such as company names receive metadata boosts to improve precision.

## Outputs

Each run writes an Excel report in `artifacts/results/` with:

- `responses`
- `retrieval`
- `references`
- `ragas_per_question`
- `ragas_summary`

Per-run response exports are written to `artifacts/correct_responses/` by default:

- `all_runs_responses.csv`
- `<run_name>_responses.csv`
- `<run_name>_responses.md`

## Notes

- The pipeline is workbook-driven, so you can point it at updated Excel files without changing code.
- RAGAS requires the configured judge model and embedding backend to be available.
- Non-RAG runs are still scored against the same question and reference answer, which makes model-to-model comparison easier.
- The checked-in `Sample questions.xlsx` currently contains 1 question. Use a workbook with 20 rows in the question column when you are ready to run the 20-question comparison.
