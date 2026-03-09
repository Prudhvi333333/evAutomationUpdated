# EV LLM Comparison Pipeline

This repository is organized around one production use case: compare multiple LLMs on Excel-based EV supply-chain questions, with and without RAG, and export evaluation workbooks with responses, retrieval evidence, reference answers, and RAGAS scores.

## Supported runs

- Qwen with RAG
- Qwen without RAG
- Gemma 3 12B with RAG
- Gemma 3 12B without RAG
- Gemini 2.5 Flash with RAG
- Gemini 2.5 Flash without RAG

## What changed

- Replaced the generic document-QA prototype with an Excel-first pipeline under `src/ev_llm_compare/`
- Added structured row-aware chunking plus hybrid dense + lexical retrieval
- Added query planning, structured metadata routing, reranking, and safer context selection
- Added a single CLI entrypoint for repeatable comparisons on changing workbooks
- Added golden-answer loading with fallback reference generation for missing questions
- Batched RAGAS scoring with `answer_accuracy`, `faithfulness`, and `response_groundedness`

## Project layout

- `src/ev_llm_compare/`: active production code
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
export GEMMA_MODEL=gemma3:12b
export RAGAS_JUDGE_PROVIDER=ollama
export RAGAS_JUDGE_MODEL=llama3.1:8b
export RAGAS_EMBEDDING_PROVIDER=huggingface
export RAGAS_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Make sure the local Ollama models are pulled:

```bash
ollama pull qwen2.5:14b
ollama pull gemma3:12b
ollama pull llama3.1:8b
```

## Run

```bash
python main.py \
  --data-workbook "GNEM updated excel (1).xlsx" \
  --question-workbook "Sample questions.xlsx"
```

To compare the first 10 questions with a single-sheet output that includes each model response plus three metric columns per model:

```bash
python main.py \
  --data-workbook "GNEM updated excel (1).xlsx" \
  --question-workbook "Sample questions.xlsx" \
  --question-limit 10 \
  --run-name qwen_rag \
  --run-name qwen_no_rag \
  --run-name gemma_rag \
  --run-name gemma_no_rag \
  --run-name gemini_rag \
  --run-name gemini_no_rag \
  --golden-workbook "artifacts/Golden_answers.xlsx" \
  --output-dir "artifacts/results/sample_run" \
  --single-sheet-only \
  --no-response-exports
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
- Retrieval uses:
  - query planning and metadata-aware routing
  - sentence-transformer dense search
  - lexical overlap scoring
  - reciprocal-rank fusion
  - cross-encoder reranking when enabled
- Exact entity mentions such as company names receive metadata boosts to improve precision.
- Context selection caps duplicate companies and trims oversized structured summaries.

## Outputs

The default full workbook writes these sheets in `artifacts/results/`:

- `responses`
- `all_in_one`
- `retrieval`
- `references`
- `ragas_per_question`
- `ragas_summary`

If you pass `--single-sheet-only`, the output workbook contains only `all_in_one`.
That sheet includes:

- `Question`
- `reference_answer`
- `reference_source`
- one response column per model
- three metric columns per model: `answer_accuracy`, `faithfulness`, `response_groundedness`

Per-run response exports are written to `artifacts/correct_responses/` by default:

- `all_runs_responses.csv`
- `all_runs_metrics.xlsx`
- `<run_name>_responses.csv`
- `<run_name>_responses.md`

## Notes

- The pipeline is workbook-driven, so you can point it at updated Excel files without changing code.
- RAGAS requires the configured judge model and embedding backend to be available.
- If `artifacts/Golden_answers.xlsx` exists, it is used automatically for `answer_accuracy`.
- Any question missing from the golden workbook falls back to generated reference answers.
- Non-RAG runs only receive `answer_accuracy`; context-grounded RAG metrics are reserved for RAG runs.
- The checked-in `Sample questions.xlsx` currently contains 1 question. Use a workbook with 20 rows in the question column when you are ready to run the 20-question comparison.
