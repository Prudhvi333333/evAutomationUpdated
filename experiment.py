#!/usr/bin/env python3
"""experiment.py — Authoritative 5-model × 2-mode × 50-question runner.

This is the single entry-point for the full comparative study.

Usage
-----
python experiment.py \
  --data-workbook   "data/GNEM auto landscape.xlsx" \
  --questions       "data/GNEM_Golden_Questions.xlsx" \
  --golden-answers  "data/human_validated_answers.xlsx" \
  --output-dir      artifacts/study_$(date +%Y%m%d_%H%M%S) \
  --seed            42

To run only specific models or modes:
  --run-names qwen14b_rag qwen14b_no_rag gemini_flash_rag gemini_flash_no_rag

To skip evaluation metrics (faster, generation only):
  --skip-evaluation

To disable actual ragas (if ragas library not installed):
  --no-ragas
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

import pandas as pd

from ev_llm_compare.chunking import ExcelChunkBuilder
from ev_llm_compare.derived_analytics import build_derived_summary_chunks
from ev_llm_compare.evaluation import (
    ALL_METRIC_NAMES,
    run_evaluation_metrics,
)
from ev_llm_compare.excel_loader import (
    EvalQuestion,
    join_golden_answers,
    load_eval_questions,
    load_golden_answers,
    load_workbook,
)
from ev_llm_compare.models import create_client, safe_generate_with_metadata
from ev_llm_compare.prompts import (
    NON_RAG_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    build_non_rag_prompt,
    build_rag_prompt,
    format_context,
)
from ev_llm_compare.ragas_integration import (
    ragas_available,
    run_ragas_evaluation,
    _response_key,
)
from ev_llm_compare.retrieval import HybridRetriever
from ev_llm_compare.schemas import ModelResponse, RetrievalResult
from ev_llm_compare.settings import AppConfig, ModelSpec, load_config

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the 5-model × 2-mode × 50-question EV RAG study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-workbook",
        default=None,
        help="Excel workbook containing the EV supply chain data. "
             "Auto-detected from data/ if omitted.",
    )
    p.add_argument(
        "--questions",
        default=None,
        help="Excel/CSV/JSON file with 50 evaluation questions. "
             "Auto-detected from data/ if omitted.",
    )
    p.add_argument(
        "--golden-answers",
        default=None,
        help="Excel/CSV file with human-validated answers for the 50 questions. "
             "Auto-detected from data/ if omitted.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory where all outputs are written. "
             "Defaults to artifacts/study_<timestamp>.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global random seed for generation. Sets EXPERIMENT_SEED env-var.",
    )
    p.add_argument(
        "--run-names",
        nargs="+",
        default=None,
        metavar="RUN_NAME",
        help="Limit execution to specific run names "
             "(e.g. qwen14b_rag qwen14b_no_rag). Default: all enabled runs.",
    )
    p.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Cap the number of questions (useful for smoke testing).",
    )
    p.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip LLM-judge evaluation metrics. Only generates answers.",
    )
    p.add_argument(
        "--no-ragas",
        action="store_true",
        help="Disable ragas library metrics even if ragas is installed.",
    )
    p.add_argument(
        "--question-limit",
        type=int,
        default=None,
        dest="max_questions",
        help="Alias for --max-questions (backward compat).",
    )
    return p


# ──────────────────────────────────────────────────────────────────────────────
# File auto-detection helpers
# ──────────────────────────────────────────────────────────────────────────────

def _find_data_workbook(hint: str | None) -> Path:
    if hint:
        p = Path(hint).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Data workbook not found: {p}")
        return p
    candidates = [
        PROJECT_ROOT / "data" / "GNEM - Auto Landscape Lat Long Updated.xlsx",
        PROJECT_ROOT / "data" / "GNEM auto landscape.xlsx",
        PROJECT_ROOT / "data" / "GNEM_auto_landscape.xlsx",
        PROJECT_ROOT / "data" / "GNEM updated excel.xlsx",
        PROJECT_ROOT / "GNEM updated excel (1).xlsx",
        PROJECT_ROOT / "GNEM updated excel.xlsx",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        f"Could not auto-detect data workbook. Checked: "
        + ", ".join(str(c) for c in candidates)
    )


def _find_questions(hint: str | None) -> Path:
    if hint:
        p = Path(hint).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Questions file not found: {p}")
        return p
    candidates = [
        PROJECT_ROOT / "data" / "Human validated 50 questions.xlsx",
        PROJECT_ROOT / "data" / "GNEM_Golden_Questions.xlsx",
        PROJECT_ROOT / "data" / "Sample questions.xlsx",
        PROJECT_ROOT / "data" / "questions.xlsx",
        PROJECT_ROOT / "data" / "questions.csv",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        f"Could not auto-detect questions file. Checked: "
        + ", ".join(str(c) for c in candidates)
    )


def _find_golden_answers(hint: str | None) -> Path | None:
    if hint:
        p = Path(hint).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Golden answers file not found: {p}")
        return p
    candidates = [
        PROJECT_ROOT / "data" / "Human validated 50 questions.xlsx",
        PROJECT_ROOT / "data" / "human_validated_answers.xlsx",
        PROJECT_ROOT / "data" / "Golden_answers.xlsx",
        PROJECT_ROOT / "data" / "golden_answers.xlsx",
        PROJECT_ROOT / "data" / "Golden_answers.csv",
        PROJECT_ROOT / "artifacts" / "Golden_answers_updated.xlsx",
        PROJECT_ROOT / "artifacts" / "Golden_answers.xlsx",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None  # no golden answers; evaluation will use auto-generated references


# ──────────────────────────────────────────────────────────────────────────────
# Pre-flight validation
# ──────────────────────────────────────────────────────────────────────────────

def _validate_env(config: AppConfig) -> list[str]:
    """Check that required env-vars are set. Returns list of problems."""
    issues: list[str] = []
    for spec in config.models:
        if spec.provider == "gemini":
            if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
                issues.append(
                    f"Model '{spec.run_name}': GEMINI_API_KEY or GOOGLE_API_KEY not set."
                )
        elif spec.provider == "openai_compatible":
            if spec.api_key_env and not os.getenv(spec.api_key_env):
                issues.append(
                    f"Model '{spec.run_name}': env-var '{spec.api_key_env}' not set."
                )
    return issues


# ──────────────────────────────────────────────────────────────────────────────
# Experiment manifest
# ──────────────────────────────────────────────────────────────────────────────

def _build_manifest(
    config: AppConfig,
    questions: list[EvalQuestion],
    data_workbook_path: Path,
    questions_path: Path,
    golden_path: Path | None,
    seed: int | None,
    output_dir: Path,
    prompt_version: str = "v2",
) -> dict[str, Any]:
    return {
        "experiment_version": "2.0",
        "prompt_version": prompt_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "output_dir": str(output_dir),
        "data_workbook": str(data_workbook_path),
        "questions_file": str(questions_path),
        "golden_answers_file": str(golden_path) if golden_path else None,
        "n_questions": len(questions),
        "question_ids": [q.q_id for q in questions],
        "models": [
            {
                "run_name": m.run_name,
                "provider": m.provider,
                "model_name": m.model_name,
                "rag_enabled": m.rag_enabled,
                "temperature": m.temperature,
                "max_tokens": m.max_tokens,
                "seed": m.seed,
            }
            for m in config.models
        ],
        "retrieval": {
            "embedding_model": config.retrieval.embedding_model,
            "dense_top_k": config.retrieval.dense_top_k,
            "final_top_k": config.retrieval.final_top_k,
            "reranker_enabled": config.retrieval.reranker_enabled,
            "reranker_model": config.retrieval.reranker_model,
            "lexical_weight": config.retrieval.lexical_weight,
            "dense_weight": config.retrieval.dense_weight,
        },
        "evaluation": {
            "judge_provider": config.evaluation.judge_provider,
            "judge_model": config.evaluation.judge_model,
            "embedding_model": config.evaluation.embedding_model,
            "ragas_enabled": config.evaluation.ragas_enabled,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Citation extraction helper
# ──────────────────────────────────────────────────────────────────────────────

def _extract_citation_ids(answer: str, n_chunks: int) -> list[str]:
    import re
    found = re.findall(r"\[E(\d+)\]", answer)
    valid = [f"E{i}" for i in found if 1 <= int(i) <= n_chunks]
    return list(dict.fromkeys(valid))  # deduplicated, order-preserving


# ──────────────────────────────────────────────────────────────────────────────
# Core generation loop
# ──────────────────────────────────────────────────────────────────────────────

def _generate_responses(
    config: AppConfig,
    questions: list[EvalQuestion],
    retriever: HybridRetriever,
    *,
    prompt_version: str = "v2",
) -> tuple[list[ModelResponse], list[dict[str, Any]]]:
    """Run all enabled model × mode combinations over all questions.

    Returns:
        responses  – list[ModelResponse] for evaluation
        flat_rows  – list[dict] with full per-response metadata for CSV export
    """
    # Pre-compute retrieval for all questions (shared across all models)
    print("[experiment] Pre-computing retrieval for all questions…", flush=True)
    retrievals: dict[str, list[RetrievalResult]] = {}
    for q in questions:
        retrievals[q.question] = retriever.retrieve(q.question)
    print(f"[experiment] Retrieval complete for {len(questions)} questions.", flush=True)

    responses: list[ModelResponse] = []
    flat_rows: list[dict[str, Any]] = []

    for spec_idx, spec in enumerate(config.models, start=1):
        print(
            f"\n[experiment] ── Model {spec_idx}/{len(config.models)}: "
            f"{spec.run_name} ({spec.model_name}, rag={spec.rag_enabled}) ──",
            flush=True,
        )
        client = create_client(spec, config.runtime)

        for q_idx, q in enumerate(questions, start=1):
            print(
                f"[experiment]   Q{q.q_id} ({q_idx}/{len(questions)}): "
                f"{q.question[:80]}{'…' if len(q.question) > 80 else ''}",
                flush=True,
            )

            retrieved = retrievals[q.question]

            if spec.rag_enabled:
                context_str = format_context(
                    retrieved,
                    question=q.question,
                    max_results=config.retrieval.generation_context_result_limit,
                    max_chars=config.retrieval.generation_context_char_budget,
                    compact=config.retrieval.compact_context_enabled,
                )
                prompt = build_rag_prompt(q.question, context_str)
                system_prompt = RAG_SYSTEM_PROMPT
                stored_chunks = retrieved
            else:
                prompt = build_non_rag_prompt(q.question)
                system_prompt = NON_RAG_SYSTEM_PROMPT
                stored_chunks = []

            answer, latency, success, error, meta = safe_generate_with_metadata(
                client,
                prompt,
                temperature=spec.temperature,
                max_tokens=spec.max_tokens,
                system_prompt=system_prompt,
                seed=spec.seed,
            )

            citations = (
                _extract_citation_ids(answer, len(retrieved))
                if spec.rag_enabled
                else []
            )
            abstained = "insufficient evidence" in answer.lower() if success else False

            response = ModelResponse(
                run_name=spec.run_name,
                provider=spec.provider,
                model_name=spec.model_name,
                rag_enabled=spec.rag_enabled,
                question=q.question,
                answer=answer,
                latency_seconds=latency,
                retrieved_chunks=stored_chunks,
                prompt_tokens_estimate=max(1, len(prompt) // 4),
                success=success,
                error_message=error,
            )
            responses.append(response)

            flat_rows.append({
                "q_id": q.q_id,
                "question": q.question,
                "golden_answer": q.golden_answer or "",
                "run_name": spec.run_name,
                "model_name": spec.model_name,
                "provider": spec.provider,
                "mode": "rag" if spec.rag_enabled else "no_rag",
                "answer": answer,
                "success": success,
                "abstained": abstained,
                "error_message": error or "",
                "latency_seconds": latency,
                "tokens_in": meta.prompt_tokens,
                "tokens_out": meta.completion_tokens,
                "tokens_total": meta.total_tokens,
                "finish_reason": meta.finish_reason or "",
                "n_retrieved_chunks": len(retrieved),
                "retrieved_chunk_ids": ",".join(r.chunk_id for r in retrieved[:8]),
                "citations_in_answer": ",".join(citations),
                "citation_count": len(citations),
                "response_length_words": len(answer.split()),
                "prompt_version": prompt_version,
                "seed": spec.seed,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })

    return responses, flat_rows


# ──────────────────────────────────────────────────────────────────────────────
# Main orchestration
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(
    config: AppConfig,
    data_workbook_path: Path,
    questions_path: Path,
    golden_path: Path | None,
    output_dir: Path,
    seed: int | None = None,
    max_questions: int | None = None,
    skip_evaluation: bool = False,
    disable_ragas: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # ── Load questions ─────────────────────────────────────────────────────────
    print(f"[experiment] Loading questions from {questions_path}", flush=True)
    questions = load_eval_questions(
        questions_path, max_questions=max_questions
    )
    print(f"[experiment] Loaded {len(questions)} questions.", flush=True)

    # ── Load & join golden answers ─────────────────────────────────────────────
    golden_by_id: dict[str, dict[str, str]] = {}
    if golden_path is not None:
        print(f"[experiment] Loading golden answers from {golden_path}", flush=True)
        golden_by_id = load_golden_answers(golden_path)
        questions, unmatched = join_golden_answers(questions, golden_by_id)
        n_matched = sum(1 for q in questions if q.golden_answer)
        print(
            f"[experiment] Golden answers: {n_matched}/{len(questions)} matched. "
            f"Unmatched IDs: {unmatched or 'none'}",
            flush=True,
        )
        if unmatched:
            print(
                f"[experiment] WARNING: {len(unmatched)} questions have no golden "
                f"answer. They will receive only deterministic metrics if available.",
                flush=True,
            )

    # golden_answers as {question_text: answer} for evaluation functions
    golden_text_map: dict[str, str] = {
        q.question: (q.golden_answer or "") for q in questions
    }

    # ── Build retrieval index ─────────────────────────────────────────────────
    print(f"[experiment] Loading data workbook: {data_workbook_path}", flush=True)
    rows, notes = load_workbook(data_workbook_path)
    print(
        f"[experiment] Workbook: {len(rows)} rows, {len(notes)} note sheets.",
        flush=True,
    )

    print("[experiment] Building chunks…", flush=True)
    chunks = ExcelChunkBuilder(config.retrieval).build(rows, notes)
    derived = build_derived_summary_chunks(rows)
    chunks.extend(derived)
    print(
        f"[experiment] {len(chunks)} total chunks "
        f"({len(derived)} derived analytics).",
        flush=True,
    )

    print("[experiment] Initialising HybridRetriever…", flush=True)
    retriever = HybridRetriever(
        chunks=chunks,
        settings=config.retrieval,
        qdrant_path=config.runtime.qdrant_path,
    )

    # ── Save experiment manifest ───────────────────────────────────────────────
    manifest = _build_manifest(
        config=config,
        questions=questions,
        data_workbook_path=data_workbook_path,
        questions_path=questions_path,
        golden_path=golden_path,
        seed=seed,
        output_dir=output_dir,
    )
    manifest_path = output_dir / "experiment_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[experiment] Manifest saved → {manifest_path}", flush=True)

    try:
        # ── Generation ────────────────────────────────────────────────────────
        print("\n[experiment] ═══ GENERATION PHASE ═══\n", flush=True)
        gen_start = time.perf_counter()
        responses, flat_rows = _generate_responses(
            config=config,
            questions=questions,
            retriever=retriever,
        )
        gen_elapsed = round(time.perf_counter() - gen_start, 1)
        print(
            f"\n[experiment] Generation complete: {len(responses)} responses "
            f"in {gen_elapsed}s.",
            flush=True,
        )

        # ── Save generation checkpoint ────────────────────────────────────────
        responses_csv = output_dir / f"responses_raw_{ts}.csv"
        pd.DataFrame(flat_rows).to_csv(responses_csv, index=False)
        print(f"[experiment] Raw responses CSV → {responses_csv}", flush=True)

        responses_jsonl = output_dir / f"responses_raw_{ts}.jsonl"
        with responses_jsonl.open("w", encoding="utf-8") as fh:
            for row in flat_rows:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(f"[experiment] Raw responses JSONL → {responses_jsonl}", flush=True)

        if skip_evaluation:
            print("[experiment] --skip-evaluation set; stopping after generation.", flush=True)
            return responses_csv

        # ── Custom LLM-judge + reference metrics evaluation ───────────────────
        print("\n[experiment] ═══ EVALUATION PHASE (custom metrics) ═══\n", flush=True)
        eval_start = time.perf_counter()

        def _log_progress(completed: int, total: int, resp: ModelResponse) -> None:
            if completed % 10 == 0 or completed == total:
                print(
                    f"[experiment]   Evaluation: {completed}/{total} "
                    f"({resp.run_name}: {resp.question[:60]}…)",
                    flush=True,
                )

        metrics_per_run, metrics_summary = run_evaluation_metrics(
            responses=responses,
            reference_answers=golden_text_map,
            judge_provider=config.evaluation.judge_provider,
            judge_model=config.evaluation.judge_model,
            max_retries=config.evaluation.max_retries,
            context_result_limit=config.retrieval.evaluation_context_result_limit,
            context_char_budget=config.retrieval.evaluation_context_char_budget,
            compact_context=config.retrieval.compact_context_enabled,
            parallelism=config.evaluation.parallelism,
            embedding_model=config.evaluation.embedding_model,
            progress_callback=_log_progress,
            judge_base_url=config.evaluation.judge_base_url,
            judge_api_key_env=config.evaluation.judge_api_key_env,
        )
        eval_elapsed = round(time.perf_counter() - eval_start, 1)
        print(f"[experiment] Custom evaluation complete in {eval_elapsed}s.", flush=True)

        # ── Actual ragas evaluation (RAG responses only) ───────────────────────
        ragas_scores: dict[str, dict[str, float | None]] = {}
        if config.evaluation.ragas_enabled and not disable_ragas:
            if ragas_available():
                print("\n[experiment] ═══ EVALUATION PHASE (ragas library) ═══\n", flush=True)
                ragas_start = time.perf_counter()
                ragas_scores = run_ragas_evaluation(
                    responses=responses,
                    golden_answers=golden_text_map,
                    judge_provider=config.evaluation.judge_provider,
                    judge_model=config.evaluation.judge_model,
                    embedding_model=config.evaluation.embedding_model,
                    context_result_limit=config.retrieval.evaluation_context_result_limit,
                    context_char_budget=config.retrieval.evaluation_context_char_budget,
                )
                ragas_elapsed = round(time.perf_counter() - ragas_start, 1)
                print(
                    f"[experiment] ragas evaluation complete in {ragas_elapsed}s. "
                    f"Scored {len(ragas_scores)} responses.",
                    flush=True,
                )
            else:
                print(
                    "[experiment] ragas library not installed; "
                    "skipping ragas metrics. Install with: pip install ragas",
                    flush=True,
                )

        # ── Merge all metrics into the flat CSV ───────────────────────────────
        flat_df = pd.DataFrame(flat_rows)

        # Add custom metrics
        if metrics_per_run is not None and not metrics_per_run.empty:
            for col in metrics_per_run.columns:
                if col in {"run_name", "question"}:
                    continue
                flat_df = flat_df.merge(
                    metrics_per_run[["run_name", "question", col]].rename(
                        columns={"run_name": "run_name", "question": "question"}
                    ),
                    left_on=["run_name", "question"],
                    right_on=["run_name", "question"],
                    how="left",
                    suffixes=("", f"_metric"),
                )
                # If a duplicate column was created, resolve it
                if f"{col}_metric" in flat_df.columns:
                    flat_df[col] = flat_df[f"{col}_metric"]
                    flat_df.drop(columns=[f"{col}_metric"], inplace=True)

        # Add ragas scores
        if ragas_scores:
            ragas_col_names = set()
            for score_dict in ragas_scores.values():
                ragas_col_names.update(score_dict.keys())
            for col in ragas_col_names:
                flat_df[col] = None

            for response in responses:
                key = _response_key(response)
                if key not in ragas_scores:
                    continue
                mask = (flat_df["run_name"] == response.run_name) & (
                    flat_df["question"] == response.question
                )
                for col, val in ragas_scores[key].items():
                    flat_df.loc[mask, col] = val

        # ── Save merged results CSV ───────────────────────────────────────────
        results_csv = output_dir / f"results_all_{ts}.csv"
        flat_df.to_csv(results_csv, index=False)
        print(f"\n[experiment] Full results CSV → {results_csv}", flush=True)

        # ── Save per-run metrics ───────────────────────────────────────────────
        if metrics_per_run is not None:
            per_q_csv = output_dir / f"metrics_per_question_{ts}.csv"
            metrics_per_run.to_csv(per_q_csv, index=False)
            print(f"[experiment] Per-question metrics → {per_q_csv}", flush=True)

        if metrics_summary is not None:
            summary_csv = output_dir / f"metrics_summary_{ts}.csv"
            metrics_summary.to_csv(summary_csv, index=False)
            print(f"[experiment] Metrics summary → {summary_csv}", flush=True)

        return results_csv

    finally:
        retriever.close()


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    # Propagate seed to env so _build_models() picks it up
    if args.seed is not None:
        os.environ["EXPERIMENT_SEED"] = str(args.seed)

    config = load_config()

    # Filter models by run-name if requested
    if args.run_names:
        requested = set(args.run_names)
        available = {m.run_name for m in config.models}
        unknown = requested - available
        if unknown:
            print(
                f"[experiment] ERROR: Unknown run name(s): {sorted(unknown)}. "
                f"Available: {sorted(available)}",
                file=sys.stderr,
            )
            return 1
        config.models = [m for m in config.models if m.run_name in requested]

    if not config.models:
        print(
            "[experiment] ERROR: No models are enabled. Check ENABLE_* env-vars.",
            file=sys.stderr,
        )
        return 1

    # Pre-flight env validation
    env_issues = _validate_env(config)
    if env_issues:
        print("[experiment] ERROR: Missing environment variables:", file=sys.stderr)
        for issue in env_issues:
            print(f"  • {issue}", file=sys.stderr)
        print(
            "\nSet the missing env-vars in your .env file or environment and retry.",
            file=sys.stderr,
        )
        return 1

    # Resolve file paths
    try:
        data_workbook_path = _find_data_workbook(args.data_workbook)
        questions_path = _find_questions(args.questions)
        golden_path = _find_golden_answers(args.golden_answers)
    except FileNotFoundError as exc:
        print(f"[experiment] ERROR: {exc}", file=sys.stderr)
        return 1

    # Resolve output directory
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        PROJECT_ROOT / "artifacts" / "study" / f"study_{ts}"
    )

    print("\n[experiment] ═══════════════════════════════════════════")
    print(f"[experiment] EV LLM Comparative Study")
    print(f"[experiment] Models:          {len(config.models)}")
    print(f"[experiment] Data workbook:   {data_workbook_path.name}")
    print(f"[experiment] Questions:       {questions_path.name}")
    print(f"[experiment] Golden answers:  {golden_path.name if golden_path else 'none (auto-generation)'}")
    print(f"[experiment] Output dir:      {output_dir}")
    print(f"[experiment] Seed:            {args.seed}")
    print(f"[experiment] Skip eval:       {args.skip_evaluation}")
    print(f"[experiment] ragas enabled:   {config.evaluation.ragas_enabled and not args.no_ragas}")
    print("[experiment] ═══════════════════════════════════════════\n")

    result_path = run_experiment(
        config=config,
        data_workbook_path=data_workbook_path,
        questions_path=questions_path,
        golden_path=golden_path,
        output_dir=output_dir,
        seed=args.seed,
        max_questions=args.max_questions,
        skip_evaluation=args.skip_evaluation,
        disable_ragas=args.no_ragas,
    )

    print(f"\n[experiment] ✓ Study complete. Primary output: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
