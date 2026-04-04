from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

from ev_llm_compare.chunking import ExcelChunkBuilder
from ev_llm_compare.derived_analytics import build_derived_summary_chunks
from ev_llm_compare.excel_loader import (
    join_golden_answers,
    load_eval_questions,
    load_golden_answers,
    load_workbook,
)
from ev_llm_compare.ragas_integration import (
    AnswerCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    RAGAS_COLUMN_MAP,
    RunConfig,
    _build_evaluation_dataset,
    _build_ragas_embedder,
    _build_ragas_llm,
    _ragas_evaluate,
)
from ev_llm_compare.retrieval import HybridRetriever
from ev_llm_compare.schemas import ModelResponse
from ev_llm_compare.settings import load_config


def _log(msg: str) -> None:
    print(msg, flush=True)


def _save_progress(df: pd.DataFrame, out_dir: Path, tag: str) -> None:
    progress_xlsx = out_dir / "results_all_gemma27b_ragas_progress.xlsx"
    progress_csv = out_dir / "gemma27b_vs_golden_ragas_progress.csv"
    comparison_cols = [
        "q_id",
        "question",
        "golden_answer",
        "answer",
        "ragas_answer_correctness",
        "ragas_faithfulness",
        "ragas_context_precision",
        "ragas_context_recall",
    ]
    comp = df[[c for c in comparison_cols if c in df.columns]].copy()
    df.to_excel(progress_xlsx, index=False, engine="openpyxl")
    comp.to_csv(progress_csv, index=False)
    nn = {
        c: int(df[c].notna().sum())
        for c in [
            "ragas_faithfulness",
            "ragas_context_precision",
            "ragas_context_recall",
            "ragas_answer_correctness",
        ]
    }
    _log(f"[resume] saved progress after {tag}: {nn}")


def _apply_metric(
    *,
    metric_name: str,
    metric_obj,
    target_col: str,
    target_responses: list[ModelResponse],
    golden_map: dict[str, str],
    config,
    run_cfg: RunConfig,
    df: pd.DataFrame,
    out_dir: Path,
) -> None:
    total = len(target_responses)
    existing = int(df[target_col].notna().sum())
    if existing >= total:
        _log(f"[resume] skip {metric_name}: already complete ({existing}/{total})")
        return

    _log(f"[resume] running {metric_name} ({existing}/{total} already scored)...")
    dataset = _build_evaluation_dataset(
        target_responses,
        golden_map,
        context_result_limit=config.retrieval.evaluation_context_result_limit,
        context_char_budget=config.retrieval.evaluation_context_char_budget,
    )
    t0 = time.time()
    result = _ragas_evaluate(
        dataset=dataset,
        metrics=[metric_obj],
        batch_size=1,
        run_config=run_cfg,
    )
    elapsed = round(time.time() - t0, 1)
    metric_df = result.to_pandas()
    col_candidates = [
        c for c in metric_df.columns if RAGAS_COLUMN_MAP.get(c.lower()) == target_col
    ]
    if not col_candidates:
        _log(
            f"[resume] WARNING {metric_name}: no target column in {list(metric_df.columns)}"
        )
        _save_progress(df, out_dir, f"{metric_name}_no_col")
        return

    src_col = col_candidates[0]
    for idx, response in enumerate(target_responses):
        if idx >= len(metric_df):
            break
        val = metric_df.iloc[idx].get(src_col)
        if val is None:
            continue
        try:
            df.loc[df["question"] == response.question, target_col] = float(val)
        except Exception:
            df.loc[df["question"] == response.question, target_col] = None

    _log(f"[resume] {metric_name} done in {elapsed}s")
    _save_progress(df, out_dir, metric_name)


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    out_dir = project / "newArchEval" / "gemma27b_ragas_20260403_183145"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_results = out_dir / "results_all_20260403_223220.xlsx"
    if not base_results.exists():
        _log(f"[resume] missing base results: {base_results}")
        return 2

    _log("[resume] loading base results...")
    progress_xlsx = out_dir / "results_all_gemma27b_ragas_progress.xlsx"
    if progress_xlsx.exists():
        df = pd.read_excel(progress_xlsx)
        _log("[resume] loaded existing progress file")
    else:
        df = pd.read_excel(base_results)
        df = df[df["run_name"] == "gemma27b_rag"].copy().reset_index(drop=True)
        for c in [
            "ragas_faithfulness",
            "ragas_context_precision",
            "ragas_context_recall",
            "ragas_answer_correctness",
        ]:
            if c not in df.columns:
                df[c] = None
        _log("[resume] initialized from base results")

    config = load_config()

    qa_wb = project / "data" / "Human validated 50 questions.xlsx"
    questions = load_eval_questions(qa_wb)
    golden = load_golden_answers(qa_wb)
    questions, _ = join_golden_answers(questions, golden)
    golden_map = {q.question: (q.golden_answer or "") for q in questions}

    _log("[resume] building retrieval index...")
    rows, notes = load_workbook(
        project / "data" / "GNEM - Auto Landscape Lat Long Updated.xlsx"
    )
    chunks = ExcelChunkBuilder(config.retrieval).build(rows, notes)
    chunks.extend(build_derived_summary_chunks(rows))

    responses: list[ModelResponse] = []
    retriever = HybridRetriever(
        chunks=chunks,
        settings=config.retrieval,
        qdrant_path=config.runtime.qdrant_path,
    )
    try:
        for _, row in df.iterrows():
            q = str(row["question"])
            responses.append(
                ModelResponse(
                    run_name="gemma27b_rag",
                    provider="ollama",
                    model_name="gemma3:27b",
                    rag_enabled=True,
                    question=q,
                    answer=str(row.get("answer") or ""),
                    latency_seconds=float(row.get("latency_seconds") or 0.0),
                    retrieved_chunks=retriever.retrieve(q),
                    prompt_tokens_estimate=int(row.get("tokens_in") or 1),
                    success=bool(row.get("success")),
                    error_message=str(row.get("error_message") or ""),
                )
            )
    finally:
        retriever.close()

    rag_responses = [r for r in responses if r.rag_enabled and r.success]
    scored_responses = [
        r
        for r in responses
        if r.success and str(r.answer).strip() and golden_map.get(r.question, "")
    ]

    _log("[resume] building RAGAS judge...")
    judge_provider = os.getenv("RAGAS_JUDGE_PROVIDER", "ollama")
    judge_model = os.getenv("RAGAS_JUDGE_MODEL", "gemma3:27b")
    ragas_llm = _build_ragas_llm(judge_provider, judge_model)
    ragas_embedder = _build_ragas_embedder(config.evaluation.embedding_model)

    run_cfg = RunConfig(
        timeout=int(os.getenv("RAGAS_TIMEOUT", "600")),
        max_retries=int(
            os.getenv("RAGAS_MAX_RETRIES", str(config.evaluation.max_retries))
        ),
        max_wait=120,
        max_workers=1,
    )

    try:
        _apply_metric(
            metric_name="Faithfulness",
            metric_obj=Faithfulness(llm=ragas_llm),
            target_col="ragas_faithfulness",
            target_responses=rag_responses,
            golden_map=golden_map,
            config=config,
            run_cfg=run_cfg,
            df=df,
            out_dir=out_dir,
        )
    except Exception as exc:
        _log(f"[resume] Faithfulness failed: {exc!r}")
        traceback.print_exc()
        _save_progress(df, out_dir, "faithfulness_failed")

    try:
        _apply_metric(
            metric_name="ContextPrecision",
            metric_obj=LLMContextPrecisionWithReference(llm=ragas_llm),
            target_col="ragas_context_precision",
            target_responses=rag_responses,
            golden_map=golden_map,
            config=config,
            run_cfg=run_cfg,
            df=df,
            out_dir=out_dir,
        )
    except Exception as exc:
        _log(f"[resume] ContextPrecision failed: {exc!r}")
        traceback.print_exc()
        _save_progress(df, out_dir, "context_precision_failed")

    try:
        _apply_metric(
            metric_name="ContextRecall",
            metric_obj=LLMContextRecall(llm=ragas_llm),
            target_col="ragas_context_recall",
            target_responses=rag_responses,
            golden_map=golden_map,
            config=config,
            run_cfg=run_cfg,
            df=df,
            out_dir=out_dir,
        )
    except Exception as exc:
        _log(f"[resume] ContextRecall failed: {exc!r}")
        traceback.print_exc()
        _save_progress(df, out_dir, "context_recall_failed")

    try:
        ac = AnswerCorrectness(llm=ragas_llm)
        if ragas_embedder:
            ac.embeddings = ragas_embedder
        _apply_metric(
            metric_name="AnswerCorrectness",
            metric_obj=ac,
            target_col="ragas_answer_correctness",
            target_responses=scored_responses,
            golden_map=golden_map,
            config=config,
            run_cfg=run_cfg,
            df=df,
            out_dir=out_dir,
        )
    except Exception as exc:
        _log(f"[resume] AnswerCorrectness failed: {exc!r}")
        traceback.print_exc()
        _save_progress(df, out_dir, "answer_correctness_failed")

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    final_results = out_dir / f"results_all_gemma27b_ragas_fixed_{stamp}.xlsx"
    final_comp_xlsx = out_dir / f"gemma27b_vs_golden_ragas_comparison_{stamp}.xlsx"
    final_comp_csv = out_dir / f"gemma27b_vs_golden_ragas_comparison_{stamp}.csv"

    comparison_cols = [
        "q_id",
        "question",
        "golden_answer",
        "answer",
        "ragas_answer_correctness",
        "ragas_faithfulness",
        "ragas_context_precision",
        "ragas_context_recall",
    ]
    comp = df[[c for c in comparison_cols if c in df.columns]].copy()

    df.to_excel(final_results, index=False, engine="openpyxl")
    comp.to_excel(final_comp_xlsx, index=False, engine="openpyxl")
    comp.to_csv(final_comp_csv, index=False)

    _log(f"[resume] FINAL_RESULTS {final_results}")
    _log(f"[resume] FINAL_COMP_XLSX {final_comp_xlsx}")
    _log(f"[resume] FINAL_COMP_CSV {final_comp_csv}")
    for c in [
        "ragas_answer_correctness",
        "ragas_faithfulness",
        "ragas_context_precision",
        "ragas_context_recall",
    ]:
        _log(f"[resume] {c} non_null {int(df[c].notna().sum())}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        traceback.print_exc()
        print(f"[resume] FATAL: {exc!r}", flush=True)
        raise
