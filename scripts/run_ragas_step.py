from __future__ import annotations

import os
import signal
import time
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
    AnswerAccuracy,
    AnswerCorrectness,
    AnswerSimilarity,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    NonLLMContextPrecisionWithReference,
    NonLLMContextRecall,
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

FAIL_SENTINEL = -1.0
ALL_RAGAS_COLS = [
    "ragas_answer_accuracy",
    "ragas_answer_similarity",
    "ragas_answer_correctness",
    "ragas_faithfulness",
    "ragas_context_precision",
    "ragas_context_recall",
]
CONTEXT_METRICS = {
    "ragas_faithfulness",
    "ragas_context_precision",
    "ragas_context_recall",
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _metric_order(profile: str) -> list[str]:
    if profile in {"lite", "similarity_grounded"}:
        return [
            "ragas_answer_similarity",
            "ragas_context_precision",
            "ragas_context_recall",
        ]
    if profile in {"asg", "accuracy_similarity_grounded"}:
        return [
            "ragas_answer_accuracy",
            "ragas_answer_similarity",
            "ragas_context_precision",
            "ragas_context_recall",
        ]
    return [
        "ragas_faithfulness",
        "ragas_context_precision",
        "ragas_context_recall",
        "ragas_answer_correctness",
    ]


def _metric_profile() -> str:
    return os.getenv("RAGAS_METRIC_PROFILE", "legacy").strip().lower()


def _save(df: pd.DataFrame, out_dir: Path) -> None:
    progress_xlsx = out_dir / "results_all_gemma27b_ragas_progress.xlsx"
    progress_csv = out_dir / "gemma27b_vs_golden_ragas_progress.csv"
    comparison_cols = [
        "q_id",
        "question",
        "golden_answer",
        "answer",
        "ragas_answer_accuracy",
        "ragas_answer_similarity",
        "ragas_answer_correctness",
        "ragas_faithfulness",
        "ragas_context_precision",
        "ragas_context_recall",
    ]
    comp = df[[c for c in comparison_cols if c in df.columns]].copy()
    df.to_excel(progress_xlsx, index=False, engine="openpyxl")
    comp.to_csv(progress_csv, index=False)


def _load_df(out_dir: Path) -> pd.DataFrame:
    progress_xlsx = out_dir / "results_all_gemma27b_ragas_progress.xlsx"

    if progress_xlsx.exists():
        df = pd.read_excel(progress_xlsx)
        _log("[step] loaded progress file")
    else:
        # Auto-detect the base results file — find the most recent results_all_*.xlsx
        candidates = sorted(out_dir.glob("results_all_*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
        # Skip the progress file itself if somehow included
        candidates = [p for p in candidates if "ragas_progress" not in p.name]
        if not candidates:
            raise FileNotFoundError(
                f"No results_all_*.xlsx found in {out_dir}. "
                "Run main.py first to generate results."
            )
        base_results = candidates[0]
        _log(f"[step] using base results: {base_results.name}")
        df = pd.read_excel(base_results)
        if "run_name" in df.columns:
            rag_runs = df[df["run_name"].str.contains("rag", case=False, na=False) &
                         ~df["run_name"].str.contains("no_rag", case=False, na=False)]
            if not rag_runs.empty:
                run = rag_runs["run_name"].iloc[0]
                df = df[df["run_name"] == run].copy().reset_index(drop=True)
                _log(f"[step] filtered to run: {run}")
        _log("[step] initialized from base results")

    for c in ALL_RAGAS_COLS:
        if c not in df.columns:
            df[c] = None
    return df


def _build_responses(df: pd.DataFrame, config) -> list[ModelResponse]:
    project = Path(__file__).resolve().parents[1]
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
            latency_raw = row.get("latency_seconds")
            latency = (
                float(latency_raw)
                if latency_raw is not None and not pd.isna(latency_raw)
                else 0.0
            )
            tokens_raw = row.get("tokens_in")
            tokens = (
                int(float(tokens_raw))
                if tokens_raw is not None and not pd.isna(tokens_raw)
                else 1
            )
            responses.append(
                ModelResponse(
                    run_name="gemma27b_rag",
                    provider="ollama",
                    model_name="gemma3:27b",
                    rag_enabled=True,
                    question=q,
                    answer=str(row.get("answer") or ""),
                    latency_seconds=latency,
                    retrieved_chunks=retriever.retrieve(q),
                    prompt_tokens_estimate=tokens,
                    success=bool(row.get("success")),
                    error_message=str(row.get("error_message") or ""),
                )
            )
    finally:
        retriever.close()
    return responses


def _pick_next(
    df: pd.DataFrame,
    responses: list[ModelResponse],
    golden_map: dict[str, str],
    metric_order: list[str],
) -> tuple[str, int] | None:
    by_q = {r.question: r for r in responses}
    needs_golden = {
        "ragas_answer_correctness",
        "ragas_answer_similarity",
        "ragas_answer_accuracy",
    }
    for metric in metric_order:
        for idx, row in df.iterrows():
            q = str(row["question"])
            resp = by_q.get(q)
            if resp is None or not resp.success:
                continue
            if metric in CONTEXT_METRICS and not resp.rag_enabled:
                continue
            if metric in needs_golden and not golden_map.get(q, ""):
                continue
            val = row.get(metric)
            if pd.isna(val):
                return metric, idx
    return None


def _metric_obj(metric: str, llm, embedder):
    context_mode = os.getenv("RAGAS_CONTEXT_MODE", "").strip().lower()
    if not context_mode:
        context_mode = (
            "nonllm" if os.getenv("RAGAS_CONTEXT_NONLLM", "0").strip() == "1" else "llm"
        )
    if metric == "ragas_answer_accuracy":
        return AnswerAccuracy(llm=llm)
    if metric == "ragas_answer_similarity":
        sim = AnswerSimilarity()
        if embedder is not None:
            sim.embeddings = embedder
        return sim
    if metric == "ragas_faithfulness":
        return Faithfulness(llm=llm)
    if metric == "ragas_context_precision":
        if context_mode == "nonllm":
            return NonLLMContextPrecisionWithReference()
        if context_mode == "simple":
            return ContextPrecision(llm=llm)
        return LLMContextPrecisionWithReference(llm=llm)
    if metric == "ragas_context_recall":
        if context_mode == "nonllm":
            return NonLLMContextRecall()
        if context_mode == "simple":
            return ContextRecall(llm=llm)
        return LLMContextRecall(llm=llm)
    if metric == "ragas_answer_correctness":
        ac = AnswerCorrectness(llm=llm)
        if embedder is not None:
            ac.embeddings = embedder
        return ac
    raise ValueError(metric)


def _evaluate_with_wall_timeout(dataset, metric_obj, run_cfg: RunConfig, wall_timeout_s: int):
    if wall_timeout_s <= 0:
        return _ragas_evaluate(
            dataset=dataset,
            metrics=[metric_obj],
            batch_size=1,
            run_config=run_cfg,
        )

    def _timeout_handler(_signum, _frame):
        raise TimeoutError(f"wall timeout exceeded ({wall_timeout_s}s)")

    prev_handler = signal.getsignal(signal.SIGALRM)
    prev_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, float(wall_timeout_s))
    try:
        return _ragas_evaluate(
            dataset=dataset,
            metrics=[metric_obj],
            batch_size=1,
            run_config=run_cfg,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev_handler)
        if prev_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, prev_timer[0], prev_timer[1])


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    out_dir = Path(
        os.getenv(
            "RAGAS_OUT_DIR",
            str(project / "newArchEval" / "gemma27b_ragas_20260403_183145"),
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    df = _load_df(out_dir)

    qa_wb = project / "data" / "Human validated 50 questions.xlsx"
    questions = load_eval_questions(qa_wb)
    golden = load_golden_answers(qa_wb)
    questions, _ = join_golden_answers(questions, golden)
    golden_map = {q.question: (q.golden_answer or "") for q in questions}

    _log("[step] building responses/retrieval context...")
    responses = _build_responses(df, config)
    by_q = {r.question: r for r in responses}

    llm = _build_ragas_llm(
        os.getenv("RAGAS_JUDGE_PROVIDER", "ollama"),
        os.getenv("RAGAS_JUDGE_MODEL", "gemma3:27b"),
    )
    embedder = _build_ragas_embedder(config.evaluation.embedding_model)
    run_cfg = RunConfig(
        timeout=int(os.getenv("RAGAS_TIMEOUT", "180")),
        max_retries=int(
            os.getenv("RAGAS_MAX_RETRIES", str(config.evaluation.max_retries))
        ),
        max_wait=120,
        max_workers=1,
    )
    wall_timeout_s = int(
        os.getenv("RAGAS_WALL_TIMEOUT", str(max(run_cfg.timeout + 60, 240)))
    )
    metric_profile = _metric_profile()
    metric_order = _metric_order(metric_profile)
    _log(f"[step] metric_profile={metric_profile} metric_order={metric_order}")

    max_steps = int(os.getenv("RAGAS_STEP_MAX_STEPS", "1"))
    completed = 0
    while completed < max_steps:
        nxt = _pick_next(df, responses, golden_map, metric_order)
        if nxt is None:
            _log("[step] all metrics already processed.")
            break
        metric, idx = nxt
        row = df.iloc[idx]
        question = str(row["question"])
        response = by_q[question]
        _log(f"[step] next metric={metric} q_index={idx} q={question[:90]}")

        ds = _build_evaluation_dataset(
            [response],
            golden_map,
            context_result_limit=config.retrieval.evaluation_context_result_limit,
            context_char_budget=config.retrieval.evaluation_context_char_budget,
        )
        metric_obj = _metric_obj(metric, llm, embedder)
        t0 = time.time()
        score_val = FAIL_SENTINEL
        try:
            result = _evaluate_with_wall_timeout(
                dataset=ds,
                metric_obj=metric_obj,
                run_cfg=run_cfg,
                wall_timeout_s=wall_timeout_s,
            )
            elapsed = round(time.time() - t0, 1)
            mdf = result.to_pandas()
            src_cols = [c for c in mdf.columns if RAGAS_COLUMN_MAP.get(c.lower()) == metric]
            if src_cols:
                raw = mdf.iloc[0].get(src_cols[0])
                if raw is not None and not pd.isna(raw):
                    score_val = float(raw)
            _log(f"[step] metric done in {elapsed}s score={score_val}")
        except Exception as exc:
            elapsed = round(time.time() - t0, 1)
            _log(
                f"[step] metric exception in {elapsed}s: {exc!r}; marking {FAIL_SENTINEL}"
            )

        df.loc[idx, metric] = score_val
        _save(df, out_dir)
        completed += 1
        nn = {c: int(df[c].notna().sum()) for c in metric_order}
        _log(f"[step] progress non_null={nn} completed_steps={completed}/{max_steps}")

    remaining = _pick_next(df, responses, golden_map, metric_order)
    if remaining is None:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        final_results = out_dir / f"results_all_gemma27b_ragas_fixed_{stamp}.xlsx"
        final_comp_xlsx = out_dir / f"gemma27b_vs_golden_ragas_comparison_{stamp}.xlsx"
        final_comp_csv = out_dir / f"gemma27b_vs_golden_ragas_comparison_{stamp}.csv"
        comparison_cols = [
            "q_id",
            "question",
            "golden_answer",
            "answer",
            "ragas_answer_accuracy",
            "ragas_answer_similarity",
            "ragas_answer_correctness",
            "ragas_faithfulness",
            "ragas_context_precision",
            "ragas_context_recall",
        ]
        comp = df[[c for c in comparison_cols if c in df.columns]].copy()
        df.to_excel(final_results, index=False, engine="openpyxl")
        comp.to_excel(final_comp_xlsx, index=False, engine="openpyxl")
        comp.to_csv(final_comp_csv, index=False)
        _log(f"[step] FINAL_RESULTS {final_results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
