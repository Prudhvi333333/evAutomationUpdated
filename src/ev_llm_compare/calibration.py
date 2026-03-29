from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from .evaluation import RAGAS_METRIC_NAMES, run_evaluation_metrics
from .schemas import ModelResponse, RetrievalResult


EXPECTED_COLUMN_BY_METRIC = {
    metric_name: f"expected_{metric_name}" for metric_name in RAGAS_METRIC_NAMES
}


@dataclass(slots=True)
class CalibrationRunResult:
    scored_rows: pd.DataFrame
    summary: pd.DataFrame


def run_judge_calibration(
    calibration_path: str | Path,
    *,
    judge_provider: str,
    judge_model: str,
    max_retries: int = 2,
    sheet_name: str | None = None,
) -> CalibrationRunResult:
    frame = _load_calibration_frame(calibration_path, sheet_name=sheet_name)
    responses, references = _build_calibration_payload(frame)
    metrics_per_run, _ = run_evaluation_metrics(
        responses=responses,
        reference_answers=references,
        judge_provider=judge_provider,
        judge_model=judge_model,
        max_retries=max_retries,
    )

    merged = frame.merge(
        metrics_per_run,
        how="left",
        left_on=["run_name", "question"],
        right_on=["run_name", "question"],
    )
    for metric_name, expected_column in EXPECTED_COLUMN_BY_METRIC.items():
        if expected_column not in merged.columns:
            continue
        merged[expected_column] = pd.to_numeric(merged[expected_column], errors="coerce")
        merged[metric_name] = pd.to_numeric(merged[metric_name], errors="coerce")
        merged[f"{metric_name}_error"] = merged[metric_name] - merged[expected_column]
        merged[f"{metric_name}_abs_error"] = (
            merged[f"{metric_name}_error"].abs()
        )

    summary_rows: list[dict[str, Any]] = []
    for metric_name, expected_column in EXPECTED_COLUMN_BY_METRIC.items():
        if expected_column not in merged.columns:
            continue
        comparable = merged.dropna(subset=[expected_column, metric_name]).copy()
        if comparable.empty:
            continue
        correlation = (
            comparable[[expected_column, metric_name]].corr().iloc[0, 1]
            if len(comparable) > 1
            else None
        )
        summary_rows.append(
            {
                "metric": metric_name,
                "rows_compared": len(comparable),
                "mean_expected": round(float(comparable[expected_column].mean()), 4),
                "mean_judged": round(float(comparable[metric_name].mean()), 4),
                "mae": round(float(comparable[f"{metric_name}_abs_error"].mean()), 4),
                "mean_signed_error": round(float(comparable[f"{metric_name}_error"].mean()), 4),
                "correlation": round(float(correlation), 4) if correlation is not None and not pd.isna(correlation) else None,
            }
        )

    return CalibrationRunResult(
        scored_rows=merged,
        summary=pd.DataFrame(summary_rows),
    )


def export_calibration_workbook(
    output_dir: str | Path,
    result: CalibrationRunResult,
    *,
    filename_prefix: str = "judge_calibration",
) -> Path:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%S")
    workbook_path = output_path / f"{filename_prefix}_{timestamp}.xlsx"

    instructions = pd.DataFrame(
        [
            {
                "required_columns": "question, answer/model_response, reference_answer/reference",
                "optional_columns": ", ".join(
                    [
                        "run_name",
                        "retrieved_contexts",
                        *EXPECTED_COLUMN_BY_METRIC.values(),
                    ]
                ),
                "retrieved_contexts_format": "JSON list or blocks separated by blank lines / ---",
                "note": "Use a small human-labeled set before trusting local-judge scores in production.",
            }
        ]
    )

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        result.scored_rows.to_excel(writer, sheet_name="calibration_rows", index=False)
        result.summary.to_excel(writer, sheet_name="metric_summary", index=False)
        instructions.to_excel(writer, sheet_name="instructions", index=False)
    return workbook_path


def _load_calibration_frame(
    calibration_path: str | Path,
    *,
    sheet_name: str | None = None,
) -> pd.DataFrame:
    path = Path(calibration_path).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, sheet_name=sheet_name or 0)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("Calibration file must be .xlsx, .xls, or .csv")

    frame = frame.copy()
    frame.columns = [_normalize_column_name(column) for column in frame.columns]
    if "question" not in frame.columns:
        raise ValueError("Calibration file must contain a question column.")

    answer_column = _first_present(frame.columns.tolist(), ["answer", "model_response", "response"])
    if answer_column is None:
        raise ValueError("Calibration file must contain an answer/model_response column.")
    if answer_column != "answer":
        frame["answer"] = frame[answer_column]

    reference_column = _first_present(
        frame.columns.tolist(),
        ["reference_answer", "reference", "golden_answer", "expected_answer"],
    )
    if reference_column is None:
        raise ValueError("Calibration file must contain a reference_answer/reference column.")
    if reference_column != "reference_answer":
        frame["reference_answer"] = frame[reference_column]

    if "run_name" not in frame.columns:
        frame["run_name"] = "calibration_local_judge"
    if "retrieved_contexts" not in frame.columns:
        context_column = _first_present(
            frame.columns.tolist(),
            ["context", "combined_context_used", "retrieved_evidence"],
        )
        frame["retrieved_contexts"] = frame[context_column] if context_column else ""

    frame["question"] = frame["question"].fillna("").astype(str).str.strip()
    frame["answer"] = frame["answer"].fillna("").astype(str).str.strip()
    frame["reference_answer"] = frame["reference_answer"].fillna("").astype(str).str.strip()
    frame["retrieved_contexts"] = frame["retrieved_contexts"].fillna("")
    frame = frame[frame["question"].ne("") & frame["answer"].ne("")].copy()
    if frame.empty:
        raise ValueError("Calibration file did not contain any usable rows.")
    return frame


def _build_calibration_payload(
    frame: pd.DataFrame,
) -> tuple[list[ModelResponse], dict[str, str]]:
    responses: list[ModelResponse] = []
    references: dict[str, str] = {}
    for row_index, row in enumerate(frame.to_dict(orient="records"), start=1):
        contexts = _parse_context_blocks(row.get("retrieved_contexts", ""))
        retrieved_chunks = [
            RetrievalResult(
                chunk_id=f"calibration_ctx_{row_index}_{context_index}",
                text=context_text,
                metadata={
                    "chunk_type": "calibration_context",
                    "source_file": "calibration",
                    "sheet_name": "calibration",
                    "rank": context_index,
                },
                dense_score=0.0,
                lexical_score=0.0,
                final_score=1.0 / context_index,
            )
            for context_index, context_text in enumerate(contexts, start=1)
        ]
        question = str(row["question"])
        responses.append(
            ModelResponse(
                run_name=str(row.get("run_name", "calibration_local_judge")),
                provider="calibration",
                model_name="calibration",
                rag_enabled=bool(retrieved_chunks),
                question=question,
                answer=str(row["answer"]),
                latency_seconds=0.0,
                retrieved_chunks=retrieved_chunks,
                prompt_tokens_estimate=0,
                success=True,
            )
        )
        references[question] = str(row.get("reference_answer", ""))
    return responses, references


def _parse_context_blocks(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, list):
        return [str(item).strip() for item in payload if str(item).strip()]

    blocks = re.split(r"\n\s*\n+|\n---+\n", text)
    return [re.sub(r"\s+", " ", block).strip() for block in blocks if block.strip()]


def _normalize_column_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _first_present(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None
