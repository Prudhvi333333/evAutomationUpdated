from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
import re
from typing import Any

import pandas as pd

from .models import LLMClient, create_client, safe_generate
from .prompts import build_reference_prompt, compact_context_segments, format_context
from .schemas import ModelResponse
from .settings import ModelSpec, RuntimeSettings

REPORT_METRIC_NAMES = [
    "answer_accuracy",
    "faithfulness",
    "response_groundedness",
    "grounded_claim_ratio",
    "unsupported_claim_ratio",
    "contradicted_claim_ratio",
]

LLM_JUDGE_SYSTEM_PROMPT = """You are a strict evaluation judge for workbook question answering.
Return only one line in the form SCORE=<value>.
Use a continuous score from 0.00 to 1.00 with exactly two decimal places.
Do not add explanation, JSON, markdown, or extra text."""
LLM_JUDGE_PACKET_SYSTEM_PROMPT = """You are a strict evaluation judge for workbook question answering.
Return only the requested metric lines with no explanation, JSON, markdown, or extra text.
Use continuous scores from 0.00 to 1.00 with exactly two decimal places."""


def build_reference_answers(
    questions: list[str],
    retrievals: dict[str, list[Any]],
    judge_client: LLMClient,
    context_result_limit: int = 5,
    context_char_budget: int = 4200,
    compact_context: bool = True,
) -> dict[str, str]:
    references: dict[str, str] = {}
    for question in questions:
        context = format_context(
            retrievals[question],
            question=question,
            max_results=context_result_limit,
            max_chars=context_char_budget,
            compact=compact_context,
        )
        prompt = build_reference_prompt(question, context)
        answer, _, success, _ = safe_generate(judge_client, prompt, temperature=0.0, max_tokens=500)
        references[question] = answer if success else "Reference generation failed."
    return references


def _make_judge_client(provider: str, model: str) -> LLMClient:
    runtime = RuntimeSettings()
    runtime.ollama_base_url = os.getenv("OLLAMA_BASE_URL", runtime.ollama_base_url)
    spec = ModelSpec(
        run_name="llm_judge",
        provider=provider,
        model_name=model,
        rag_enabled=False,
    )
    return create_client(spec, runtime)


def _llm_judge_prompt_answer_accuracy(
    question: str,
    answer: str,
    reference_answer: str,
) -> str:
    return (
        "Task: score how well the candidate answer matches the reference answer for a workbook question.\n"
        "Scoring rubric:\n"
        "- 1.00 means fully correct, all major requested facts present, no material errors.\n"
        "- Use the full continuous range 0.00 to 1.00; do not restrict yourself to quarter-step buckets.\n"
        "- Prefer precise decimals like 0.13, 0.42, 0.68, or 0.91 when appropriate.\n"
        "- For list, grouped, and count questions, score by factual coverage and precision.\n"
        "- Missing requested items should reduce the score proportionally.\n"
        "- Wrong extra entities or wrong numbers should reduce the score.\n"
        "- If the answer explicitly says it lacks the workbook/dataset, gives a general method, or gives generic industry examples instead of the requested workbook answer, score 0.00.\n"
        "- If the answer is largely a non-answer or refusal, score 0.00.\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference_answer}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        "Return exactly one line: SCORE=<0.00-1.00>"
    )


def _llm_judge_prompt_grounding_packet(
    question: str,
    answer: str,
    retrieved_contexts: list[str],
) -> str:
    context = "\n\n".join(retrieved_contexts)
    return (
        "Task: evaluate the candidate answer against the retrieved workbook evidence.\n"
        "Use continuous scores from 0.00 to 1.00 with exactly two decimal places.\n"
        "Definitions:\n"
        "- FAITHFULNESS: how free the answer is from claims contradicted by evidence.\n"
        "- RESPONSE_GROUNDEDNESS: how much of the answer is grounded in evidence.\n"
        "- GROUNDED_CLAIM_RATIO: fraction of substantive claims supported by evidence.\n"
        "- UNSUPPORTED_CLAIM_RATIO: fraction of substantive claims not supported by evidence.\n"
        "- CONTRADICTED_CLAIM_RATIO: fraction of substantive claims contradicted by evidence.\n"
        "Consistency rules:\n"
        "- Prefer precise decimals like 0.18, 0.37, 0.64, or 0.92 when appropriate.\n"
        "- RESPONSE_GROUNDEDNESS should usually match GROUNDED_CLAIM_RATIO.\n"
        "- Higher CONTRADICTED_CLAIM_RATIO should lower FAITHFULNESS.\n"
        "- GROUNDED_CLAIM_RATIO, UNSUPPORTED_CLAIM_RATIO, and CONTRADICTED_CLAIM_RATIO should approximately sum to 1.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved evidence:\n{context}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        "Return exactly these five lines:\n"
        "FAITHFULNESS=<0.00-1.00>\n"
        "RESPONSE_GROUNDEDNESS=<0.00-1.00>\n"
        "GROUNDED_CLAIM_RATIO=<0.00-1.00>\n"
        "UNSUPPORTED_CLAIM_RATIO=<0.00-1.00>\n"
        "CONTRADICTED_CLAIM_RATIO=<0.00-1.00>"
    )


def _parse_llm_judge_score(raw_text: str) -> float | None:
    if not raw_text:
        return None
    match = re.search(r"(?<![\d.])(0(?:\.\d+)?|1(?:\.0+)?)\b", raw_text)
    if not match:
        return None
    try:
        score = float(match.group(1))
    except ValueError:
        return None
    if not (0.0 <= score <= 1.0):
        return None
    return round(score, 4)


def _parse_llm_judge_packet(raw_text: str) -> dict[str, float] | None:
    if not raw_text:
        return None
    metric_map = {
        "FAITHFULNESS": "faithfulness",
        "RESPONSE_GROUNDEDNESS": "response_groundedness",
        "GROUNDED_CLAIM_RATIO": "grounded_claim_ratio",
        "UNSUPPORTED_CLAIM_RATIO": "unsupported_claim_ratio",
        "CONTRADICTED_CLAIM_RATIO": "contradicted_claim_ratio",
    }
    parsed: dict[str, float] = {}
    for label, metric_name in metric_map.items():
        match = re.search(
            rf"{label}\s*=\s*(0(?:\.\d+)?|1(?:\.0+)?)\b",
            raw_text,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        score = float(match.group(1))
        if not (0.0 <= score <= 1.0):
            return None
        parsed[metric_name] = round(score, 4)
    return parsed


def _llm_judge_metric(
    judge_client: LLMClient,
    prompt: str,
    retries: int = 2,
) -> float | None:
    for _ in range(max(1, retries + 1)):
        answer, _, success, _ = safe_generate(
            judge_client,
            prompt,
            temperature=0.0,
            max_tokens=12,
            system_prompt=LLM_JUDGE_SYSTEM_PROMPT,
        )
        if not success:
            continue
        parsed = _parse_llm_judge_score(answer)
        if parsed is not None:
            return parsed
    return None


def _llm_judge_grounding_packet(
    judge_client: LLMClient,
    prompt: str,
    retries: int = 2,
) -> dict[str, float] | None:
    for _ in range(max(1, retries + 1)):
        answer, _, success, _ = safe_generate(
            judge_client,
            prompt,
            temperature=0.0,
            max_tokens=80,
            system_prompt=LLM_JUDGE_PACKET_SYSTEM_PROMPT,
        )
        if not success:
            continue
        parsed = _parse_llm_judge_packet(answer)
        if parsed is not None:
            return parsed
    return None


def run_ragas(
    responses: list[ModelResponse],
    reference_answers: dict[str, str],
    judge_provider: str,
    judge_model: str,
    embedding_provider: str,
    embedding_model: str,
    timeout: int = 600,
    max_retries: int = 2,
    max_wait: int = 60,
    max_workers: int = 1,
    context_result_limit: int = 4,
    context_char_budget: int = 2600,
    compact_context: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    judge_client = _make_judge_client(judge_provider, judge_model)

    record_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for response in responses:
        base_record = {
            "run_name": response.run_name,
            "question": response.question,
            **{metric_name: None for metric_name in REPORT_METRIC_NAMES},
        }
        reference_answer = reference_answers.get(response.question, "")
        if response.success:
            if reference_answer:
                base_record["answer_accuracy"] = _llm_judge_metric(
                    judge_client,
                    _llm_judge_prompt_answer_accuracy(
                        response.question,
                        response.answer,
                        reference_answer,
                    ),
                )
        if response.success and response.rag_enabled and response.retrieved_chunks:
            retrieved_contexts = (
                compact_context_segments(
                    response.question,
                    response.retrieved_chunks,
                    max_results=context_result_limit,
                    max_chars=context_char_budget,
                )
                if compact_context
                else [chunk.text for chunk in response.retrieved_chunks]
            )
            grounding_packet = _llm_judge_grounding_packet(
                judge_client,
                _llm_judge_prompt_grounding_packet(
                    response.question,
                    response.answer,
                    retrieved_contexts,
                ),
            )
            if grounding_packet is not None:
                base_record.update(grounding_packet)
        record_lookup[(response.run_name, response.question)] = {
            **base_record,
        }

    per_run_records = [
        record_lookup[(response.run_name, response.question)]
        for response in responses
    ]

    per_run_df = pd.DataFrame(
        per_run_records,
        columns=["run_name", "question", *REPORT_METRIC_NAMES],
    )
    for metric_name in REPORT_METRIC_NAMES:
        per_run_df[metric_name] = pd.to_numeric(per_run_df[metric_name], errors="coerce")
    summary_df = (
        per_run_df.groupby("run_name", dropna=False)
        .agg({metric_name: "mean" for metric_name in REPORT_METRIC_NAMES})
        .reset_index()
        .sort_values(by=REPORT_METRIC_NAMES[0], ascending=False, na_position="last")
    )
    return per_run_df, summary_df


def export_results(
    output_dir: Path,
    responses: list[ModelResponse],
    retrievals: dict[str, list[Any]],
    references: dict[str, str],
    reference_sources: dict[str, str],
    ragas_per_run: pd.DataFrame | None,
    ragas_summary: pd.DataFrame | None,
    filename_prefix: str = "comparison_report",
    single_sheet_only: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    workbook_path = output_dir / f"{filename_prefix}_{timestamp}.xlsx"

    response_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    for response in responses:
        response_rows.append(
            {
                "run_name": response.run_name,
                "provider": response.provider,
                "model_name": response.model_name,
                "rag_enabled": response.rag_enabled,
                "question": response.question,
                "reference_answer": references.get(response.question, ""),
                "reference_source": reference_sources.get(response.question, ""),
                "answer": response.answer,
                "latency_seconds": response.latency_seconds,
                "prompt_tokens_estimate": response.prompt_tokens_estimate,
                "success": response.success,
                "error_message": response.error_message,
            }
        )

    for question, chunks in retrievals.items():
        for rank, chunk in enumerate(chunks, start=1):
            retrieval_rows.append(
                {
                    "question": question,
                    "rank": rank,
                    "company": chunk.metadata.get("company"),
                    "sheet_name": chunk.metadata.get("sheet_name"),
                    "chunk_type": chunk.metadata.get("chunk_type"),
                    "final_score": chunk.final_score,
                    "dense_score": chunk.dense_score,
                    "lexical_score": chunk.lexical_score,
                    "text": chunk.text,
                }
            )

    comparison_df = _build_comparison_sheet(response_rows, ragas_per_run)
    single_sheet_df = _build_single_sheet_report(response_rows, ragas_per_run)

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        single_sheet_df.to_excel(writer, sheet_name="all_in_one", index=False)
        if not single_sheet_only:
            comparison_df.to_excel(writer, sheet_name="responses", index=False)
            pd.DataFrame(response_rows).to_excel(writer, sheet_name="responses_raw", index=False)
            pd.DataFrame(retrieval_rows).to_excel(writer, sheet_name="retrieval", index=False)
            pd.DataFrame(
                [
                    {
                        "question": question,
                        "reference_answer": answer,
                        "reference_source": reference_sources.get(question, ""),
                    }
                    for question, answer in references.items()
                ]
            ).to_excel(writer, sheet_name="references", index=False)
            if ragas_per_run is not None:
                ragas_per_run.to_excel(writer, sheet_name="metrics_per_question", index=False)
            if ragas_summary is not None:
                ragas_summary.to_excel(writer, sheet_name="metrics_summary", index=False)

    return workbook_path


def export_metrics_workbook(
    output_dir: Path,
    ragas_per_run: pd.DataFrame | None,
    ragas_summary: pd.DataFrame | None,
    filename_prefix: str = "metrics_report",
) -> Path | None:
    if ragas_per_run is None and ragas_summary is None:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    workbook_path = output_dir / f"{filename_prefix}_{timestamp}.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        if ragas_per_run is not None:
            ragas_per_run.to_excel(writer, sheet_name="metrics_per_question", index=False)
        if ragas_summary is not None:
            ragas_summary.to_excel(writer, sheet_name="metrics_summary", index=False)
    return workbook_path


def export_response_sets(
    output_dir: Path,
    responses: list[ModelResponse],
    references: dict[str, str],
    reference_sources: dict[str, str],
    ragas_per_run: pd.DataFrame | None = None,
    ragas_summary: pd.DataFrame | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"response_set_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metric_lookup = _metric_lookup(ragas_per_run)
    all_rows: list[dict[str, Any]] = []
    grouped: defaultdict[str, list[ModelResponse]] = defaultdict(list)
    for response in responses:
        grouped[response.run_name].append(response)
        all_rows.append(
            {
                "run_name": response.run_name,
                "provider": response.provider,
                "model_name": response.model_name,
                "rag_enabled": response.rag_enabled,
                "question": response.question,
                "reference_answer": references.get(response.question, ""),
                "reference_source": reference_sources.get(response.question, ""),
                "answer": response.answer,
                "latency_seconds": response.latency_seconds,
                "success": response.success,
                "error_message": response.error_message,
                **_response_metrics(response, metric_lookup),
            }
        )

    pd.DataFrame(all_rows).to_csv(run_dir / "all_runs_responses.csv", index=False)
    single_sheet_df = _build_single_sheet_report(all_rows, ragas_per_run)
    with pd.ExcelWriter(run_dir / "all_runs_single_sheet.xlsx", engine="openpyxl") as writer:
        single_sheet_df.to_excel(writer, sheet_name="all_in_one", index=False)
    if ragas_per_run is not None or ragas_summary is not None:
        with pd.ExcelWriter(run_dir / "all_runs_metrics.xlsx", engine="openpyxl") as writer:
            if ragas_per_run is not None:
                ragas_per_run.to_excel(writer, sheet_name="metrics_per_question", index=False)
            if ragas_summary is not None:
                ragas_summary.to_excel(writer, sheet_name="metrics_summary", index=False)

    for run_name, run_responses in grouped.items():
        rows = [
            {
                "question": response.question,
                "reference_answer": references.get(response.question, ""),
                "reference_source": reference_sources.get(response.question, ""),
                "answer": response.answer,
                "latency_seconds": response.latency_seconds,
                "success": response.success,
                "error_message": response.error_message,
                **_response_metrics(response, metric_lookup),
            }
            for response in run_responses
        ]
        pd.DataFrame(rows).to_csv(run_dir / f"{run_name}_responses.csv", index=False)

        markdown_lines = [f"# {run_name}", ""]
        for index, response in enumerate(run_responses, start=1):
            markdown_lines.extend(
                [
                    f"## Question {index}",
                    f"Question: {response.question}",
                    "",
                    "Answer:",
                    response.answer or "(empty)",
                    "",
                ]
            )
        (run_dir / f"{run_name}_responses.md").write_text(
            "\n".join(markdown_lines),
            encoding="utf-8",
        )

    return run_dir


def _metric_lookup(
    ragas_per_run: pd.DataFrame | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    if ragas_per_run is None or ragas_per_run.empty:
        return {}
    return {
        (row["question"], row["run_name"]): row
        for row in ragas_per_run.to_dict(orient="records")
    }


def _response_metrics(
    response: ModelResponse,
    metric_lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    metrics = metric_lookup.get((response.question, response.run_name), {})
    return {metric_name: metrics.get(metric_name) for metric_name in REPORT_METRIC_NAMES}


def _build_comparison_sheet(
    response_rows: list[dict[str, Any]],
    ragas_per_run: pd.DataFrame | None,
) -> pd.DataFrame:
    question_order = list(dict.fromkeys(row["question"] for row in response_rows))
    run_order = list(dict.fromkeys(row["run_name"] for row in response_rows))

    response_lookup = {
        (row["question"], row["run_name"]): row
        for row in response_rows
    }

    metric_names: list[str] = list(REPORT_METRIC_NAMES)
    metric_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if ragas_per_run is not None and not ragas_per_run.empty:
        metric_lookup = _metric_lookup(ragas_per_run)

    comparison_rows: list[dict[str, Any]] = []
    for question in question_order:
        first_response = next(
            (response_lookup[(question, run_name)] for run_name in run_order if (question, run_name) in response_lookup),
            {},
        )
        row: dict[str, Any] = {
            "Question": question,
            "reference_answer": first_response.get("reference_answer", ""),
            "reference_source": first_response.get("reference_source", ""),
        }

        for run_name in run_order:
            response = response_lookup.get((question, run_name), {})
            row[run_name] = response.get("answer", "")

        for run_name in run_order:
            metrics = metric_lookup.get((question, run_name), {})
            for metric_name in metric_names:
                row[f"{run_name}_{metric_name}"] = metrics.get(metric_name)

        for run_name in run_order:
            response = response_lookup.get((question, run_name), {})
            row[f"{run_name}_latency_seconds"] = response.get("latency_seconds")

        for run_name in run_order:
            response = response_lookup.get((question, run_name), {})
            row[f"{run_name}_prompt_tokens_estimate"] = response.get("prompt_tokens_estimate")

        comparison_rows.append(row)

    return pd.DataFrame(comparison_rows)


def _build_single_sheet_report(
    response_rows: list[dict[str, Any]],
    ragas_per_run: pd.DataFrame | None,
) -> pd.DataFrame:
    question_order = list(dict.fromkeys(row["question"] for row in response_rows))
    run_order = list(dict.fromkeys(row["run_name"] for row in response_rows))

    response_lookup = {
        (row["question"], row["run_name"]): row
        for row in response_rows
    }

    metric_names: list[str] = list(REPORT_METRIC_NAMES)
    metric_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if ragas_per_run is not None and not ragas_per_run.empty:
        metric_lookup = _metric_lookup(ragas_per_run)

    rows: list[dict[str, Any]] = []
    for question in question_order:
        first_response = next(
            (response_lookup[(question, run_name)] for run_name in run_order if (question, run_name) in response_lookup),
            {},
        )
        row: dict[str, Any] = {
            "Question": question,
            "reference_answer": first_response.get("reference_answer", ""),
            "reference_source": first_response.get("reference_source", ""),
        }
        for run_name in run_order:
            response = response_lookup.get((question, run_name), {})
            row[run_name] = response.get("answer", "")
            metrics = metric_lookup.get((question, run_name), {})
            for metric_name in metric_names:
                row[f"{run_name}_{metric_name}"] = metrics.get(metric_name)
        rows.append(row)

    ordered_columns = ["Question", "reference_answer", "reference_source"]
    for run_name in run_order:
        ordered_columns.append(run_name)
        for metric_name in metric_names:
            ordered_columns.append(f"{run_name}_{metric_name}")
    return pd.DataFrame(rows, columns=ordered_columns)
