from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
from typing import Any
import warnings

import pandas as pd

from .models import LLMClient, safe_generate
from .prompts import build_reference_prompt, format_context
from .schemas import ModelResponse

RAGAS_METRIC_NAMES = [
    "answer_accuracy",
    "faithfulness",
    "response_groundedness",
]


def build_reference_answers(
    questions: list[str],
    retrievals: dict[str, list[Any]],
    judge_client: LLMClient,
) -> dict[str, str]:
    references: dict[str, str] = {}
    for question in questions:
        context = format_context(retrievals[question])
        prompt = build_reference_prompt(question, context)
        answer, _, success, _ = safe_generate(judge_client, prompt, temperature=0.0, max_tokens=500)
        references[question] = answer if success else "Reference generation failed."
    return references


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*google\\.generativeai.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module=r"instructor\.providers\.gemini\.client",
        )
        warnings.filterwarnings(
            "ignore",
            message=".*HuggingFaceEmbeddings.*deprecated.*",
        )
        warnings.filterwarnings(
            "ignore",
            message="Importing .* from 'ragas\\.metrics' is deprecated.*",
            category=DeprecationWarning,
        )
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import AnswerAccuracy, Faithfulness, ResponseGroundedness
        from ragas.run_config import RunConfig

    evaluator_llm = LangchainLLMWrapper(_make_langchain_llm(judge_provider, judge_model))
    run_config = RunConfig(
        timeout=timeout,
        max_retries=max_retries,
        max_wait=max_wait,
        max_workers=max_workers,
    )

    record_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for response in responses:
        record_lookup[(response.run_name, response.question)] = {
            "run_name": response.run_name,
            "question": response.question,
            **{metric_name: None for metric_name in RAGAS_METRIC_NAMES},
        }
    accuracy_keys: list[tuple[str, str]] = []
    accuracy_rows: list[dict[str, Any]] = []
    rag_keys: list[tuple[str, str]] = []
    rag_rows: list[dict[str, Any]] = []
    for response in responses:
        if not response.success:
            continue
        reference_answer = reference_answers.get(response.question, "")
        if reference_answer:
            accuracy_keys.append((response.run_name, response.question))
            accuracy_rows.append(
                {
                    "user_input": response.question,
                    "response": response.answer,
                    "reference": reference_answer,
                }
            )
        if response.rag_enabled and response.retrieved_chunks:
            rag_keys.append((response.run_name, response.question))
            rag_rows.append(
                {
                    "user_input": response.question,
                    "response": response.answer,
                    "retrieved_contexts": [chunk.text for chunk in response.retrieved_chunks],
                    "reference": reference_answer,
                }
            )

    if accuracy_rows:
        accuracy_results = _evaluate_ragas_batch(
            evaluate=evaluate,
            dataset_builder=EvaluationDataset,
            dataset_rows=accuracy_rows,
            metrics=[AnswerAccuracy(llm=evaluator_llm, name="answer_accuracy")],
            llm=evaluator_llm,
            embeddings=None,
            run_config=run_config,
            batch_size=min(len(accuracy_rows), max(1, max_workers * 4)),
        )
        _merge_metric_rows(record_lookup, accuracy_keys, accuracy_results)

    if rag_rows:
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            _make_langchain_embeddings(embedding_provider, embedding_model)
        )
        rag_metric_results = _evaluate_ragas_batch(
            evaluate=evaluate,
            dataset_builder=EvaluationDataset,
            dataset_rows=rag_rows,
            metrics=[
                Faithfulness(llm=evaluator_llm, name="faithfulness"),
                ResponseGroundedness(llm=evaluator_llm, name="response_groundedness"),
            ],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=run_config,
            batch_size=min(len(rag_rows), max(1, max_workers * 4)),
        )
        _merge_metric_rows(record_lookup, rag_keys, rag_metric_results)

    per_run_records = [
        record_lookup[(response.run_name, response.question)]
        for response in responses
    ]

    per_run_df = pd.DataFrame(
        per_run_records,
        columns=["run_name", "question", *RAGAS_METRIC_NAMES],
    )
    for metric_name in RAGAS_METRIC_NAMES:
        per_run_df[metric_name] = pd.to_numeric(per_run_df[metric_name], errors="coerce")
    summary_df = (
        per_run_df.groupby("run_name", dropna=False)
        .agg({metric_name: "mean" for metric_name in RAGAS_METRIC_NAMES})
        .reset_index()
        .sort_values(by=RAGAS_METRIC_NAMES[0], ascending=False, na_position="last")
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
                ragas_per_run.to_excel(writer, sheet_name="ragas_per_question", index=False)
            if ragas_summary is not None:
                ragas_summary.to_excel(writer, sheet_name="ragas_summary", index=False)

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
            ragas_per_run.to_excel(writer, sheet_name="ragas_per_question", index=False)
        if ragas_summary is not None:
            ragas_summary.to_excel(writer, sheet_name="ragas_summary", index=False)
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
                ragas_per_run.to_excel(writer, sheet_name="ragas_per_question", index=False)
            if ragas_summary is not None:
                ragas_summary.to_excel(writer, sheet_name="ragas_summary", index=False)

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


def _make_langchain_llm(provider: str, model: str):
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=0.0)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            temperature=0.0,
            base_url=os.getenv("OLLAMA_BASE_URL"),
        )

    raise ValueError(f"Unsupported RAGAS judge provider: {provider}")


def _make_langchain_embeddings(provider: str, model: str):
    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=model)

    if provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model)

    raise ValueError(f"Unsupported RAGAS embedding provider: {provider}")


def _result_to_rows(result: Any) -> list[dict[str, Any]]:
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        return frame.to_dict(orient="records")
    if hasattr(result, "scores"):
        scores = getattr(result, "scores")
        if isinstance(scores, list):
            return [dict(item) for item in scores]
        if isinstance(scores, dict):
            return [dict(scores)]
    if isinstance(result, dict):
        return [result]
    return []


def _evaluate_ragas_batch(
    evaluate: Any,
    dataset_builder: Any,
    dataset_rows: list[dict[str, Any]],
    metrics: list[Any],
    llm: Any,
    embeddings: Any,
    run_config: Any,
    batch_size: int,
) -> list[dict[str, Any]]:
    try:
        result = evaluate(
            dataset=dataset_builder.from_list(dataset_rows),
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
            batch_size=max(1, batch_size),
            raise_exceptions=False,
            show_progress=False,
            allow_nest_asyncio=False,
        )
        return _result_to_rows(result)
    except Exception as exc:
        print(f"[ev-llm-compare] RAGAS batch evaluation failed: {exc}", flush=True)
        return []


def _merge_metric_rows(
    record_lookup: dict[tuple[str, str], dict[str, Any]],
    keys: list[tuple[str, str]],
    metric_rows: list[dict[str, Any]],
) -> None:
    for index, key in enumerate(keys):
        if index >= len(metric_rows):
            continue
        metrics = metric_rows[index]
        record = record_lookup[key]
        for metric_name in RAGAS_METRIC_NAMES:
            if metric_name in metrics:
                record[metric_name] = metrics.get(metric_name)


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
    return {metric_name: metrics.get(metric_name) for metric_name in RAGAS_METRIC_NAMES}


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

    metric_names: list[str] = list(RAGAS_METRIC_NAMES)
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

    metric_names: list[str] = list(RAGAS_METRIC_NAMES)
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
