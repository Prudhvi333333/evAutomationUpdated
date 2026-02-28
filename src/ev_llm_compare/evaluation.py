from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .models import LLMClient, safe_generate
from .prompts import build_reference_prompt, format_context
from .schemas import ModelResponse


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    metric_names: list[str]
    try:
        from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextRecall, FactualCorrectness

        metrics = [
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextRecall(),
            FactualCorrectness(),
        ]
        metric_names = [
            "faithfulness",
            "answer_relevancy",
            "context_recall",
            "factual_correctness",
        ]
    except ImportError:
        from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

        metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
        metric_names = [
            "faithfulness",
            "answer_relevancy",
            "context_recall",
            "context_precision",
        ]

    evaluator_llm = LangchainLLMWrapper(_make_langchain_llm(judge_provider, judge_model))
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        _make_langchain_embeddings(embedding_provider, embedding_model)
    )

    per_run_records: list[dict[str, Any]] = []
    for response in responses:
        dataset = EvaluationDataset.from_list(
            [
                {
                    "user_input": response.question,
                    "response": response.answer,
                    "retrieved_contexts": [chunk.text for chunk in response.retrieved_chunks],
                    "reference": reference_answers.get(response.question, ""),
                }
            ]
        )
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
        metric_values = _result_to_dict(result)
        per_run_records.append(
            {
                "run_name": response.run_name,
                "question": response.question,
                **{name: metric_values.get(name) for name in metric_names},
            }
        )

    per_run_df = pd.DataFrame(per_run_records)
    summary_df = (
        per_run_df.groupby("run_name", dropna=False)[metric_names]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(by=metric_names[0], ascending=False)
    )
    return per_run_df, summary_df


def export_results(
    output_dir: Path,
    responses: list[ModelResponse],
    retrievals: dict[str, list[Any]],
    references: dict[str, str],
    ragas_per_run: pd.DataFrame | None,
    ragas_summary: pd.DataFrame | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    workbook_path = output_dir / f"comparison_report_{timestamp}.xlsx"

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

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        comparison_df.to_excel(writer, sheet_name="responses", index=False)
        pd.DataFrame(response_rows).to_excel(writer, sheet_name="responses_raw", index=False)
        pd.DataFrame(retrieval_rows).to_excel(writer, sheet_name="retrieval", index=False)
        pd.DataFrame(
            [{"question": question, "reference_answer": answer} for question, answer in references.items()]
        ).to_excel(writer, sheet_name="references", index=False)
        if ragas_per_run is not None:
            ragas_per_run.to_excel(writer, sheet_name="ragas_per_question", index=False)
        if ragas_summary is not None:
            ragas_summary.to_excel(writer, sheet_name="ragas_summary", index=False)

    return workbook_path


def _make_langchain_llm(provider: str, model: str):
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=0.0)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=0.0)

    raise ValueError(f"Unsupported RAGAS judge provider: {provider}")


def _make_langchain_embeddings(provider: str, model: str):
    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=model)

    if provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model)

    raise ValueError(f"Unsupported RAGAS embedding provider: {provider}")


def _result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        return frame.iloc[0].to_dict()
    if hasattr(result, "scores"):
        return dict(result.scores)
    if isinstance(result, dict):
        return result
    return {}


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

    metric_names: list[str] = []
    metric_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if ragas_per_run is not None and not ragas_per_run.empty:
        metric_names = [
            column
            for column in ragas_per_run.columns
            if column not in {"run_name", "question"}
        ]
        metric_lookup = {
            (row["question"], row["run_name"]): row
            for row in ragas_per_run.to_dict(orient="records")
        }

    comparison_rows: list[dict[str, Any]] = []
    for question in question_order:
        row: dict[str, Any] = {"Question": question}

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
