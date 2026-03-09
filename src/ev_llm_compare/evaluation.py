from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any
import warnings

import pandas as pd

from .models import LLMClient, safe_generate
from .prompts import build_reference_prompt, compact_context_segments, format_context
from .schemas import ModelResponse

REPORT_METRIC_NAMES = [
    "answer_accuracy",
    "faithfulness",
    "response_groundedness",
    "grounded_claim_ratio",
    "unsupported_claim_ratio",
    "contradicted_claim_ratio",
    "ragas_answer_accuracy",
    "ragas_faithfulness",
    "ragas_response_groundedness",
]

RAGAS_OUTPUT_METRICS = [
    "ragas_answer_accuracy",
    "ragas_faithfulness",
    "ragas_response_groundedness",
]

FIELD_ALIASES = {
    "company": "company",
    "category": "category",
    "industry group": "industry_group",
    "ev supply chain role": "ev_supply_chain_role",
    "product / service": "product_service",
    "product service": "product_service",
    "primary oems": "primary_oems",
    "primary oem": "primary_oems",
    "location": "location",
    "primary facility type": "primary_facility_type",
    "employment": "employment",
    "ev / battery relevant": "ev_battery_relevant",
    "ev battery relevant": "ev_battery_relevant",
    "supplier or affiliation type": "supplier_or_affiliation_type",
    "classification method": "classification_method",
}

STOPWORD_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "to",
    "with",
}

STRUCTURED_MISSING_PATTERNS = (
    "without specific workbook evidence",
    "without access to the workbook",
    "without access to the excel workbook",
    "without access to the relevant workbook data",
    "without the workbook",
    "without specific workbook data",
    "please provide the excel workbook",
    "i cannot answer this question",
    "i am unable to answer this question",
    "i do not have access",
    "exact dataset answer is unknown",
    "general approach",
    "you would need to refer",
    "if you have access to the workbook",
    "cannot provide a definitive list",
    "can t provide a definitive list",
    "can t give you exact",
    "cannot give you exact",
    "filter your dataset",
    "countif",
    "pivot table",
)

NEGATIVE_PATTERNS = (
    "there are no companies",
    "there are no matching companies",
    "no companies",
    "no matching companies",
    "no entries",
    "none",
    "not possible to provide",
    "does not contain any companies",
)


@dataclass(frozen=True, slots=True)
class StructuredClaim:
    kind: str
    subject: str = ""
    field: str = ""
    label: str = ""
    value: str = ""
    weight: float = 1.0


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


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _normalize_value(value: str) -> str:
    cleaned = value.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", cleaned).strip()


def _normalize_label(value: str) -> str:
    return _normalize_text(re.sub(r"\[\d+\]", "", value))


def _claim_key(claim: StructuredClaim) -> tuple[str, str, str, str, str]:
    return (claim.kind, claim.subject, claim.field, claim.label, claim.value)


def _add_claim(
    claims: dict[tuple[str, str, str, str, str], StructuredClaim],
    claim: StructuredClaim,
) -> None:
    claims[_claim_key(claim)] = claim


def _strip_markup(text: str) -> str:
    cleaned = text.replace("**", " ").replace("`", " ")
    cleaned = re.sub(r"^[\-\*\u2022]+\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _canonical_field_name(label: str) -> str | None:
    normalized = _normalize_text(label)
    return FIELD_ALIASES.get(normalized)


def _split_candidate_segments(text: str) -> list[str]:
    cleaned = text.replace("\r", "\n")
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    cleaned = re.sub(r"\s+\|\s+", " | ", cleaned)
    prepared = re.sub(r"(?m)^\s*[\-\*\u2022]\s*", "", cleaned)
    prepared = re.sub(
        r"(?<!^)\s+(?=Company:\s*[A-Z0-9])",
        "\n",
        prepared,
    )
    prepared = re.sub(
        r"(?<!^)\s+(?=[A-Z0-9][A-Za-z0-9&().,'/\- ]{1,90}\s+\|\s+"
        r"(?:Category|Location|Primary OEMs|Primary Facility Type|Employment|EV Supply Chain Role):)",
        "\n",
        prepared,
    )
    prepared = re.sub(
        r"(?<!^)\s+(?=[A-Z0-9][A-Za-z0-9&().,'/\- ]{1,90}\s+->)",
        "\n",
        prepared,
    )
    segments = [segment.strip() for segment in prepared.split("\n") if segment.strip()]
    if segments:
        return segments
    return [prepared.strip()] if prepared.strip() else []


def _split_group_items(value: str) -> list[str]:
    if ";" in value:
        items = value.split(";")
    else:
        items = re.split(r",\s+(?=[A-Z0-9])", value)
    return [item.strip(" -") for item in items if item.strip(" -")]


def _extract_structured_claims(question: str, text: str) -> dict[tuple[str, str, str, str, str], StructuredClaim]:
    claims: dict[tuple[str, str, str, str, str], StructuredClaim] = {}
    if not text or not text.strip():
        return claims

    normalized_question = _normalize_text(question)
    stripped = _strip_markup(text)
    lowered = _normalize_text(stripped)

    if any(pattern in lowered for pattern in NEGATIVE_PATTERNS):
        _add_claim(claims, StructuredClaim(kind="negative", value="no_matches", weight=1.5))

    for segment in _split_candidate_segments(stripped):
        segment = segment.strip()
        if not segment:
            continue

        if " | " in segment or "->" in segment:
            _extract_record_claims(segment, claims)
            continue

        label_matches = list(
            re.finditer(
                r"([A-Z][A-Za-z0-9 /()&+\-\.]{1,80}?)(?: \[\d+\])?:\s*",
                segment,
            )
        )
        if label_matches:
            parsed_group = False
            for index, match in enumerate(label_matches):
                label = match.group(1).strip()
                label_field = _canonical_field_name(label)
                if label_field:
                    continue
                start = match.end()
                end = label_matches[index + 1].start() if index + 1 < len(label_matches) else len(segment)
                value = segment[start:end].strip(" ;")
                if not value:
                    continue
                parsed_group = True
                normalized_label = _normalize_label(label)
                numeric_value = re.sub(r"[^0-9]", "", value)
                if numeric_value and re.sub(r"[\d,\s]", "", value) == "":
                    _add_claim(
                        claims,
                        StructuredClaim(
                            kind="count",
                            label=normalized_label,
                            value=numeric_value,
                            weight=2.0,
                        ),
                    )
                    continue
                for item in _split_group_items(value):
                    company, category = _extract_company_with_optional_category(item)
                    normalized_company = _normalize_text(company or item)
                    if not normalized_company:
                        continue
                    _add_claim(
                        claims,
                        StructuredClaim(
                            kind="group_item",
                            label=normalized_label,
                            value=normalized_company,
                            weight=1.25,
                        ),
                    )
                    if category:
                        _add_claim(
                            claims,
                            StructuredClaim(
                                kind="group_item_category",
                                label=normalized_label,
                                subject=normalized_company,
                                value=_normalize_text(category),
                                weight=1.35,
                            ),
                        )
            if parsed_group:
                continue

        if (
            any(term in normalized_question for term in {"identify the set", "represented", "union"})
            and ";" in segment
        ):
            for item in _split_group_items(segment):
                normalized_item = _normalize_text(item)
                if normalized_item:
                    _add_claim(
                        claims,
                        StructuredClaim(kind="set_item", value=normalized_item, weight=1.0),
                    )

    return claims


def _extract_record_claims(
    segment: str,
    claims: dict[tuple[str, str, str, str, str], StructuredClaim],
) -> None:
    normalized_segment = _normalize_value(segment)
    if "->" in normalized_segment and "|" not in normalized_segment:
        subject_raw, remainder = normalized_segment.split("->", 1)
        subject = _normalize_text(subject_raw)
        if subject:
            _add_claim(claims, StructuredClaim(kind="entity", subject=subject, weight=0.5))
        if ":" in remainder:
            field_label, value = remainder.split(":", 1)
            field_name = _canonical_field_name(field_label)
            if field_name and subject:
                _add_claim(
                    claims,
                    StructuredClaim(
                        kind="field",
                        subject=subject,
                        field=field_name,
                        value=_normalize_text(value),
                        weight=1.5,
                    ),
                )
        return

    parts = [part.strip(" -") for part in normalized_segment.split("|") if part.strip(" -")]
    if not parts:
        return

    subject = ""
    first_part = parts[0]
    if ":" not in first_part:
        subject = _normalize_text(first_part)
        _add_claim(claims, StructuredClaim(kind="entity", subject=subject, weight=0.5))
        parts = parts[1:]
    elif _canonical_field_name(first_part.split(":", 1)[0]) == "company":
        subject = _normalize_text(first_part.split(":", 1)[1])
        _add_claim(claims, StructuredClaim(kind="entity", subject=subject, weight=0.5))
        parts = parts[1:]

    for part in parts:
        if ":" not in part:
            continue
        field_label, value = part.split(":", 1)
        field_name = _canonical_field_name(field_label)
        if field_name == "company" and not subject:
            subject = _normalize_text(value)
            _add_claim(claims, StructuredClaim(kind="entity", subject=subject, weight=0.5))
            continue
        if not field_name or not subject:
            continue
        normalized_value = _normalize_text(value)
        if not normalized_value:
            continue
        _add_claim(
            claims,
            StructuredClaim(
                kind="field",
                subject=subject,
                field=field_name,
                value=normalized_value,
                weight=1.5,
            ),
        )


def _extract_company_with_optional_category(item: str) -> tuple[str, str]:
    match = re.match(r"(.+?)\s*\(([^)]+)\)\s*$", item.strip())
    if not match:
        return item.strip(), ""
    company, category = match.groups()
    return company.strip(), category.strip()


def _weighted_f1(
    gold_claims: dict[tuple[str, str, str, str, str], StructuredClaim],
    predicted_claims: dict[tuple[str, str, str, str, str], StructuredClaim],
) -> float:
    if not gold_claims and not predicted_claims:
        return 1.0
    if not gold_claims or not predicted_claims:
        return 0.0

    matched_weight = sum(
        min(gold_claims[key].weight, predicted_claims[key].weight)
        for key in set(gold_claims) & set(predicted_claims)
    )
    gold_weight = sum(claim.weight for claim in gold_claims.values())
    predicted_weight = sum(claim.weight for claim in predicted_claims.values())
    if gold_weight <= 0 or predicted_weight <= 0:
        return 0.0
    precision = matched_weight / predicted_weight
    recall = matched_weight / gold_weight
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _tokenize_for_scoring(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", _normalize_text(text))
    return {
        token
        for token in tokens
        if token not in STOPWORD_TOKENS and (token.isdigit() or len(token) >= 3)
    }


def _token_f1(reference_text: str, answer_text: str) -> float:
    reference_tokens = _tokenize_for_scoring(reference_text)
    answer_tokens = _tokenize_for_scoring(answer_text)
    if not reference_tokens or not answer_tokens:
        return 0.0
    overlap = len(reference_tokens & answer_tokens)
    precision = overlap / len(answer_tokens)
    recall = overlap / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _looks_like_structured_question(question: str, reference_answer: str) -> bool:
    normalized_question = _normalize_text(question)
    normalized_reference = _normalize_text(reference_answer)
    if "|" in reference_answer or "->" in reference_answer or "[" in reference_answer:
        return True
    if any(term in normalized_question for term in {"count", "group", "list", "show all", "map", "provide"}):
        return True
    if any(term in normalized_reference for term in {"category", "location", "employment", "primary oems"}):
        return True
    return False


def _is_missing_dataset_answer(answer: str) -> bool:
    normalized = _normalize_text(answer)
    return any(pattern in normalized for pattern in STRUCTURED_MISSING_PATTERNS)


def _score_answer_accuracy(question: str, answer: str, reference_answer: str) -> float | None:
    if not reference_answer:
        return None

    normalized_question = _normalize_text(question)
    gold_claims = _extract_structured_claims(question, reference_answer)
    answer_claims = _extract_structured_claims(question, answer)
    structured_question = _looks_like_structured_question(question, reference_answer)

    if structured_question and _is_missing_dataset_answer(answer):
        return 0.0

    deterministic_count_question = any(
        term in normalized_question
        for term in {
            "for each category",
            "count how many",
            "how many",
        }
    )

    if deterministic_count_question and gold_claims:
        return round(_weighted_f1(gold_claims, answer_claims), 4)

    if structured_question:
        if any(claim.kind == "negative" for claim in answer_claims.values()) and reference_answer.strip():
            return 0.0
        return round(_token_f1(reference_answer, answer), 4)

    return None


def _claim_family_key(claim: StructuredClaim) -> tuple[str, str, str]:
    if claim.kind == "field":
        return (claim.kind, claim.subject, claim.field)
    if claim.kind == "count":
        return (claim.kind, claim.label, "")
    if claim.kind in {"group_item", "group_item_category"}:
        return (claim.kind, claim.label, claim.subject)
    return (claim.kind, claim.subject or claim.label or claim.value, "")


def _derive_grounding_metrics(
    question: str,
    answer: str,
    retrieved_contexts: list[str],
) -> dict[str, float | None]:
    response_claims = list(_extract_structured_claims(question, answer).values())
    context_claim_lookup = _extract_structured_claims(question, "\n".join(retrieved_contexts))
    context_claims = list(context_claim_lookup.values())

    if not response_claims:
        return {
            "faithfulness": None,
            "response_groundedness": None,
            "grounded_claim_ratio": None,
            "unsupported_claim_ratio": None,
            "contradicted_claim_ratio": None,
        }

    supported = 0
    unsupported = 0
    contradicted = 0
    context_families: defaultdict[tuple[str, str, str], list[StructuredClaim]] = defaultdict(list)
    for claim in context_claims:
        context_families[_claim_family_key(claim)].append(claim)

    for claim in response_claims:
        key = _claim_key(claim)
        if key in context_claim_lookup:
            supported += 1
            continue

        family_matches = context_families.get(_claim_family_key(claim), [])
        if claim.kind == "negative":
            if context_claims:
                contradicted += 1
            else:
                supported += 1
            continue

        if claim.kind in {"field", "count"} and family_matches:
            contradicted += 1
            continue

        unsupported += 1

    total = max(1, len(response_claims))
    grounded_ratio = supported / total
    unsupported_ratio = unsupported / total
    contradicted_ratio = contradicted / total
    faithfulness = max(0.0, 1.0 - contradicted_ratio)
    response_groundedness = grounded_ratio
    return {
        "faithfulness": round(faithfulness, 4),
        "response_groundedness": round(response_groundedness, 4),
        "grounded_claim_ratio": round(grounded_ratio, 4),
        "unsupported_claim_ratio": round(unsupported_ratio, 4),
        "contradicted_claim_ratio": round(contradicted_ratio, 4),
    }


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
        base_record = {
            "run_name": response.run_name,
            "question": response.question,
            **{metric_name: None for metric_name in REPORT_METRIC_NAMES},
        }
        reference_answer = reference_answers.get(response.question, "")
        if response.success:
            trusted_accuracy = _score_answer_accuracy(
                response.question,
                response.answer,
                reference_answer,
            )
            base_record["answer_accuracy"] = trusted_accuracy
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
            base_record.update(
                _derive_grounding_metrics(
                    response.question,
                    response.answer,
                    retrieved_contexts,
                )
            )
        record_lookup[(response.run_name, response.question)] = {
            **base_record,
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
            rag_keys.append((response.run_name, response.question))
            rag_rows.append(
                {
                    "user_input": response.question,
                    "response": response.answer,
                    "retrieved_contexts": retrieved_contexts,
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
        _merge_metric_rows(
            record_lookup,
            accuracy_keys,
            accuracy_results,
            {"answer_accuracy": "ragas_answer_accuracy"},
        )

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
        _merge_metric_rows(
            record_lookup,
            rag_keys,
            rag_metric_results,
            {
                "faithfulness": "ragas_faithfulness",
                "response_groundedness": "ragas_response_groundedness",
            },
        )

    for record in record_lookup.values():
        if record.get("answer_accuracy") is None:
            record["answer_accuracy"] = record.get("ragas_answer_accuracy")
        if record.get("faithfulness") is None:
            record["faithfulness"] = record.get("ragas_faithfulness")
        if record.get("response_groundedness") is None:
            record["response_groundedness"] = record.get("ragas_response_groundedness")

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
    metric_name_map: dict[str, str],
) -> None:
    for index, key in enumerate(keys):
        if index >= len(metric_rows):
            continue
        metrics = metric_rows[index]
        record = record_lookup[key]
        for source_name, target_name in metric_name_map.items():
            if source_name in metrics:
                record[target_name] = metrics.get(source_name)


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
