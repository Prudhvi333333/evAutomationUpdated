from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import threading
from typing import Any

import pandas as pd

from .models import LLMClient, create_client, safe_generate
from .prompts import build_reference_prompt, compact_context_segments, format_context
from .reference_metrics import compute_all_reference_metrics
from .schemas import ModelResponse
from .settings import ModelSpec, RuntimeSettings

# ──────────────────────────────────────────────────────────────────────────────
# Metric name registries
#
# Three clearly-separated groups:
#
# REFERENCE_METRIC_NAMES   – applicable to ALL responses (RAG + no-RAG),
#                            compare against golden answers.
#                            These are the only metrics valid for cross-mode
#                            comparison.
#
# CUSTOM_RAG_METRIC_NAMES  – applicable ONLY to RAG responses (rag_enabled=True).
#                            Undefined / null for no-RAG.
#                            Renamed from legacy "RAGAS_METRIC_NAMES" to avoid
#                            confusion with the actual ragas library.
#
# RAGAS_LIBRARY_METRIC_NAMES – metrics produced by the actual ragas library,
#                              RAG responses only (or answer_correctness for all).
#
# DIAGNOSTIC_METRIC_NAMES  – claim-level signals, RAG only.
#
# OPERATIONAL_METRIC_NAMES – always available (success, latency, length, etc.).
# ──────────────────────────────────────────────────────────────────────────────

# Cross-mode reference metrics (both RAG and no-RAG)
REFERENCE_METRIC_NAMES: list[str] = [
    "answer_accuracy",        # LLM dual-judge vs golden answer
    "rouge_l",                # deterministic ROUGE-L F-measure
    "token_f1",               # deterministic token F1
    "semantic_similarity",    # sentence-embedding cosine similarity
    "normalized_exact_match", # exact match after normalisation
]

# Custom LLM-judge grounding metrics — RAG only, clearly labelled as CUSTOM
CUSTOM_RAG_METRIC_NAMES: list[str] = [
    "custom_faithfulness",         # grounded claim ratio (claim-level attribution)
    "custom_response_groundedness",# holistic 0/1/2 → normalised score
    "custom_context_precision",    # average precision of ranked contexts
    "custom_context_recall",       # fraction of reference claims in context
    "unsupported_claim_ratio",     # 1 - faithfulness (diagnostic)
    "contradicted_claim_ratio",    # claims contradicted by context
]

# Actual ragas library metrics — RAG only except answer_correctness
RAGAS_LIBRARY_METRIC_NAMES: list[str] = [
    "ragas_faithfulness",
    "ragas_context_precision",
    "ragas_context_recall",
    "ragas_answer_correctness",  # all responses with golden answer
]

# Operational / diagnostic metrics
OPERATIONAL_METRIC_NAMES: list[str] = [
    "success",
    "abstained",             # True when model said "Insufficient evidence:"
    "response_length",
    "citation_coverage",     # fraction of evidence IDs cited by model
    "judge_failure_count",   # number of judge calls that had to be retried
]

# Full list used for DataFrame column creation
ALL_METRIC_NAMES: list[str] = [
    *REFERENCE_METRIC_NAMES,
    *CUSTOM_RAG_METRIC_NAMES,
    *RAGAS_LIBRARY_METRIC_NAMES,
    *OPERATIONAL_METRIC_NAMES,
]

# ──────────────────────────────────────────────────────────────────────────────
# Mode-aware composite scores
#
# These are SEPARATE scores for each mode — they are NOT combined into a single
# leaderboard score across modes, which would be methodologically invalid.
#
# rag_composite_score:   weighted combination of RAG-mode metrics only
# answer_quality_score:  cross-mode score using reference metrics only
# ──────────────────────────────────────────────────────────────────────────────

RAG_COMPOSITE_COMPONENTS: dict[str, tuple[float, bool]] = {
    # (weight, invert)  — invert=True means lower is better
    "answer_accuracy":               (0.30, False),
    "custom_faithfulness":           (0.20, False),
    "custom_response_groundedness":  (0.15, False),
    "custom_context_precision":      (0.12, False),
    "custom_context_recall":         (0.12, False),
    "unsupported_claim_ratio":       (0.06, True),
    "contradicted_claim_ratio":      (0.05, True),
}

ANSWER_QUALITY_COMPONENTS: dict[str, tuple[float, bool]] = {
    "answer_accuracy":        (0.40, False),
    "semantic_similarity":    (0.30, False),
    "rouge_l":                (0.20, False),
    "token_f1":               (0.10, False),
}

# Legacy aliases — kept for backward compatibility with existing callers
RAGAS_METRIC_NAMES = REFERENCE_METRIC_NAMES  # re-mapped, no longer raw RAGAS
REPORT_METRIC_NAMES = ALL_METRIC_NAMES
DIAGNOSTIC_METRIC_NAMES = ["unsupported_claim_ratio", "contradicted_claim_ratio"]
DERIVED_METRIC_NAMES = ["rag_composite_score", "answer_quality_score"]
# The old field names are mapped to their new canonical names
_LEGACY_FIELD_MAP = {
    "faithfulness": "custom_faithfulness",
    "response_groundedness": "custom_response_groundedness",
    "context_precision": "custom_context_precision",
    "context_recall": "custom_context_recall",
    "overall_metric_score_pct": "rag_composite_score",
}

# Keep OVERALL_SCORE_COMPONENTS for backward compat in callers (now unused internally)
OVERALL_SCORE_COMPONENTS: dict[str, tuple[float, bool]] = RAG_COMPOSITE_COMPONENTS

LLM_JUDGE_SCORE_SYSTEM_PROMPT = """You are a strict evaluation judge for workbook question answering.
Judge only from the texts provided in the user prompt.
Do not use world knowledge, prior knowledge, or unstated assumptions.
Be conservative: if support is incomplete or uncertain, choose the lower score.
Return only one line in the form SCORE=<value>.
Use a continuous score from 0.00 to 1.00 with exactly two decimal places.
Do not add explanation, JSON, markdown, or extra text."""
LLM_JUDGE_RATING_SYSTEM_PROMPT = """You are a strict evaluation judge for workbook question answering.
Judge only from the texts provided in the user prompt.
Do not use world knowledge, prior knowledge, or unstated assumptions.
Be conservative: if support is incomplete or uncertain, choose the lower rating.
Return only one line in the form RATING=<value>.
Use only the allowed discrete rating values described in the prompt.
Do not add explanation, JSON, markdown, or extra text."""
LLM_ATTRIBUTION_SYSTEM_PROMPT = """You are performing claim-level provenance attribution for workbook question answering.
Judge only from the retrieved evidence and response units provided in the prompt.
Do not use world knowledge, prior knowledge, or unstated assumptions.
Return only valid JSON.
Do not add markdown, explanation, or prose before or after the JSON."""
LLM_JSON_SYSTEM_PROMPT = """You are a strict evaluation judge for workbook question answering.
Judge only from the texts provided in the user prompt.
Do not use world knowledge, prior knowledge, or unstated assumptions.
Be conservative: if support is incomplete or uncertain, prefer the lower-support label.
Return only valid JSON.
Do not add markdown, explanation, or prose before or after the JSON."""

ATTRIBUTION_BATCH_SIZE = 24
CLAIM_CLASSIFICATION_BATCH_SIZE = 16
CONTEXT_RELEVANCE_BATCH_SIZE = 10

_THREAD_LOCAL_STATE = threading.local()


def _split_text_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.strip():
            current.append(line.rstrip())
            continue
        if current:
            blocks.append("\n".join(current).strip())
            current = []
    if current:
        blocks.append("\n".join(current).strip())
    return blocks


def _is_list_like_line(line: str) -> bool:
    stripped = line.lstrip()
    if not stripped:
        return False
    if stripped.startswith(("-", "*", "•")):
        return True
    if stripped[:1].isdigit():
        return True
    return False


def _sentence_units(text: str) -> list[str]:
    if not text.strip():
        return []

    abbreviations = {
        "co",
        "corp",
        "inc",
        "llc",
        "ltd",
        "mr",
        "mrs",
        "ms",
        "dr",
        "st",
        "vs",
        "etc",
    }
    units: list[str] = []
    current: list[str] = []
    length = len(text)

    for index, char in enumerate(text):
        current.append(char)
        if char not in {".", "!", "?", ";"}:
            continue

        segment = "".join(current).strip()
        if not segment:
            current = []
            continue

        if char == ".":
            tail = segment.rstrip(".").split()[-1].lower() if segment.rstrip(".").split() else ""
            if tail in abbreviations:
                continue

        next_non_space = ""
        cursor = index + 1
        while cursor < length:
            candidate = text[cursor]
            if not candidate.isspace():
                next_non_space = candidate
                break
            cursor += 1
        if next_non_space and next_non_space.islower():
            continue

        units.append(segment)
        current = []

    trailing = "".join(current).strip()
    if trailing:
        units.append(trailing)
    return units


def _clean_segmented_unit(text: str) -> str:
    cleaned = re.sub(r"^\s*(?:[-*]\s+|\d+[.)]\s+)", "", text.strip())
    return cleaned.strip()


def _split_structured_catalog_units(text: str) -> list[str]:
    normalized = " ".join((text or "").split())
    if not normalized:
        return []

    prefix_units: list[str] = []
    tail_text = normalized
    prefix_match = re.match(r"^(.*?\.)(?=\s+[A-Z0-9&].*\[[^\]]+\]\s*\|)", normalized)
    if prefix_match:
        prefix_units = _sentence_units(prefix_match.group(1).strip())
        tail_text = normalized[prefix_match.end() :].strip()

    if tail_text.count(";") >= 1:
        candidates = [segment.strip() for segment in re.split(r"\s*;\s*", tail_text) if segment.strip()]
        if (
            len(candidates) >= 2
            and sum("|" in segment for segment in candidates) >= len(candidates) - 1
            and any("Company:" in segment or "[" in segment for segment in candidates)
        ):
            return prefix_units + candidates

    company_token = r"[A-Z0-9&(][A-Za-z0-9&/().,'-]*"
    company_connector = r"(?:of|and|the|for|&)"
    company_name_pattern = rf"{company_token}(?:\s+(?:{company_token}|{company_connector})){{0,11}}"
    start_patterns = [
        rf"(?<!\S){company_name_pattern}\s*\[[^\]]+\]\s*\|\s*(?:Role|Category|EV Supply Chain Role|Primary OEMs|Updated Location|Location|Primary Facility Type|Employment|Product(?:\s*/\s*Service)?)\s*:",
        rf"(?<!\S)(?:Company:\s*)?{company_name_pattern}\s*\|\s*(?:Category|EV Supply Chain Role|Role|Primary OEMs|Updated Location|Location|Primary Facility Type|Employment|Product(?:\s*/\s*Service)?)\s*:",
    ]
    for pattern in start_patterns:
        matches = list(re.finditer(pattern, tail_text))
        if len(matches) < 2:
            continue
        units: list[str] = list(prefix_units)
        for index, match in enumerate(matches):
            end_index = matches[index + 1].start() if index + 1 < len(matches) else len(tail_text)
            segment = tail_text[match.start() : end_index].strip(" ;,-")
            if segment:
                units.append(segment)
        return units

    if tail_text.count("|") >= 4 and tail_text.count(";") >= 1:
        candidates = [segment.strip() for segment in re.split(r"\s*;\s*", tail_text) if segment.strip()]
        if len(candidates) >= 2 and sum("|" in segment for segment in candidates) >= len(candidates) - 1:
            return prefix_units + candidates

    return []

def _segment_response_units(answer: str) -> list[str]:
    text = (answer or "").replace("\r\n", "\n").strip()
    if not text:
        return []

    units: list[str] = []
    for block in _split_text_blocks(text):
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue

        expanded_lines: list[str] = []
        structured_line_found = False
        for line in lines:
            structured_units = _split_structured_catalog_units(line)
            if structured_units:
                expanded_lines.extend(structured_units)
                structured_line_found = True
            else:
                expanded_lines.append(line)

        if len(expanded_lines) > 1 and sum(1 for line in expanded_lines if _is_list_like_line(line)) >= max(1, len(expanded_lines) // 2):
            units.extend(expanded_lines)
            continue
        if len(expanded_lines) > 1 and (structured_line_found or all(len(line) <= 180 for line in expanded_lines)):
            units.extend(expanded_lines)
            continue

        for line in expanded_lines:
            if " | " in line and (":" in line or line.count("|") >= 2):
                units.append(line)
            else:
                units.extend(_sentence_units(line))

    cleaned_units = [_clean_segmented_unit(unit) for unit in units if unit.strip()]
    return [unit for unit in cleaned_units if unit]

def _build_attribution_prompt(
    question: str,
    retrieved_contexts: list[str],
    unit_batch: list[tuple[int, str]],
) -> str:
    context = "\n\n".join(retrieved_contexts) if retrieved_contexts else "No retrieved evidence."
    units_text = "\n".join(f"{unit_id}. {text}" for unit_id, text in unit_batch)
    return (
        "Task: attribute each response unit to one provenance label.\n"
        "Labels:\n"
        "- knowledge_source: the unit is directly supported by the retrieved workbook evidence.\n"
        "- pretrained: the unit depends on general model knowledge, unsupported synthesis, filler, or anything not directly grounded in the retrieved evidence.\n"
        "Rules:\n"
        "- Use only the two labels above.\n"
        "- If a unit mixes supported and unsupported material, label it pretrained.\n"
        "- If you are uncertain, label it pretrained.\n"
        "- Preserve the unit ids exactly.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved evidence:\n{context}\n\n"
        f"Response units:\n{units_text}\n\n"
        "Return JSON in exactly this shape:\n"
        '{"labels":[{"unit_id":1,"label":"knowledge_source"}]}'
    )


def _extract_json_payload(raw_text: str) -> Any | None:
    if not raw_text:
        return None
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    start_positions = [
        index for index, char in enumerate(raw_text) if char in {"{", "["}
    ]
    for start in start_positions:
        stack: list[str] = []
        in_string = False
        escape = False
        for end in range(start, len(raw_text)):
            char = raw_text[end]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char in {"{", "["}:
                stack.append("}" if char == "{" else "]")
                continue
            if char in {"}", "]"}:
                if not stack or char != stack[-1]:
                    break
                stack.pop()
                if not stack:
                    candidate = raw_text[start : end + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    return None


def _parse_attribution_labels(raw_text: str) -> dict[int, str] | None:
    payload = _extract_json_payload(raw_text)
    if not isinstance(payload, dict):
        return None
    labels = payload.get("labels")
    if not isinstance(labels, list):
        return None

    parsed: dict[int, str] = {}
    for item in labels:
        if not isinstance(item, dict):
            return None
        unit_id = item.get("unit_id")
        label = str(item.get("label", "")).strip().lower()
        if not isinstance(unit_id, int):
            return None
        if label not in {"knowledge_source", "pretrained"}:
            return None
        parsed[unit_id] = label
    return parsed


def _llm_judge_attribution(
    judge_client: LLMClient,
    prompt: str,
    retries: int = 2,
) -> dict[int, str] | None:
    for _ in range(max(1, retries + 1)):
        answer, _, success, _ = safe_generate(
            judge_client,
            prompt,
            temperature=0.0,
            max_tokens=1200,
            system_prompt=LLM_ATTRIBUTION_SYSTEM_PROMPT,
        )
        if not success:
            continue
        parsed = _parse_attribution_labels(answer)
        if parsed is not None:
            return parsed
    return None


def attribute_response_sources(
    response: ModelResponse,
    judge_client: LLMClient | None,
    context_result_limit: int = 4,
    context_char_budget: int = 2600,
    compact_context: bool = True,
    max_retries: int = 2,
) -> dict[str, Any]:
    overall_response = response.answer or ""
    if not overall_response.strip():
        return {
            "overall_response": overall_response,
            "knowledge_source_data": "",
            "pretrained_data": "",
            "attribution_units": [],
        }

    # No external knowledge source was available to the model, so the whole answer is treated as pretrained/general.
    if not response.rag_enabled or not response.retrieved_chunks or judge_client is None:
        return {
            "overall_response": overall_response,
            "knowledge_source_data": "",
            "pretrained_data": overall_response,
            "attribution_units": [
                {"unit_id": 1, "unit_text": overall_response, "label": "pretrained"}
            ],
        }

    units = _segment_response_units(overall_response)
    if not units:
        units = [overall_response]

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

    labels_by_id: dict[int, str] = {}
    indexed_units = list(enumerate(units, start=1))
    for start in range(0, len(indexed_units), ATTRIBUTION_BATCH_SIZE):
        batch = indexed_units[start : start + ATTRIBUTION_BATCH_SIZE]
        prompt = _build_attribution_prompt(
            question=response.question,
            retrieved_contexts=retrieved_contexts,
            unit_batch=batch,
        )
        parsed = _llm_judge_attribution(
            judge_client,
            prompt,
            retries=max_retries,
        )
        batch_ids = {unit_id for unit_id, _ in batch}
        if parsed is None:
            for unit_id in batch_ids:
                labels_by_id[unit_id] = "pretrained"
            continue
        for unit_id in batch_ids:
            labels_by_id[unit_id] = parsed.get(unit_id, "pretrained")

    knowledge_units: list[str] = []
    pretrained_units: list[str] = []
    attribution_units: list[dict[str, Any]] = []
    for unit_id, unit_text in indexed_units:
        label = labels_by_id.get(unit_id, "pretrained")
        attribution_units.append(
            {
                "unit_id": unit_id,
                "unit_text": unit_text,
                "label": label,
            }
        )
        if label == "knowledge_source":
            knowledge_units.append(unit_text)
        else:
            pretrained_units.append(unit_text)

    return {
        "overall_response": overall_response,
        "knowledge_source_data": "\n".join(knowledge_units).strip(),
        "pretrained_data": "\n".join(pretrained_units).strip(),
        "attribution_units": attribution_units,
    }


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


def _make_judge_client(
    provider: str,
    model: str,
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> LLMClient:
    runtime = RuntimeSettings()
    runtime.ollama_base_url = os.getenv("OLLAMA_BASE_URL", runtime.ollama_base_url)
    spec = ModelSpec(
        run_name="llm_judge",
        provider=provider,
        model_name=model,
        rag_enabled=False,
        base_url=base_url,
        api_key_env=api_key_env,
    )
    return create_client(spec, runtime)


def _thread_local_judge_client(
    provider: str,
    model: str,
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> LLMClient:
    cache = getattr(_THREAD_LOCAL_STATE, "judge_clients", None)
    if cache is None:
        cache = {}
        _THREAD_LOCAL_STATE.judge_clients = cache
    key = (provider, model)
    client = cache.get(key)
    if client is None:
        client = _make_judge_client(provider, model, base_url=base_url, api_key_env=api_key_env)
        cache[key] = client
    return client


def _llm_judge_prompt_answer_accuracy(
    question: str,
    answer: str,
    reference_answer: str,
) -> str:
    return (
        "Task: score how well the candidate answer matches the reference answer for a workbook question.\n"
        "Use a continuous score from 0.00 to 1.00.\n"
        "Scoring guidance:\n"
        "- 0.00 to 0.19 = inaccurate, off-task, or materially answering a different question.\n"
        "- 0.20 to 0.49 = partially correct but with major omissions, wrong entities, or wrong numbers.\n"
        "- 0.50 to 0.79 = mostly correct but still missing requested facts or containing noticeable mistakes.\n"
        "- 0.80 to 0.94 = strong match with only minor omissions or formatting differences.\n"
        "- 0.95 to 1.00 = fully aligned with the reference on the requested facts with no material mistakes.\n"
        "Rules:\n"
        "- For list, grouped, and count questions, judge factual coverage and precision.\n"
        "- Missing requested items should lower the rating.\n"
        "- Wrong extra entities or wrong numbers should lower the rating.\n"
        "- Do not reward writing style, verbosity, or fluent paraphrasing.\n"
        "- Do not infer missing facts that are not explicitly present in the candidate or reference.\n"
        "- If the answer is mostly a refusal, generic method, or off-task, use a very low score.\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference_answer}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        "Return exactly one line: SCORE=<value>"
    )


def _llm_judge_prompt_answer_accuracy_reverse(
    question: str,
    answer: str,
    reference_answer: str,
) -> str:
    return (
        "Task: independently re-evaluate answer accuracy by swapping roles.\n"
        "Score how well the reference answer covers and matches the candidate answer for the same workbook question.\n"
        "Use a continuous score from 0.00 to 1.00.\n"
        "Scoring guidance:\n"
        "- 0.00 to 0.19 = the answers do not materially align.\n"
        "- 0.20 to 0.49 = there is only limited overlap with major gaps or mistakes.\n"
        "- 0.50 to 0.79 = there is meaningful overlap but material gaps remain.\n"
        "- 0.80 to 0.94 = the answers mostly align with minor gaps.\n"
        "- 0.95 to 1.00 = the answers materially align on the requested facts.\n"
        "- Do not reward writing style, verbosity, or fluent paraphrasing.\n"
        "- Do not infer missing facts that are not explicitly present in the two answers.\n"
        "This prompt intentionally swaps the roles to reduce judge-position bias.\n\n"
        f"Question:\n{question}\n\n"
        f"Candidate answer:\n{reference_answer}\n\n"
        f"Reference answer:\n{answer}\n\n"
        "Return exactly one line: SCORE=<value>"
    )


def _llm_judge_prompt_response_groundedness(
    question: str,
    answer: str,
    retrieved_contexts: list[str],
) -> str:
    context = "\n\n".join(retrieved_contexts)
    return (
        "Task: rate how grounded the candidate answer is in the retrieved workbook evidence.\n"
        "Use only these ratings:\n"
        "- 0 = the answer is largely unsupported or contradicted by the evidence.\n"
        "- 1 = the answer is partially grounded, but some material content is unsupported or unclear.\n"
        "- 2 = the answer is well grounded in the retrieved evidence with no material unsupported content.\n"
        "Focus on support from the retrieved evidence, not world knowledge.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved evidence:\n{context}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        "Return exactly one line: RATING=<0|1|2>"
    )


def _llm_judge_prompt_response_groundedness_reverse(
    question: str,
    answer: str,
    retrieved_contexts: list[str],
) -> str:
    context = "\n\n".join(retrieved_contexts)
    return (
        "Task: independently re-evaluate groundedness from the evidence side.\n"
        "Ask whether the retrieved workbook evidence would justify producing the candidate answer without adding unstated facts.\n"
        "Use only these ratings:\n"
        "- 0 = the evidence would not justify the answer.\n"
        "- 1 = the evidence justifies only part of the answer.\n"
        "- 2 = the evidence materially justifies the answer.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved evidence:\n{context}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        "Return exactly one line: RATING=<0|1|2>"
    )


def _parse_llm_judge_score(raw_text: str) -> float | None:
    if not raw_text:
        return None
    stripped = raw_text.strip()

    explicit_match = re.fullmatch(
        r"SCORE\s*=\s*(0(?:\.\d+)?|1(?:\.0+)?)",
        stripped,
        flags=re.IGNORECASE,
    )
    bare_match = re.fullmatch(r"(0(?:\.\d+)?|1(?:\.0+)?)", stripped)
    match = explicit_match or bare_match
    if match is None:
        return None

    try:
        score = float(match.group(1))
    except ValueError:
        return None
    if not (0.0 <= score <= 1.0):
        return None
    return round(score, 4)


def _parse_llm_judge_rating(raw_text: str, allowed_values: set[int]) -> int | None:
    if not raw_text:
        return None
    match = re.search(r"RATING\s*=\s*([0-9]+)\b", raw_text, flags=re.IGNORECASE)
    if not match:
        return None
    rating = int(match.group(1))
    if rating not in allowed_values:
        return None
    return rating


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
            system_prompt=LLM_JUDGE_SCORE_SYSTEM_PROMPT,
        )
        if not success:
            continue
        parsed = _parse_llm_judge_score(answer)
        if parsed is not None:
            return parsed
    return None


def _llm_judge_rating(
    judge_client: LLMClient,
    prompt: str,
    allowed_values: set[int],
    retries: int = 2,
) -> int | None:
    for _ in range(max(1, retries + 1)):
        answer, _, success, _ = safe_generate(
            judge_client,
            prompt,
            temperature=0.0,
            max_tokens=8,
            system_prompt=LLM_JUDGE_RATING_SYSTEM_PROMPT,
        )
        if not success:
            continue
        parsed = _parse_llm_judge_rating(answer, allowed_values=allowed_values)
        if parsed is not None:
            return parsed
    return None


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _dual_judge_metric(
    judge_client: LLMClient,
    prompts: list[str],
    allowed_values: set[int],
    rating_scale: int,
    retries: int,
) -> float | None:
    normalized_scores: list[float] = []
    for prompt in prompts:
        rating = _llm_judge_rating(
            judge_client,
            prompt,
            allowed_values=allowed_values,
            retries=retries,
        )
        if rating is None:
            continue
        normalized_scores.append(round(rating / rating_scale, 4))
    return _average(normalized_scores)


def _dual_judge_score_metric(
    judge_client: LLMClient,
    prompts: list[str],
    retries: int,
) -> float | None:
    scores: list[float] = []
    for prompt in prompts:
        score = _llm_judge_metric(
            judge_client,
            prompt,
            retries=retries,
        )
        if score is not None:
            scores.append(score)
    return _average(scores)


def _build_claim_classification_prompt(
    question: str,
    retrieved_contexts: list[str],
    claim_batch: list[tuple[int, str]],
) -> str:
    context = "\n\n".join(retrieved_contexts) if retrieved_contexts else "No retrieved evidence."
    claims_text = "\n".join(f"{claim_id}. {text}" for claim_id, text in claim_batch)
    return (
        "Task: classify each claim against the retrieved workbook evidence.\n"
        "Labels:\n"
        "- supported: the claim is directly supported by the evidence.\n"
        "- unsupported: the evidence does not support the claim, or the claim goes beyond the evidence.\n"
        "- contradicted: the evidence directly conflicts with the claim.\n"
        "Rules:\n"
        "- If a claim mixes supported and unsupported content, use unsupported.\n"
        "- If you are uncertain, use unsupported.\n"
        "- Preserve the claim ids exactly.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved evidence:\n{context}\n\n"
        f"Claims:\n{claims_text}\n\n"
        'Return JSON in exactly this shape:\n{"labels":[{"claim_id":1,"label":"supported"}]}'
    )


def _parse_claim_labels(raw_text: str) -> dict[int, str] | None:
    payload = _extract_json_payload(raw_text)
    if not isinstance(payload, dict):
        return None
    labels = payload.get("labels")
    if not isinstance(labels, list):
        return None

    parsed: dict[int, str] = {}
    allowed = {"supported", "unsupported", "contradicted"}
    for item in labels:
        if not isinstance(item, dict):
            return None
        claim_id = item.get("claim_id")
        label = str(item.get("label", "")).strip().lower()
        if not isinstance(claim_id, int) or label not in allowed:
            return None
        parsed[claim_id] = label
    return parsed


def _llm_judge_claim_labels(
    judge_client: LLMClient,
    prompt: str,
    retries: int = 2,
) -> dict[int, str] | None:
    for _ in range(max(1, retries + 1)):
        answer, _, success, _ = safe_generate(
            judge_client,
            prompt,
            temperature=0.0,
            max_tokens=1200,
            system_prompt=LLM_JSON_SYSTEM_PROMPT,
        )
        if not success:
            continue
        parsed = _parse_claim_labels(answer)
        if parsed is not None:
            return parsed
    return None


def _build_context_relevance_prompt(
    question: str,
    reference_answer: str,
    ranked_contexts: list[tuple[int, str]],
) -> str:
    contexts_text = "\n\n".join(
        f"{context_id}. {context_text}" for context_id, context_text in ranked_contexts
    )
    return (
        "Task: judge whether each retrieved context chunk is relevant for answering the workbook question.\n"
        "A context is relevant if it contains information that supports at least one material part of the reference answer or would directly help answer the question.\n"
        "Labels:\n"
        "- relevant\n"
        "- irrelevant\n"
        "Rules:\n"
        "- Boilerplate, tangential details, and duplicate filler should be labeled irrelevant.\n"
        "- If you are uncertain, label the context irrelevant.\n"
        "- Preserve the context ids exactly.\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference_answer}\n\n"
        f"Retrieved contexts:\n{contexts_text}\n\n"
        'Return JSON in exactly this shape:\n{"labels":[{"context_id":1,"label":"relevant"}]}'
    )


def _parse_context_labels(raw_text: str) -> dict[int, bool] | None:
    payload = _extract_json_payload(raw_text)
    if not isinstance(payload, dict):
        return None
    labels = payload.get("labels")
    if not isinstance(labels, list):
        return None

    parsed: dict[int, bool] = {}
    for item in labels:
        if not isinstance(item, dict):
            return None
        context_id = item.get("context_id")
        label = str(item.get("label", "")).strip().lower()
        if not isinstance(context_id, int) or label not in {"relevant", "irrelevant"}:
            return None
        parsed[context_id] = label == "relevant"
    return parsed


def _llm_judge_context_labels(
    judge_client: LLMClient,
    prompt: str,
    retries: int = 2,
) -> dict[int, bool] | None:
    for _ in range(max(1, retries + 1)):
        answer, _, success, _ = safe_generate(
            judge_client,
            prompt,
            temperature=0.0,
            max_tokens=1000,
            system_prompt=LLM_JSON_SYSTEM_PROMPT,
        )
        if not success:
            continue
        parsed = _parse_context_labels(answer)
        if parsed is not None:
            return parsed
    return None


def _truncate_context_value(text: str, limit: int = 1200) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _classify_claims_against_context(
    judge_client: LLMClient,
    question: str,
    retrieved_contexts: list[str],
    claims: list[str],
    retries: int,
) -> dict[int, str]:
    labels_by_id: dict[int, str] = {}
    indexed_claims = list(enumerate(claims, start=1))
    for start in range(0, len(indexed_claims), CLAIM_CLASSIFICATION_BATCH_SIZE):
        batch = indexed_claims[start : start + CLAIM_CLASSIFICATION_BATCH_SIZE]
        prompt = _build_claim_classification_prompt(
            question=question,
            retrieved_contexts=retrieved_contexts,
            claim_batch=batch,
        )
        parsed = _llm_judge_claim_labels(
            judge_client,
            prompt,
            retries=retries,
        )
        batch_ids = {claim_id for claim_id, _ in batch}
        if parsed is None:
            for claim_id in batch_ids:
                labels_by_id[claim_id] = "unsupported"
            continue
        for claim_id in batch_ids:
            labels_by_id[claim_id] = parsed.get(claim_id, "unsupported")
    return labels_by_id


def _compute_claim_ratios(labels_by_id: dict[int, str], claim_count: int) -> dict[str, float] | None:
    if claim_count <= 0:
        return None
    supported = sum(1 for label in labels_by_id.values() if label == "supported")
    unsupported = sum(1 for label in labels_by_id.values() if label == "unsupported")
    contradicted = sum(1 for label in labels_by_id.values() if label == "contradicted")
    grounded_ratio = round(supported / claim_count, 4)
    unsupported_ratio = round(unsupported / claim_count, 4)
    contradicted_ratio = round(contradicted / claim_count, 4)
    return {
        "faithfulness": grounded_ratio,
        "grounded_claim_ratio": grounded_ratio,
        "unsupported_claim_ratio": unsupported_ratio,
        "contradicted_claim_ratio": contradicted_ratio,
    }


def _average_precision(relevance_flags: list[bool]) -> float | None:
    if not relevance_flags:
        return None
    relevant_seen = 0
    precision_values: list[float] = []
    for index, is_relevant in enumerate(relevance_flags, start=1):
        if not is_relevant:
            continue
        relevant_seen += 1
        precision_values.append(relevant_seen / index)
    if not precision_values:
        return 0.0
    return round(sum(precision_values) / len(precision_values), 4)


def _evaluate_context_precision(
    judge_client: LLMClient,
    question: str,
    reference_answer: str,
    retrieved_contexts: list[str],
    retries: int,
) -> float | None:
    if not reference_answer.strip() or not retrieved_contexts:
        return None

    ranked_contexts = [
        (index, _truncate_context_value(str(context_text)))
        for index, context_text in enumerate(retrieved_contexts, start=1)
        if str(context_text).strip()
    ]
    if not ranked_contexts:
        return None

    relevance_by_id: dict[int, bool] = {}
    for start in range(0, len(ranked_contexts), CONTEXT_RELEVANCE_BATCH_SIZE):
        batch = ranked_contexts[start : start + CONTEXT_RELEVANCE_BATCH_SIZE]
        prompt = _build_context_relevance_prompt(
            question=question,
            reference_answer=reference_answer,
            ranked_contexts=batch,
        )
        parsed = _llm_judge_context_labels(
            judge_client,
            prompt,
            retries=retries,
        )
        batch_ids = {context_id for context_id, _ in batch}
        if parsed is None:
            for context_id in batch_ids:
                relevance_by_id[context_id] = False
            continue
        for context_id in batch_ids:
            relevance_by_id[context_id] = parsed.get(context_id, False)

    flags = [relevance_by_id.get(index, False) for index, _ in ranked_contexts]
    return _average_precision(flags)


def _evaluate_context_recall(
    judge_client: LLMClient,
    question: str,
    reference_answer: str,
    retrieved_contexts: list[str],
    retries: int,
) -> float | None:
    reference_claims = _segment_response_units(reference_answer)
    if not reference_claims:
        return None
    labels_by_id = _classify_claims_against_context(
        judge_client=judge_client,
        question=question,
        retrieved_contexts=retrieved_contexts,
        claims=reference_claims,
        retries=retries,
    )
    supported = sum(1 for label in labels_by_id.values() if label == "supported")
    return round(supported / len(reference_claims), 4)


def _compute_rag_composite_score(record: dict[str, Any]) -> float | None:
    """Weighted composite for RAG responses only.

    Uses only RAG-applicable metrics.  Returns None if no RAG metrics are
    available (e.g., for no-RAG responses — callers must not store this value
    on no-RAG records to avoid invalid cross-mode comparison).
    """
    weighted_total = 0.0
    available_weight = 0.0
    for metric_name, (weight, invert) in RAG_COMPOSITE_COMPONENTS.items():
        raw_value = record.get(metric_name)
        if raw_value is None or pd.isna(raw_value):
            continue
        value = float(raw_value)
        value = 1.0 - value if invert else value
        value = min(1.0, max(0.0, value))
        weighted_total += weight * value
        available_weight += weight
    if available_weight < 0.20:   # require at least 20% weight coverage
        return None
    return round((weighted_total / available_weight) * 100.0, 2)


def _compute_answer_quality_score(record: dict[str, Any]) -> float | None:
    """Weighted composite for cross-mode comparison.

    Uses ONLY reference metrics (applicable to both RAG and no-RAG).
    This is the only composite score that can legitimately be used to
    compare RAG vs no-RAG responses directly.
    """
    weighted_total = 0.0
    available_weight = 0.0
    for metric_name, (weight, _invert) in ANSWER_QUALITY_COMPONENTS.items():
        raw_value = record.get(metric_name)
        if raw_value is None or pd.isna(raw_value):
            continue
        value = float(raw_value)
        value = min(1.0, max(0.0, value))
        weighted_total += weight * value
        available_weight += weight
    if available_weight < 0.30:   # require at least 30% weight coverage
        return None
    return round((weighted_total / available_weight) * 100.0, 2)


# Legacy alias — kept for backward compatibility
def _compute_overall_metric_score_pct(record: dict[str, Any]) -> float | None:
    """Deprecated: use _compute_rag_composite_score or _compute_answer_quality_score."""
    return _compute_rag_composite_score(record)


def _score_response_metrics(
    response: ModelResponse,
    reference_answers: dict[str, str],
    judge_client: LLMClient,
    max_retries: int,
    context_result_limit: int,
    context_char_budget: int,
    compact_context: bool,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """Score a single response with mode-aware, properly-labelled metrics.

    Metric groups:
      REFERENCE  – computed for all responses with a golden answer
      CUSTOM_RAG – computed only for RAG responses
      COMPOSITE  – mode-specific; rag_composite_score for RAG only,
                   answer_quality_score for all

    No retrieval-dependent score is ever set on a no-RAG response.
    The composite scores use different formulas to prevent invalid
    cross-mode comparison.
    """
    record: dict[str, Any] = {
        "run_name": response.run_name,
        "question": response.question,
        "rag_enabled": response.rag_enabled,
        **{metric_name: None for metric_name in ALL_METRIC_NAMES},
    }

    # ── Operational metrics (always available) ────────────────────────────────
    record["success"] = response.success
    record["response_length"] = len((response.answer or "").split())
    record["abstained"] = bool(
        response.success
        and "insufficient evidence" in (response.answer or "").lower()
    )

    # Citation coverage: fraction of evidence tags [E1],[E2]… cited by model
    if response.rag_enabled and response.retrieved_chunks:
        n_chunks = len(response.retrieved_chunks)
        if n_chunks > 0:
            cited_count = sum(
                1 for i in range(1, n_chunks + 1)
                if f"[E{i}]" in (response.answer or "")
            )
            record["citation_coverage"] = round(cited_count / n_chunks, 4)

    if not response.success:
        return record

    reference_answer = reference_answers.get(response.question, "")

    # ── Reference metrics: applicable to ALL modes ────────────────────────────
    if reference_answer:
        # Deterministic metrics — no LLM needed
        ref_metrics = compute_all_reference_metrics(
            prediction=response.answer,
            reference=reference_answer,
            embedding_model=embedding_model,
        )
        record["rouge_l"] = ref_metrics.get("rouge_l")
        record["token_f1"] = ref_metrics.get("token_f1")
        record["semantic_similarity"] = ref_metrics.get("semantic_similarity")
        record["normalized_exact_match"] = ref_metrics.get("normalized_exact_match")

        # LLM dual-judge answer accuracy
        record["answer_accuracy"] = _dual_judge_score_metric(
            judge_client=judge_client,
            prompts=[
                _llm_judge_prompt_answer_accuracy(
                    response.question,
                    response.answer,
                    reference_answer,
                ),
                _llm_judge_prompt_answer_accuracy_reverse(
                    response.question,
                    response.answer,
                    reference_answer,
                ),
            ],
            retries=max_retries,
        )

    # ── Custom RAG grounding metrics: RAG responses only ─────────────────────
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

        answer_claims = _segment_response_units(response.answer)
        if answer_claims:
            claim_labels = _classify_claims_against_context(
                judge_client=judge_client,
                question=response.question,
                retrieved_contexts=retrieved_contexts,
                claims=answer_claims,
                retries=max_retries,
            )
            claim_metrics = _compute_claim_ratios(claim_labels, len(answer_claims))
            if claim_metrics is not None:
                # Map old names → new canonical names
                record["custom_faithfulness"] = claim_metrics.get("faithfulness")
                record["unsupported_claim_ratio"] = claim_metrics.get("unsupported_claim_ratio")
                record["contradicted_claim_ratio"] = claim_metrics.get("contradicted_claim_ratio")

        record["custom_response_groundedness"] = _dual_judge_metric(
            judge_client=judge_client,
            prompts=[
                _llm_judge_prompt_response_groundedness(
                    response.question,
                    response.answer,
                    retrieved_contexts,
                ),
                _llm_judge_prompt_response_groundedness_reverse(
                    response.question,
                    response.answer,
                    retrieved_contexts,
                ),
            ],
            allowed_values={0, 1, 2},
            rating_scale=2,
            retries=max_retries,
        )

        if reference_answer:
            record["custom_context_precision"] = _evaluate_context_precision(
                judge_client=judge_client,
                question=response.question,
                reference_answer=reference_answer,
                retrieved_contexts=retrieved_contexts,
                retries=max_retries,
            )
            record["custom_context_recall"] = _evaluate_context_recall(
                judge_client=judge_client,
                question=response.question,
                reference_answer=reference_answer,
                retrieved_contexts=retrieved_contexts,
                retries=max_retries,
            )

    # ── Composite scores — strictly mode-specific ─────────────────────────────
    # answer_quality_score: cross-mode valid (reference metrics only)
    record["answer_quality_score"] = _compute_answer_quality_score(record)

    # rag_composite_score: RAG only — do NOT set this on no-RAG responses
    if response.rag_enabled and response.retrieved_chunks:
        record["rag_composite_score"] = _compute_rag_composite_score(record)
    else:
        record["rag_composite_score"] = None  # explicitly null for no-RAG

    # Legacy field for backward compat callers
    record["overall_metric_score_pct"] = record["rag_composite_score"]

    return record


def run_evaluation_metrics(
    responses: list[ModelResponse],
    reference_answers: dict[str, str],
    judge_provider: str,
    judge_model: str,
    max_retries: int = 2,
    context_result_limit: int = 4,
    context_char_budget: int = 2600,
    compact_context: bool = True,
    parallelism: int = 1,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    progress_callback: Any | None = None,
    judge_base_url: str | None = None,
    judge_api_key_env: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all responses with mode-aware metrics.

    Returns:
        (per_run_df, summary_df)

    per_run_df has one row per (run_name, question) with all metric columns.
    summary_df has one row per run_name with mean metrics.

    IMPORTANT: The summary_df contains two separate composite scores:
      - answer_quality_score: valid for cross-mode comparison (reference only)
      - rag_composite_score:  RAG mode only, null for no-RAG rows
    Do NOT sort by rag_composite_score across both modes.
    Use answer_quality_score for cross-mode leaderboards.
    """
    total = len(responses)

    def score_one(response: ModelResponse) -> dict[str, Any]:
        judge_client = _thread_local_judge_client(
            judge_provider, judge_model,
            base_url=judge_base_url, api_key_env=judge_api_key_env,
        )
        return _score_response_metrics(
            response=response,
            reference_answers=reference_answers,
            judge_client=judge_client,
            max_retries=max_retries,
            context_result_limit=context_result_limit,
            context_char_budget=context_char_budget,
            compact_context=compact_context,
            embedding_model=embedding_model,
        )

    per_run_records: list[dict[str, Any]]
    if parallelism <= 1 or total <= 1:
        judge_client = _make_judge_client(
            judge_provider, judge_model,
            base_url=judge_base_url, api_key_env=judge_api_key_env,
        )
        _THREAD_LOCAL_STATE.judge_clients = {(judge_provider, judge_model): judge_client}
        per_run_records = []
        for index, response in enumerate(responses, start=1):
            per_run_records.append(
                _score_response_metrics(
                    response=response,
                    reference_answers=reference_answers,
                    judge_client=judge_client,
                    max_retries=max_retries,
                    context_result_limit=context_result_limit,
                    context_char_budget=context_char_budget,
                    compact_context=compact_context,
                    embedding_model=embedding_model,
                )
            )
            if progress_callback is not None:
                progress_callback(index, total, response)
    else:
        records_by_index: list[dict[str, Any] | None] = [None] * total
        completed = 0
        with ThreadPoolExecutor(max_workers=max(1, parallelism)) as executor:
            future_to_index = {
                executor.submit(score_one, response): index
                for index, response in enumerate(responses)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                records_by_index[index] = future.result()
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total, responses[index])
        per_run_records = [record for record in records_by_index if record is not None]

    # Build DataFrame with all canonical metric columns
    extra_cols = ["rag_enabled", "answer_quality_score", "rag_composite_score"]
    all_cols = ["run_name", "question", *ALL_METRIC_NAMES, *extra_cols]
    per_run_df = pd.DataFrame(per_run_records)
    # Ensure all expected columns exist
    for col in all_cols:
        if col not in per_run_df.columns:
            per_run_df[col] = None
    per_run_df = per_run_df[[c for c in all_cols if c in per_run_df.columns]]

    numeric_cols = [c for c in per_run_df.columns if c not in {"run_name", "question", "rag_enabled"}]
    for col in numeric_cols:
        per_run_df[col] = pd.to_numeric(per_run_df[col], errors="coerce")

    # Summary: mean per run_name; sort by answer_quality_score for cross-mode validity
    summary_df = (
        per_run_df.groupby("run_name", dropna=False)
        .agg({col: "mean" for col in numeric_cols if col in per_run_df.columns})
        .reset_index()
        .sort_values(by="answer_quality_score", ascending=False, na_position="last")
    )
    return per_run_df, summary_df


def export_results(
    output_dir: Path,
    responses: list[ModelResponse],
    retrievals: dict[str, list[Any]],
    references: dict[str, str],
    reference_sources: dict[str, str],
    metrics_per_run: pd.DataFrame | None = None,
    metrics_summary: pd.DataFrame | None = None,
    filename_prefix: str = "comparison_report",
    single_sheet_only: bool = False,
    **legacy_kwargs: Any,
) -> Path:
    if "ragas_per_run" in legacy_kwargs and metrics_per_run is None:
        metrics_per_run = legacy_kwargs.pop("ragas_per_run")
    if "ragas_summary" in legacy_kwargs and metrics_summary is None:
        metrics_summary = legacy_kwargs.pop("ragas_summary")
    if legacy_kwargs:
        unknown = ", ".join(sorted(legacy_kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%S")
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

    comparison_df = _build_comparison_sheet(response_rows, metrics_per_run)
    single_sheet_df = _build_single_sheet_report(response_rows, metrics_per_run)

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
            if metrics_per_run is not None:
                metrics_per_run.to_excel(writer, sheet_name="metrics_per_question", index=False)
            if metrics_summary is not None:
                metrics_summary.to_excel(writer, sheet_name="metrics_summary", index=False)

    return workbook_path


def export_metrics_workbook(
    output_dir: Path,
    metrics_per_run: pd.DataFrame | None = None,
    metrics_summary: pd.DataFrame | None = None,
    filename_prefix: str = "metrics_report",
    **legacy_kwargs: Any,
) -> Path | None:
    if "ragas_per_run" in legacy_kwargs and metrics_per_run is None:
        metrics_per_run = legacy_kwargs.pop("ragas_per_run")
    if "ragas_summary" in legacy_kwargs and metrics_summary is None:
        metrics_summary = legacy_kwargs.pop("ragas_summary")
    if legacy_kwargs:
        unknown = ", ".join(sorted(legacy_kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}")

    if metrics_per_run is None and metrics_summary is None:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%S")
    workbook_path = output_dir / f"{filename_prefix}_{timestamp}.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        if metrics_per_run is not None:
            metrics_per_run.to_excel(writer, sheet_name="metrics_per_question", index=False)
        if metrics_summary is not None:
            metrics_summary.to_excel(writer, sheet_name="metrics_summary", index=False)
    return workbook_path


def export_single_model_report(
    output_dir: Path,
    responses: list[ModelResponse],
    references: dict[str, str],
    reference_sources: dict[str, str],
    judge_provider: str,
    judge_model: str,
    max_retries: int = 2,
    context_result_limit: int = 4,
    context_char_budget: int = 2600,
    compact_context: bool = True,
    metrics_per_run: pd.DataFrame | None = None,
    filename_prefix: str | None = None,
    parallelism: int = 1,
    progress_callback: Any | None = None,
    judge_base_url: str | None = None,
    judge_api_key_env: str | None = None,
) -> Path:
    if not responses:
        raise ValueError("Cannot export a single-model report with no responses.")

    run_names = sorted({response.run_name for response in responses})
    if len(run_names) != 1:
        joined = ", ".join(run_names)
        raise ValueError(
            "Single-model report requires exactly one run. "
            f"Received: {joined}"
        )

    run_name = run_names[0]
    metric_lookup = _metric_lookup(metrics_per_run)
    needs_judge = any(response.rag_enabled and response.retrieved_chunks for response in responses)

    def build_one(response: ModelResponse) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        judge_client = None
        if needs_judge and response.rag_enabled and response.retrieved_chunks:
            judge_client = _thread_local_judge_client(
                judge_provider, judge_model,
                base_url=judge_base_url, api_key_env=judge_api_key_env,
            )
        attribution = attribute_response_sources(
            response,
            judge_client=judge_client,
            context_result_limit=context_result_limit,
            context_char_budget=context_char_budget,
            compact_context=compact_context,
            max_retries=max_retries,
        )
        metrics = metric_lookup.get((response.question, response.run_name), {})
        report_row: dict[str, Any] = {
            "Question": response.question,
            "reference_answer": references.get(response.question, ""),
            "reference_source": reference_sources.get(response.question, ""),
            "overall_response": attribution["overall_response"],
            "knowledge_source_data": attribution["knowledge_source_data"],
            "pretrained_data": attribution["pretrained_data"],
        }
        for metric_name in REPORT_METRIC_NAMES:
            report_row[metric_name] = metrics.get(metric_name)

        attribution_entries = [
            {
                "question": response.question,
                "run_name": response.run_name,
                "unit_id": unit["unit_id"],
                "label": unit["label"],
                "unit_text": unit["unit_text"],
            }
            for unit in attribution["attribution_units"]
        ]
        return report_row, attribution_entries

    report_rows: list[dict[str, Any]]
    attribution_rows: list[dict[str, Any]]
    if parallelism <= 1 or len(responses) <= 1:
        if needs_judge:
            judge_client = _make_judge_client(
                judge_provider, judge_model,
                base_url=judge_base_url, api_key_env=judge_api_key_env,
            )
            _THREAD_LOCAL_STATE.judge_clients = {(judge_provider, judge_model): judge_client}
        report_rows = []
        attribution_rows = []
        for index, response in enumerate(responses, start=1):
            report_row, attribution_entries = build_one(response)
            report_rows.append(report_row)
            attribution_rows.extend(attribution_entries)
            if progress_callback is not None:
                progress_callback(index, len(responses), response)
    else:
        ordered_reports: list[dict[str, Any] | None] = [None] * len(responses)
        ordered_attribution: list[list[dict[str, Any]] | None] = [None] * len(responses)
        completed = 0
        with ThreadPoolExecutor(max_workers=max(1, parallelism)) as executor:
            future_to_index = {
                executor.submit(build_one, response): index
                for index, response in enumerate(responses)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                report_row, attribution_entries = future.result()
                ordered_reports[index] = report_row
                ordered_attribution[index] = attribution_entries
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, len(responses), responses[index])
        report_rows = [row for row in ordered_reports if row is not None]
        attribution_rows = []
        for entries in ordered_attribution:
            if entries:
                attribution_rows.extend(entries)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%S")
    prefix = filename_prefix or f"{run_name}_single_model_report"
    workbook_path = output_dir / f"{prefix}_{timestamp}.xlsx"

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        pd.DataFrame(report_rows).to_excel(writer, sheet_name="report", index=False)
        pd.DataFrame(attribution_rows).to_excel(
            writer,
            sheet_name="attribution_units",
            index=False,
        )
        if metrics_per_run is not None:
            metrics_per_run.to_excel(writer, sheet_name="metrics_per_question", index=False)
            summary_df = (
                metrics_per_run.groupby("run_name", dropna=False)
                .agg({metric_name: "mean" for metric_name in REPORT_METRIC_NAMES})
                .reset_index()
            )
            summary_df.to_excel(writer, sheet_name="metrics_summary", index=False)

    return workbook_path


def export_response_sets(
    output_dir: Path,
    responses: list[ModelResponse],
    references: dict[str, str],
    reference_sources: dict[str, str],
    metrics_per_run: pd.DataFrame | None = None,
    metrics_summary: pd.DataFrame | None = None,
    **legacy_kwargs: Any,
) -> Path:
    if "ragas_per_run" in legacy_kwargs and metrics_per_run is None:
        metrics_per_run = legacy_kwargs.pop("ragas_per_run")
    if "ragas_summary" in legacy_kwargs and metrics_summary is None:
        metrics_summary = legacy_kwargs.pop("ragas_summary")
    if legacy_kwargs:
        unknown = ", ".join(sorted(legacy_kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir

    metric_lookup = _metric_lookup(metrics_per_run)
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
    single_sheet_df = _build_single_sheet_report(all_rows, metrics_per_run)
    with pd.ExcelWriter(run_dir / "all_runs_single_sheet.xlsx", engine="openpyxl") as writer:
        single_sheet_df.to_excel(writer, sheet_name="all_in_one", index=False)
    if metrics_per_run is not None or metrics_summary is not None:
        with pd.ExcelWriter(run_dir / "all_runs_metrics.xlsx", engine="openpyxl") as writer:
            if metrics_per_run is not None:
                metrics_per_run.to_excel(writer, sheet_name="metrics_per_question", index=False)
            if metrics_summary is not None:
                metrics_summary.to_excel(writer, sheet_name="metrics_summary", index=False)

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
    metrics_per_run: pd.DataFrame | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    if metrics_per_run is None or metrics_per_run.empty:
        return {}
    return {
        (row["question"], row["run_name"]): row
        for row in metrics_per_run.to_dict(orient="records")
    }


def _response_metrics(
    response: ModelResponse,
    metric_lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    metrics = metric_lookup.get((response.question, response.run_name), {})
    return {metric_name: metrics.get(metric_name) for metric_name in REPORT_METRIC_NAMES}


def _build_comparison_sheet(
    response_rows: list[dict[str, Any]],
    metrics_per_run: pd.DataFrame | None,
) -> pd.DataFrame:
    question_order = list(dict.fromkeys(row["question"] for row in response_rows))
    run_order = list(dict.fromkeys(row["run_name"] for row in response_rows))

    response_lookup = {
        (row["question"], row["run_name"]): row
        for row in response_rows
    }

    metric_names: list[str] = list(REPORT_METRIC_NAMES)
    metric_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if metrics_per_run is not None and not metrics_per_run.empty:
        metric_lookup = _metric_lookup(metrics_per_run)

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
    metrics_per_run: pd.DataFrame | None,
) -> pd.DataFrame:
    question_order = list(dict.fromkeys(row["question"] for row in response_rows))
    run_order = list(dict.fromkeys(row["run_name"] for row in response_rows))

    response_lookup = {
        (row["question"], row["run_name"]): row
        for row in response_rows
    }

    metric_names: list[str] = list(REPORT_METRIC_NAMES)
    metric_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if metrics_per_run is not None and not metrics_per_run.empty:
        metric_lookup = _metric_lookup(metrics_per_run)

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


run_ragas = run_evaluation_metrics
