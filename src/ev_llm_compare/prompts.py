from __future__ import annotations

import re

from .schemas import RetrievalResult



# ──────────────────────────────────────────────────────────────────────────────
# System prompts
#
# RAG_SYSTEM_PROMPT   – used when retrieved workbook evidence is injected.
#                       Evidence-only; no silent fallback to general knowledge.
# NON_RAG_SYSTEM_PROMPT – used when no evidence is provided.
#                         General-knowledge baseline; explicit about limitations.
# ──────────────────────────────────────────────────────────────────────────────

# Legacy alias kept for backward compatibility
SYSTEM_PROMPT = (
    "You answer questions about EV supply chain workbook data.\n"
    "Use ONLY the retrieved evidence to answer. Do not answer from general knowledge.\n"
    "Cite each factual claim with its evidence tag, e.g. [E1], [E2].\n"
    "ALWAYS attempt to answer using whatever relevant evidence is available, even if partial.\n"
    "Only say 'Insufficient evidence' if the evidence contains ZERO relevant information."
)

RAG_SYSTEM_PROMPT = (
    "You answer questions about EV supply chain workbook data.\n"
    "Use ONLY the retrieved evidence provided in the prompt. "
    "Do NOT use general world knowledge, prior training knowledge, or unstated assumptions.\n"
    "Rules:\n"
    "- Cite every factual claim with the evidence tag shown in the context, e.g. [E1], [E2].\n"
    "- Use exact values, company names, counts, locations, roles, and employment figures from the evidence.\n"
    "- ALWAYS attempt to answer using whatever relevant evidence is available, even if partial.\n"
    "- List ALL matching companies and data points found in the evidence — do not omit any.\n"
    "- Provide detailed, comprehensive answers. Include every relevant field: "
    "company name, Updated Location, role, tier, employment, product/service, OEMs.\n"
    "- Only say 'Insufficient evidence' if the evidence contains ZERO relevant information "
    "for the question. Partial evidence should produce a partial answer, not abstention.\n"
    "- Do not ask the user to upload files. Do not mention workbook filenames.\n"
    "- Do not repeat the evidence headers verbatim.\n"
    "- CRITICAL: Always use 'Updated Location' field values (NOT 'Location'). "
    "Updated Location contains the corrected location data.\n"
    "- CRITICAL: When analyzing Employment for Georgia questions, exclude global headcount "
    "outliers (>100,000 employees) as these represent global corporate employment, "
    "not local Georgia facility employment."
)

NON_RAG_SYSTEM_PROMPT = (
    "You answer EV supply chain questions from your general model knowledge.\n"
    "Rules:\n"
    "- Do NOT mention missing files, workbooks, or datasets.\n"
    "- Do NOT provide spreadsheet or filtering instructions unless explicitly asked.\n"
    "- If the question asks for exact dataset-specific facts you cannot verify, "
    "state briefly that the specific data is unavailable from general knowledge, "
    "then give the closest general domain answer without inventing specifics.\n"
    "- Prefer a direct, substantive answer over vague process descriptions.\n"
    "- Do not invent company names, exact counts, or specific locations."
)

FIELD_LABELS = {
    "category": "Category",
    "industry_group": "Industry Group",
    "ev_supply_chain_role": "EV Supply Chain Role",
    "product_service": "Product / Service",
    "primary_oems": "Primary OEMs",
    "location": "Updated Location",
    "primary_facility_type": "Primary Facility Type",
    "employment": "Employment",
    "ev_battery_relevant": "EV / Battery Relevant",
    "supplier_or_affiliation_type": "Supplier or Affiliation Type",
    "classification_method": "Classification Method",
}


def compact_context_segments(
    question: str,
    results: list[RetrievalResult],
    max_results: int = 5,
    max_chars: int = 4200,
) -> list[str]:
    if not results:
        return ["No retrieved evidence."]

    normalized_question = _normalize_question(question)
    effective_max_results = _effective_compact_result_limit(
        normalized_question,
        max_results,
    )
    selected = _select_compact_results(
        normalized_question,
        results,
        effective_max_results,
    )
    blocks: list[str] = []
    char_budget = max(600, max_chars)
    reserved_summary_budget = int(char_budget * 0.45)

    for index, result in enumerate(selected, start=1):
        remaining = char_budget - sum(len(block) for block in blocks)
        if remaining < 180:
            break
        if index == 1 and _chunk_type(result) == "structured_match_summary":
            if len(selected) == 1:
                summary_budget = remaining
            else:
                summary_budget = min(remaining, reserved_summary_budget)
            block = _render_structured_summary(
                result,
                summary_budget,
                index=index,
            )
        else:
            block = _render_compact_block(
                normalized_question,
                result,
                remaining,
                index=index,
            )
        if not block:
            continue
        if len(block) > remaining:
            block = block[: max(0, remaining - 3)].rstrip() + "..."
        blocks.append(block)
        if len(blocks) >= effective_max_results:
            break

    return blocks or ["No retrieved evidence."]


def format_context(
    results: list[RetrievalResult],
    question: str | None = None,
    max_results: int = 5,
    max_chars: int = 4200,
    compact: bool = True,
) -> str:
    """Format retrieved evidence blocks for injection into the RAG prompt.

    Each block is prefixed with a citation tag [E1], [E2], … so that the
    model can cite specific evidence in its answer, enabling citation
    coverage tracking during evaluation.
    """
    if not results:
        return "No retrieved evidence."

    if question and compact:
        raw_blocks = compact_context_segments(
            question,
            results,
            max_results=max_results,
            max_chars=max_chars,
        )
        # Prepend citation tags to each compact block
        tagged: list[str] = []
        for idx, block in enumerate(raw_blocks, start=1):
            tagged.append(f"[E{idx}] {block}")
        return "\n\n".join(tagged)

    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        company = result.metadata.get("company") or "n/a"
        chunk_type = result.metadata.get("chunk_type") or "retrieved_chunk"
        source = (
            f"{result.metadata.get('source_file')}::"
            f"{result.metadata.get('sheet_name')}"
        )
        row_number = result.metadata.get("row_number")
        header = (
            f"[E{index}] type={chunk_type} company={company} source={source}"
            + (f" row={row_number}" if row_number else "")
        )
        blocks.append(f"{header}\n{result.text}")
    return "\n\n".join(blocks)


def build_rag_prompt(question: str, context: str) -> str:
    """Construct the RAG user prompt.

    Evidence blocks are pre-tagged [E1], [E2], … by format_context().
    The model is required to cite every factual claim with these tags.
    If the evidence does not support the question, the model must abstain
    with the exact phrase 'Insufficient evidence:'.
    """
    return (
        "Use ONLY the retrieved evidence below. Do not use general knowledge.\n\n"
        f"Evidence:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- Cite every factual claim with the evidence tag, e.g. [E1] or [E1][E2].\n"
        "- Use exact values from the evidence: company names, counts, locations, roles, employment numbers.\n"
        "- If the structured summary already states the final answer, copy it directly; do not recompute.\n"
        "- Include ALL evidence-supported matches; do not omit any company or data point.\n"
        "- For each company or entity, include as many fields as the evidence provides: "
        "company name, Updated Location, EV Supply Chain Role, Supplier/Affiliation Type, "
        "Employment, Product/Service, Primary OEMs, Primary Facility Type.\n"
        "- Group results when the question asks for grouping.\n"
        "- If the evidence has SOME relevant information, give a partial answer using it. "
        "Only abstain if the evidence contains ZERO relevant data.\n"
        "- Do not invent values. Do not pad with general domain knowledge.\n"
        "- Give a thorough, detailed answer. Do not give one-line answers when the evidence "
        "supports a longer response.\n"
        "- Start directly with the answer; do not repeat question text.\n"
        "- IMPORTANT: Always use 'Updated Location' field values (NOT the 'Location' field). "
        "The Updated Location contains the corrected/most recent location data.\n"
        "- IMPORTANT: When analyzing Employment data for Georgia-based questions, "
        "exclude global headcount outliers (>100,000 employees). Companies like WIKA USA (250k) "
        "and Yazaki (230k) report global corporate employment, not local Georgia facility employment. "
        "For local employment analysis, only consider values under 100,000.\n\n"
        "Answer:"
    )


def build_non_rag_prompt(question: str) -> str:
    """Construct the no-RAG user prompt.

    The model answers from general knowledge only.
    It must not pretend to have workbook data it does not have.
    """
    return (
        "Answer from your general model knowledge.\n"
        "Do not mention missing files, workbooks, or datasets.\n"
        "If the question requires exact dataset-specific facts that you cannot verify, "
        "state briefly that the specific data is unavailable from general knowledge, "
        "then give the closest general EV supply chain domain answer without inventing specifics.\n"
        "Prefer a direct, substantive answer.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def build_reference_prompt(question: str, context: str) -> str:
    return (
        "Create a high-quality reference answer from the workbook evidence.\n"
        "This answer will be used as the ground truth for evaluation.\n\n"
        f"Evidence:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Requirements:\n"
        "- Use only supported facts.\n"
        "- If the evidence does not fully answer the question, state the limitation.\n"
        "- Keep the answer concise but complete.\n\n"
        "Reference answer:"
    )


def _normalize_question(question: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()


def _chunk_type(result: RetrievalResult) -> str:
    return str(result.metadata.get("chunk_type", "") or "retrieved_chunk")


def _select_compact_results(
    normalized_question: str,
    results: list[RetrievalResult],
    max_results: int,
) -> list[RetrievalResult]:
    analytic_question = _is_analytic_question(normalized_question)
    broad_context_required = _needs_broad_context(normalized_question)
    relationship_heavy = _is_relationship_heavy_question(normalized_question)
    grouped_listing_question = (
        "ev supply chain role" in normalized_question
        and "group" in normalized_question
        and any(term in normalized_question for term in {"show all", "list all", "provide all"})
    )
    definition_question = any(
        term in normalized_question for term in {"methodology", "define", "definition", "meaning"}
    )
    has_structured_summary = any(_chunk_type(result) == "structured_match_summary" for result in results)
    summary_result = next(
        (result for result in results if _chunk_type(result) == "structured_match_summary"),
        None,
    )
    if broad_context_required or relationship_heavy:
        result_limit = max_results
    elif (
        (analytic_question or grouped_listing_question)
        and summary_result
        and _summary_is_self_sufficient(summary_result.text)
    ):
        result_limit = max(max_results, 5)
    elif analytic_question and has_structured_summary:
        result_limit = max_results
    else:
        result_limit = max_results
    ranked = sorted(
        results,
        key=lambda result: (_compact_priority(normalized_question, result), result.final_score),
        reverse=True,
    )
    selected: list[RetrievalResult] = []
    seen_rows: set[str] = set()
    seen_companies: set[str] = set()
    note_used = False
    allow_repeated_company_rows = _needs_multi_record_context(normalized_question)

    for result in ranked:
        chunk_type = _chunk_type(result)
        row_key = str(result.metadata.get("row_key", "")).strip()
        company = str(result.metadata.get("company", "")).strip()
        if chunk_type == "note_reference" and not definition_question:
            continue
        if chunk_type == "note_reference" and note_used:
            continue
        if row_key and row_key in seen_rows:
            continue
        if (
            company
            and company in seen_companies
            and chunk_type in {"row_full", "company_profile", "supply_chain_theme", "location_theme"}
            and not allow_repeated_company_rows
        ):
            continue
        selected.append(result)
        if row_key:
            seen_rows.add(row_key)
        if company:
            seen_companies.add(company)
        if chunk_type == "note_reference":
            note_used = True
        if len(selected) >= result_limit:
            break
    return selected


def _needs_broad_context(normalized_question: str) -> bool:
    padded_question = f" {normalized_question} "
    return any(
        term in padded_question
        for term in {
            " all entries ",
            " all companies ",
            " all suppliers ",
            "full set",
            "supplier network",
            "network",
            "connected to each",
            "linked to each",
            "linked suppliers",
            "suppliers linked to",
            "for each oem",
            "for each company",
            "for each location",
            "broken down by",
            "map all",
            "identify all",
            "list all",
            "show all",
            "find all",
            "which companies",
            "which georgia companies",
            "which areas",
            "which county",
            "which counties",
            "how many",
            "every ",
            "each ",
            "classified under",
            "classified as",
            "involved in",
            "produce ",
            "manufacture ",
            "currently ",
        }
    )


def _is_relationship_heavy_question(normalized_question: str) -> bool:
    padded_question = f" {normalized_question} "
    return any(
        term in padded_question
        for term in {
            " connected to ",
            " linked to ",
            " linked suppliers ",
            " supply to ",
            " supplier network ",
            " full set ",
            " for each ",
            " relationship ",
            " connected with ",
        }
    )


def _needs_multi_record_context(normalized_question: str) -> bool:
    if _needs_broad_context(normalized_question) or _is_relationship_heavy_question(normalized_question):
        return True
    return any(
        term in normalized_question
        for term in {
            "locations",
            "each location",
            "primary facility type",
            "multiple times",
            "appear multiple times",
            "county",
            "counties",
            "cities",
            "areas",
            "employment",
            "tier",
            "role",
            "suppliers",
            "companies",
            "battery",
            "thermal",
            "wiring",
            "recycl",
        }
    )


def _effective_compact_result_limit(normalized_question: str, max_results: int) -> int:
    if _needs_broad_context(normalized_question) or _is_relationship_heavy_question(normalized_question):
        return max(max_results, 15)
    return max_results


def _compact_priority(normalized_question: str, result: RetrievalResult) -> float:
    chunk_type = _chunk_type(result)
    if chunk_type == "structured_match_summary":
        return 4.0
    if chunk_type == "structured_row_match":
        return 3.3
    if chunk_type == "note_reference":
        return 3.0 if any(
            term in normalized_question for term in {"methodology", "define", "definition", "meaning"}
        ) else -1.0
    if _is_relationship_heavy_question(normalized_question):
        if chunk_type == "supply_chain_theme":
            return 3.1
        if chunk_type == "location_theme":
            return 2.8
        if chunk_type in {"company_profile", "row_full"}:
            if str(result.metadata.get("primary_oems", "")).strip():
                return 2.7
            return 2.3
    if chunk_type in {"location_theme", "identity_theme", "supply_chain_theme", "product_theme"}:
        return 2.4
    if chunk_type == "company_profile":
        return 2.0
    if chunk_type == "row_full":
        return 1.0
    return 1.4


def _render_structured_summary(result: RetrievalResult, budget: int, index: int) -> str:
    header = _evidence_header(index, result)
    lines = [line.rstrip() for line in result.text.splitlines() if line.strip()]
    kept: list[str] = []
    total = len(header) + 1
    for line in lines:
        projected = total + len(line) + 1
        if projected > budget:
            break
        kept.append(line)
        total = projected
    if not kept:
        kept = [result.text[: max(0, budget - len(header) - 5)].rstrip() + "..."]
    elif len(kept) < len(lines):
        kept.append("...")
    return f"{header}\n" + "\n".join(kept)


def _render_compact_block(
    normalized_question: str,
    result: RetrievalResult,
    budget: int,
    index: int,
) -> str:
    header = _evidence_header(index, result)
    chunk_type = _chunk_type(result)
    if chunk_type == "note_reference":
        body = result.text
    elif _should_use_rich_row_context(normalized_question, result):
        body = str(result.metadata.get("row_summary", "")).strip() or result.text
    else:
        body = _compact_metadata_line(normalized_question, result)
    if not body or _looks_sparse_compact_body(body, result):
        body = str(result.metadata.get("row_summary", "")).strip() or result.text
    allowance = max(120, budget - len(header) - 1)
    if len(body) > allowance:
        body = body[: max(0, allowance - 3)].rstrip() + "..."
    return f"{header}\n{body}"


def _evidence_header(index: int | None, result: RetrievalResult) -> str:
    company = result.metadata.get("company") or "n/a"
    chunk_type = _chunk_type(result)
    source = f"{result.metadata.get('source_file')}::{result.metadata.get('sheet_name')}"
    row_number = result.metadata.get("row_number")
    prefix = f"[Evidence {index}]" if index is not None else "[Evidence]"
    return (
        f"{prefix} type={chunk_type} company={company} source={source}"
        + (f" row={row_number}" if row_number else "")
    )


def _compact_metadata_line(normalized_question: str, result: RetrievalResult) -> str:
    metadata = result.metadata
    company = str(metadata.get("company", "")).strip()
    chunk_type = _chunk_type(result)
    requested_fields = _requested_fields(normalized_question, chunk_type)
    parts: list[str] = []
    if company:
        parts.append(f"Company: {company}")
    for field_name in requested_fields:
        value = str(metadata.get(field_name, "")).strip()
        if not value:
            continue
        parts.append(f"{FIELD_LABELS[field_name]}: {value}")
    return " | ".join(dict.fromkeys(parts))


def _should_use_rich_row_context(normalized_question: str, result: RetrievalResult) -> bool:
    chunk_type = _chunk_type(result)
    if chunk_type == "structured_row_match":
        return True
    if _is_relationship_heavy_question(normalized_question):
        return chunk_type in {
            "row_full",
            "company_profile",
            "supply_chain_theme",
            "location_theme",
            "structured_row_match",
        }
    if _needs_multi_record_context(normalized_question):
        return chunk_type in {"row_full", "company_profile", "location_theme"}
    return False


def _looks_sparse_compact_body(body: str, result: RetrievalResult) -> bool:
    normalized_body = " ".join(body.split()).strip().lower()
    if not normalized_body:
        return True
    company = str(result.metadata.get("company", "")).strip().lower()
    if company and normalized_body == f"company: {company}":
        return True
    parts = [part.strip() for part in body.split("|") if part.strip()]
    informative_parts = [
        part for part in parts if not part.lower().startswith("company:")
    ]
    return len(informative_parts) == 0


def _requested_fields(normalized_question: str, chunk_type: str) -> list[str]:
    fields: list[str] = []
    if "category" in normalized_question:
        fields.append("category")
    if "industry group" in normalized_question:
        fields.append("industry_group")
    if "ev supply chain role" in normalized_question or "role" in normalized_question:
        fields.append("ev_supply_chain_role")
    if "product / service" in normalized_question or "product service" in normalized_question:
        fields.append("product_service")
    if "primary oem" in normalized_question or "linked to" in normalized_question:
        fields.append("primary_oems")
    if any(term in normalized_question for term in {"location", "county", "city"}):
        fields.append("location")
    if "primary facility type" in normalized_question or "facility" in normalized_question:
        fields.append("primary_facility_type")
    if "employment" in normalized_question:
        fields.append("employment")
    if "ev / battery relevant" in normalized_question or "ev battery relevant" in normalized_question:
        fields.append("ev_battery_relevant")
    if "supplier or affiliation type" in normalized_question:
        fields.append("supplier_or_affiliation_type")
    if "classification method" in normalized_question:
        fields.append("classification_method")
    if _is_relationship_heavy_question(normalized_question):
        fields.extend(
            [
                "category",
                "ev_supply_chain_role",
                "primary_oems",
                "location",
                "primary_facility_type",
                "employment",
                "supplier_or_affiliation_type",
            ]
        )
    # Always include core fields so the model has enough information to answer
    core_fields = [
        "ev_supply_chain_role",
        "location",
        "employment",
        "supplier_or_affiliation_type",
        "product_service",
        "primary_oems",
        "primary_facility_type",
        "ev_battery_relevant",
    ]
    for f in core_fields:
        if f not in fields:
            fields.append(f)
    if chunk_type == "identity_theme":
        for f in ["category", "industry_group", "classification_method"]:
            if f not in fields:
                fields.append(f)
    return list(dict.fromkeys(fields))


def _is_analytic_question(normalized_question: str) -> bool:
    return any(
        term in normalized_question
        for term in {
            "count",
            "how many",
            "total",
            "average",
            "median",
            "highest",
            "lowest",
            "top",
            "bottom",
            "range",
            "represented",
            "summary",
            "summarize",
            "compare",
            "only one",
            "at least one company",
            "matching companies",
            "list all companies",
            "show all",
            "group by",
            "group them by",
        }
    )


def _summary_is_self_sufficient(summary_text: str) -> bool:
    normalized = summary_text.lower()
    return any(
        marker in normalized
        for marker in {
            "counts by ",
            "total employment by ",
            "average employment by ",
            "industry groups represented",
            "public oem footprint / supplier listing",
            "cities with both tier 1 and tier 2/3 companies",
            "companies appearing multiple times",
            "ev / battery relevant groups",
            "ev / battery relevant = yes companies",
            "ev supply chain roles with ev / battery relevant = yes companies",
            "matching product / service entries",
            "primary facility type matches",
            "employment-ranked companies",
            "linked companies:",
            "grouped by ev supply chain role",
        }
    )
