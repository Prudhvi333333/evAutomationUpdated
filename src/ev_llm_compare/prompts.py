from __future__ import annotations

import json
import re
from typing import Any, Callable

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
    "company name, location, role, tier, employment, product/service, OEMs.\n"
    "- Only say 'Insufficient evidence' if the evidence contains ZERO relevant information "
    "for the question. Partial evidence should produce a partial answer, not abstention.\n"
    "- Do not ask the user to upload files. Do not mention workbook filenames.\n"
    "- Do not repeat the evidence headers verbatim."
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


def _detect_answer_mode(question: str) -> str:
    """Route question to one of 3 generation modes.

    Returns one of:
        'structured_list'  – exhaustive listing / counting questions
        'comparison'       – compare across categories, states, roles
        'freeform'         – definitions, relationships, open-ended explanations
    """
    q = question.lower()

    # Structured list: questions asking for a complete enumeration or count
    structured_list_signals = [
        "list all", "show all", "identify all", "map all",
        "list every", "show every", "identify every", "find every",
        "full list", "complete list", "how many", "what is the count",
        "which companies", "which georgia companies",
        "classified under", "classified as",
        "show all tier", "list tier",
    ]
    if any(s in q for s in structured_list_signals):
        return "structured_list"

    # Comparison: questions that ask to compare across groups
    comparison_signals = [
        "compare", "versus", " vs ", "difference between",
        "for each category", "for each role", "for each tier",
        "by category", "by role", "by tier", "by state",
        "breakdown", "broken down",
        "highest", "lowest", "top ", "bottom ",
        "average", "total employment by",
    ]
    if any(s in q for s in comparison_signals):
        return "comparison"

    return "freeform"


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic renderers
# Code extracts rows from structured_row_match metadata, computes metrics,
# and builds a pre-verified data block. LLM only does final wording.
# ──────────────────────────────────────────────────────────────────────────────

def _extract_structured_rows(retrieved_chunks: list[RetrievalResult]) -> list[dict]:
    """Pull structured_row_match metadata from retrieved chunks.

    Returns a deduplicated list of row dicts ordered by company name.
    Each dict contains all available metadata fields.
    """
    seen_companies: set[str] = set()
    rows: list[dict] = []
    for chunk in retrieved_chunks:
        if str(chunk.metadata.get("chunk_type", "")) != "structured_row_match":
            continue
        company = str(chunk.metadata.get("company", "")).strip()
        if not company or company in seen_companies:
            continue
        seen_companies.add(company)
        rows.append({
            "company":             company,
            "category":            str(chunk.metadata.get("category", "") or ""),
            "ev_supply_chain_role":str(chunk.metadata.get("ev_supply_chain_role", "") or ""),
            "employment":          str(chunk.metadata.get("employment", "") or ""),
            "location":            str(chunk.metadata.get("location", "") or ""),
            "product_service":     str(chunk.metadata.get("product_service", "") or ""),
            "primary_oems":        str(chunk.metadata.get("primary_oems", "") or ""),
            "primary_facility_type": str(chunk.metadata.get("primary_facility_type", "") or ""),
            "supplier_type":       str(chunk.metadata.get("supplier_or_affiliation_type", "") or ""),
            "ev_battery_relevant": str(chunk.metadata.get("ev_battery_relevant", "") or ""),
            "classification_method": str(chunk.metadata.get("classification_method", "") or ""),
            "state":               str(chunk.metadata.get("state", "") or ""),
        })
    rows.sort(key=lambda r: r["company"].lower())
    return rows


def _fields_requested(question: str) -> list[str]:
    """Detect which fields the question asks for, to keep the table focused."""
    q = question.lower()
    fields: list[str] = ["company"]  # always included

    if any(k in q for k in ["role", "supply chain role"]):
        fields.append("ev_supply_chain_role")
    if any(k in q for k in ["product", "service", "product / service"]):
        fields.append("product_service")
    if any(k in q for k in ["employment", "employee", "employees", "size"]):
        fields.append("employment")
    if any(k in q for k in ["location", "county", "city", "state", "where"]):
        fields.append("location")
    if any(k in q for k in ["oem", "linked to", "supply to", "supplier"]):
        fields.append("primary_oems")
    if any(k in q for k in ["tier", "category"]):
        fields.append("category")
    if any(k in q for k in ["facility", "facility type"]):
        fields.append("primary_facility_type")
    if any(k in q for k in ["affiliation", "supplier type", "affiliation type"]):
        fields.append("supplier_type")
    if any(k in q for k in ["ev / battery", "ev battery", "battery relevant"]):
        fields.append("ev_battery_relevant")
    if any(k in q for k in ["classification", "classified"]):
        fields.append("classification_method")

    # If no specific fields beyond company detected, include role as default
    if fields == ["company"]:
        fields.append("ev_supply_chain_role")
    return fields


def _render_row_table(rows: list[dict], fields: list[str]) -> str:
    """Render a tab-aligned table of rows for the requested fields."""
    if not rows:
        return "(no matching rows found in structured data)"
    lines: list[str] = []
    for row in rows:
        parts: list[str] = []
        for f in fields:
            label = f.replace("_", " ").replace("ev supply chain role", "Role") \
                      .replace("product service", "Product/Service") \
                      .replace("primary oems", "Primary OEMs") \
                      .replace("primary facility type", "Facility Type") \
                      .replace("supplier type", "Affiliation Type") \
                      .replace("ev battery relevant", "EV/Battery Relevant") \
                      .replace("classification method", "Classification") \
                      .title()
            val = row.get(f, "")
            if val:
                parts.append(f"{label}: {val}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _detect_group_field(question: str) -> str | None:
    """For comparison questions, detect what field to group by."""
    q = question.lower()
    if any(k in q for k in ["by role", "each role", "per role", "supply chain role"]):
        return "ev_supply_chain_role"
    if any(k in q for k in ["by category", "each category", "per category", "tier"]):
        return "category"
    if any(k in q for k in ["by state", "each state", "per state"]):
        return "state"
    if any(k in q for k in ["by location", "each location", "per location", "county"]):
        return "location"
    if any(k in q for k in ["by oem", "each oem", "per oem"]):
        return "primary_oems"
    if any(k in q for k in ["ev / battery", "ev battery", "battery relevant"]):
        return "ev_battery_relevant"
    if any(k in q for k in ["facility type", "facility"]):
        return "primary_facility_type"
    return None


def build_deterministic_list_context(
    question: str,
    retrieved_chunks: list[RetrievalResult],
) -> tuple[str, int]:
    """Build a code-computed data block for structured_list questions.

    Returns:
        (data_block, row_count)

    The data_block is passed to build_structured_list_prompt() instead of
    the raw mixed-chunk context string.
    row_count is the authoritative count computed from metadata, not from LLM.
    """
    rows = _extract_structured_rows(retrieved_chunks)
    fields = _fields_requested(question)
    # Always include ev_supply_chain_role in the rendered context so the
    # extraction LLM can verify which companies match a role-based filter.
    render_fields = fields[:]
    if "ev_supply_chain_role" not in render_fields:
        render_fields.insert(1, "ev_supply_chain_role")  # after company
    table = _render_row_table(rows, render_fields)
    count = len(rows)
    block = (
        f"VERIFIED DATA — {count} matching {'company' if count == 1 else 'companies'} "
        f"extracted from workbook metadata (do not add or remove any row):\n\n"
        f"{table}"
    )
    return block, count


def build_deterministic_comparison_context(
    question: str,
    retrieved_chunks: list[RetrievalResult],
) -> str:
    """Build a code-computed grouped metrics block for comparison questions.

    Groups rows by the detected field and computes count + total employment
    per group in Python — no LLM arithmetic needed.
    """
    rows = _extract_structured_rows(retrieved_chunks)
    group_field = _detect_group_field(question)
    fields = _fields_requested(question)

    if not group_field or not rows:
        # Fall back to flat table if grouping can't be determined
        table = _render_row_table(rows, fields)
        return f"VERIFIED DATA — {len(rows)} rows:\n\n{table}"

    # Group in code
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = row.get(group_field, "") or "Unspecified"
        groups[key].append(row)

    lines: list[str] = [
        f"VERIFIED GROUPED DATA — grouped by {group_field.replace('_', ' ')} "
        f"(counts and totals computed from workbook metadata):\n"
    ]
    for group_key in sorted(groups):
        group_rows = groups[group_key]
        count = len(group_rows)
        # Compute total employment if numeric values available
        emp_values = []
        for r in group_rows:
            try:
                emp_values.append(int(float(r["employment"].replace(",", ""))))
            except (ValueError, AttributeError):
                pass
        emp_line = f" | Total Employment: {sum(emp_values):,}" if emp_values else ""
        lines.append(f"\n{group_key} — Count: {count}{emp_line}")
        for r in group_rows:
            company_fields = [f for f in fields if f != "company"]
            parts = [f"Company: {r['company']}"]
            for f in company_fields:
                v = r.get(f, "")
                if v:
                    label = f.replace("_", " ").title()
                    parts.append(f"{label}: {v}")
            lines.append("  " + " | ".join(parts))

    return "\n".join(lines)


def build_rag_prompt(
    question: str,
    context: str,
    retrieved_chunks: list[RetrievalResult] | None = None,
) -> str:
    """Route to the correct specialised prompt builder based on question intent.

    If retrieved_chunks is provided, structured_list and comparison questions
    use the deterministic renderer — code computes the row set and metrics,
    LLM only does final formatting.

    If retrieved_chunks is None (legacy call), falls back to prompt-only path.
    """
    mode = _detect_answer_mode(question)

    if mode == "structured_list":
        if retrieved_chunks is not None:
            data_block, row_count = build_deterministic_list_context(question, retrieved_chunks)
            return build_structured_list_prompt(question, data_block, row_count=row_count)
        return build_structured_list_prompt(question, context)

    if mode == "comparison":
        if retrieved_chunks is not None:
            data_block = build_deterministic_comparison_context(question, retrieved_chunks)
            return build_comparison_prompt(question, data_block)
        return build_comparison_prompt(question, context)

    return build_freeform_prompt(question, context)


def build_structured_list_prompt(question: str, context: str, row_count: int = 0) -> str:
    """Prompt for exhaustive list / count questions.

    When called from the deterministic path, context is the pre-computed
    verified data block (not raw mixed chunks), and row_count is the
    authoritative count already computed in Python.
    """
    count_instruction = (
        f"- Start your answer with exactly this sentence: "
        f"'There are {row_count} {'company' if row_count == 1 else 'companies'} matching this query.'\n"
        if row_count > 0
        else
        "- Start your answer with a count sentence: "
        "'There are N companies/suppliers/entries matching this query.'\n"
    )
    return (
        "Use ONLY the verified data below. Do not use general knowledge.\n\n"
        f"Data:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- The data block above is the COMPLETE and VERIFIED set of matching rows "
        "extracted directly from the workbook. It is authoritative.\n"
        "- DO NOT add any company that is not in the data block.\n"
        "- DO NOT remove any company that is in the data block.\n"
        f"{count_instruction}"
        "- Then list EVERY company on its own line in this exact format:\n"
        "  CompanyName | Role: <role> | <other requested fields>\n"
        "- Include ONLY the fields the question asks for.\n"
        "- If the question asks to group by role or category, group the lines "
        "under bold role/category headers before listing.\n"
        "- Do not add commentary or extra fields not requested.\n\n"
        "Answer:"
    )


def build_comparison_prompt(question: str, context: str) -> str:
    """Prompt for comparison / aggregation questions.

    When called from the deterministic path, context is the pre-computed
    grouped metrics block with counts and totals already calculated in Python.
    """
    return (
        "Use ONLY the verified data below. Do not use general knowledge.\n\n"
        f"Data:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- The data block above contains groups with counts and totals already "
        "computed from the workbook. These numbers are authoritative — do not "
        "recompute or change them.\n"
        "- Format the data as a clean comparison answer:\n"
        "  Group A: count=N, total_employment=X — companies: ...\n"
        "  Group B: count=N, total_employment=X — companies: ...\n"
        "- Do not invent metrics. Use exactly the numbers in the data block.\n"
        "- If a metric is not in the data block, say 'Not available'.\n"
        "- End with a one-sentence summary identifying the largest/highest group "
        "if the question asks for it.\n\n"
        "Answer:"
    )


def build_freeform_prompt(question: str, context: str) -> str:
    """Prompt for definitions, relationship explanations, and open-ended questions.

    Strategy:
    - Allow natural language prose.
    - Still require citations.
    - Faithful to evidence, no hallucination.
    """
    return (
        "Use ONLY the retrieved evidence below. Do not use general knowledge.\n\n"
        f"Evidence:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- Cite every factual claim with the evidence tag, e.g. [E1] or [E1][E2].\n"
        "- Use exact values from the evidence: company names, counts, locations, "
        "roles, employment numbers.\n"
        "- Include ALL evidence-supported matches; do not omit any company or data point.\n"
        "- If the evidence has SOME relevant information, give a partial answer. "
        "Only say 'Insufficient evidence' if the evidence contains ZERO relevant data.\n"
        "- Do not invent values. Do not pad with general domain knowledge.\n"
        "- Start directly with the answer; do not repeat the question.\n\n"
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


# ──────────────────────────────────────────────────────────────────────────────
# Two-pass generation
# Pass 1: LLM extracts to strict JSON schema (constrained, no hallucination)
# Pass 2: LLM verbalizes JSON into final user-facing answer (natural language)
# ──────────────────────────────────────────────────────────────────────────────

_EXTRACTION_SYSTEM_PROMPT = (
    "You are a precise data extraction assistant. "
    "You extract structured data from evidence and return ONLY valid JSON. "
    "Do not add any text before or after the JSON. "
    "Do not invent values. Only extract what is explicitly stated in the evidence."
)

_VERBALIZATION_SYSTEM_PROMPT = (
    "You are a clear technical writer. "
    "You convert structured JSON data into a well-formatted natural language answer. "
    "Use only the data provided in the JSON. Do not add, remove, or change any values."
)


def _build_extraction_schema_for_list(fields: list[str]) -> str:
    """Build the JSON schema description for pass 1 extraction."""
    item_fields = []
    for f in fields:
        if f == "company":
            item_fields.append('    "company": "exact company name from evidence"')
        elif f == "ev_supply_chain_role":
            item_fields.append('    "role": "EV Supply Chain Role from evidence"')
        elif f == "product_service":
            item_fields.append('    "product_service": "Product / Service from evidence"')
        elif f == "employment":
            item_fields.append('    "employment": "Employment value from evidence (number or string)"')
        elif f == "location":
            item_fields.append('    "location": "Updated Location from evidence"')
        elif f == "primary_oems":
            item_fields.append('    "primary_oems": "Primary OEMs from evidence"')
        elif f == "category":
            item_fields.append('    "category": "Category / Tier from evidence"')
        elif f == "primary_facility_type":
            item_fields.append('    "facility_type": "Primary Facility Type from evidence"')
        elif f == "supplier_type":
            item_fields.append('    "affiliation_type": "Supplier or Affiliation Type from evidence"')
        elif f == "ev_battery_relevant":
            item_fields.append('    "ev_battery_relevant": "EV / Battery Relevant value"')
        elif f == "classification_method":
            item_fields.append('    "classification": "Classification Method from evidence"')
    item_schema = ",\n".join(item_fields)
    return (
        "{\n"
        '  "count": <integer — total number of matching companies>,\n'
        '  "items": [\n'
        "    {\n"
        f"{item_schema},\n"
        '      "evidence": ["E1"]  // evidence tag(s) supporting this row\n'
        "    }\n"
        "  ]\n"
        "}"
    )


def _build_extraction_schema_for_comparison(group_field: str) -> str:
    return (
        "{\n"
        f'  "group_by": "{group_field}",\n'
        '  "total_rows": <integer>,\n'
        '  "groups": [\n'
        "    {\n"
        '      "group_name": "...",\n'
        '      "count": <integer>,\n'
        '      "companies": ["company1", "company2"],\n'
        '      "total_employment": <integer or null if not available>,\n'
        '      "evidence": ["E1"]\n'
        "    }\n"
        "  ]\n"
        "}"
    )


def build_pass1_extraction_prompt(
    question: str,
    context: str,
    mode: str,
    fields: list[str] | None = None,
    group_field: str | None = None,
) -> str:
    """Pass 1: ask the LLM to extract strict JSON from the evidence.

    The LLM reads the context (with [E1] structured summary as authoritative)
    and returns only valid JSON matching the schema. No prose, no explanation.
    """
    if mode == "structured_list":
        schema = _build_extraction_schema_for_list(fields or ["company", "ev_supply_chain_role"])
        task = (
            "Extract every matching company from the evidence into the JSON schema below.\n"
            "The evidence is the COMPLETE and VERIFIED set — do not add or invent companies.\n"
            "For each company, extract ONLY the fields listed in the schema.\n"
            "CRITICAL: Each company must appear EXACTLY ONCE in the items array.\n"
            "Do NOT duplicate a company even if it appears under multiple roles.\n"
            "Only include companies whose role EXPLICITLY matches the question filter "
            "as stated in the evidence — do not infer or assume roles.\n"
            "Set 'count' to the exact number of items in your 'items' array.\n"
            "Return ONLY the JSON. No text before or after."
        )
    else:  # comparison
        gf = group_field or "category"
        schema = _build_extraction_schema_for_comparison(gf)
        task = (
            f"Group all companies from the evidence by '{gf}' into the JSON schema below.\n"
            "The evidence is the COMPLETE and VERIFIED set — do not add or invent companies.\n"
            "CRITICAL: Each company must appear in EXACTLY ONE group — "
            "the group matching its actual value in the evidence.\n"
            "Do NOT place the same company in multiple groups.\n"
            "For each group: count the companies, list their names, sum employment if available.\n"
            "Set 'total_rows' to the total number of companies across all groups.\n"
            "Return ONLY the JSON. No text before or after."
        )

    return (
        f"Evidence:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Task: {task}\n\n"
        f"JSON Schema to fill:\n{schema}\n\n"
        "JSON output:"
    )


def build_pass2_verbalization_prompt(
    question: str,
    extracted_json: dict[str, Any],
    mode: str,
) -> str:
    """Pass 2: ask the LLM to convert validated JSON into a natural language answer.

    The LLM receives only the validated JSON — no raw context. It cannot
    hallucinate companies because the set is already fixed.
    """
    json_str = json.dumps(extracted_json, indent=2)

    if mode == "structured_list":
        count = extracted_json.get("count", len(extracted_json.get("items", [])))
        return (
            f"The following verified data answers the question below.\n\n"
            f"Verified data (JSON):\n{json_str}\n\n"
            f"Question: {question}\n\n"
            "Instructions:\n"
            f"- Start with exactly: 'There are {count} matching "
            f"{'entry' if count == 1 else 'entries'}'\n"
            "- Then list every item from the JSON on its own line.\n"
            "- Use this format: CompanyName | Field: Value | Field: Value\n"
            "- Do NOT include any citation markers or evidence labels "
            "(such as [E1], [E2], [E12]) anywhere in your output.\n"
            "- If 'role' contains grouping information (e.g. Battery Pack vs Battery Cell), "
            "group items under bold headers.\n"
            "- Do not add, remove, or change any company or value.\n"
            "- Do not add commentary.\n\n"
            "Answer:"
        )
    else:  # comparison
        group_by = extracted_json.get("group_by", "category")
        total = extracted_json.get("total_rows", "?")
        return (
            f"The following verified grouped data answers the question below.\n\n"
            f"Verified data (JSON):\n{json_str}\n\n"
            f"Question: {question}\n\n"
            "Instructions:\n"
            f"- Total companies: {total}. Use this exact number.\n"
            f"- Present one paragraph per group, grouped by '{group_by}'.\n"
            "- Format: **GroupName** — Count: N, Total Employment: X\n"
            "  Companies: company1, company2, ...\n"
            "- Do NOT include any citation markers or evidence labels "
            "(such as [E1], [E2], [E12]) anywhere in your output.\n"
            "- Do not change any counts, totals, or company names.\n"
            "- End with a one-sentence summary of the result "
            "(e.g. which group is largest).\n\n"
            "Answer:"
        )


def _parse_json_from_llm_output(raw: str) -> dict[str, Any] | None:
    """Extract and parse JSON from LLM output, handling common formatting issues."""
    if not raw:
        return None
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def build_two_pass_answer(
    question: str,
    context: str,
    retrieved_chunks: list[RetrievalResult],
    generate_fn: Callable[[str, str, float, int], tuple[str, float, bool, str | None]],
    mode: str,
) -> tuple[str, float, bool, str | None]:
    """Run the 2-pass generation for structured_list and comparison questions.

    Pass 1: LLM extracts verified JSON from evidence (temperature=0)
    Pass 2: LLM verbalizes JSON into final answer (temperature=0)

    Falls back to single-pass freeform prompt if JSON extraction fails.

    Args:
        question:        the user question
        context:         pre-formatted evidence string (from format_context)
        retrieved_chunks: raw RetrievalResult list for field detection
        generate_fn:     callable(prompt, system_prompt, temperature, max_tokens)
                         → (answer, latency, success, error)
        mode:            'structured_list' or 'comparison'

    Returns:
        (final_answer, total_latency, success, error_message)
    """
    fields = _fields_requested(question)
    group_field = _detect_group_field(question) if mode == "comparison" else None

    # ── Build deterministic data block as Pass 1 input ────────────────────────
    # Using the deterministic renderer means Pass 1 sees ONLY the pre-filtered
    # structured_row_match rows (clean, no noise), not the full raw context.
    # This prevents the LLM from hallucinating companies or duplicating rows.
    python_row_count: int | None = None
    if retrieved_chunks:
        if mode == "structured_list":
            det_context, python_row_count = build_deterministic_list_context(
                question, retrieved_chunks
            )
        else:
            det_context = build_deterministic_comparison_context(question, retrieved_chunks)
        pass1_context = det_context
    else:
        pass1_context = context  # fallback to raw context if no chunks

    # ── Pass 1: extraction ────────────────────────────────────────────────────
    pass1_prompt = build_pass1_extraction_prompt(
        question, pass1_context, mode,
        fields=fields,
        group_field=group_field,
    )
    raw1, latency1, ok1, err1 = generate_fn(
        pass1_prompt,
        _EXTRACTION_SYSTEM_PROMPT,
        0.0,   # always deterministic for extraction
        2500,  # large lists (18+ companies) need room; JSON is verbose
    )

    def _direct_fallback(reason: str):
        row_count = python_row_count or 0
        fb = build_structured_list_prompt(question, pass1_context, row_count=row_count)
        ans, lat, ok, err = generate_fn(fb, RAG_SYSTEM_PROMPT, 0.0, 1500)
        return ans, latency1 + lat, ok, f"{reason}; direct prompt used"

    if not ok1:
        return _direct_fallback(f"pass1_failed: {err1}")

    extracted = _parse_json_from_llm_output(raw1)
    if extracted is None:
        return _direct_fallback("pass1_json_parse_failed")

    # ── Verify extraction completeness ────────────────────────────────────────
    # If extraction returned fewer items than Python found in metadata,
    # the LLM truncated or missed companies. Fall back to direct single-pass
    # rather than producing an answer with wrong count or missing entries.
    if mode == "structured_list":
        items = extracted.get("items", [])
        n_extracted = len(items)
        n_expected = python_row_count if python_row_count is not None else n_extracted
        if n_extracted < n_expected:
            return _direct_fallback(
                f"extraction_partial: got {n_extracted} of {n_expected} companies"
            )
        extracted["count"] = n_expected  # Python count is authoritative

    if mode == "comparison":
        groups = extracted.get("groups", [])
        python_total = sum(len(g.get("companies", [])) for g in groups)
        if python_total > 0:
            extracted["total_rows"] = python_total

    # ── Strip internal evidence fields before verbalization ───────────────────
    # Remove any "evidence" keys so citation markers can't leak into Pass 2 output.
    if mode == "structured_list":
        for item in extracted.get("items", []):
            item.pop("evidence", None)
    else:
        for group in extracted.get("groups", []):
            group.pop("evidence", None)

    # ── Pass 2: verbalization ─────────────────────────────────────────────────
    pass2_prompt = build_pass2_verbalization_prompt(question, extracted, mode)
    raw2, latency2, ok2, err2 = generate_fn(
        pass2_prompt,
        _VERBALIZATION_SYSTEM_PROMPT,
        0.0,
        1500,
    )
    total_latency = latency1 + latency2
    return raw2, total_latency, ok2, err2


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
    structured_match_companies: set[str] = set()
    note_used = False
    allow_repeated_company_rows = _needs_multi_record_context(normalized_question)
    # Pre-compute which companies already have a structured_row_match chunk
    # so we can skip lower-priority duplicates (row_full, company_profile)
    for result in ranked:
        if _chunk_type(result) == "structured_row_match":
            c = str(result.metadata.get("company", "")).strip()
            if c:
                structured_match_companies.add(c)

    for result in ranked:
        chunk_type = _chunk_type(result)
        row_key = str(result.metadata.get("row_key", "")).strip()
        company = str(result.metadata.get("company", "")).strip()
        if chunk_type == "note_reference" and not definition_question:
            continue
        # Skip row_full / company_profile duplicates when a structured_row_match
        # already covers this company — it's higher quality and more compact.
        if chunk_type in {"row_full", "company_profile"} and company in structured_match_companies:
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


def _is_exhaustive_listing_question(normalized_question: str) -> bool:
    """Return True for questions that ask for a complete list of companies/entries."""
    if any(
        term in normalized_question
        for term in {
            "list all", "show all", "identify all", "map all",
            "list every", "show every", "identify every", "find every",
            "full list of", "complete list of", "every company",
        }
    ):
        return True
    import re as _re
    if _re.search(r"\ball\s+\w+\s+(companies|suppliers|manufacturers|firms)\b", normalized_question):
        return True
    return False


def _effective_compact_result_limit(normalized_question: str, max_results: int) -> int:
    if _is_exhaustive_listing_question(normalized_question):
        return max(max_results, 30)
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
