from __future__ import annotations

from .schemas import RetrievalResult


SYSTEM_PROMPT = """You answer questions about one or more Excel workbooks.
Use the provided evidence when available. Prefer exact values, company names, counts,
locations, roles, and employment numbers from the source data.
When no workbook evidence is provided, answer from general knowledge as helpfully as possible.
Do not ask the user to re-upload the workbook unless the task explicitly requires exact workbook-only facts."""


def format_context(results: list[RetrievalResult]) -> str:
    if not results:
        return "No retrieved evidence."
    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        company = result.metadata.get("company") or "n/a"
        chunk_type = result.metadata.get("chunk_type") or "retrieved_chunk"
        source = f"{result.metadata.get('source_file')}::{result.metadata.get('sheet_name')}"
        row_number = result.metadata.get("row_number")
        header = (
            f"[Evidence {index}] type={chunk_type} company={company} source={source}"
            + (f" row={row_number}" if row_number else "")
        )
        blocks.append(
            f"{header}\n{result.text}"
        )
    return "\n\n".join(blocks)


def build_rag_prompt(question: str, context: str) -> str:
    return (
        "Answer the user question using only the retrieved workbook evidence.\n"
        "If a structured workbook match summary is present, treat it as the primary evidence.\n\n"
        f"Retrieved evidence:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- Be specific and concise.\n"
        "- When listing companies, preserve names exactly.\n"
        "- Include all supported matches from the evidence, not just one example.\n"
        "- If the evidence already groups companies by EV Supply Chain Role, copy that grouping directly.\n"
        "- Group results when the question asks for grouping.\n"
        "- Do not repeat evidence headers such as [Evidence 1].\n"
        "- Start directly with the answer.\n"
        "- Prefer EV Supply Chain Role over Product / Service when both appear.\n"
        "- Mention if the workbook evidence is incomplete for the question.\n"
        "- Do not invent values that are not in evidence.\n\n"
        "Answer:"
    )


def build_non_rag_prompt(question: str) -> str:
    return (
        "Answer the user question directly without relying on retrieved workbook evidence.\n"
        "Do not say that the workbook is missing or ask the user to provide it.\n"
        "If the question depends on exact workbook-specific facts, give your best general answer and mention uncertainty briefly.\n\n"
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
