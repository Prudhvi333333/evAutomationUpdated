from __future__ import annotations

from .schemas import RetrievalResult


SYSTEM_PROMPT = """You answer questions about one or more Excel workbooks.
Use the provided evidence when available. Prefer exact values, company names, counts,
locations, roles, and employment numbers from the source data.
If the answer is not supported by the evidence, say that clearly instead of guessing."""


def format_context(results: list[RetrievalResult]) -> str:
    if not results:
        return "No retrieved evidence."
    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        company = result.metadata.get("company") or "n/a"
        source = f"{result.metadata.get('source_file')}::{result.metadata.get('sheet_name')}"
        blocks.append(
            f"[Chunk {index}] company={company} source={source} row={result.metadata.get('row_number', 'n/a')}\n"
            f"{result.text}"
        )
    return "\n\n".join(blocks)


def build_rag_prompt(question: str, context: str) -> str:
    return (
        "Answer the user question using only the retrieved workbook evidence.\n\n"
        f"Retrieved evidence:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- Be specific and concise.\n"
        "- When listing companies, preserve names exactly.\n"
        "- Mention if the workbook evidence is incomplete for the question.\n"
        "- Do not invent values that are not in evidence.\n\n"
        "Answer:"
    )


def build_non_rag_prompt(question: str) -> str:
    return (
        "Answer the user question directly. If you are uncertain, say so.\n\n"
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
