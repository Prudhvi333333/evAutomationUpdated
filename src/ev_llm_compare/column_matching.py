"""Centralised, case-insensitive column matching helpers.

This keeps the heuristics for locating question, ID, answer, and key-facts
columns consistent across loaders and entry points.
"""

from __future__ import annotations

from typing import Iterable


QUESTION_CANDIDATES = {
    "question",
    "questions",
    "query",
    "prompt",
    "sample query",
}

QID_CANDIDATES = {"q_id", "id", "question id", "question_id", "#", "no", "num"}

KEY_FACTS_CANDIDATES = {"key_facts", "key facts", "facts"}

ANSWER_CANDIDATES = {
    "golden_answer",
    "golden answer",
    "reference_answer",
    "reference answer",
    "reference",
    "answer",
    "human validated answers",
    "human validated answer",
    "golden_summary",
    "golden summary",
    "expected answer",
    "expected_answer",
}


def _normalise(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def pick_column(columns: Iterable[str], candidates: set[str]) -> str | None:
    normalised = { _normalise(col): str(col) for col in columns }
    for candidate in candidates:
        if candidate in normalised:
            return normalised[candidate]
    return None

