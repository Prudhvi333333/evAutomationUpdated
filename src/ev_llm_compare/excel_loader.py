from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import pandas as pd

from .schemas import TableRow, WorkbookNote
from .column_matching import (
    pick_column,
    QUESTION_CANDIDATES,
    QID_CANDIDATES,
    KEY_FACTS_CANDIDATES,
    ANSWER_CANDIDATES,
)
from .column_matching import pick_column, QUESTION_CANDIDATES, QID_CANDIDATES, KEY_FACTS_CANDIDATES, ANSWER_CANDIDATES


# ──────────────────────────────────────────────────────────────────────────────
# EvalQuestion – carries a stable ID so golden-answer matching is ID-based,
# not fragile string-equality-based.
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EvalQuestion:
    """A single evaluation question with a stable numeric/string ID."""

    q_id: str          # e.g. "Q01", "1", "42"
    question: str      # normalised question text
    golden_answer: str | None = None   # populated after golden-answer join
    key_facts: Any = None              # optional structured key-facts column


US_STATE_ABBREVIATIONS: dict[str, str] = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "DC": "District of Columbia",
}
US_STATE_NAMES = tuple(sorted(US_STATE_ABBREVIATIONS.values(), key=len, reverse=True))
US_STATE_NAME_LOOKUP = {name.lower(): name for name in US_STATE_NAMES}
STATE_ABBREVIATION_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(US_STATE_ABBREVIATIONS)) + r")\b(?:\s+\d{5}(?:-\d{4})?)?"
)


def normalize_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_reference_answer(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        text = str(int(value))
    else:
        text = str(value)

    blocks: list[str] = []
    current: list[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        cleaned = re.sub(r"[ \t]+", " ", line).strip()
        if cleaned:
            current.append(cleaned)
            continue
        if current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return "\n\n".join(blocks).strip()


def _canonicalize_state_token(value: object) -> str:
    cleaned = normalize_cell(value).strip(" ,")
    if not cleaned:
        return ""
    if cleaned.upper() in US_STATE_ABBREVIATIONS:
        return US_STATE_ABBREVIATIONS[cleaned.upper()]
    return US_STATE_NAME_LOOKUP.get(cleaned.lower(), "")


def extract_state_from_text(value: object) -> str:
    cleaned = normalize_cell(value)
    if not cleaned:
        return ""

    direct = _canonicalize_state_token(cleaned)
    if direct:
        return direct

    lowered = cleaned.lower()
    for state_name in US_STATE_NAMES:
        if re.search(rf"\b{re.escape(state_name.lower())}\b", lowered):
            return state_name

    abbreviation_match = STATE_ABBREVIATION_PATTERN.search(cleaned)
    if abbreviation_match:
        return US_STATE_ABBREVIATIONS.get(abbreviation_match.group(1), "")

    return ""


def infer_state(values: dict[str, object]) -> str:
    for field_name in (
        "State",
        "Updated State",
        "Address",
        "Updated Location",
        "Location",
    ):
        inferred = extract_state_from_text(values.get(field_name))
        if inferred:
            return inferred
    return ""


def preferred_location(values: dict[str, object]) -> str:
    updated = normalize_cell(values.get("Updated Location"))
    if updated:
        return updated
    return normalize_cell(values.get("Location"))


def _is_tabular_sheet(df: pd.DataFrame) -> bool:
    populated_columns = [
        column for column in df.columns if normalize_cell(column) and not str(column).startswith("Unnamed")
    ]
    return len(populated_columns) >= 2 and len(df.index) > 0


def load_workbook(workbook_path: str | Path) -> tuple[list[TableRow], list[WorkbookNote]]:
    path = Path(workbook_path).expanduser().resolve()
    rows: list[TableRow] = []
    notes: list[WorkbookNote] = []

    with pd.ExcelFile(path) as excel_file:
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            df = df.dropna(how="all").dropna(axis=1, how="all")
            if df.empty:
                continue

            cleaned_columns = [normalize_cell(column) for column in df.columns]
            df.columns = cleaned_columns

            if _is_tabular_sheet(df):
                for row_idx, (_, series) in enumerate(df.iterrows(), start=1):
                    values = {
                        column: normalize_cell(value)
                        for column, value in series.to_dict().items()
                        if normalize_cell(value)
                    }
                    if values:
                        rows.append(
                            TableRow(
                                workbook_path=path,
                                sheet_name=sheet_name,
                                row_number=row_idx,
                                values=values,
                            )
                        )
            else:
                parts: list[str] = []
                for column in df.columns:
                    for value in df[column].tolist():
                        text = normalize_cell(value)
                        if text:
                            parts.append(text)
                if parts:
                    notes.append(
                        WorkbookNote(
                            workbook_path=path,
                            sheet_name=sheet_name,
                            text="\n".join(parts),
                        )
                    )

    if not rows and not notes:
        raise ValueError(f"No usable content found in workbook: {path}")

    return rows, notes


def load_questions(question_workbook: str | Path, sheet_name: str | None = None) -> list[str]:
    """Legacy loader — returns plain question strings (deduped, no IDs).

    Prefer ``load_eval_questions`` for new code so that stable IDs are
    preserved throughout the evaluation pipeline.
    """
    questions = load_eval_questions(question_workbook, sheet_name=sheet_name)
    return [q.question for q in questions]


# QID column name candidates (case-insensitive after normalisation)
_QID_CANDIDATES = QID_CANDIDATES
_QUESTION_CANDIDATES = QUESTION_CANDIDATES
_KEY_FACTS_CANDIDATES = KEY_FACTS_CANDIDATES


def load_eval_questions(
    question_workbook: str | Path,
    sheet_name: str | None = None,
    max_questions: int | None = None,
) -> list[EvalQuestion]:
    """Load evaluation questions with stable IDs from an Excel or CSV file.

    The workbook should have at minimum a 'question' column.
    An optional 'q_id' / 'id' column is used for stable ID assignment;
    if absent, row index (1-based) is used.

    Duplicate questions (same text) are preserved because each has its own
    stable ID. Call sites that need deduplication must do so explicitly.
    """
    path = Path(question_workbook).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name=sheet_name or 0)

    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        raise ValueError(f"No questions found in workbook: {path}")

    cleaned_columns = [normalize_cell(col) for col in df.columns]
    df.columns = cleaned_columns  # type: ignore[assignment]

    # Find question column
    q_col = _pick_column(cleaned_columns, _QUESTION_CANDIDATES)
    if q_col is None:
        raise ValueError(
            f"Could not find a question column in {path}. "
            f"Expected one of: {sorted(_QUESTION_CANDIDATES)}. "
            f"Found columns: {cleaned_columns}"
        )

    id_col = _pick_column(cleaned_columns, _QID_CANDIDATES)
    kf_col = _pick_column(cleaned_columns, _KEY_FACTS_CANDIDATES)

    questions: list[EvalQuestion] = []
    for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
        text = normalize_cell(row.get(q_col))
        if len(text) < 5:
            continue
        if id_col:
            raw_id = normalize_cell(row.get(id_col))
            q_id = raw_id if raw_id else f"Q{row_idx:02d}"
        else:
            q_id = f"Q{row_idx:02d}"

        key_facts = row.get(kf_col) if kf_col else None
        if isinstance(key_facts, float) and pd.isna(key_facts):
            key_facts = None

        questions.append(
            EvalQuestion(
                q_id=q_id,
                question=text,
                key_facts=key_facts,
            )
        )
        if max_questions is not None and len(questions) >= max_questions:
            break

    if not questions:
        raise ValueError(f"No valid questions found in workbook: {path}")

    return questions


def _pick_column(columns: list[str], candidates: set[str]) -> str | None:
    return pick_column(columns, candidates)


_ANSWER_CANDIDATES = ANSWER_CANDIDATES


def load_reference_answers(
    reference_workbook: str | Path,
    sheet_name: str | None = None,
) -> dict[str, str]:
    """Legacy loader — returns {question_text: answer}.

    Prefer ``load_golden_answers`` for new code so that ID-based joining
    is used instead of fragile exact-string matching on question text.
    """
    golden = load_golden_answers(reference_workbook, sheet_name=sheet_name)
    # Fall back to text-keyed dict for backward compatibility
    return {entry["question"]: entry["answer"] for entry in golden.values()}


def load_golden_answers(
    reference_workbook: str | Path,
    sheet_name: str | None = None,
) -> dict[str, dict[str, str]]:
    """Load golden answers keyed by stable question ID.

    Returns:
        dict[q_id, {"q_id": str, "question": str, "answer": str}]

    The workbook must have:
        - A 'question' column (or 'query' / 'prompt')
        - An 'answer' column (or 'golden_answer' / 'reference_answer' / etc.)

    If a 'q_id' / 'id' column is present it is used as the join key;
    otherwise row index (Q01, Q02, …) is used.

    Raises ValueError immediately if:
        - No question column is found
        - No answer column is found
        - Zero rows pass the minimum length filter
    """
    path = Path(reference_workbook).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name=sheet_name or 0)

    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        raise ValueError(f"No reference answers found in workbook: {path}")

    cleaned_columns = [normalize_cell(col) for col in df.columns]
    df.columns = cleaned_columns  # type: ignore[assignment]

    q_col = _pick_column(cleaned_columns, _QUESTION_CANDIDATES)
    if q_col is None:
        raise ValueError(
            f"No question column found in golden-answers workbook: {path}. "
            f"Expected one of: {sorted(_QUESTION_CANDIDATES)}."
        )
    ans_col = _pick_column(cleaned_columns, _ANSWER_CANDIDATES)
    if ans_col is None:
        if len(df.columns) >= 2:
            ans_col = df.columns[1]
        else:
            raise ValueError(
                f"No answer column found in golden-answers workbook: {path}. "
                f"Expected one of: {sorted(_ANSWER_CANDIDATES)}."
            )
    id_col = _pick_column(cleaned_columns, _QID_CANDIDATES)

    golden: dict[str, dict[str, str]] = {}
    for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
        question = normalize_cell(row.get(q_col))
        answer = normalize_reference_answer(row.get(ans_col))
        if len(question) < 5 or not answer:
            continue
        if id_col:
            raw_id = normalize_cell(row.get(id_col))
            q_id = raw_id if raw_id else f"Q{row_idx:02d}"
        else:
            q_id = f"Q{row_idx:02d}"
        golden[q_id] = {"q_id": q_id, "question": question, "answer": answer}

    if not golden:
        raise ValueError(
            f"No usable reference answers found in golden-answers workbook: {path}"
        )
    return golden


def join_golden_answers(
    questions: list[EvalQuestion],
    golden: dict[str, dict[str, str]],
) -> tuple[list[EvalQuestion], list[str]]:
    """Join golden answers onto EvalQuestion objects using stable IDs.

    Falls back to normalised question-text matching for questions whose ID
    is not found in the golden dict (handles mismatched ID columns).

    Returns:
        (enriched_questions, unmatched_q_ids)

    The caller should inspect ``unmatched_q_ids`` and decide whether to
    abort or continue with generated references.
    """
    # Build a secondary index by normalised question text for fallback
    text_index: dict[str, dict[str, str]] = {
        _norm_text(entry["question"]): entry for entry in golden.values()
    }

    enriched: list[EvalQuestion] = []
    unmatched: list[str] = []

    for q in questions:
        # Primary: ID-based match
        match = golden.get(q.q_id)
        # Secondary: text-based match (catches minor formatting differences)
        if match is None:
            match = text_index.get(_norm_text(q.question))
        if match is not None:
            enriched.append(
                EvalQuestion(
                    q_id=q.q_id,
                    question=q.question,
                    golden_answer=match["answer"],
                    key_facts=q.key_facts,
                )
            )
        else:
            enriched.append(q)
            unmatched.append(q.q_id)

    return enriched, unmatched


def _norm_text(text: str) -> str:
    """Collapse whitespace and lower-case for loose text matching."""
    return re.sub(r"\s+", " ", text).strip().lower()
